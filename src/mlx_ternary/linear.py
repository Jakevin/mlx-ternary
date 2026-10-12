# Copyright 2026 Jakevin
# SPDX-License-Identifier: Apache-2.0
"""A linear layer that runs the ternary kernel.

``TernaryLinear`` follows ``nn.Linear`` on the forward: ``y = x W^T + b``,
with ``W`` stored packed. It also replaces an ``nn.QuantizedLinear`` whose
affine 2-bit parameters are ternary (codes in {0, 1, 2}, bias = -scale).
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .convert import from_mlx_2bit
from .kernel import tern_matmul
from .pack import GROUP, pack_trits


class TernaryLinear(nn.Module):
    """Packed ternary affine layer.

    ``weight`` is uint8 ``[output_dims, input_dims / 4]``. ``scales`` is
    float32 ``[output_dims, input_dims / 64]``. Parameters are frozen.
    """

    def __init__(self, weight, scales, bias=None):
        super().__init__()
        if not isinstance(weight, mx.array):
            weight = mx.array(np.ascontiguousarray(weight))
        if weight.dtype != mx.uint8:
            weight = weight.astype(mx.uint8)
        if not isinstance(scales, mx.array):
            scales = mx.array(np.asarray(scales, dtype=np.float32))
        if scales.dtype != mx.float32:
            scales = scales.astype(mx.float32)
        weight = mx.contiguous(weight)
        scales = mx.contiguous(scales)
        if weight.ndim != 2 or scales.ndim != 2:
            raise ValueError("weight and scales must be matrices")
        n, row_bytes = (int(v) for v in weight.shape)
        k = row_bytes * 4
        if k % GROUP or scales.shape != (n, k // GROUP):
            raise ValueError(
                f"weight {tuple(weight.shape)} and scales {tuple(scales.shape)} "
                f"are not a group-{GROUP} ternary matrix"
            )
        self.weight = weight
        self.scales = scales
        self.input_dims = k
        self.output_dims = n
        if bias is not None:
            if not isinstance(bias, mx.array):
                bias = mx.array(bias)
            if tuple(bias.shape) != (n,):
                raise ValueError(f"bias shape {tuple(bias.shape)} != ({n},)")
            self.bias = bias
        self.freeze()

    def _extra_repr(self) -> str:
        return (
            f"input_dims={self.input_dims}, output_dims={self.output_dims}, "
            f"bias={'bias' in self}, group_size={GROUP}, bits=2"
        )

    def __call__(self, x):
        if int(x.shape[-1]) != self.input_dims:
            raise ValueError(f"x last dim {x.shape[-1]} != input_dims {self.input_dims}")
        lead = x.shape[:-1]
        flat = x.reshape(-1, self.input_dims)
        y = tern_matmul(flat, self.weight, self.scales, self.input_dims)
        if y.dtype != x.dtype:
            y = y.astype(x.dtype)
        if "bias" in self:
            b = self.bias
            if b.dtype != y.dtype:
                b = b.astype(y.dtype)
            y = y + b
        return y.reshape(*lead, self.output_dims)

    @classmethod
    def from_trits(cls, trits, scales, bias=None) -> "TernaryLinear":
        """Build from int8 trits ``{-1,0,1}`` of shape ``[output, input]``."""
        packed, _row = pack_trits(np.asarray(trits))
        return cls(packed, scales, bias)

    @classmethod
    def from_mlx_2bit(cls, weight, scales, biases, bias=None) -> "TernaryLinear":
        """Build from MLX affine 2-bit ternary tensors."""
        packed, scale_f = from_mlx_2bit(weight, scales, biases)
        return cls(packed, scale_f, bias)

    @classmethod
    def from_quantized_linear(cls, layer: nn.Module) -> "TernaryLinear":
        """Replace one affine 2-bit ``QuantizedLinear`` that stores ternary weights."""
        if not isinstance(layer, nn.QuantizedLinear):
            raise TypeError(f"expected QuantizedLinear, got {type(layer).__name__}")
        if layer.bits != 2 or layer.group_size != GROUP or layer.mode != "affine":
            raise ValueError(
                f"expected affine 2-bit group {GROUP}, got bits={layer.bits} "
                f"group_size={layer.group_size} mode={layer.mode}"
            )
        if layer.biases is None:
            raise ValueError("affine 2-bit ternary storage needs biases equal to -scales")
        extra = layer.bias if "bias" in layer else None
        return cls.from_mlx_2bit(layer.weight, layer.scales, layer.biases, bias=extra)


def _replace_linear(value, path: str):
    if isinstance(value, nn.QuantizedLinear):
        try:
            return TernaryLinear.from_quantized_linear(value), 1
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
    if isinstance(value, list):
        n = 0
        for i, item in enumerate(list(value)):
            new, k = _replace_linear(item, f"{path}.{i}")
            if k:
                value[i] = new
                n += k
        return value, n
    if isinstance(value, dict) and not isinstance(value, nn.Module):
        n = 0
        for key, item in list(value.items()):
            new, k = _replace_linear(item, f"{path}.{key}")
            if k:
                value[key] = new
                n += k
        return value, n
    return value, 0


def replace_quantized_linears(model: nn.Module) -> int:
    """Swap ternary affine 2-bit ``QuantizedLinear`` modules inside ``model``.

    The model object is updated in place. A ``QuantizedLinear`` that is not
    ternary raises. Returns how many layers were replaced.
    """

    replaced = 0

    def visit(path, mod):
        nonlocal replaced
        for key, value in list(mod.items()):
            where = f"{path}.{key}" if path else str(key)
            new, n = _replace_linear(value, where)
            if n and new is not value:
                mod[key] = new
            replaced += n

    model.apply_to_modules(visit)
    return replaced
