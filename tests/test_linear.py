# Copyright 2026 Jakevin
# SPDX-License-Identifier: Apache-2.0
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_ternary import TernaryLinear, dequantize, replace_quantized_linears


def _words(trits: np.ndarray) -> np.ndarray:
    stored = trits.astype(np.int32) + 1
    lanes = 16
    grouped = stored.astype(np.uint32).reshape(stored.shape[0], stored.shape[1] // lanes, lanes)
    shifts = (np.arange(lanes, dtype=np.uint32) * np.uint32(2))
    return np.sum(grouped << shifts, axis=-1).astype(np.uint32)


def _quantized(n, k, seed, bias=False):
    rng = np.random.default_rng(seed)
    trits = rng.integers(-1, 2, size=(n, k), dtype=np.int8)
    scales = rng.uniform(0.05, 0.3, size=(n, k // 64)).astype(np.float32)
    layer = nn.QuantizedLinear(k, n, bias=bias, group_size=64, bits=2, mode="affine")
    layer.weight = mx.array(_words(trits))
    layer.scales = mx.array(scales)
    layer.biases = mx.array(-scales)
    extra = None
    if bias:
        extra = mx.array(rng.standard_normal(n).astype(np.float32) * 0.1)
        layer.bias = extra
    return layer, trits, scales, extra


def test_readme_load_example():
    n, k = 64, 256
    trits = np.random.default_rng(0).integers(-1, 2, size=(n, k), dtype=np.int8)
    scales = mx.ones((n, k // 64))
    layer = TernaryLinear.from_trits(trits, scales)
    x = mx.random.normal(shape=(1, k))
    y = layer(x)
    mx.eval(y)
    assert y.shape == (1, n)


def test_matches_dequant_and_mlx_2bit():
    layer_q, trits, scales, extra = _quantized(32, 128, 3, bias=True)
    layer = TernaryLinear.from_quantized_linear(layer_q)
    x = mx.random.normal(shape=(2, 5, 128))
    y = layer(x)
    ref_w = mx.array(dequantize(trits, scales))
    ref = x @ ref_w.T + extra
    q = mx.quantized_matmul(
        x.reshape(-1, 128),
        layer_q.weight,
        scales=layer_q.scales,
        biases=layer_q.biases,
        transpose=True,
        group_size=64,
        bits=2,
        mode="affine",
    ).reshape(2, 5, 32) + extra
    mx.eval(y, ref, q)
    denom = float(mx.max(mx.abs(ref)).item())
    assert float(mx.max(mx.abs(y - ref)).item()) / denom < 1e-3
    assert float(mx.max(mx.abs(y - q)).item()) / denom < 1e-3


def test_replace_nested_and_list():
    direct, trits, scales, _extra = _quantized(16, 64, 4)
    a, _, _, _ = _quantized(16, 64, 5)
    b, _, _, _ = _quantized(16, 64, 6)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = direct
            self.linears = [a, b]

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = Block()

    model = Model()
    assert replace_quantized_linears(model) == 3
    assert isinstance(model.block.proj, TernaryLinear)
    assert all(isinstance(item, TernaryLinear) for item in model.block.linears)
    x = mx.ones((4, 64))
    y = model.block.proj(x)
    ref = x @ mx.array(dequantize(trits, scales)).T
    mx.eval(y, ref)
    assert float(mx.max(mx.abs(y - ref)).item()) == pytest.approx(0.0, abs=1e-3)


def test_set_cache_limit_is_opt_in():
    import mlx_ternary

    mlx_ternary.set_cache_limit(2)
    assert mlx_ternary.set_cache_limit(2) == 2 * 1024**3
    assert mlx_ternary.configure() == 2 * 1024**3
