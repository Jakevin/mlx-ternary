# Copyright 2026 Jakevin
# SPDX-License-Identifier: Apache-2.0
"""Convert ternary weights stored as MLX affine 2-bit.

MLX affine 2-bit keeps 16 codes in each uint32, lane ``i`` in bits ``[2i, 2i+2)``.
Ternary storage uses codes ``{0, 1, 2}`` and a per-group bias equal to ``-scale``,
so ``mx.dequantize`` returns ``(code - 1) * scale``. This module checks that
contract and repacks the codes into the 4-per-byte kernel layout.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

from .pack import GROUP, GROUPS, pack_trits

_LANES = 16


def _as_numpy(value, mlx_dtype, np_dtype) -> np.ndarray:
    if isinstance(value, mx.array):
        if value.dtype != mlx_dtype:
            value = value.astype(mlx_dtype)
        return np.array(value)
    arr = np.asarray(value)
    if arr.dtype != np_dtype:
        arr = arr.astype(np_dtype, copy=False)
    return arr


def unpack_mlx_2bit(words: np.ndarray, k: int) -> np.ndarray:
    """Unpack MLX affine 2-bit words to integer codes of length ``k``."""
    if k % _LANES:
        raise ValueError(f"K {k} is not a multiple of {_LANES}")
    view = np.ascontiguousarray(words, dtype=np.uint32)
    if int(view.shape[-1]) != k // _LANES:
        raise ValueError(f"uint32 last dim {view.shape[-1]} != K/16 {k // _LANES}")
    out = np.empty(view.shape[:-1] + (k,), dtype=np.int16)
    for lane in range(_LANES):
        out[..., lane::_LANES] = (view >> np.uint32(2 * lane)) & np.uint32(3)
    return out


def from_mlx_2bit(weight, scales, biases, group_size: int = GROUP):
    """Repack MLX affine 2-bit ternary weights.

    ``weight`` is uint32 ``[N, K/16]``. ``scales`` and ``biases`` are
    ``[N, K/group_size]`` with ``group_size`` 64 (default) or 128. Codes must lie in ``{0, 1, 2}`` and ``biases``
    must equal ``-scales``. Returns uint8 codes ``[N, K/4]`` and float32 scales.
    """
    if group_size not in GROUPS:
        raise ValueError(f"group_size {group_size} not in {GROUPS}")
    words = _as_numpy(weight, mx.uint32, np.uint32)
    scale_f = np.ascontiguousarray(_as_numpy(scales, mx.float32, np.float32), dtype=np.float32)
    bias_f = np.ascontiguousarray(_as_numpy(biases, mx.float32, np.float32), dtype=np.float32)
    if words.ndim != 2 or scale_f.ndim != 2:
        raise ValueError("expected weight [N, K/16] and scales [N, K/group_size]")
    if bias_f.shape != scale_f.shape:
        raise ValueError(f"biases shape {bias_f.shape} != scales shape {scale_f.shape}")
    n, nwords = (int(v) for v in words.shape)
    k = nwords * _LANES
    groups = k // group_size
    if k % group_size or scale_f.shape != (n, groups):
        raise ValueError(
            f"shapes weight {words.shape} scales {scale_f.shape} are not group-{group_size} ternary"
        )
    stored = unpack_mlx_2bit(words, k)
    if stored.size and (int(stored.min()) < 0 or int(stored.max()) > 2):
        raise ValueError("MLX 2-bit codes are outside {0, 1, 2}")
    if not np.array_equal(bias_f, -scale_f):
        raise ValueError("MLX 2-bit biases are not -scales")
    packed, _row = pack_trits((stored - 1).astype(np.int8))
    return packed, scale_f
