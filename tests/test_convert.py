# Copyright 2026 Jakevin
# SPDX-License-Identifier: Apache-2.0
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_ternary import TernaryLinear, from_mlx_2bit, pack_trits, unpack_mlx_2bit, unpack_trits


def _words(stored: np.ndarray) -> np.ndarray:
    lanes = 16
    grouped = stored.astype(np.uint32).reshape(*stored.shape[:-1], stored.shape[-1] // lanes, lanes)
    shifts = (np.arange(lanes, dtype=np.uint32) * np.uint32(2))
    return np.sum(grouped << shifts, axis=-1).astype(np.uint32)


def test_layout_matches_mx_quantize():
    mx.random.seed(1)
    weight = mx.random.normal(shape=(4, 128)).astype(mx.float32)
    packed, scales, biases = mx.quantize(weight, group_size=64, bits=2, mode="affine")
    mx.eval(packed, scales, biases)
    words = np.array(packed)
    codes = unpack_mlx_2bit(words, 128)
    assert np.array_equal(_words(codes.astype(np.int32)), words)
    manual = (codes.reshape(4, 2, 64) * np.array(scales)[:, :, None] + np.array(biases)[:, :, None]).reshape(4, 128)
    got = np.array(mx.dequantize(packed, scales, biases, group_size=64, bits=2, mode="affine"))
    assert np.max(np.abs(manual - got)) < 1e-5


def test_from_mlx_2bit_roundtrip():
    rng = np.random.default_rng(2)
    n, k = 6, 128
    trits = rng.integers(-1, 2, size=(n, k), dtype=np.int8)
    scales = rng.uniform(0.05, 0.5, size=(n, k // 64)).astype(np.float32)
    words = _words(trits.astype(np.int32) + 1)
    packed, scale_f = from_mlx_2bit(mx.array(words), mx.array(scales).astype(mx.bfloat16), mx.array(-scales).astype(mx.bfloat16))
    assert np.array_equal(unpack_trits(packed, k), trits)
    direct, _row = pack_trits(trits)
    assert np.array_equal(packed, direct)
    # bf16 widened back to the exact ternary scale the kernel multiplies.
    assert scale_f.dtype == np.float32
    deq = mx.dequantize(
        mx.array(words),
        mx.array(scales).astype(mx.bfloat16),
        mx.array(-scales).astype(mx.bfloat16),
        group_size=64,
        bits=2,
        mode="affine",
    )
    ref = trits.reshape(n, k // 64, 64).astype(np.float32) * scale_f.reshape(n, k // 64, 1)
    assert np.max(np.abs(np.array(deq.astype(mx.float32)) - ref.reshape(n, k))) < 1e-6


def test_rejects_code_3_and_bad_bias():
    stored = np.zeros((1, 64), dtype=np.int32)
    stored[0, 0] = 3
    words = _words(stored)
    scales = np.ones((1, 1), dtype=np.float32)
    with pytest.raises(ValueError):
        from_mlx_2bit(words, scales, -scales)
    stored[0, 0] = 1
    words = _words(stored)
    with pytest.raises(ValueError):
        from_mlx_2bit(words, scales, scales)


def test_quantized_linear_must_be_ternary():
    layer = nn.QuantizedLinear(64, 32, bias=False, group_size=64, bits=2, mode="affine")
    with pytest.raises(ValueError):
        TernaryLinear.from_quantized_linear(layer)
    wide = nn.QuantizedLinear(64, 32, bias=False, group_size=64, bits=4, mode="affine")
    with pytest.raises(ValueError):
        TernaryLinear.from_quantized_linear(wide)
