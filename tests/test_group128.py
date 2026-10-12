# Copyright 2026 Jakevin
# SPDX-License-Identifier: Apache-2.0
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_ternary import TernaryLinear, dequantize, from_mlx_2bit, pack_trits, tern_matmul


def _case(n, k, group, seed):
    rng = np.random.default_rng(seed)
    trits = rng.integers(-1, 2, size=(n, k), dtype=np.int8)
    # Half-exact scales so the half-staged GEMM is compared without rounding noise.
    scales = (rng.integers(1, 64, size=(n, k // group)) / 512.0).astype(np.float32)
    packed, _ = pack_trits(trits)
    return trits, scales, packed


def _check(y, ref, tol=1e-3):
    y = np.array(y)
    err = np.max(np.abs(y - ref)) / max(np.max(np.abs(ref)), 1e-12)
    assert err < tol, err


@pytest.mark.parametrize("group", [64, 128])
@pytest.mark.parametrize(
    "n,k,b",
    [(64, 640, 1), (96, 2048, 3), (128, 512, 32), (64, 1024, 64), (64, 640, 100), (40, 512, 100), (32, 384, 37)],
)
def test_matmul_groups(group, n, k, b):
    if k % group:
        pytest.skip("K not a multiple of group")
    trits, scales, packed = _case(n, k, group, n + k + b + group)
    x = np.random.default_rng(b).standard_normal((b, k)).astype(np.float32)
    ref = x @ dequantize(trits, scales, group).T
    y = tern_matmul(mx.array(x), mx.array(packed), mx.array(scales), k, group)
    _check(y, ref)


def test_group128_differs_from_64_layout():
    # Same codes, scales that differ between the two halves of a 128 group.
    n, k = 32, 512
    trits, s128, packed = _case(n, k, 128, 7)
    x = np.random.default_rng(0).standard_normal((1, k)).astype(np.float32)
    ref = x @ dequantize(trits, s128, 128).T
    _check(tern_matmul(mx.array(x), mx.array(packed), mx.array(s128), k, group=128), ref)
    with pytest.raises(ValueError):
        tern_matmul(mx.array(x), mx.array(packed), mx.array(s128), k)  # g64 shape mismatch


def test_bad_group():
    with pytest.raises(ValueError):
        tern_matmul(mx.zeros((1, 256)), mx.zeros((4, 64), dtype=mx.uint8), mx.zeros((4, 8)), 256, group=32)
    TernaryLinear(np.zeros((4, 32), np.uint8), np.zeros((4, 1), np.float32), group_size=128)
    with pytest.raises(ValueError):
        TernaryLinear(np.zeros((4, 16), np.uint8), np.zeros((4, 1), np.float32), group_size=128)


def test_linear_and_convert_group128():
    n, k = 64, 1024
    trits, scales, _ = _case(n, k, 128, 3)
    w = mx.array(dequantize(trits, scales, 128))
    stored = (trits.astype(np.uint32) + 1).reshape(n, k // 16, 16)
    q = mx.array(np.sum(stored << (2 * np.arange(16, dtype=np.uint32)), axis=-1).astype(np.uint32))
    qs, qb = mx.array(scales), mx.array(-scales)
    layer = nn.QuantizedLinear(k, n, bias=False, group_size=128, bits=2, mode="affine")
    layer.weight, layer.scales, layer.biases = q, qs, qb
    tl = TernaryLinear.from_quantized_linear(layer)
    assert tl.group_size == 128
    x = mx.array(np.random.default_rng(1).standard_normal((5, k)).astype(np.float32))
    _check(tl(x), np.array(layer(x)))
    packed, sf = from_mlx_2bit(q, qs, qb, group_size=128)
    assert sf.shape == (n, k // 128)
    tl2 = TernaryLinear.from_trits(trits, scales, group_size=128)
    _check(tl2(x), np.array(x) @ np.array(w).T)


@pytest.mark.parametrize("b", [1, 4, 64])
def test_new_weights_after_delete(b):
    # A same-shape weight created after the old one is freed must be used,
    # including through the compiled small-batch path.
    n, k = 64, 512
    x = np.random.default_rng(5).standard_normal((b, k)).astype(np.float32)
    t1, s1, p1 = _case(n, k, 64, 11)
    layer = TernaryLinear(p1, s1)
    _check(layer(mx.array(x)), x @ dequantize(t1, s1).T)
    del layer, p1, s1
    mx.clear_cache()
    t2, s2, p2 = _case(n, k, 64, 12)
    assert not np.array_equal(t1, t2)
    layer = TernaryLinear(p2, s2)
    ref2 = x @ dequantize(t2, s2).T
    _check(layer(mx.array(x)), ref2)
    _check(tern_matmul(mx.array(x), mx.array(p2), mx.array(s2), k), ref2)
