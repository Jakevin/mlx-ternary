# Copyright 2026 Jakevin
# SPDX-License-Identifier: Apache-2.0
import mlx.core as mx
import numpy as np
import pytest

from mlx_ternary import dequantize, pack_trits, tern_matmul

# Shapes probed with the v3 kernel. The published maximum was 1.44e-6.
CASES = [
    (8, 64, 1),
    (16, 2048, 1),
    (16, 2048, 128),
    (40, 128, 3),
    (64, 256, 64),
    (64, 2048, 32),
    (128, 192, 32),
    (64, 6144, 32),
    (41, 512, 1),
    (41, 512, 7),
    (640, 2560, 1),
    (640, 2560, 128),
    (2560, 640, 1),
    (2560, 640, 128),
    (2048, 6144, 1),
]


def _rel(y, ref) -> float:
    err = mx.abs(y - ref)
    mx.eval(err)
    denom = float(mx.max(mx.abs(ref)).item())
    return float(mx.max(err).item()) / max(denom, 1e-8)


def _layer(n, k, trits, scales):
    packed, _row = pack_trits(trits)
    return mx.array(packed), mx.array(np.asarray(scales, dtype=np.float32)), mx.array(dequantize(trits, scales))


def test_exact_ones_cover_qmv_and_gemm():
    # Tail GEMV (K=64), 512-wide GEMV, 640 = 512+128, and the 32x32 GEMM.
    specs = [
        (2, 64, 1, 0.5, 32.0),
        (1, 512, 1, 0.25, 128.0),
        (4, 640, 1, 1.0, 640.0),
        (32, 64, 32, 0.5, 32.0),
        (32, 128, 32, 0.25, 32.0),
    ]
    for n, k, batch, scale, expect in specs:
        trits = np.ones((n, k), dtype=np.int8)
        scales = np.full((n, k // 64), scale, dtype=np.float32)
        codes, scale_mx, _ref = _layer(n, k, trits, scales)
        x = mx.ones((batch, k), dtype=mx.float32)
        y = tern_matmul(x, codes, scale_mx, k)
        mx.eval(y)
        assert y.shape == (batch, n)
        assert float(mx.max(mx.abs(y - expect)).item()) == pytest.approx(0.0, abs=1e-4)


def test_negative_and_zero_trits():
    n, k = 2, 64
    trits = np.zeros((n, k), dtype=np.int8)
    trits[0] = -1
    scales = np.ones((n, 1), dtype=np.float32)
    codes, scale_mx, _ref = _layer(n, k, trits, scales)
    x = mx.ones((1, k), dtype=mx.float32)
    y = tern_matmul(x, codes, scale_mx, k)
    mx.eval(y)
    assert float(y[0, 0].item()) == pytest.approx(-64.0, abs=1e-4)
    assert float(y[0, 1].item()) == pytest.approx(0.0, abs=1e-4)


def test_vector_input_is_one_row():
    trits = np.ones((3, 64), dtype=np.int8)
    scales = np.ones((3, 1), dtype=np.float32)
    codes, scale_mx, _ref = _layer(3, 64, trits, scales)
    y = tern_matmul(mx.ones((64,), dtype=mx.float32), codes, scale_mx, 64)
    mx.eval(y)
    assert y.shape == (1, 3)


def test_random_shapes_under_1e_3():
    rng = np.random.default_rng(7)
    worst = 0.0
    for n, k, batch in CASES:
        trits = rng.integers(-1, 2, size=(n, k), dtype=np.int8)
        scales = rng.uniform(0.02, 0.2, size=(n, k // 64)).astype(np.float32)
        codes, scale_mx, ref_w = _layer(n, k, trits, scales)
        x = mx.array(rng.standard_normal((batch, k)).astype(np.float32))
        y = tern_matmul(x, codes, scale_mx, k)
        ref = x @ ref_w.T
        mx.eval(y, ref)
        rel = _rel(y, ref)
        worst = max(worst, rel)
        assert np.isfinite(rel) and rel < 1e-3, f"{n}x{k} B={batch} rel={rel:.3e}"
        del codes, scale_mx, ref_w, x, y, ref
        mx.clear_cache()
    assert worst < 1e-3


def test_rejects_bad_shapes():
    trits = np.ones((2, 64), dtype=np.int8)
    codes, scale_mx, _ref = _layer(2, 64, trits, np.ones((2, 1), np.float32))
    with pytest.raises(ValueError):
        tern_matmul(mx.ones((2, 3, 64)), codes, scale_mx, 64)
    with pytest.raises(ValueError):
        tern_matmul(mx.ones((1, 32)), codes, scale_mx, 32)
