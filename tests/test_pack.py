# Copyright 2026 Jakevin
# SPDX-License-Identifier: Apache-2.0
import math

import numpy as np
import pytest

from mlx_ternary import dequantize, info_bits_per_weight, pack_trits, stored_bits_per_weight, unpack_trits


def test_bits_about_1_85():
    bits = info_bits_per_weight()
    assert bits == pytest.approx(math.log2(3) + 16 / 64)
    assert 1.83 < bits < 1.85
    assert stored_bits_per_weight() == pytest.approx(2.25)


def test_roundtrip():
    rng = np.random.default_rng(0)
    for k in (4, 8, 64, 640, 2048):
        codes = rng.integers(-1, 2, size=(5, k), dtype=np.int8)
        packed, row = pack_trits(codes)
        assert row == k // 4
        assert packed.dtype == np.uint8
        assert packed.shape == (5, row)
        assert np.array_equal(unpack_trits(packed, k), codes)
        # Low bits of the first byte are the first trit + 1.
        assert int(packed.reshape(-1)[0] & 3) == int(codes.reshape(-1)[0] + 1)


def test_rejects_bad_codes():
    with pytest.raises(ValueError):
        pack_trits(np.array([[0, 1, 2]], dtype=np.int8))
    with pytest.raises(ValueError):
        pack_trits(np.array([[-2, 0, 1, 1]], dtype=np.int8))
    with pytest.raises(ValueError):
        pack_trits(np.array([[0.5, 0, 0, 0]], dtype=np.float32))


def test_rejects_code_3():
    packed = np.array([[0x03]], dtype=np.uint8)
    with pytest.raises(ValueError):
        unpack_trits(packed, 4)


def test_dequantize_groups():
    trits = np.array([[1, 0, -1, 1] * 32], dtype=np.int8)
    scales = np.array([[0.5, -2.0]], dtype=np.float32)
    out = dequantize(trits, scales)
    assert out.shape == trits.shape
    assert out[0, 0] == pytest.approx(0.5)
    assert out[0, 1] == pytest.approx(0.0)
    assert out[0, 63] == pytest.approx(0.5)
    assert out[0, 64] == pytest.approx(-2.0)
    assert out[0, 66] == pytest.approx(2.0)
