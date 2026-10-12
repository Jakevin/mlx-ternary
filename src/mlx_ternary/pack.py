# Copyright 2026 Jakevin
# SPDX-License-Identifier: Apache-2.0
"""2-bit ternary pack format.

Four codes share a byte, low bits first. The stored code is ``trit + 1``
in {0, 1, 2}. One scale covers 64 weights (128 as an option) and there is no zero point:

    weight = (code - 1) * scale
"""
from __future__ import annotations

import math

import numpy as np

GROUP = 64
# Supported group sizes. 128 halves the scale overhead (2.125 bits stored).
GROUPS = (64, 128)


def info_bits_per_weight(group_size: int = GROUP) -> float:
    """``log2(3)`` plus one 16-bit scale per group.

    At group 64 this is about 1.85 bits.
    """
    if group_size <= 0:
        raise ValueError(f"group_size {group_size} must be positive")
    return math.log2(3.0) + 16.0 / group_size


def stored_bits_per_weight(group_size: int = GROUP) -> float:
    """Two-bit slot plus one 16-bit scale per group."""
    if group_size <= 0:
        raise ValueError(f"group_size {group_size} must be positive")
    return 2.0 + 16.0 / group_size


def pack_trits(codes: np.ndarray) -> tuple[np.ndarray, int]:
    """Pack ``{-1,0,1}`` as 2-bit codes ``{0,1,2}``.

    The last axis must be a multiple of 4. Slot 0 sits in bits 0-1, slot 1
    in bits 2-3, slot 2 in bits 4-5, slot 3 in bits 6-7. Returns the uint8
    array with that axis replaced by ``K/4``, and ``K/4``.
    """
    raw = np.asarray(codes)
    flat_shape = raw.shape[:-1]
    k = int(raw.shape[-1])
    if k % 4:
        raise ValueError(f"K {k} is not a multiple of 4")
    if raw.size == 0:
        return np.zeros((*flat_shape, k // 4), dtype=np.uint8), k // 4
    if np.issubdtype(raw.dtype, np.floating) and not np.all(raw == np.trunc(raw)):
        raise ValueError("codes are not ternary {-1,0,1}")
    c = raw.reshape(-1, k).astype(np.int16) + 1
    if int(c.min()) < 0 or int(c.max()) > 2:
        raise ValueError("codes are not ternary {-1,0,1}")
    c = c.reshape(-1, 4)
    packed = (c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6)).astype(np.uint8)
    row_bytes = k // 4
    return np.ascontiguousarray(packed.reshape(*flat_shape, row_bytes)), row_bytes


def unpack_trits(packed: np.ndarray, k: int) -> np.ndarray:
    """Inverse of :func:`pack_trits`. Returns int8 trits in ``{-1,0,1}``."""
    if k % 4:
        raise ValueError(f"K {k} is not a multiple of 4")
    raw = np.ascontiguousarray(packed)
    row_bytes = k // 4
    if raw.shape[-1] != row_bytes:
        raise ValueError(f"packed last dim {raw.shape[-1]} != K/4 {row_bytes}")
    if raw.size == 0:
        return np.zeros((*raw.shape[:-1], k), dtype=np.int8)
    b = raw.reshape(-1).astype(np.uint8, copy=False)
    slots = np.empty(b.shape[0] * 4, dtype=np.int16)
    slots[0::4] = b & np.uint8(3)
    slots[1::4] = (b >> np.uint8(2)) & np.uint8(3)
    slots[2::4] = (b >> np.uint8(4)) & np.uint8(3)
    slots[3::4] = (b >> np.uint8(6)) & np.uint8(3)
    if int(slots.max()) > 2:
        raise ValueError("packed byte holds code 3, which is not a ternary code")
    trits = (slots - 1).astype(np.int8)
    return trits.reshape(raw.shape[:-1] + (k,))


def dequantize(trits: np.ndarray, scales: np.ndarray, group_size: int = GROUP) -> np.ndarray:
    """Expand trits and per-group scales to a float32 matrix."""
    w = np.asarray(trits)
    s = np.asarray(scales, dtype=np.float32)
    k = int(w.shape[-1])
    if group_size <= 0 or k % group_size:
        raise ValueError(f"K {k} is not a multiple of group_size {group_size}")
    groups = k // group_size
    if s.shape[-1] != groups or w.shape[:-1] != s.shape[:-1]:
        raise ValueError(
            f"scales shape {s.shape} does not match trits {w.shape} at group {group_size}"
        )
    grouped = w.reshape(*w.shape[:-1], groups, group_size).astype(np.float32)
    return (grouped * s[..., None]).reshape(w.shape)
