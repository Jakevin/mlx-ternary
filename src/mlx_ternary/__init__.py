# Copyright 2026 Jakevin
# SPDX-License-Identifier: Apache-2.0
"""Ternary linear layers for MLX, about 1.85 bits per weight."""

from mlx_ternary.convert import from_mlx_2bit, unpack_mlx_2bit
from mlx_ternary.kernel import configure, set_cache_limit, tern_matmul
from mlx_ternary.linear import TernaryLinear, replace_quantized_linears
from mlx_ternary.pack import (
    GROUP,
    dequantize,
    info_bits_per_weight,
    pack_trits,
    stored_bits_per_weight,
    unpack_trits,
)

__version__ = "0.3.0"

__all__ = [
    "GROUP",
    "TernaryLinear",
    "configure",
    "dequantize",
    "from_mlx_2bit",
    "info_bits_per_weight",
    "pack_trits",
    "replace_quantized_linears",
    "set_cache_limit",
    "stored_bits_per_weight",
    "tern_matmul",
    "unpack_mlx_2bit",
    "unpack_trits",
]
