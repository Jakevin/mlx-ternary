# Copyright 2026 Jakevin
# SPDX-License-Identifier: Apache-2.0
import mlx_ternary

# Importing does not touch the MLX cache; cap it explicitly for test runs.
mlx_ternary.set_cache_limit(2)
