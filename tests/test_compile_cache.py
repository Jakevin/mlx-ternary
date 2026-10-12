# Copyright 2026 Jakevin
# SPDX-License-Identifier: Apache-2.0
import gc
import weakref

import mlx.core as mx
import numpy as np

from mlx_ternary import kernel, pack_trits, tern_matmul


def _weights(n=64, k=256, seed=0):
    rng = np.random.default_rng(seed)
    trits = rng.integers(-1, 2, size=(n, k), dtype=np.int8)
    packed, _ = pack_trits(trits)
    scales = rng.uniform(0.02, 0.08, size=(n, k // 64)).astype(np.float32)
    return mx.array(packed), mx.array(scales)


def test_cache_bounded_across_many_batches():
    kernel._COMPILED.clear()
    codes, scales = _weights()
    for b in range(1, 300):
        mx.eval(tern_matmul(mx.ones((b, 256)), codes, scales, 256))
    assert len(kernel._COMPILED) <= kernel.COMPILE_MAX_BATCH
    mx.synchronize()
    mx.clear_cache()
    base = mx.get_active_memory()
    for b in range(300, 600):
        mx.eval(tern_matmul(mx.ones((b, 256)), codes, scales, 256))
    mx.synchronize()
    mx.clear_cache()
    assert mx.get_active_memory() <= base + (1 << 20)
    assert len(kernel._COMPILED) <= kernel.COMPILE_CACHE_SIZE


def test_cache_does_not_keep_weights_alive():
    kernel._COMPILED.clear()
    for seed in range(3):
        codes, scales = _weights(seed=seed)
        ref = weakref.ref(codes)
        for b in (1, 4, 64):
            mx.eval(tern_matmul(mx.ones((b, 256)), codes, scales, 256))
        del codes, scales
        gc.collect()
        assert ref() is None
    assert len(kernel._COMPILED) <= 2
