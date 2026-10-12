# Copyright 2026 Jakevin
# SPDX-License-Identifier: Apache-2.0
"""Time the v3 kernel against MLX 2-bit and 4-bit on the published shapes.

The gate is the same order of magnitude as the Apple M4 Pro numbers in the
kernel report: each median must fall within 10x of that measurement, and the
relative error against the dequantized ternary matrix must stay under 1e-3.
"""
from __future__ import annotations

import json
import platform
import subprocess
import time

import mlx.core as mx
import numpy as np

from mlx_ternary import dequantize, pack_trits, set_cache_limit, tern_matmul
from mlx_ternary.pack import GROUP

# Published v3 medians, milliseconds, Apple M4 Pro.
PUBLISHED_MS = {
    ("gate", 1): 0.126,
    ("gate", 128): 0.223,
    ("down", 1): 0.130,
    ("down", 128): 0.219,
    ("in_proj_a", 1): 0.130,
    ("in_proj_a", 128): 0.147,
    ("mlp.gate", 1): 0.165,
    ("mlp.gate", 128): 0.842,
}

SHAPES = {
    "gate": (640, 2560),
    "down": (2560, 640),
    "in_proj_a": (16, 2048),
    "mlp.gate": (6144, 2048),
}


def _machine() -> str:
    try:
        chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        chip = platform.processor() or platform.machine()
    return chip


def _rss_gb() -> float:
    try:
        kb = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(__import__("os").getpid())], text=True).strip() or 0)
        return kb / 1024 / 1024
    except Exception:
        return 0.0


def _rel(y, ref) -> float:
    err = mx.abs(y - ref)
    mx.eval(err)
    denom = float(mx.max(mx.abs(ref)).item())
    return float(mx.max(err).item()) / max(denom, 1e-8)


def _paired(fns: list, repeats: int, warmup: int = 4) -> list[float]:
    for fn in fns:
        for _ in range(warmup):
            mx.eval(fn())
    mx.synchronize()
    samples = [[] for _ in fns]
    for rep in range(repeats):
        order = list(range(len(fns)))
        if rep % 2:
            order.reverse()
        for i in order:
            mx.synchronize()
            t0 = time.perf_counter()
            mx.eval(fns[i]())
            mx.synchronize()
            samples[i].append((time.perf_counter() - t0) * 1e3)
    medians = []
    for s in samples:
        s.sort()
        medians.append(s[len(s) // 2])
    return medians


def _prepare(name: str, n: int, k: int, seed: int):
    rng = np.random.default_rng(seed)
    trits = rng.integers(-1, 2, size=(n, k), dtype=np.int8)
    scales = rng.uniform(0.02, 0.08, size=(n, k // GROUP)).astype(np.float32)
    ref = dequantize(trits, scales)
    packed, _row = pack_trits(trits)
    codes = mx.array(packed)
    scale_mx = mx.array(scales)
    weight = mx.array(ref)
    q2 = mx.quantize(weight, group_size=GROUP, bits=2, mode="affine")
    q4 = mx.quantize(weight, group_size=GROUP, bits=4, mode="affine")
    mx.eval(codes, scale_mx, weight, *q2, *q4)
    return codes, scale_mx, mx.array(ref), q2, q4


def main() -> None:
    set_cache_limit(2)
    chip = _machine()
    rows = []
    failures = []
    print(f"machine: {chip}", flush=True)
    for index, (name, shape) in enumerate(SHAPES.items()):
        n, k = shape
        codes, scales, ref_w, q2, q4 = _prepare(name, n, k, 100 + index)
        q2_w, q2_s, q2_b = q2
        q4_w, q4_s, q4_b = q4
        for batch, repeats in ((1, 40), (128, 12)):
            mx.random.seed(1000 + batch + n)
            x = mx.random.normal(shape=(batch, k)).astype(mx.float32)
            y_ref = x @ ref_w.T
            y = tern_matmul(x, codes, scales, k)
            mx.eval(x, y, y_ref)
            rel = _rel(y, y_ref)

            def tern(x=x, codes=codes, scales=scales, k=k):
                return tern_matmul(x, codes, scales, k)

            def mlx2(x=x, q2_w=q2_w, q2_s=q2_s, q2_b=q2_b):
                return mx.quantized_matmul(
                    x, q2_w, scales=q2_s, biases=q2_b, transpose=True, group_size=GROUP, bits=2, mode="affine"
                )

            def mlx4(x=x, q4_w=q4_w, q4_s=q4_s, q4_b=q4_b):
                return mx.quantized_matmul(
                    x, q4_w, scales=q4_s, biases=q4_b, transpose=True, group_size=GROUP, bits=4, mode="affine"
                )

            tern_ms, mlx2_ms, mlx4_ms = _paired([tern, mlx2, mlx4], repeats)
            published = PUBLISHED_MS[(name, batch)]
            ratio = tern_ms / mlx2_ms
            row = {
                "name": name,
                "shape": [n, k],
                "batch": batch,
                "tern_ms": tern_ms,
                "mlx2_ms": mlx2_ms,
                "mlx4_ms": mlx4_ms,
                "ratio_mlx2": ratio,
                "published_tern_ms": published,
                "rel": rel,
            }
            rows.append(row)
            print(
                f"{name} {n}x{k} B={batch} tern={tern_ms:.3f} ms "
                f"mlx2={mlx2_ms:.3f} mlx4={mlx4_ms:.3f} "
                f"ratio={ratio:.3f} published={published:.3f} rel={rel:.3e}",
                flush=True,
            )
            if not np.isfinite(rel) or rel >= 1e-3:
                failures.append(f"{name} B={batch} rel {rel:.3e}")
            if not (published / 10.0 <= tern_ms <= published * 10.0):
                failures.append(f"{name} B={batch} tern {tern_ms:.3f} ms vs published {published:.3f} ms")
        del codes, scales, ref_w, q2, q4
        mx.clear_cache()
    payload = {"machine": chip, "rss_gb": _rss_gb(), "rows": rows, "failures": failures}
    print(json.dumps(payload), flush=True)
    if failures:
        raise SystemExit("benchmark failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
