# mlx_ternary

Apple M4 Pro. Times are interleaved medians in milliseconds. The ratio is v3 divided by MLX 2-bit f32 (`mx.quantized_matmul`, bits=2, group 64) on the same weight. Ternary weights carry about 1.85 bits of information but are stored at 2.25 bits per weight, the same as MLX 2-bit (see below). Kernel error against the dequantized ternary matrix stays under 1e-3.

| Mode | Matrix | v3 (ms) | MLX 2-bit f32 (ms) | MLX 4-bit (ms) | v3 / MLX 2-bit |
| --- | --- | ---: | ---: | ---: | ---: |
| Generation (batch 1) | gate 640×2560 | 0.126 | 0.124 | 0.125 | 1.009 |
| Prefill (batch 128) | gate 640×2560 | 0.223 | 0.224 | 0.224 | 0.996 |
| Generation (batch 1) | down 2560×640 | 0.130 | 0.137 | 0.133 | 0.949 |
| Prefill (batch 128) | down 2560×640 | 0.219 | 0.221 | 0.228 | 0.991 |

Whole layer, one graph, same machine. These runs compare v3 with MLX 2-bit f32.

| Mode | Scope | v3 (ms) | MLX 2-bit f32 (ms) | v3 / MLX 2-bit |
| --- | --- | ---: | ---: | ---: |
| Generation (batch 1) | expert gate+down | 0.167–0.186 | 0.177–0.193 | 0.944–0.961 |
| Prefill (batch 128) | expert gate+down | 0.341–0.360 | 0.346–0.368 | 0.979–0.985 |
| Generation (batch 1) | d3-nano layer 0, eight projections | 0.301–0.317 | 0.308–0.316 | 0.979–1.006 |
| Prefill (batch 128) | d3-nano layer 0, eight projections | 3.300–3.312 | 3.474–3.480 | 0.950–0.952 |

Generation matches MLX 2-bit. Prefill of the whole d3-nano layer is faster (ratio 0.950–0.952). Expert prefill is ratio 0.979–0.985.

A ternary value carries log2(3) ≈ 1.585 bits, plus one 16-bit scale for every 64 weights (16/64 = 0.25). That is about 1.85 bits per weight. The kernel stores each trit in a 2-bit slot, so the resident size is 2.25 bits per weight.

Ternary currently occupies 2.25 bits per weight resident, the same as MLX 2-bit. The point is an extra supported format with no speed loss, not smaller files.

The reference is the dequantized ternary matrix, `max|y − ref| / max|ref|`. The bound is 1e-3. The largest probed case was 1.44e-6 (64×6144, batch 32). Errors on the timed shapes were at most 4e-7. Those probes used scales that are exact in half precision. The batch ≥ 32 GEMM stages dequantized weights as half, so with arbitrary float32 scales its error is set by half rounding of `scale` (seen up to about 3e-4, still under 1e-3); the GEMV path stays near 1e-7.

## Load a layer

```python
import mlx.core as mx
import numpy as np
from mlx_ternary import TernaryLinear

n, k = 64, 256
trits = np.random.default_rng(0).integers(-1, 2, size=(n, k), dtype=np.int8)
scales = mx.ones((n, k // 64))
layer = TernaryLinear.from_trits(trits, scales)
y = layer(mx.random.normal(shape=(1, k)))
```

An `mlx.nn.QuantizedLinear` that already stores ternary weights as affine 2-bit (codes in {0, 1, 2}, per-group bias equal to `-scale`, group size 64) swaps in place:

```python
from mlx_ternary import TernaryLinear, replace_quantized_linears

layer = TernaryLinear.from_quantized_linear(q_linear)
# or, for a whole module tree:
n_replaced = replace_quantized_linears(model)
```

`TernaryLinear` follows `nn.Linear` on the forward: `y = x Wᵀ + b`, and accepts the same leading batch dimensions. Importing the package does not change the MLX cache limit; call `mlx_ternary.set_cache_limit(gb=2)` to opt in to a cap (default 2GB).

## Measured projections, d3-nano layer 0

v3 time and the ratio to MLX 2-bit f32. Summing the eight medians gives batch 1 = 1.168 ms versus 1.143 ms (1.022×) and batch 128 = 4.293 ms versus 4.480 ms (0.958×).

| Projection | Batch | v3 (ms) | v3 / MLX 2-bit |
| --- | ---: | ---: | ---: |
| in_proj_a 16×2048 | 1 | 0.130 | 1.023 |
| in_proj_a 16×2048 | 128 | 0.147 | 0.817 |
| in_proj_b 16×2048 | 1 | 0.127 | 1.028 |
| in_proj_b 16×2048 | 128 | 0.141 | 0.808 |
| in_proj_qkv 6144×2048 | 1 | 0.150 | 1.020 |
| in_proj_qkv 6144×2048 | 128 | 0.793 | 0.961 |
| in_proj_z 2048×2048 | 1 | 0.135 | 0.998 |
| in_proj_z 2048×2048 | 128 | 0.364 | 0.985 |
| out_proj 2048×2048 | 1 | 0.137 | 1.036 |
| out_proj 2048×2048 | 128 | 0.362 | 0.989 |
| mlp.down 2048×6144 | 1 | 0.154 | 1.029 |
| mlp.down 2048×6144 | 128 | 0.831 | 0.981 |
| mlp.gate 6144×2048 | 1 | 0.165 | 1.010 |
| mlp.gate 6144×2048 | 128 | 0.842 | 0.967 |
| mlp.up 6144×2048 | 1 | 0.170 | 1.029 |
| mlp.up 6144×2048 | 128 | 0.813 | 0.960 |

Single batch-1 calls sit near 0.13 ms, where 0.004 ms is about 3%, so one projection moves between about 0.98× and 1.04×. The fused graph above is the stable comparison.

## Packed format

Codes are int8 `{-1, 0, +1}`, stored as `{0, 1, 2}`. Four codes fill one uint8, low bits first:

| Bits | Weight index |
| --- | --- |
| 0–1 | 4t |
| 2–3 | 4t+1 |
| 4–5 | 4t+2 |
| 6–7 | 4t+3 |

`weight` has shape `[output, K/4]` uint8. `scales` has shape `[output, K/64]`. `K` is a multiple of 64. The value is `(code - 1) * scale`. Code 3 is rejected.

MLX affine 2-bit stores the same codes, 16 per uint32, lane `i` in bits `[2i, 2i+2)`, with a scale and a bias per group. Ternary checkpoints use bias = `-scale`. `from_mlx_2bit` checks that and returns the byte layout above. On this machine the uint32 bytes and the 4-per-byte layout are the same little-endian sequence.

## What the kernels do

* Batch 1, or any batch whose size, row count, or `K` is not a multiple of 32: simdgroup GEMV. `K` multiple of 512 uses two output rows per simdgroup and 16 weights per lane. Other `K`, including 640, uses a 4-wide dot plus a scalar tail.
* Batch ≥ 32 with batch, rows, and `K` all multiples of 32: 32×32×32 simdgroup-matrix GEMM. Weights are staged as half in the threadgroup; the accumulator stays float. The 32-wide K tile never crosses a scale group of 64.
* The Metal graph is compiled once per weight buffer and batch.

## Limits

* Apple Silicon, through MLX Metal.
* Group size is 64. `K` is a multiple of 64. There is no per-group zero point.
* The layer returns the dtype of `x`. The kernel accumulates in float32.
* Inputs may be non-contiguous views; they are copied to row-major first. An empty batch returns an empty result without a kernel launch. `tern_matmul` checks that `codes` is `[N, K/4]` and `scales` is `[N, K/64]`.
* Parameters are frozen. This package does not choose scales or codes.
* `from_quantized_linear` accepts an affine 2-bit `QuantizedLinear` whose codes are in {0, 1, 2} and whose per-group bias equals `-scale`. Other bit widths and other biases raise `ValueError`.
* The 1e-3 figure is the kernel's error against the packed ternary matrix.
* Importing `mlx_ternary` does not change the MLX cache limit. Call `mlx_ternary.set_cache_limit(gb=2)` to opt in; it returns the previous limit.

## Install

```bash
pip install -e .
python -m pytest
python -m mlx_ternary.bench
```

`mlx-ternary-bench` is the same benchmark. It times gate 640×2560, down 2560×640, in_proj_a 16×2048, and mlp.gate 6144×2048 at batch 1 and batch 128, and checks that each v3 median is within 10× of the table above and that the relative error is under 1e-3.

License: Apache-2.0, Copyright 2026 Jakevin.
