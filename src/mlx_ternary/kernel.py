# Copyright 2026 Jakevin
# SPDX-License-Identifier: Apache-2.0
"""Ternary GEMV and GEMM for MLX.

Codes {-1,0,1} are stored as 2-bit values {0,1,2}, four per byte, low bits
first. One scale per 64 weights (or 128, as an option) and no zero point: value = (code - 1) * scale.
The inner loop is the MLX 2-bit mask dot: pre-scale x by 1, 1/4, 1/16, 1/64
and AND the byte.

Small batches use a simdgroup GEMV. K multiple of 512 follows the MLX
qmv_fast walk (two rows per simdgroup, 16 weights per lane). Other K,
including 640, uses a 4-wide dot. Batch >= 32 with the row count and K
divisible by 32 uses a 32x32x32 simdgroup-matrix GEMM; if the batch is not a
multiple of 32, the leftover rows go through the GEMV. The weight
tile is half, the accumulator is float, and a tile never crosses a scale group.

Batches up to 8 run through a small LRU of compiled graphs keyed by shape.
Weights are passed as arguments, so the cache holds no weight arrays.
Larger batches dispatch directly.
"""
from __future__ import annotations

import mlx.core as mx

from .pack import GROUP, GROUPS


def set_cache_limit(gb: float = 2) -> int:
    """Opt-in: cap the MLX cache at ``gb`` gigabytes. Returns the previous limit.

    Importing the package does not change the MLX cache limit.
    """
    return mx.set_cache_limit(int(gb * 1024**3))


def configure() -> int:
    """Cap the MLX cache at 2GB. Same as ``set_cache_limit(2)``."""
    return set_cache_limit(2)

_KERNELS: dict = {}
_META: dict = {}

QMV_ROWS = 2
QMV_NSG = 1
GEMM_BM = 32
GEMM_BN = 32
GEMM_BK = 32
GEMM_MIN_BATCH = 32

_HEADER = r"""
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;

// code*x for four 2-bit lanes packed in one byte. x1/x2/x3 are already
// divided by 4, 16 and 64 so the masked field does not need a shift.
#define TERN_DOT4(b, x0, x1, x2, x3) \
    (((x0) * float((b) & 3u) + (x1) * float((b) & 0x0cu)) \
     + ((x2) * float((b) & 0x30u) + (x3) * float((b) & 0xc0u)))
#define TERN_DOT4H(b, x0, x1, x2, x3) \
    (((x0) * half((b) & 3u) + (x1) * half((b) & 0x0cu)) \
     + ((x2) * half((b) & 0x30u) + (x3) * half((b) & 0xc0u)))
"""


def metal(name: str, source: str, header: str = _HEADER):
    # MLX 0.22 metal_kernel has no compile_options. The cache key does not
    # include the source, so a body change must use a new name.
    key = (name, header[:40])
    if key not in _KERNELS:
        _KERNELS[key] = mx.fast.metal_kernel(
            name=name,
            input_names=["x", "codes", "scales", "meta"],
            output_names=["y"],
            header=header,
            source=source,
            ensure_row_contiguous=False,
        )
    return _KERNELS[key]


def meta_u32(values: list[int]) -> mx.array:
    key = tuple(int(v) for v in values)
    buf = _META.get(key)
    if buf is None:
        buf = mx.array(list(key), dtype=mx.uint32)
        mx.eval(buf)
        _META[key] = buf
    return buf


def _prepare(x, codes, scales, k, group):
    if group not in GROUPS:
        raise ValueError(f"group {group} not in {GROUPS}")
    if k % group != 0:
        raise ValueError(f"K {k} is not a multiple of {group}")
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if not isinstance(x, mx.array):
        x = mx.array(x)
    if x.dtype != mx.float32:
        x = x.astype(mx.float32)
    if not isinstance(codes, mx.array):
        codes = mx.array(codes)
    if codes.dtype != mx.uint8:
        codes = codes.astype(mx.uint8)
    if not isinstance(scales, mx.array):
        scales = mx.array(scales)
    if int(x.shape[-1]) != k:
        raise ValueError(f"x last dim {x.shape[-1]} != K {k}")
    # The kernels index raw row-major buffers (ensure_row_contiguous=False),
    # so transposed or sliced views must be copied first. No-op if contiguous.
    x = mx.contiguous(x)
    codes = mx.contiguous(codes)
    scales = mx.contiguous(scales)
    return x, codes, scales, int(x.shape[0]), int(codes.shape[0])


# Kernel sources are written for group 64. Group 128 only changes how a
# weight index maps to its scale: k >> 7 instead of k >> 6, and one more
# lane bit per scale. Group 64 sources are left byte-identical.
_REGROUP_128 = (
    ("k >> 6", "k >> 7"),
    ("t >> 6", "t >> 7"),
    ("(lane >> 2)", "(lane >> 3)"),
    ("(lane >> 4)", "(lane >> 5)"),
    ("((k0 + wc) >> 6)", "((k0 + wc) >> 7)"),
)


def _regroup(source: str, group: int) -> str:
    if group == 64:
        return source
    if group != 128:
        raise ValueError(f"group {group} not in {GROUPS}")
    for old, new in _REGROUP_128:
        source = source.replace(old, new)
    if ">> 6" in source:
        raise AssertionError("group 128 source still indexes scales by 64")
    return source


def _gname(group: int) -> str:
    return "" if group == 64 else f"_gs{group}"


def _dot16_expr(w: str, vecdot: bool = False) -> str:
    if not vecdot:
        return (
            f"TERN_DOT4({w} & 0xffu, xv0, xv1, xv2, xv3)"
            f" + TERN_DOT4(({w} >> 8u) & 0xffu, xv4, xv5, xv6, xv7)"
            f" + TERN_DOT4(({w} >> 16u) & 0xffu, xv8, xv9, xv10, xv11)"
            f" + TERN_DOT4(({w} >> 24u) & 0xffu, xv12, xv13, xv14, xv15)"
        )
    # Packed dot so the four masked lanes stay in one float4.
    def one(byte: str, xs: str) -> str:
        return (
            f"dot(float4(float({byte} & 3u), float({byte} & 0x0cu), "
            f"float({byte} & 0x30u), float({byte} & 0xc0u)), float4({xs}))"
        )
    return (
        f"{one(f'({w} & 0xffu)', 'xv0, xv1, xv2, xv3')}"
        f" + {one(f'(({w} >> 8u) & 0xffu)', 'xv4, xv5, xv6, xv7')}"
        f" + {one(f'(({w} >> 16u) & 0xffu)', 'xv8, xv9, xv10, xv11')}"
        f" + {one(f'(({w} >> 24u) & 0xffu)', 'xv12, xv13, xv14, xv15')}"
    )


def _qmv_source(nsg: int, rows: int, guarded: bool, blocks: int | None = None, vecdot: bool = False, narrow: bool = False, half_dot: bool = False, chunk128: bool = False) -> str:
    if rows not in (2, 4, 8) or nsg not in (1, 2, 4):
        raise ValueError(f"qmv rows={rows} nsg={nsg}")
    if blocks is not None and blocks < 1:
        raise ValueError(f"qmv blocks={blocks}")
    acc_init = "\n".join(f"    float acc{r} = 0.0f;" for r in range(rows))

    def w16_loads() -> str:
        lines = []
        for r in range(rows):
            addr = (
                f"*reinterpret_cast<const device uint*>(codes + (row0 + {r}u) * row_bytes + byte0 + (lane << 2))"
            )
            if guarded:
                lines.append(f"uint w{r} = 0u; if (row0 + {r}u < N) w{r} = {addr};")
            else:
                lines.append(f"uint w{r} = {addr};")
        return "\n        ".join(lines)

    def w4_loads() -> str:
        lines = []
        for r in range(rows):
            addr = f"uint(codes[(row0 + {r}u) * row_bytes + byte0 + lane])"
            if guarded:
                lines.append(f"uint c{r} = 0u; if (row0 + {r}u < N) c{r} = {addr};")
            else:
                lines.append(f"uint c{r} = {addr};")
        return "\n        ".join(lines)

    def apply(scale_shift: int, src: str) -> str:
        lines = []
        for r in range(rows):
            fma = (
                f"acc{r} = fma(float(scales[(row0 + {r}u) * n_groups + g0 + (lane >> {scale_shift})]), "
                f"d{r} - sumx, acc{r});"
            )
            if guarded:
                lines.append(f"if (row0 + {r}u < N) {{ {fma} }}")
            else:
                lines.append(fma)
        return "\n        ".join(lines)

    def tail_body() -> str:
        lines = [
            "uint sh = (t & 3u) * 2u;",
            "uint byte = t >> 2;",
            "uint g = t >> 6;",
        ]
        for r in range(rows):
            body = (
                f"uint code = (uint(codes[(row0 + {r}u) * row_bytes + byte]) >> sh) & 3u;"
                f"acc{r} = fma((float(code) - 1.0f) * float(scales[(row0 + {r}u) * n_groups + g]), xv, acc{r});"
            )
            if guarded:
                lines.append(f"if (row0 + {r}u < N) {{ {body} }}")
            else:
                lines.append("{ " + body + " }")
        return "\n        ".join(lines)

    red = "\n    ".join(f"float red{r} = simd_sum(acc{r});" for r in range(rows))
    writes = []
    for r in range(rows):
        cond = f"lane == 0u && row0 + {r}u < N" if guarded else "lane == 0u"
        writes.append(f"if ({cond}) y[b * N + row0 + {r}u] = red{r};")
    store = red + "\n    " + "\n    ".join(writes)
    dots16 = "\n        ".join(
        f"float d{r} = {_dot16_expr(f'w{r}', vecdot)};" for r in range(rows)
    )
    dots4 = "\n        ".join(
        f"float d{r} = TERN_DOT4(c{r}, xv0, xv1, xv2, xv3);" for r in range(rows)
    )
    step512 = f"""        const device float* xp = x + b * K + k + (lane << 4);
        float4 a0 = *reinterpret_cast<const device float4*>(xp);
        float4 a1 = *reinterpret_cast<const device float4*>(xp + 4);
        float4 a2 = *reinterpret_cast<const device float4*>(xp + 8);
        float4 a3 = *reinterpret_cast<const device float4*>(xp + 12);
        float sumx = ((a0.x + a0.y) + (a0.z + a0.w)) + ((a1.x + a1.y) + (a1.z + a1.w));
        sumx += ((a2.x + a2.y) + (a2.z + a2.w)) + ((a3.x + a3.y) + (a3.z + a3.w));
        float xv0 = a0.x; float xv1 = a0.y * 0.25f; float xv2 = a0.z * 0.0625f; float xv3 = a0.w * 0.015625f;
        float xv4 = a1.x; float xv5 = a1.y * 0.25f; float xv6 = a1.z * 0.0625f; float xv7 = a1.w * 0.015625f;
        float xv8 = a2.x; float xv9 = a2.y * 0.25f; float xv10 = a2.z * 0.0625f; float xv11 = a2.w * 0.015625f;
        float xv12 = a3.x; float xv13 = a3.y * 0.25f; float xv14 = a3.z * 0.0625f; float xv15 = a3.w * 0.015625f;
        uint byte0 = k >> 2;
        uint g0 = k >> 6;
        {w16_loads()}
        {dots16}
        {apply(2, "w")}"""
    if narrow:
        # One float4 of x is live at a time so the large-N grid can stay resident.
        shifts = (0, 8, 16, 24)
        nlines = [
            "const device float* xp = x + b * K + k + (lane << 4);",
            "uint byte0 = k >> 2;",
            "uint g0 = k >> 6;",
            w16_loads(),
        ]
        for r in range(rows):
            nlines.append(f"float part{r} = 0.0f;")
        for bi, sh in enumerate(shifts):
            nlines.append("{")
            nlines.append(
                f"float4 a = *reinterpret_cast<const device float4*>(xp + {bi * 4});"
            )
            nlines.append("float sum4 = (a.x + a.y) + (a.z + a.w);")
            nlines.append(
                "float u0 = a.x; float u1 = a.y * 0.25f; "
                "float u2 = a.z * 0.0625f; float u3 = a.w * 0.015625f;"
            )
            for r in range(rows):
                expr = (
                    f"part{r} += TERN_DOT4((w{r} >> {sh}u) & 0xffu, u0, u1, u2, u3) - sum4;"
                )
                if guarded:
                    nlines.append(f"if (row0 + {r}u < N) {{ {expr} }}")
                else:
                    nlines.append(expr)
            nlines.append("}")
        for r in range(rows):
            fma = (
                f"acc{r} = fma(float(scales[(row0 + {r}u) * n_groups + g0 + (lane >> 2)]), "
                f"part{r}, acc{r});"
            )
            if guarded:
                nlines.append(f"if (row0 + {r}u < N) {{ {fma} }}")
            else:
                nlines.append(fma)
        step512 = "\n        ".join(nlines)
    if half_dot:
        # Half products, float partial. One float4 of x is live per byte.
        shifts = (0, 8, 16, 24)
        hlines = [
            "const device float* xp = x + b * K + k + (lane << 4);",
            "uint byte0 = k >> 2;",
            "uint g0 = k >> 6;",
            w16_loads(),
        ]
        for r in range(rows):
            hlines.append(f"float part{r} = 0.0f;")
        for bi, sh in enumerate(shifts):
            hlines.append("{")
            hlines.append(
                f"float4 a = *reinterpret_cast<const device float4*>(xp + {bi * 4});"
            )
            hlines.append(
                "half hx = half(a.x); half hy = half(a.y); "
                "half hz = half(a.z); half hw = half(a.w);"
            )
            hlines.append("half sum4 = (hx + hy) + (hz + hw);")
            hlines.append(
                "half u0 = hx; half u1 = hy * 0.25h; "
                "half u2 = hz * 0.0625h; half u3 = hw * 0.015625h;"
            )
            for r in range(rows):
                expr = (
                    f"part{r} += float(TERN_DOT4H((w{r} >> {sh}u) & 0xffu, u0, u1, u2, u3)) "
                    f"- float(sum4);"
                )
                if guarded:
                    hlines.append(f"if (row0 + {r}u < N) {{ {expr} }}")
                else:
                    hlines.append(expr)
            hlines.append("}")
        for r in range(rows):
            fma = (
                f"acc{r} = fma(float(scales[(row0 + {r}u) * n_groups + g0 + (lane >> 2)]), "
                f"part{r}, acc{r});"
            )
            if guarded:
                hlines.append(f"if (row0 + {r}u < N) {{ {fma} }}")
            else:
                hlines.append(fma)
        step512 = "\n        ".join(hlines)
    if chunk128:
        # Four values per lane. Fewer live registers than the 16-wide body.
        k_loops = (
            "    uint k = 0u;\n"
            "    for (; k + 128u <= K; k += 128u) {\n"
            "        const device float* xp = x + b * K + k + (lane << 2);\n"
            "        float4 a0 = *reinterpret_cast<const device float4*>(xp);\n"
            "        float sumx = (a0.x + a0.y) + (a0.z + a0.w);\n"
            "        float xv0 = a0.x; float xv1 = a0.y * 0.25f; float xv2 = a0.z * 0.0625f; float xv3 = a0.w * 0.015625f;\n"
            "        uint byte0 = k >> 2;\n"
            "        uint g0 = k >> 6;\n"
            f"        {w4_loads()}\n"
            f"        {dots4}\n"
            f"        {apply(4, 'c')}\n"
            "    }\n"
            "    for (uint t = k + lane; t < K; t += 32u) {\n"
            "        float xv = x[b * K + t];\n"
            f"        {tail_body()}\n"
            "    }\n"
        )
    elif blocks is None:
        # K is not a multiple of 512 (expert down is 640). Keep the short tail.
        k_loops = (
            "    uint k = 0u;\n"
            "    for (; k + 512u <= K; k += 512u) {\n"
            f"{step512}\n"
            "    }\n"
            "    for (; k + 128u <= K; k += 128u) {\n"
            "        const device float* xp = x + b * K + k + (lane << 2);\n"
            "        float4 a0 = *reinterpret_cast<const device float4*>(xp);\n"
            "        float sumx = (a0.x + a0.y) + (a0.z + a0.w);\n"
            "        float xv0 = a0.x; float xv1 = a0.y * 0.25f; float xv2 = a0.z * 0.0625f; float xv3 = a0.w * 0.015625f;\n"
            "        uint byte0 = k >> 2;\n"
            "        uint g0 = k >> 6;\n"
            f"        {w4_loads()}\n"
            f"        {dots4}\n"
            f"        {apply(4, 'c')}\n"
            "    }\n"
            "    for (uint t = k + lane; t < K; t += 32u) {\n"
            "        float xv = x[b * K + t];\n"
            f"        {tail_body()}\n"
            "    }\n"
        )
    else:
        # Bake the trip count and drop the tail so the mask body can unroll.
        k_loops = (
            "    #pragma unroll\n"
            f"    for (uint blk = 0u; blk < {int(blocks)}u; ++blk) {{\n"
            "        uint k = blk * 512u;\n"
            f"{step512}\n"
            "    }\n"
        )
    return f"""
    constexpr uint NSG = {nsg}u;
    constexpr uint ROWS = {rows}u;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint tile = threadgroup_position_in_grid.x;
    uint b = threadgroup_position_in_grid.y;
    uint N = meta[0];
    uint K = meta[1];
    uint row_bytes = meta[2];
    uint n_groups = meta[3];
    uint row0 = (tile * NSG + sg) * ROWS;

    {acc_init}

    {k_loops}
    {store}
"""



def _gemm_source(bm: int, bn: int, bk: int, half_w: bool) -> str:
    if bm % 16 or bn % 16 or bk % 32 or 64 % bk:
        raise ValueError(f"bad tile {bm} {bn} {bk}")
    nsg_n = bn // 16
    nsg_m = bm // 16
    tg = nsg_n * nsg_m * 32
    per_w = (bn * bk) // tg
    per_x = (bm * bk) // tg
    if per_w % 4 or per_x % 4 or bk % per_w or bk % per_x or per_w > bk or per_x > bk:
        raise ValueError(f"tile {bm}x{bn}x{bk} does not split across {tg} threads")
    lda_x = bk + 4
    lda_w = bk + (8 if half_w else 4)
    wtype = "half" if half_w else "float"
    nbytes = per_w // 4
    if nbytes % 4 == 0:
        w_load = f"""
        uint e = tid * {per_w}u;
        uint wr = e / BK;
        uint wc = e - wr * BK;
        uint grow = n0 + wr;
        const device uint* wp = reinterpret_cast<const device uint*>(
            codes + grow * row_bytes + ((k0 + wc) >> 2));
        float s = float(scales[grow * n_groups + ((k0 + wc) >> 6)]);
        float s1 = s * 0.25f;
        float s2 = s * 0.0625f;
        float s3 = s * 0.015625f;
        #pragma unroll
        for (uint j = 0u; j < {nbytes // 4}u; ++j) {{
            uint w32 = wp[j];
            #pragma unroll
            for (uint i = 0u; i < 4u; ++i) {{
                uint bcode = (w32 >> (i * 8u)) & 0xffu;
                float d0 = s * float(bcode & 3u) - s;
                float d1 = s1 * float(bcode & 0x0cu) - s;
                float d2 = s2 * float(bcode & 0x30u) - s;
                float d3 = s3 * float(bcode & 0xc0u) - s;
                uint kc = wc + (j * 4u + i) * 4u;
                *reinterpret_cast<threadgroup {wtype}4*>(&WBUF[wr][kc]) = {wtype}4({wtype}(d0), {wtype}(d1), {wtype}(d2), {wtype}(d3));
            }}
        }}
"""
    else:
        w_load = f"""
        uint e = tid * {per_w}u;
        uint wr = e / BK;
        uint wc = e - wr * BK;
        uint grow = n0 + wr;
        const device uint8_t* wp = codes + grow * row_bytes + ((k0 + wc) >> 2);
        float s = float(scales[grow * n_groups + ((k0 + wc) >> 6)]);
        float s1 = s * 0.25f;
        float s2 = s * 0.0625f;
        float s3 = s * 0.015625f;
        #pragma unroll
        for (uint i = 0u; i < {nbytes}u; ++i) {{
            uint bcode = uint(wp[i]);
            float d0 = s * float(bcode & 3u) - s;
            float d1 = s1 * float(bcode & 0x0cu) - s;
            float d2 = s2 * float(bcode & 0x30u) - s;
            float d3 = s3 * float(bcode & 0xc0u) - s;
            uint kc = wc + i * 4u;
            *reinterpret_cast<threadgroup {wtype}4*>(&WBUF[wr][kc]) = {wtype}4({wtype}(d0), {wtype}(d1), {wtype}(d2), {wtype}(d3));
        }}
"""
    n4 = per_x // 4
    x_load = f"""
        uint ex = tid * {per_x}u;
        uint xr = ex / BK;
        uint xc = ex - xr * BK;
        const device float* xp = x + (b0 + xr) * K + (k0 + xc);
        #pragma unroll
        for (uint i = 0u; i < {n4}u; ++i) {{
            float4 v = *reinterpret_cast<const device float4*>(xp + i * 4u);
            *reinterpret_cast<threadgroup float4*>(&XBUF[xr][xc + i * 4u]) = v;
        }}
"""
    return f"""
    constexpr uint BM = {bm}u;
    constexpr uint BN = {bn}u;
    constexpr uint BK = {bk}u;
    constexpr uint LDAX = {lda_x}u;
    constexpr uint LDAW = {lda_w}u;

    uint tid = thread_position_in_threadgroup.x;
    uint sg = simdgroup_index_in_threadgroup;
    uint N = meta[0];
    uint K = meta[1];
    uint row_bytes = meta[2];
    uint n_groups = meta[3];
    uint B = meta[4];
    uint n0 = threadgroup_position_in_grid.x * BN;
    uint b0 = threadgroup_position_in_grid.y * BM;

    threadgroup float XBUF[BM][LDAX];
    threadgroup {wtype} WBUF[BN][LDAW];

    uint sg_n = sg % {nsg_n}u;
    uint sg_m = sg / {nsg_n}u;
    uint mbase = sg_m * 16u;
    uint nbase = sg_n * 16u;
    ushort lane = thread_index_in_simdgroup;
    short qid = short(lane) >> 2;
    short fm = (qid & 4) + ((short(lane) >> 1) & 3);
    short fn = ((qid & 2) << 1) + ((short(lane) & 1) << 1);
    uint ufm = uint(fm);
    uint ufn = uint(fn);

    simdgroup_matrix<float, 8, 8> c00(0.0f);
    simdgroup_matrix<float, 8, 8> c01(0.0f);
    simdgroup_matrix<float, 8, 8> c10(0.0f);
    simdgroup_matrix<float, 8, 8> c11(0.0f);

    for (uint k0 = 0u; k0 < K; k0 += BK) {{
        {x_load}
        {w_load}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint kk = 0u; kk < BK; kk += 8u) {{
            simdgroup_matrix<float, 8, 8> a0(0.0f);
            simdgroup_matrix<float, 8, 8> a1(0.0f);
            simdgroup_matrix<float, 8, 8> b0m(0.0f);
            simdgroup_matrix<float, 8, 8> b1m(0.0f);
            float2 av0 = *reinterpret_cast<threadgroup float2*>(&XBUF[mbase + ufm][kk + ufn]);
            float2 av1 = *reinterpret_cast<threadgroup float2*>(&XBUF[mbase + 8u + ufm][kk + ufn]);
            a0.thread_elements()[0] = av0.x;
            a0.thread_elements()[1] = av0.y;
            a1.thread_elements()[0] = av1.x;
            a1.thread_elements()[1] = av1.y;
            b0m.thread_elements()[0] = float(WBUF[nbase + ufn][kk + ufm]);
            b0m.thread_elements()[1] = float(WBUF[nbase + ufn + 1u][kk + ufm]);
            b1m.thread_elements()[0] = float(WBUF[nbase + 8u + ufn][kk + ufm]);
            b1m.thread_elements()[1] = float(WBUF[nbase + 8u + ufn + 1u][kk + ufm]);
            simdgroup_multiply_accumulate(c00, a0, b0m, c00);
            simdgroup_multiply_accumulate(c01, a0, b1m, c01);
            simdgroup_multiply_accumulate(c10, a1, b0m, c10);
            simdgroup_multiply_accumulate(c11, a1, b1m, c11);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    uint gm = b0 + mbase + ufm;
    uint gn = n0 + nbase + ufn;
    if (gm < B) {{
        if (gn < N) y[gm * N + gn] = c00.thread_elements()[0];
        if (gn + 1u < N) y[gm * N + gn + 1u] = c00.thread_elements()[1];
        if (gn + 8u < N) y[gm * N + gn + 8u] = c01.thread_elements()[0];
        if (gn + 9u < N) y[gm * N + gn + 9u] = c01.thread_elements()[1];
    }}
    uint gm8 = gm + 8u;
    if (gm8 < B) {{
        if (gn < N) y[gm8 * N + gn] = c10.thread_elements()[0];
        if (gn + 1u < N) y[gm8 * N + gn + 1u] = c10.thread_elements()[1];
        if (gn + 8u < N) y[gm8 * N + gn + 8u] = c11.thread_elements()[0];
        if (gn + 9u < N) y[gm8 * N + gn + 9u] = c11.thread_elements()[1];
    }}
"""


def _qmv_mlx_source(nsg: int, rows: int, guarded: bool) -> str:
    """MLX qmv_fast layout: 16 values per lane, one K loop, byte loads."""
    if rows not in (2, 4, 8) or nsg not in (1, 2, 4):
        raise ValueError(f"qmv mlx rows={rows} nsg={nsg}")
    init = "\n    ".join(f"float result{r} = 0.0f;" for r in range(rows))
    dots = []
    for r in range(rows):
        body = (
            f"const device uint8_t* wl = codes + (out_row + {r}u) * row_bytes + (lane << 2) + (k >> 2);"
            f"float s = float(scales[(out_row + {r}u) * n_groups + (k >> 6) + (lane >> 2)]);"
            "float accum = 0.0f;"
            "accum += xt0 * float(wl[0] & 3u) + xt1 * float(wl[0] & 0x0cu);"
            "accum += xt2 * float(wl[0] & 0x30u) + xt3 * float(wl[0] & 0xc0u);"
            "accum += xt4 * float(wl[1] & 3u) + xt5 * float(wl[1] & 0x0cu);"
            "accum += xt6 * float(wl[1] & 0x30u) + xt7 * float(wl[1] & 0xc0u);"
            "accum += xt8 * float(wl[2] & 3u) + xt9 * float(wl[2] & 0x0cu);"
            "accum += xt10 * float(wl[2] & 0x30u) + xt11 * float(wl[2] & 0xc0u);"
            "accum += xt12 * float(wl[3] & 3u) + xt13 * float(wl[3] & 0x0cu);"
            "accum += xt14 * float(wl[3] & 0x30u) + xt15 * float(wl[3] & 0xc0u);"
            f"result{r} = fma(s, accum - sum, result{r});"
        )
        if guarded:
            dots.append(f"if (out_row + {r}u < N) {{ {body} }}")
        else:
            dots.append("{ " + body + " }")
    row_gate = "\n      ".join(dots)
    if guarded:
        store = "\n    ".join(
            f"float red{r} = simd_sum(result{r}); if (lane == 0u && out_row + {r}u < N) y[b * N + out_row + {r}u] = red{r};"
            for r in range(rows)
        )
    else:
        store = "\n    ".join(
            f"float red{r} = simd_sum(result{r}); if (lane == 0u) y[b * N + out_row + {r}u] = red{r};"
            for r in range(rows)
        )
    return f"""
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint tile = threadgroup_position_in_grid.x;
    uint b = threadgroup_position_in_grid.y;
    uint N = meta[0];
    uint K = meta[1];
    uint row_bytes = meta[2];
    uint n_groups = meta[3];
    uint out_row = (tile * {nsg}u + sg) * {rows}u;
    {init}
    for (uint k = 0u; k < K; k += 512u) {{
      const device float* xp = x + b * K + k + (lane << 4);
      float4 a0 = *reinterpret_cast<const device float4*>(xp);
      float4 a1 = *reinterpret_cast<const device float4*>(xp + 4);
      float4 a2 = *reinterpret_cast<const device float4*>(xp + 8);
      float4 a3 = *reinterpret_cast<const device float4*>(xp + 12);
      float sum = ((a0.x + a0.y) + (a0.z + a0.w)) + ((a1.x + a1.y) + (a1.z + a1.w));
      sum += ((a2.x + a2.y) + (a2.z + a2.w)) + ((a3.x + a3.y) + (a3.z + a3.w));
      float xt0 = a0.x; float xt1 = a0.y * 0.25f; float xt2 = a0.z * 0.0625f; float xt3 = a0.w * 0.015625f;
      float xt4 = a1.x; float xt5 = a1.y * 0.25f; float xt6 = a1.z * 0.0625f; float xt7 = a1.w * 0.015625f;
      float xt8 = a2.x; float xt9 = a2.y * 0.25f; float xt10 = a2.z * 0.0625f; float xt11 = a2.w * 0.015625f;
      float xt12 = a3.x; float xt13 = a3.y * 0.25f; float xt14 = a3.z * 0.0625f; float xt15 = a3.w * 0.015625f;
      {row_gate}
    }}
    {store}
"""


def run_qmv_mlx(x, codes, scales, k, rows: int = 4, nsg: int = 2, group: int = GROUP):
    x, codes, scales, b, n = _prepare(x, codes, scales, k, group)
    if k % 512:
        raise ValueError(f"qmv mlx needs K multiple of 512, got {k}")
    rows_per_tg = nsg * rows
    guarded = (n % rows_per_tg) != 0
    tg = nsg * 32
    n_tiles = (n + rows_per_tg - 1) // rows_per_tg
    name = f"mlxtern3_qmvmlx_r{rows}_s{nsg}" + ("_g" if guarded else "_a") + _gname(group)
    ker = metal(name, _regroup(_qmv_mlx_source(nsg, rows, guarded), group))
    (y,) = ker(
        inputs=[x, codes, scales, meta_u32([n, k, k // 4, k // group, b])],
        template=[("T", mx.float32)],
        grid=(n_tiles * tg, b, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(b, n)],
        output_dtypes=[mx.float32],
    )
    return y


def run_qmv(x, codes, scales, k, rows: int = QMV_ROWS, nsg: int = QMV_NSG, group: int = GROUP, vecdot: bool = False, specialize: bool = True, narrow: bool = False, half_dot: bool = False, chunk128: bool = False):
    x, codes, scales, b, n = _prepare(x, codes, scales, k, group)
    if rows not in (2, 4, 8) or nsg not in (1, 2, 4):
        raise ValueError(f"qmv rows={rows} nsg={nsg}")
    rows_per_tg = nsg * rows
    guarded = (n % rows_per_tg) != 0
    tg = nsg * 32
    n_tiles = (n + rows_per_tg - 1) // rows_per_tg
    blocks = None if chunk128 else ((k // 512) if (specialize and k % 512 == 0) else None)
    name = f"mlxtern3_u2q2_r{rows}_s{nsg}"
    if chunk128:
        name += "_c128"
    elif blocks is not None:
        name += f"_k{k}"
    if vecdot:
        name += "_v"
    if narrow:
        name += "_n"
    if half_dot:
        name += "_h"
    name += "_g" if guarded else "_a"
    name += "_fast" + _gname(group)
    ker = metal(name, _regroup(_qmv_source(nsg, rows, guarded, blocks, vecdot, narrow, half_dot, chunk128), group))
    (y,) = ker(
        inputs=[x, codes, scales, meta_u32([n, k, k // 4, k // group, b])],
        template=[("T", mx.float32)],
        grid=(n_tiles * tg, b, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(b, n)],
        output_dtypes=[mx.float32],
    )
    return y


def run_gemm(x, codes, scales, k, bm: int = GEMM_BM, bn: int = GEMM_BN, bk: int = GEMM_BK, group: int = GROUP, half_w: bool = True):
    x, codes, scales, b, n = _prepare(x, codes, scales, k, group)
    if k % bk or n % bn or b % bm:
        raise ValueError(f"gemm tile {bm}x{bn}x{bk} does not cover N={n} K={k} B={b}")
    nsg = (bm // 16) * (bn // 16)
    tg = nsg * 32
    name = f"mlxtern3_u2g2_{bm}x{bn}x{bk}" + ("" if half_w else "_fw") + "_fast" + _gname(group)
    ker = metal(name, _regroup(_gemm_source(bm, bn, bk, half_w), group))
    (y,) = ker(
        inputs=[x, codes, scales, meta_u32([n, k, k // 4, k // group, b])],
        template=[("T", mx.float32)],
        grid=((n // bn) * tg, b // bm, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(b, n)],
        output_dtypes=[mx.float32],
    )
    return y


from collections import OrderedDict

COMPILE_MAX_BATCH = 8
COMPILE_CACHE_SIZE = 64
_COMPILED: OrderedDict = OrderedDict()


def _compiled(k: int, n: int, b: int, group: int, x_dtype, c_dtype, s_dtype):
    key = (k, n, b, group, x_dtype, c_dtype, s_dtype)
    fn = _COMPILED.get(key)
    if fn is not None:
        _COMPILED.move_to_end(key)
        return fn

    def _run(z, c, s):
        return _dispatch(z, c, s, k, group)

    fn = mx.compile(_run)
    _COMPILED[key] = fn
    while len(_COMPILED) > COMPILE_CACHE_SIZE:
        _COMPILED.popitem(last=False)
    return fn


def _dispatch(x, codes, scales, k: int, group: int):
    b = 1 if x.ndim == 1 else int(x.shape[0])
    n = int(codes.shape[0])
    tile = _gemm_launch(b, n, k)
    if tile is not None:
        bm, bn, bk = tile
        return run_gemm(x, codes, scales, k, bm, bn, bk, group=group)
    head = _gemm_head(b, n, k)
    if head:
        # Batch >= 32 but not a multiple of 32: the leading multiple of 32
        # rows go through the GEMM, the remainder through the GEMV.
        top = run_gemm(x[:head], codes, scales, k, GEMM_BM, GEMM_BN, GEMM_BK, group=group)
        rest = _gemv(x[head:], codes, scales, k, group)
        return mx.concatenate([top, rest], axis=0)
    return _gemv(x, codes, scales, k, group)


def _gemv(x, codes, scales, k: int, group: int):
    # K multiple of 512 uses the MLX qmv_fast walk (2 rows, 1 simdgroup).
    # Shorter K, including the 640-wide expert down projection, keeps the
    # 128-wide body, which is faster there.
    if k % 512 == 0:
        return run_qmv_mlx(x, codes, scales, k, rows=2, nsg=1, group=group)
    return run_qmv(x, codes, scales, k, group=group)


def _gemm_head(b: int, n: int, k: int) -> int:
    """Rows of a ragged batch that the GEMM can take; 0 if none."""
    if b < GEMM_MIN_BATCH or n % 32 or k % 32:
        return 0
    return (b // 32) * 32


def _gemm_launch(b: int, n: int, k: int) -> tuple[int, int, int] | None:
    if b < GEMM_MIN_BATCH or b % 32 or n % 32 or k % 32:
        return None
    # 32x32x32 keeps the shared tile small enough to stay resident on the M4 Pro.
    # Larger tiles were slower on the expert and are re-checked on the wide d3 shapes.
    return 32, 32, 32


def tern_matmul(x, codes, scales, k: int, group: int = GROUP) -> mx.array:
    """x [B, K] or [K], codes uint8 [N, K/4], scales [N, K/group] -> [B, N].

    ``group`` is 64 (default) or 128 weights per scale.

    Batches up to ``COMPILE_MAX_BATCH`` reuse a compiled graph from a bounded
    LRU keyed by shape and dtype; weights are arguments, not captured, so the
    cache never keeps them alive. Larger batches dispatch the kernel directly.
    A vector input comes back as shape [1, N].
    """
    if not isinstance(x, mx.array):
        x = mx.array(x)
    if x.ndim == 0 or x.ndim > 2:
        raise ValueError(f"x must have shape [K] or [B, K], got ndim {x.ndim}")
    k = int(k)
    group = int(group)
    if group not in GROUPS:
        raise ValueError(f"group {group} not in {GROUPS}")
    if k <= 0 or k % group:
        raise ValueError(f"K {k} is not a positive multiple of {group}")
    if int(x.shape[-1]) != k:
        raise ValueError(f"x last dim {x.shape[-1]} != K {k}")
    if codes.ndim != 2 or int(codes.shape[1]) != k // 4:
        raise ValueError(f"codes shape {tuple(codes.shape)} != [N, K/4] with K={k}")
    n = int(codes.shape[0])
    if tuple(scales.shape) != (n, k // group):
        raise ValueError(f"scales shape {tuple(scales.shape)} != ({n}, {k // group})")
    b = 1 if x.ndim == 1 else int(x.shape[0])
    if b == 0 or n == 0:
        # An empty launch would hand Metal a zero-size buffer.
        return mx.zeros((b, n), dtype=mx.float32)
    if b > COMPILE_MAX_BATCH:
        return _dispatch(x, codes, scales, k, int(group))
    fn = _compiled(k, n, b, int(group), x.dtype, codes.dtype, scales.dtype)
    return fn(x, codes, scales)
