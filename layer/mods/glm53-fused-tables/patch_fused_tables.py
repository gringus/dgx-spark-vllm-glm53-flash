#!/usr/bin/env python3
"""[glm53-fused-tables] Replace build_grouped_fat_tables' ~125 small device ops
with 2 Triton kernels. In the live engine each small op serializes behind the
in-flight MoE kernel (stream queue delay), so 125 ops/call costs multi-ms per
MoE layer per step; 2 launches cut that ~60x. The original body is preserved
as _build_grouped_fat_tables_reference (bit-exact validation + CPU fallback).
Fail-closed on anchor drift; marker [glm53-fused-tables]."""
from pathlib import Path

p = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py")
s = p.read_text()
if "[glm53-fused-tables]" in s:
    print("[fused-tables] already applied")
    raise SystemExit(0)

assert "import triton" not in s, "exl3.py already imports triton"
s = s.replace("\nimport torch\n", "\nimport torch\nimport triton\nimport triton.language as tl\n", 1)
assert "import triton" in s

DEF = "def build_grouped_fat_tables("
assert s.count(DEF) == 1, f"def anchor count={s.count(DEF)}"
tail_idx = s.index("def apply_exl3_grouped_fat(", s.index(DEF))
fn_text = s[s.index(DEF):tail_idx]

KERN_AND_NEW = '''@triton.jit
def _glm53_tables_prep_kernel(
    counts_ptr, prep_ptr, cap, tile_rows, n_exp, BLOCK_N: tl.constexpr,
):
    # One program over the (padded) expert axis: inclusive cumsums via tl.cumsum,
    # then the exclusive forms and the inclusive totals the main kernel needs.
    i = tl.arange(0, BLOCK_N)
    m = i < n_exp
    c = tl.load(counts_ptr + i, mask=m, other=0)
    fat = tl.where(c > cap, c, 0)
    tiles = (fat + (tile_rows - 1)) // tile_rows
    rc_incl = tl.cumsum(c, 0)
    ro_incl = tl.cumsum(fat, 0)
    to_incl = tl.cumsum(tiles, 0)
    ro = ro_incl - fat
    so = rc_incl - c
    to = to_incl - tiles
    tl.store(prep_ptr + 0 * n_exp + i, fat, mask=m)
    tl.store(prep_ptr + 1 * n_exp + i, ro, mask=m)
    tl.store(prep_ptr + 2 * n_exp + i, so, mask=m)
    tl.store(prep_ptr + 3 * n_exp + i, tiles, mask=m)
    tl.store(prep_ptr + 4 * n_exp + i, to, mask=m)
    tl.store(prep_ptr + 5 * n_exp + i, to_incl, mask=m)
    tl.store(prep_ptr + 6 * n_exp + i, ro_incl, mask=m)
    if n_exp - 1 < BLOCK_N:
        last = n_exp - 1
        tl.store(prep_ptr + 7 * n_exp, tl.load(prep_ptr + 5 * n_exp + last))
        tl.store(prep_ptr + 8 * n_exp, tl.load(prep_ptr + 6 * n_exp + last))


@triton.jit
def _glm53_tables_main_kernel(
    row_cum_ptr, tile_cum_ptr, row_off_ptr, sorted_off_ptr, fat_rows_ptr,
    tile_off_ptr, token_sorted_ptr, weight_sorted_ptr,
    row_expert_ptr, row_token_ptr, row_weight_ptr,
    seg_expert_ptr, seg_row0_ptr, seg_rows_ptr,
    n_exp, rows_cap, tile_rows, BLOCK_R: tl.constexpr,
):
    # lane r < rows_cap: the row tables (binary search over the inclusive
    # row_cum); lane r >= rows_cap: the segment tables (over tile_cum).
    pid = tl.program_id(0)
    if pid < rows_cap:
        r = pid
        lo = 0
        hi = n_exp
        for _ in range(0, 16):
            mid = (lo + hi) // 2
            v = tl.load(row_cum_ptr + mid)
            hit = v <= r
            lo = tl.where(hit, mid + 1, lo)
            hi = tl.where(hit, hi, mid)
        re = tl.minimum(lo, n_exp - 1)
        ro = tl.load(row_off_ptr + re)
        so = tl.load(sorted_off_ptr + re)
        src = tl.minimum(so + (r - ro), rows_cap - 1)
        tk = tl.load(token_sorted_ptr + src)
        w = tl.load(weight_sorted_ptr + src)
        tl.store(row_expert_ptr + r, re)
        tl.store(row_token_ptr + r, tk)
        tl.store(row_weight_ptr + r, w)
    else:
        seg = pid - rows_cap
        lo = 0
        hi = n_exp
        for _ in range(0, 16):
            mid = (lo + hi) // 2
            v = tl.load(tile_cum_ptr + mid)
            hit = v <= seg
            lo = tl.where(hit, mid + 1, lo)
            hi = tl.where(hit, hi, mid)
        e = tl.minimum(lo, n_exp - 1)
        to = tl.load(tile_off_ptr + e)
        ro = tl.load(row_off_ptr + e)
        f = tl.load(fat_rows_ptr + e)
        local_tile = seg - to
        sr = f - local_tile * tile_rows
        sr = tl.maximum(sr, 0)
        sr = tl.minimum(sr, tile_rows)
        tl.store(seg_expert_ptr + seg, e)
        tl.store(seg_row0_ptr + seg, ro + local_tile * tile_rows)
        tl.store(seg_rows_ptr + seg, sr)


def build_grouped_fat_tables(
    counts: torch.Tensor,
    cap: int,
    token_sorted: torch.Tensor,
    weight_sorted: torch.Tensor,
    rows_cap: int,
    tile_rows: int,
) -> dict[str, torch.Tensor]:
    """[glm53-fused-tables] Same outputs and semantics as
    _build_grouped_fat_tables_reference, computed in 2 kernel launches instead
    of ~125 small device ops (each of which serializes behind the in-flight
    MoE kernel on the stream). CPU tensors fall back to the reference."""
    if counts.device.type != "cuda":
        return _build_grouped_fat_tables_reference(
            counts, cap, token_sorted, weight_sorted, rows_cap, tile_rows)
    n_exp = int(counts.numel())
    device = counts.device
    BLOCK_N = max(16, triton.next_power_of_2(n_exp))
    max_segs = (rows_cap + tile_rows - 1) // tile_rows + n_exp

    prep = torch.empty((9, n_exp), device=device, dtype=torch.int64)
    _glm53_tables_prep_kernel[(1,)](
        counts, prep, cap, tile_rows, n_exp, BLOCK_N=BLOCK_N)

    row_expert = torch.empty(rows_cap, device=device, dtype=torch.int32)
    row_token = torch.empty(rows_cap, device=device, dtype=torch.int64)
    row_weight = torch.empty(rows_cap, device=device, dtype=weight_sorted.dtype)
    seg_expert = torch.empty(max_segs, device=device, dtype=torch.int32)
    seg_row0 = torch.empty(max_segs, device=device, dtype=torch.int32)
    seg_rows = torch.empty(max_segs, device=device, dtype=torch.int32)
    row_cum = prep[6, :n_exp]
    tile_cum = prep[5, :n_exp]
    row_off = prep[1, :n_exp]
    sorted_off = prep[2, :n_exp]
    fat_rows = prep[0, :n_exp]
    tile_off = prep[4, :n_exp]
    num_segs = prep[7, :1].to(torch.int32)
    num_rows = prep[8, :1].to(torch.int32)
    BLOCK_R = 1024
    total_lanes = rows_cap + max_segs
    _glm53_tables_main_kernel[(total_lanes,)](
        row_cum, tile_cum, row_off, sorted_off, fat_rows, tile_off,
        token_sorted, weight_sorted,
        row_expert, row_token, row_weight, seg_expert, seg_row0, seg_rows,
        n_exp, rows_cap, tile_rows, BLOCK_R=BLOCK_R)
    return {
        "seg_expert": seg_expert,
        "seg_row0": seg_row0,
        "seg_rows": seg_rows,
        "num_segs": num_segs,
        "num_rows": num_rows,
        "row_expert": row_expert,
        "row_token": row_token,
        "row_weight": row_weight,
    }


'''

RENAME = DEF.replace("def build_grouped_fat_tables(", "def _build_grouped_fat_tables_reference(")
new_fn = fn_text.replace(DEF, RENAME, 1)
s = s.replace(fn_text, KERN_AND_NEW + new_fn, 1)

import ast
ast.parse(s)
p.write_text(s)
print("[fused-tables] applied (ast-validated): 2 Triton kernels + dispatch wrapper")
