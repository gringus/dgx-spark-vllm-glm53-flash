#!/usr/bin/env python3
"""Route Mamba2 metadata state-index fetches through the MRV2 aligned-index
precompute (round 10-b; the low-risk core of upstream PR #58851).

THE ITL SPIKE: vllm telemetry on the 2026-10-13 sweep window shows e7/e6
ctx_tg c2 ITL at 0.136-0.158 s vs e5b-c's flat 0.098-0.104. A large share is
per-step GPU idle between eager launches in the mamba metadata path: every
decode step, EACH mamba KV group's metadata build (or update_block_table
reuse) calls ``mamba_get_block_table_tensor`` — a separate device gather per
group (4 groups on GLM-5.3-Flash).

THE MACHINERY IS ALREADY IN THIS TREE, IDLE: MambaHybridAttnMetadata's
align-mode block (model_states/mamba_hybrid.py, the mrv2state era) computes
EVERY mamba group's aligned physical state IDs in ONE Triton launch
(``MambaContext.compute_aligned_state_indices`` -> get_aligned_state_indices_
multi_group_kernel, "Compute every Mamba group's aligned physical state IDs
in one launch") and hands them to any metadata builder that carries a
``mamba_aligned_state_indices`` attribute. Upstream PR #58851's extension is
exactly that opt-in for the remaining builder classes; only GDN and nvidia
KDA carried it, so on GLM-5.3-Flash (Mamba2 builders) the loop finds no
eligible builder and the precompute never runs.

This mod: give BaseMambaAttentionMetadataBuilder the opt-in attribute and
route both gather sites (the align-mode branch of _compute_common_metadata
and update_block_table — the reuse path the base's same-spec metadata
dedup already uses) through ``_get_state_indices``, which prefers the
precomputed indices. One Triton launch replaces the per-group gathers.

PR #58851's remaining parts (the per-builder metadata cache in
CommonAttentionMetadata._cross_group_cache, the mamba_hybrid kwargs
de-slicing, the GDN/Kimi/KDA files) are deliberately NOT ported: the PR is
unmerged, and its diff bases on #57169's packed-KV mamba_attn.py rewrite
that postdates this tree — only the precompute routing above is
structure-independent and equivalence-by-construction
(compute_aligned_state_indices computes what
mamba_get_block_table_tensor computes, for all groups, in one launch; the
mrv2state machinery has run it in production for GDN/KDA).

Fails closed (count==1 anchors, ast.parse before write); idempotent via the
[glm53-alignperf] marker. GLM53_MAMBA_ALIGNPERF=0 restores the per-group
gathers (reboot only, no rebuild).
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

F = Path(
    os.environ.get(
        "GLM53_MAMBA_ATTN_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mamba_attn.py",
    )
)
MARKER = "[glm53-alignperf]"

INIT_ANCHOR = """\
import torch
"""

INIT_NEW = """\
import os  # [glm53-alignperf]
import torch
"""

OPTIN_ANCHOR = """\
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        # Enable speculative decoding support
"""

OPTIN_NEW = """\
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # [glm53-alignperf] Opt into the MRV2 aligned state-index precompute:
        # MambaHybridAttnMetadata's align-mode block fills this per step with
        # every mamba group's state IDs from ONE Triton launch, replacing the
        # per-group mamba_get_block_table_tensor gathers below.
        if torch.device(device).type == "cuda" and vllm_config.use_v2_model_runner:
            if os.environ.get("GLM53_MAMBA_ALIGNPERF", "1") != "0":
                self.mamba_aligned_state_indices: torch.Tensor | None = None

        # Enable speculative decoding support
"""

ELSE_ANCHOR = """\
        state_indices_tensor = mamba_get_block_table_tensor(
            common_attn_metadata.block_table_tensor,
            common_attn_metadata.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )
"""

ELSE_NEW = """\
        # [glm53-alignperf] Prefer the one-launch precomputed indices.
        state_indices_tensor = self._get_state_indices(
            common_attn_metadata.block_table_tensor,
            common_attn_metadata.seq_lens,
        )
"""

HELPER_ANCHOR = """\
    def update_block_table(
        self,
        metadata: M,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> M:
        state_indices_tensor = mamba_get_block_table_tensor(
            blk_table,
            metadata.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )
"""

HELPER_NEW = """\
    def _get_state_indices(  # [glm53-alignperf]
        self, blk_table: torch.Tensor, seq_lens: torch.Tensor
    ) -> torch.Tensor:
        state_indices = getattr(self, "mamba_aligned_state_indices", None)
        if state_indices is not None:
            return state_indices
        return mamba_get_block_table_tensor(
            blk_table,
            seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )

    def update_block_table(
        self,
        metadata: M,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> M:
        # [glm53-alignperf] Prefer the one-launch precomputed indices.
        state_indices_tensor = self._get_state_indices(
            blk_table,
            metadata.seq_lens,
        )
"""


def main() -> int:
    text = F.read_text()
    if MARKER in text:
        print(f"{F.name}: {MARKER} already present - no-op")
        return 0
    for old, new, label in (
        (INIT_ANCHOR, INIT_NEW, "os import"),
        (OPTIN_ANCHOR, OPTIN_NEW, "builder opt-in"),
        (ELSE_ANCHOR, ELSE_NEW, "align-mode gather routing"),
        (HELPER_ANCHOR, HELPER_NEW, "update_block_table routing + helper"),
    ):
        n = text.count(old)
        if n != 1:
            raise SystemExit(
                f"{label}: expected exactly 1 anchor occurrence, found {n}. "
                "The baked tree has drifted - re-derive the anchor."
            )
        text = text.replace(old, new, 1)
    ast.parse(text, filename=str(F))
    tmp = F.with_name(f".{F.name}.glm53-alignperf.tmp")
    try:
        tmp.write_text(text)
        os.replace(tmp, F)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(
        f"patched {F.name} (mamba metadata state-index fetches route through "
        f"the one-launch MRV2 aligned-index precompute; "
        f"GLM53_MAMBA_ALIGNPERF={os.environ.get('GLM53_MAMBA_ALIGNPERF', '1')})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
