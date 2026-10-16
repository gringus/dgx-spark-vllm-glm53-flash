#!/usr/bin/env python3
"""Harden the GLM-5.3 sparse-indexer prefill workspace (fail-closed guards).

77154ca re-derivation: the SIZING half of this mod is RETIRED -- upstream
now sizes the indexer workspace natively and better than our rightsize did:
``get_max_prefill_buffer_size`` became config-driven
(``attention_config.sparse_mla_prefill_workspace_factor``, default 5 =>
the old 40x entry count) and both GLM call sites divide by the compression
ratio (``common/attention.py``: ``// self.index_kpool``; indexer.py:1022
``// self.compress_ratio``), so the ~825 MiB stock lock is already ~4x
narrower without any patch. Tuning, if ever needed, is a recipe knob
(--attention-config sparse_mla_prefill_workspace_factor=N), not a patch.

What remains here is the SAFETY half -- three fail-closed guards against
workspace overruns, which upstream still lacks:

* a single row wider than the workspace raises instead of the stock
  ``if end == start`` fail-open (indexer.py splitter);
* every emitted chunk is re-checked before metadata construction and GPU
  gather (indexer.py pre-gather assertion, ``self.max_prefill_buffer_size``);
* the GLM kpool operator compares each chunk against its actual allocation
  argument immediately before slicing the gather buffers
  (models/glm5next/nvidia/sparse_indexer.py).

All sites are unconditional (pure safety rails; they also cover the stock
sizing). Fails closed (count==1 anchors, compile() before write); idempotent
via the [glm53-indexer-workspace] markers; atomic writes; stale pyc cleared.

v0.31.0 (e7) history: the sizing half rewrote get_max_prefill_buffer_size and
re-pointed the GLM call site at a GLM-scoped helper -- both retired here.
ANCHOR_SPLITTER/ANCHOR_CHUNK_ASSERT verified byte-identical at 77154ca.
ANCHOR_KPOOL_CAP re-derived: the chunk loop gained a DCP guard comment and the
chunk attribute is now ``max_local_total_seq_lens``.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

INDEXER_TARGET = Path(
    os.environ.get(
        "GLM53_INDEXER_BACKEND_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/"
        "indexer.py",
    )
)
KPOOL_TARGET = Path(
    os.environ.get(
        "GLM53_INDEXER_KPOOL_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/"
        "sparse_indexer.py",
    )
)

MARK_IMPORT = "import os  # [glm53-indexer-workspace]\n"
# e7 re-anchor (v0.31.0): numpy between typing and torch. Verified byte-identical
# at 77154ca.
ANCHOR_IMPORT = """import numpy as np
import torch
"""
PATCHED_IMPORT = """import numpy as np
import os  # [glm53-indexer-workspace]
import torch
"""

MARK_SPLITTER = (
    "                # [glm53-indexer-workspace] N cannot be sub-chunked.\n"
)
# Verified byte-identical at 77154ca.
ANCHOR_SPLITTER = """            # A single request can exceed the budget, requiring sub-chunking
            # on the query dimension.
            if end == start:
                chunk_m, chunk_n = (
                    prefill_query_lens_cpu[end].item(),
                    compressed_seq_lens_cpu[end].item(),
                )
                end += 1

            req_slice = slice(start + request_offset, end + request_offset)
"""
PATCHED_SPLITTER = """            # A single request can exceed the logits budget, requiring
            # sub-chunking on the query dimension. N cannot be sub-chunked: the
            # stock fail-open admitted the oversized row anyway, so right-sizing
            # could overrun the locked gather workspace.
            if end == start:
                chunk_m, chunk_n = (
                    prefill_query_lens_cpu[end].item(),
                    compressed_seq_lens_cpu[end].item(),
                )
                # [glm53-indexer-workspace] N cannot be sub-chunked.
                if chunk_n > workspace_size:
                    raise ValueError(
                        "[glm53-indexer-workspace] sparse-indexer row exceeds "
                        f"workspace: row={chunk_n}, workspace={workspace_size}, "
                        f"request={end + request_offset}"
                    )
                end += 1

            if chunk_n > workspace_size:
                raise AssertionError(
                    "[glm53-indexer-workspace] emitted chunk exceeds workspace: "
                    f"chunk={chunk_n}, workspace={workspace_size}, "
                    f"requests={start + request_offset}:{end + request_offset}"
                )
            req_slice = slice(start + request_offset, end + request_offset)
"""

MARK_CHUNK_ASSERT = (
    "                # [glm53-indexer-workspace] Final guard before GPU gather.\n"
)
# Verified byte-identical at 77154ca.
ANCHOR_CHUNK_ASSERT = """                # Skip when total_seq_lens is 0 (i.e., no compressed token).
                if metadata is not None:
                    chunks.append(metadata)
"""
PATCHED_CHUNK_ASSERT = """                # Skip when total_seq_lens is 0 (i.e., no compressed token).
                if metadata is not None:
                    # [glm53-indexer-workspace] Final guard before GPU gather.
                    if metadata.total_seq_lens > self.max_prefill_buffer_size:
                        raise AssertionError(
                            "[glm53-indexer-workspace] metadata chunk exceeds "
                            f"workspace: chunk={metadata.total_seq_lens}, "
                            f"workspace={self.max_prefill_buffer_size}"
                        )
                    chunks.append(metadata)
"""

INDEXER_SITES = (
    ("workspace environment import", MARK_IMPORT, ANCHOR_IMPORT, PATCHED_IMPORT),
    ("splitter fail-closed", MARK_SPLITTER, ANCHOR_SPLITTER, PATCHED_SPLITTER),
    (
        "pre-gather chunk assertion",
        MARK_CHUNK_ASSERT,
        ANCHOR_CHUNK_ASSERT,
        PATCHED_CHUNK_ASSERT,
    ),
)

MARK_KPOOL_CAP = (
    "            # [glm53-indexer-workspace] Guard the actual GLM allocation.\n"
)
# 77154ca re-anchor (77154ca): the loop gained the DCP guard comment and the chunk
# attribute is now max_local_total_seq_lens (DCP-aware).
ANCHOR_KPOOL_CAP = """        for chunk in prefill_metadata.chunks if not short_prefill else ():
            # Under DCP this rank only gathers the pools it owns.
            assert chunk.local_cu_seq_lens is not None
            k_quant = k_quant_full[: chunk.max_local_total_seq_lens]
            k_scale = k_scale_full[: chunk.max_local_total_seq_lens]
"""
PATCHED_KPOOL_CAP = """        for chunk in prefill_metadata.chunks if not short_prefill else ():
            # Under DCP this rank only gathers the pools it owns.
            assert chunk.local_cu_seq_lens is not None
            # [glm53-indexer-workspace] Guard the actual GLM allocation.
            if index_kpool > 1 and chunk.max_local_total_seq_lens > total_seq_lens:
                raise ValueError(
                    "[glm53-indexer-workspace] gather chunk exceeds allocated "
                    f"workspace: chunk={chunk.max_local_total_seq_lens}, "
                    f"allocated={total_seq_lens}"
                )
            k_quant = k_quant_full[: chunk.max_local_total_seq_lens]
            k_scale = k_scale_full[: chunk.max_local_total_seq_lens]
"""


def _verified(text: str, sites) -> bool:
    return all(
        text.count(mark) == 1
        and text.count(patched) == 1
        and text.count(anchor) == patched.count(anchor)
        for _name, mark, anchor, patched in sites
    )


def _prepare(text: str, sites, label: str) -> tuple[str, str]:
    marks = sum(text.count(mark) for _name, mark, _anchor, _patched in sites)
    if marks:
        if marks != len(sites) or not _verified(text, sites):
            raise ValueError(
                f"partial/inconsistent {label} patch "
                f"(marks={marks}, expected={len(sites)})"
            )
        return text, "already present"

    out = text
    for name, _mark, anchor, patched in sites:
        count = out.count(anchor)
        if count != 1:
            raise ValueError(
                f"pinned {label} anchor {name!r} drifted "
                f"(found {count}, expected 1)"
            )
        out = out.replace(anchor, patched, 1)
    if not _verified(out, sites):
        raise ValueError(f"{label} post-patch verification failed")
    return out, "patched"


def prepare_indexer(text: str) -> tuple[str, str]:
    return _prepare(text, INDEXER_SITES, "indexer")


def prepare_kpool(text: str) -> tuple[str, str]:
    sites = (
        (
            "actual allocation guard",
            MARK_KPOOL_CAP,
            ANCHOR_KPOOL_CAP,
            PATCHED_KPOOL_CAP,
        ),
    )
    return _prepare(text, sites, "kpool operator")


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-indexer-workspace.tmp")
    try:
        tmp.write_text(source)
        os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def clear_pyc(target: Path) -> None:
    cache = target.parent / "__pycache__"
    if not cache.is_dir():
        return
    for pyc in cache.glob(f"{target.stem}*.pyc"):
        pyc.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    preflight_only = "--preflight" in argv[1:]

    for target in (INDEXER_TARGET, KPOOL_TARGET):
        if not target.is_file():
            raise SystemExit(f"missing {target}")

    indexer_source = INDEXER_TARGET.read_text()
    kpool_source = KPOOL_TARGET.read_text()
    try:
        indexer_patched, indexer_action = prepare_indexer(indexer_source)
        kpool_patched, kpool_action = prepare_kpool(kpool_source)
    except ValueError as exc:
        raise SystemExit(f"indexer workspace preflight failed: {exc}") from exc

    compile(indexer_patched, str(INDEXER_TARGET), "exec")
    compile(kpool_patched, str(KPOOL_TARGET), "exec")
    if preflight_only:
        print(
            "indexer workspace preflight OK "
            f"(indexer={indexer_action}, kpool={kpool_action})"
        )
        return 0

    if indexer_patched != indexer_source:
        replace_file(INDEXER_TARGET, indexer_patched)
        clear_pyc(INDEXER_TARGET)
    if kpool_patched != kpool_source:
        replace_file(KPOOL_TARGET, kpool_patched)
        clear_pyc(KPOOL_TARGET)
    print(
        "indexer workspace guards "
        f"(indexer={indexer_action}, kpool={kpool_action}; sizing retired "
        "natively: sparse_mla_prefill_workspace_factor + //index_kpool)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
