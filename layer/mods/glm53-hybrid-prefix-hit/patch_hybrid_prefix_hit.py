#!/usr/bin/env python3
"""Keep hybrid prefix-cache hits when the DFlash2 group would zero them.

Issue #7: OpenClaw follow-ups looked like 0% APC. On this kit the MLA +
mamba groups already hit at the 3584-token hybrid align (README 3584/7760).
Two coordinator bugs then throw the extra block away:

1. ``dflash`` is ``use_eagle()``. GLM never sets ``is_eagle_group`` (that
   annotator is DeepseekV4-only), so HybridKVCacheCoordinator flags EVERY
   group. MLA drops its last scheduler-aligned block (~3584 tokens).
2. The DFlash2 SlidingWindow group still participates in the hybrid min.
   After an EAGLE one-block pop it re-aligns down by a full 3584-token
   scheduler page (block=64, align=3584), which can wipe a longer MLA hit.

KpoolTail already opts out of prefix caching (1-block circular scratch).
Mamba align-mode state *does* materialize at 896-token chunk ends, and
3584 is a multiple of 896, so mamba must stay in the min — skipping a
mamba miss is a correctness hole (vLLM #47491 / #43090).

This patch: flag only exact SlidingWindowSpec groups as EAGLE, and do not
let that drafter group shrink ``curr_hit_length``. If the drafter window
does not cover the MLA/mamba hit, leave its blocks empty so a fresh
window is allocated (zeros / new pages). Wrong indexer tail state is
fatal — we do not touch KpoolTailManager.

Round 5 hit-finding fix: the drafter group's EAGLE one-block peek (its
``_contiguous_blocks_for_hit`` +1 and the final pop) can never fire for a
replay. The sparse-retention tail (retention_interval 0) is published by
``reachable_block_mask`` with shift=1 ending one block ABOVE the reachable
boundary (for prompt 16384: blocks 9-11..13, top = 16128) only while that
peek block is still full of identical prefix content — with a short first
decode the peek block straddles the prompt end (decode-dirty, content
differs cross-request) and is never matched, so the right-to-left search
finds no complete run and reports hit 0 / one-boundary-low (14976 < 16128),
the exemption leaves the blocks empty, and DFlash2 decodes against a
freshly zeroed ~4.6k window (acceptance 1.00 no-hit vs 0.619 fully empty =
the -30% ctx_tg c2 gap). The tail blocks below the reachable boundary are
pure prefill KV of the identical prefix (decode writes land at/after the
prompt end, one block higher), so relaxing the peek for the drafter group
lets ``find_longest_cache_hit`` land the tail exactly on the candidate
boundary and the existing keep-if-covers branch resumes the window from
cache. Landing stays on the scheduler block grid, so reconciliation with
MLA/mamba is unchanged.

Round 6 rolling-window retention: the round-5 tail freezes at the request's
prompt-end hash-grid cap, so a replay whose candidate runs into the first
request's own decode territory (agent loops re-sending context+response;
the coordinator registers a shared-prefix junction deeper in decode) finds
no drafter tail at the resume position and the keep branch falls back to a
fresh zeroed drafter window - low acceptance for that request's whole life,
poisoning the per-request adaptive-k EMA (k=7 13% of steps vs 41% on
e5b-c). Fix: retain the same window-sized tail at the current settle
position each ``cache_blocks`` call, so the drafter's rolling decode window
stays cached and any later replay finds the window covered at any depth.
Also adds an env-gated (GLM53_APC_DEBUG=1) keep/fallback decision log.

Fine-grained SWA lookup (single_type_kv_cache_manager.py): on
77154ca the drafter is a NATIVE SlidingWindowSpec group (block 4608, no
slot-share) and SlidingWindowManager is the only prefix-cacheable manager
without ``supports_fine_grained_hash_lookup`` - so
``partial_hash_hits_enabled()`` fails for the WHOLE model and
``_cache_hit_alignment_tokens`` degrades to the 4608 scheduler block: the
reconciled prefix hit caps at 4608 multiples (measured 13824 for a
16384-token replay vs e7's 16128 at the 1152 grid) and every group
recomputes up to 4607 tokens per resume (hit-resume TTFT 1.42s vs e7
0.31s; the B shape 16k-hit+2048-tail 4.05s vs 1.82s). The publish side
already exists in the native manager (``cache_blocks`` ->
``_cache_partial_tail_block`` registers the prompt-tail hash boundary);
this adds the lookup side: flip the ClassVar and give
``find_longest_cache_hit`` a fine-grained branch mirroring the
FullAttentionManager shape (walk full blocks on the block view, then probe
the first non-full block's interior boundaries high-to-low). The
min-reconcile exemption was analyzed and REJECTED: the request's
num_computed_tokens would outrun the drafter's own hit, the target never
computes the shortfall region (no hidden states -> no drafter KV), and a
drafter self-prefill would be out-of-distribution (DFlash2 KV is a
function of TARGET hidden states) - the empty-window hole at the window's
recent end. Fixing the drafter's own hit is the e7 round-5 pattern.

Fail closed if the vLLM coordinator anchors drift.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

P = Path(
    os.environ.get(
        "GLM53_KV_COORDINATOR_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py",
    )
)
P2 = Path(
    os.environ.get(
        "GLM53_STYPE_MANAGER_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/single_type_kv_cache_manager.py",
    )
)
MARK = "# [glm53-hybrid-apc]"
MARK2 = "# [glm53-swa-fine-lookup]"

# --- fine-grained hash lookup for the drafter's SlidingWindowManager ---

FINE_CLASSVAR_OLD = """class SlidingWindowManager(SingleTypeKVCacheManager):
    def __init__(self, kv_cache_spec: SlidingWindowSpec, **kwargs) -> None:
"""

FINE_CLASSVAR_NEW = """class SlidingWindowManager(SingleTypeKVCacheManager):
    # [glm53-swa-fine-lookup] The drafter's SWA group is the only
    # prefix-cacheable manager without fine-grained hash lookup, so
    # partial_hash_hits_enabled() fails for the WHOLE model and
    # _cache_hit_alignment_tokens degrades to the 4608 scheduler block: the
    # reconciled prefix hit caps at 4608 multiples (measured 13824 for a
    # 16384-token replay vs e7's 16128 at the 1152 grid) and every group
    # recomputes up to 4607 tokens per resume (hit-resume TTFT 1.42s vs e7
    # 0.31s). The publish side already exists in this manager
    # (cache_blocks -> _cache_partial_tail_block registers the prompt-tail
    # hash boundary); find_longest_cache_hit gains the lookup side below.
    supports_fine_grained_hash_lookup: ClassVar[bool] = True

    def __init__(self, kv_cache_spec: SlidingWindowSpec, **kwargs) -> None:
"""

FINE_WALK_OLD = """        sliding_window_contiguous_blocks = cls._contiguous_blocks_for_hit(
            kv_cache_spec.sliding_window, kv_cache_spec.block_size, drop_eagle_block
        )

        # TODO: reduce i by sliding_window_contiguous_blocks when cache miss, to
"""

FINE_WALK_NEW = """        sliding_window_contiguous_blocks = cls._contiguous_blocks_for_hit(
            kv_cache_spec.sliding_window, kv_cache_spec.block_size, drop_eagle_block
        )

        # [glm53-swa-fine-lookup] Fine-grained partial hits: with the hash-unit
        # alignment resolve_block_hashes kept the RAW unit-granularity list, so
        # the block-granular walk below cannot run on it. Mirror the
        # FullAttentionManager shape: walk full blocks on the block view, then
        # extend into the first non-full block by probing its interior hash
        # boundaries high-to-low (this manager publishes exactly one such
        # boundary - the prompt tail - via _cache_partial_tail_block). The
        # window-coverage rule is unchanged and the returned list keeps nulls
        # below the run (positions SWA attention never reads).
        if (
            alignment_tokens < kv_cache_spec.block_size
            and kv_cache_spec.block_size % alignment_tokens == 0
        ):
            assert isinstance(block_hashes, Sequence), (
                "fine-grained SWA lookup needs a sequence hash list"
            )
            block_size = kv_cache_spec.block_size
            full_block_hashes = BlockHashListWithBlockSize(
                block_hashes, alignment_tokens, block_size
            )
            max_num_blocks = max_length // block_size
            computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
                [block_pool.null_block] * max_num_blocks
                for _ in range(len(kv_cache_group_ids))
            )
            num_contiguous_blocks = 0
            match_found = False
            for i in range(max_num_blocks - 1, -1, -1):
                if cached_block := block_pool.get_cached_block(
                    full_block_hashes[i], kv_cache_group_ids
                ):
                    if num_contiguous_blocks == 0 and block_size != alignment_tokens:
                        post_pop_blocks = i if drop_eagle_block else i + 1
                        if (post_pop_blocks * block_size) % alignment_tokens != 0:
                            continue
                    for computed, cached in zip(computed_blocks, cached_block):
                        computed[i] = cached
                    num_contiguous_blocks += 1
                    if num_contiguous_blocks >= sliding_window_contiguous_blocks:
                        for computed in computed_blocks:
                            del computed[i + num_contiguous_blocks :]
                        match_found = True
                        break
                else:
                    num_contiguous_blocks = 0
            if not match_found:
                # The trailing run is a hit even below the window floor.
                for computed in computed_blocks:
                    del computed[num_contiguous_blocks:]
            hit_length = len(computed_blocks[0]) * block_size
            # Extend into the first non-full block (the published prompt tail).
            scale_factor = block_size // alignment_tokens
            first_partial_idx = len(computed_blocks[0]) * scale_factor
            max_partial_idx = min(
                first_partial_idx + scale_factor - 1,
                max_length // alignment_tokens,
                len(block_hashes),
            )
            for fine_idx in range(max_partial_idx - 1, first_partial_idx - 1, -1):
                cached_tail = block_pool.get_cached_block(
                    block_hashes[fine_idx], kv_cache_group_ids
                )
                if not cached_tail:
                    continue
                for computed, cached in zip(computed_blocks, cached_tail):
                    computed.append(cached)
                hit_length = (fine_idx + 1) * alignment_tokens
                break
            if drop_eagle_block and hit_length > 0:
                hit_length -= min(alignment_tokens, block_size)
            hit_length -= hit_length % alignment_tokens
            num_blocks = cdiv(hit_length, block_size)
            for computed in computed_blocks:
                del computed[num_blocks:]
            return computed_blocks, hit_length

        # TODO: reduce i by sliding_window_contiguous_blocks when cache miss, to
"""


def fine_verified_state(text: str) -> bool:
    return (
        MARK2 in text
        and text.count(FINE_CLASSVAR_NEW) == 1
        and text.count(FINE_WALK_NEW) == 1
        and text.count(FINE_CLASSVAR_OLD) == 0
        and text.count(FINE_WALK_OLD) == 0
    )

HELPER = '''
def _glm53_inner_kv_spec(spec):
    specs = getattr(spec, "kv_cache_specs", None)
    if isinstance(specs, dict) and specs:
        return next(iter(specs.values()))
    return spec


def _glm53_is_draft_swa_spec(spec) -> bool:
    """DFlash2 drafter: exact SlidingWindowSpec, not the CircularBufferSpec
    tail (the #57169 ring) or an MLA/mamba group."""
    return type(_glm53_inner_kv_spec(spec)).__name__ == "SlidingWindowSpec"


# Round 6 telemetry gate (read once at import; no per-call env lookups).
_GLM53_APC_DEBUG = __import__("os").environ.get("GLM53_APC_DEBUG") == "1"


'''

EAGLE_OLD = """        # Conservatively fall back to flag all groups when no group is flagged.
        if use_eagle and not self.eagle_group_ids:
            self.eagle_group_ids = set(range(len(kv_cache_config.kv_cache_groups)))
"""

EAGLE_NEW = """        # Conservatively fall back to flag all groups when no group is flagged.
        if use_eagle and not self.eagle_group_ids:
            # [glm53-hybrid-apc] dflash is use_eagle(); GLM has no is_eagle_group
            # annotator. Flag only the drafter SlidingWindowSpec group so MLA /
            # mamba do not drop a whole scheduler page (~3584). MTP with no SWA
            # group keeps the upstream all-groups fallback.
            swa_ids = {
                i
                for i, g in enumerate(kv_cache_config.kv_cache_groups)
                if _glm53_is_draft_swa_spec(g.kv_cache_spec)
            }
            self.eagle_group_ids = swa_ids or set(
                range(len(kv_cache_config.kv_cache_groups))
            )
"""

RELAX_OLD = """                drop_eagle_block = use_eagle and idx not in eagle_verified

                _max_length = curr_hit_length
"""

RELAX_NEW = """                drop_eagle_block = use_eagle and idx not in eagle_verified
                if _glm53_is_draft_swa_spec(spec):  # [glm53-hybrid-apc]
                    # Round 5 hit-finding fix: drop the EAGLE one-block peek
                    # for the drafter group. Its sparse-retention tail ends at
                    # the reachable boundary + one peek block; that peek block
                    # straddles the first request's prompt end (decode-dirty,
                    # content differs cross-request), so requiring it reports
                    # hit 0 / one-boundary-low and the keep branch below never
                    # fires. Tail blocks below the boundary are pure identical-
                    # prefix prefill KV, so resuming straight from them is
                    # positionally exact and stays on the block grid.
                    drop_eagle_block = False

                _max_length = curr_hit_length
"""

MIN_OLD = """                if drop_eagle_block:
                    eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:
                    # length shrunk; invalidate previous eagle verifications
                    eagle_verified.clear()
                curr_hit_length = _new_hit_length
                for group_id, blocks in zip(group_ids, hit_blocks):
                    hit_blocks_by_group[group_id] = blocks
                    hit_length_by_group[group_id] = _new_hit_length

                longest_hit_length = max(longest_hit_length, curr_hit_length)
"""

MIN_NEW = """                if drop_eagle_block:
                    eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:
                    # length shrunk; invalidate previous eagle verifications
                    eagle_verified.clear()
                if _glm53_is_draft_swa_spec(spec):  # [glm53-hybrid-apc]
                    # Drafter SWA must not min() the hybrid hit. Its EAGLE pop
                    # re-aligns by LCM(window block, MLA page) = 3584. If the
                    # cached window does not cover the MLA/mamba hit, leave
                    # blocks empty so a fresh window is allocated; do not
                    # reseed the indexer tail here (KpoolTail already opted out).
                    # Round 5: with the peek relaxed, a hit that reaches the
                    # candidate boundary means the cached tail covers the
                    # drafter's whole SWA window below it — keep the blocks so
                    # the drafter resumes its window from cache instead of
                    # decoding against a freshly zeroed ~4.6k window.
                    if _new_hit_length >= curr_hit_length:
                        for group_id, blocks in zip(group_ids, hit_blocks):
                            hit_blocks_by_group[group_id] = blocks
                            hit_length_by_group[group_id] = _new_hit_length
                    if _GLM53_APC_DEBUG:  # [glm53-hybrid-apc]
                        logger.info(
                            "[glm53-apc-debug] draft hit=%d cand=%d %s",
                            _new_hit_length,
                            curr_hit_length,
                            "keep" if _new_hit_length >= curr_hit_length
                            else "FALLBACK",
                        )
                    continue
                curr_hit_length = _new_hit_length
                for group_id, blocks in zip(group_ids, hit_blocks):
                    hit_blocks_by_group[group_id] = blocks
                    hit_length_by_group[group_id] = _new_hit_length

                longest_hit_length = max(longest_hit_length, curr_hit_length)
"""

BOUND_OLD = """            manager.cache_blocks(
                request,
                self.get_num_cacheable_tokens(num_computed_tokens, group_id),
                retention_interval=self.retention_interval,
                replay_boundaries=boundaries,
            )
"""

BOUND_NEW = """            if _glm53_is_draft_swa_spec(manager.kv_cache_spec):  # [glm53-hybrid-apc]
                # Round 5 retention fix: the scheduler-grid replay boundaries
                # sit a scheduler block below where MLA/mamba actually replay
                # (the hash-unit grid), so the drafter's sparse tail lands
                # below its SWA window at the real resume position and no hit
                # can ever cover the window. Also retain the window-sized tail
                # at the hash-grid caps; drafter-only, mamba keeps its
                # conservative scheduler-grid boundaries.
                _fine = self.hash_block_size
                boundaries = boundaries + (
                    max((request.num_prompt_tokens - 1) // _fine * _fine, 0),
                    max(request.num_prompt_tokens // _fine * _fine, 0),
                    # Round 6 rolling-window retention: the prompt-end cap
                    # freezes the retained tail at admission time, so a replay
                    # whose candidate runs into this request's own decode
                    # territory (agent loops re-sending context+response)
                    # outruns the tail and the keep branch falls back to a
                    # fresh zeroed drafter window. Retain the same window-
                    # sized tail at the current settle position each step;
                    # the tail follows the decode and a later replay finds
                    # the window covered at any depth.
                    max((num_computed_tokens - 1) // _fine * _fine, 0),
                )
            manager.cache_blocks(
                request,
                self.get_num_cacheable_tokens(num_computed_tokens, group_id),
                retention_interval=self.retention_interval,
                replay_boundaries=boundaries,
            )
"""

LOG_OLD = """        # Propagate the eagle bit to each manager (default to ``use_eagle=False``).
        for group in self.attention_groups:
            if group.use_eagle:
                for gid in group.group_ids:
                    self.single_type_managers[gid].use_eagle = True
"""

LOG_NEW = """        # Propagate the eagle bit to each manager (default to ``use_eagle=False``).
        for group in self.attention_groups:
            if group.use_eagle:
                for gid in group.group_ids:
                    self.single_type_managers[gid].use_eagle = True
        logger.info(  # [glm53-hybrid-apc]
            "hybrid APC groups: %s; eagle_group_ids=%s",
            [
                (
                    type(_glm53_inner_kv_spec(g.spec)).__name__,
                    g.group_ids,
                    getattr(g.manager_cls, "__name__", type(g.manager_cls).__name__),
                    g.use_eagle,
                )
                for g in self.attention_groups
            ],
            sorted(self.eagle_group_ids),
        )
"""


LOGGER_OLD = """from vllm.v1.request import Request
"""

LOGGER_NEW = """from vllm.logger import init_logger
from vllm.v1.request import Request

logger = init_logger(__name__)
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{P}: expected one {label} target, found {n}")
    return text.replace(old, new, 1)


def verified_state(text: str) -> bool:
    return (
        MARK in text
        and "def _glm53_is_draft_swa_spec(" in text
        and "swa_ids or set(" in text
        and text.count(EAGLE_NEW) == 1
        and text.count(RELAX_NEW) == 1
        and text.count(BOUND_NEW) == 1
        and text.count(MIN_NEW) == 1
        and text.count(LOG_NEW) == 1
        and "logger = init_logger(__name__)" in text
        and text.count(EAGLE_OLD) == 0
        and text.count(RELAX_OLD) == 0
        and text.count(MIN_OLD) == 0
    )
    # NOTE: LOGGER_OLD and BOUND_OLD are substrings of their NEW texts (the
    # logger line precedes the Request import; the base cache_blocks call is
    # re-emitted verbatim below the drafter-if), so OLD==0 does not hold for
    # them post-patch -- completeness is enforced by the NEW==1 counts above.


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-hybrid-apc.tmp")
    try:
        tmp.write_text(source)
        os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def main() -> int:
    # --- P2: fine-grained SWA lookup (single_type_kv_cache_manager.py) ---
    if not P2.is_file():
        raise SystemExit(f"missing {P2}")
    text2 = P2.read_text()
    if MARK2 in text2:
        if not fine_verified_state(text2):
            raise SystemExit(
                f"{P2}: {MARK2} present but fine-lookup patch is incomplete"
            )
        print(f"{P2.name}: {MARK2} already present — skipping")
    else:
        text2 = replace_once(text2, FINE_CLASSVAR_OLD, FINE_CLASSVAR_NEW, "swa-fine-classvar")
        text2 = replace_once(text2, FINE_WALK_OLD, FINE_WALK_NEW, "swa-fine-walk")
        if not fine_verified_state(text2):
            raise SystemExit(f"{P2}: fine-lookup post-patch verification failed")
        compile(text2, str(P2), "exec")
        replace_file(P2, text2)
        print(
            f"patched {P2.name} (fine-grained SWA lookup: ClassVar flip + "
            "block-view walk with prompt-tail interior probe)"
        )

    # --- P: the coordinator hunks (unchanged since round 5/6) ---
    if not P.is_file():
        raise SystemExit(f"missing {P}")
    text = P.read_text()
    if MARK in text:
        if not verified_state(text):
            raise SystemExit(
                f"{P}: {MARK} present but hybrid APC patch is incomplete"
            )
        print(f"{P.name}: {MARK} already present — skipping")
        return 0
    needle = "def _validate_prefix_cache_retention_interval(\n"
    if text.count(needle) != 1:
        raise SystemExit(f"{P}: helper insert point not unique")
    if "def _glm53_inner_kv_spec(" not in text:
        text = text.replace(needle, HELPER + needle, 1)
    text = replace_once(text, LOGGER_OLD, LOGGER_NEW, "logger-restore")
    text = replace_once(text, EAGLE_OLD, EAGLE_NEW, "eagle-fallback")
    text = replace_once(text, RELAX_OLD, RELAX_NEW, "draft-swa-hit-finding")
    text = replace_once(text, BOUND_OLD, BOUND_NEW, "draft-swa-retention")
    text = replace_once(text, MIN_OLD, MIN_NEW, "hybrid-min")
    text = replace_once(text, LOG_OLD, LOG_NEW, "group-log")
    if not verified_state(text):
        raise SystemExit(f"{P}: hybrid APC post-patch verification failed")
    compile(text, str(P), "exec")
    replace_file(P, text)
    print(
        f"patched {P.name} (hybrid APC: drafter SWA skipped in min, eagle on SWA only, "
        "draft-swa hit-finding peek relaxed, coordinator logger restored)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
