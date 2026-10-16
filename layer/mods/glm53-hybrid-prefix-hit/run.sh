#!/bin/sh
# e6 re-derivation (upstream nightly 385dce36b) of the kit patch_hybrid_prefix_hit.py
# minimal form (@aad6481): keep hybrid prefix-cache hits when the DFlash2 drafter
# group would zero them — (1) flag only exact SlidingWindowSpec drafter groups as
# EAGLE instead of the all-groups fallback, (2) do not let that drafter SWA group
# shrink the hybrid-min hit (empty blocks -> fresh window allocation), (3) one
# observability log line of group/eagle posture.
# All 4 anchors verified byte-identical on nightly kv_cache_coordinator.py
# (eagle-fallback block, hybrid-min block, attention_groups log site,
# _validate_prefix_cache_retention_interval helper insert point; `spec` in scope
# at the min site, logger defined).
# EAGLE-STAGE REVIEW vs the native spec-driven machinery: nightly
# _annotate_eagle_groups (kv_cache_utils.py:2089-2151) sets is_eagle_group only
# for groups whose layer specs declare non_causal_multi_token_decode — which
# today identifies only Kimi-K3 DSpark drafter specs, NOT the DFlash2 drafter.
# The all-groups fallback therefore still fires for GLM and still costs MLA a
# full scheduler page (~3584 tokens), so ALL THREE kit stages remain relevant;
# nothing dropped. (The earlier drop-the-eagle-stages note applied to the big v3
# migrator, not this minimal kit form.) Fail-closed on anchor drift; idempotent.
set -eu
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python3 "$DIR/patch_hybrid_prefix_hit.py"
