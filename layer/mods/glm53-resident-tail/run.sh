# e6 re-derivation (upstream nightly 385dce36b): all four scheduler anchors verified
# (running-loop input_budget<=draft_slots block x1, running LPTT cap x1, waiting LPTT
# cap x1, class line x1). The waiters helper calls
# kv_cache_manager.coordinator.find_longest_cache_hit via the manager, no import
# retarget needed; the nightly coordinator signature (block_hashes,
# max_cache_hit_length) -> 3-tuple matches the e4 helper unpacking. Patcher verbatim.
#!/bin/sh
# Applies Reederey87 #93: bounded resident-tail scheduling - a cached follow-up (hit >= one
# page, small uncached remainder) keeps the token budget until its tail is done instead of
# paying a concurrent cold prefill's step time. Apply-time gate GLM53_RESIDENT_TAIL=1;
# fail-closed on anchor drift. Production evidence in the upstream commit.
set -eu
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
GLM53_RESIDENT_TAIL=1 python3 "$DIR/patch_resident_tail.py"
