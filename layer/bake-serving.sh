#!/bin/bash
# serving bake: applies the re-derived mod set to the 77154ca tree (0.32.0rc1.dev,
# #57169 packed-KV base) at build time, in dependency order. Every patcher is
# fail-closed (anchor count==1) and marker-tagged; a failure here fails the
# docker build. apply ALSO ends with the static marker check (fail-closed
# build tripwire). Runtime gates stay in the recipe env (run.sh sets the
# apply-time gates; the exports below are belt-and-braces).
# Ordering constraints (verified live, unchanged from e7): decode-floor MUST
# precede resident-tail; adaptive-k owns the scheduler `import os` insertion.
#
# E9 SET (77154ca bump, 2026-10-14). Retired vs e7:
#   glm53-hybrid-kv-v2      - SUBSUMED by #57169 (02e55d2af3): the generic
#                             packed KV layout + CircularBufferSpec ring replace
#                             the drafter slot-share/KpoolTail machinery.
#   glm53-upstream-pc60533  - #60533 (d8fa463e2b) native. NOTE: the coordinator
#                             logger it restored is now carried by
#                             glm53-hybrid-prefix-hit (LOGGER edit).
#   glm53-upstream-mrv2state- #60210 (8a26869bca) + #55507 native.
#   glm53-eagle3            - native (Glm5Next EagleModelMixin/SupportsEagle3).
# Re-derived vs e7: upstream-gbm (2 hunks re-anchored), upstream-alignperf (2
# gather sites), suppress-stops (factory + stop-check), indexer-rightsize
# (sizing half retired: native sparse_mla_prefill_workspace_factor +
# //index_kpool; guards kept), hybrid-prefix-hit (retention retargeted to the
# base-class cache_blocks + logger restore).
set -euo pipefail
MODS=/opt/glm53/mods
ORDER="glm53-upstream-gbm glm53-upstream-alignperf glm53-eagle-backoff glm53-cache-reset glm53-model-overrides glm53-shm-recheck glm53-role-token-cache \
glm53-video-placeholders glm53-indexer-rightsize glm53-suppress-stops glm53-prefix-admit glm53-scheduler-decode-floor \
glm53-adaptive-k-b0 glm53-upstream-grammar-resample glm53-strict-thinking glm53-nixl-ep-remove glm53-tools-grammar-coexist glm53-resident-tail glm53-sparse-miss-metric glm53-hybrid-prefix-hit glm53-fused-tables"
export GLM53_ROLE_TOKEN_CACHE=1 GLM53_SHM_RECHECK_50MS=1
mode=apply
case "${1:-}" in
  --verify-fast) mode=verify-fast ;;
esac
# Static marker check: the strongest marker per baked patch on the FINAL tree
# (one process, ~4s). Shared by apply (fail-closed build tripwire) and
# --verify-fast (manual spot-check).
marker_check() {
  python3 - <<'PYV'
from pathlib import Path
D = "/usr/local/lib/python3.12/dist-packages/vllm"
S = "/usr/local/lib/python3.12/dist-packages"
CHECKS = [
    # backport: gbm (#49760) still not native
    (D + "/v1/worker/gpu_worker.py", "cap_unified_memory_budget"),
    # native-base tripwires (must hold on the 77154ca base)
    (D + "/v1/core/kv_cache_utils.py", "partial_hash_hits_enabled"),
    (D + "/v1/core/kv_cache_utils.py", "eagle_proof_margin"),
    (D + "/models/glm5next/common/model.py", "SupportsEagle3"),
    (D + "/models/glm5next/common/attention.py", "CircularBufferSpec"),
    # retired mods must be ABSENT (their anchors no longer exist)
    (D + "/v1/core/kv_cache_utils.py", "!glm53-drafter-group"),
    (D + "/v1/core/kv_cache_utils.py", "!_get_kv_cache_groups_glm5_next"),
    # our patches
    (D + "/v1/kv_cache_interface.py", "[glm53-eagle-backoff]"),
    (D + "/v1/attention/backends/mamba_attn.py", "[glm53-alignperf]"),
    (D + "/v1/core/sched/scheduler.py", "[glm53-prefix-admit]"),
    (D + "/v1/core/sched/scheduler.py", "[glm53-resident-tail]"),
    (D + "/v1/core/sched/scheduler.py", "[glm53-adaptive-k]"),
    (D + "/v1/core/sched/scheduler.py", "[glm53-grammar-resample]"),
    (D + "/v1/structured_output/__init__.py", "[glm53-strict-thinking]"),
    (D + "/parser/abstract_parser.py", "[glm53-tools-grammar-coexist]"),
    (D + "/v1/request.py", "num_invalid_spec_tokens"),
    (D + "/v1/core/sched/scheduler.py", "# [glm53-decode-floor]"),
    (D + "/v1/attention/backends/mla/indexer.py", "[glm53-indexer-workspace]"),
    (D + "/models/glm5next/nvidia/sparse_indexer.py", "[glm53-indexer-workspace]"),
    (D + "/v1/core/kv_cache_coordinator.py", "[glm53-hybrid-apc]"),
    (D + "/v1/core/kv_cache_coordinator.py", "logger = init_logger(__name__)"),
    (D + "/v1/metrics/stats.py", "[glm53-prefix-cache-sparse-miss-metric]"),
    (D + "/models/glm5next/common/model.py", "aux_hidden_state_layers"),
    (D + "/model_executor/models/glm4_1v.py", "image_token_id"),
    (S + "/glm53_video_patch.py", "[glm53-video-placeholders]"),
    (S + "/glm53_video.pth", "import glm53_video_patch"),
    (D + "/distributed/device_communicators/shm_broadcast.py",
     "SHM_READER_RECHECK_INTERVAL_MS = 50"),
    (D + "/entrypoints/launchers/api_server/routers.py", "[glm53-cache-reset]"),
    (D + "/config/model.py", '"exl3",'),
    (D + "/v1/engine/detokenizer.py", "[suppress-stops-in-reasoning]"),
    (D + "/renderers/base.py", "[glm53-role-token-cache]"),
    (D + "/model_executor/layers/quantization/exl3.py", "[glm53-fused-tables]"),
    ("/opt/glm53/role_token_cache.py", "def make_cache"),
]
bad = []
for f, m in CHECKS:
    neg = m.startswith("!")
    probe = m[1:] if neg else m
    text = Path(f).read_text(errors="replace")
    hit = probe in text
    if neg and hit:
        bad.append((f, "RETIRED MARKER PRESENT: " + probe))
    elif not neg and not hit:
        bad.append((f, m))
if bad:
    for f, m in bad:
        print("[bake] VERIFY-FAST MISSING:", repr(m), "in", f)
    raise SystemExit(1)
print("[bake] VERIFY-FAST OK -", len(CHECKS), "markers present, gate disabled")
PYV
}
if [[ $mode == apply ]]; then
  # The GB10 persistent_topk disable rides inside video-placeholders (the nightly
  # still has the unguarded `select_k in (512,1024,2048)` path - dropping it re-arms
  # the cudagraph-FULL crash).
  for m in $ORDER; do
    echo "[bake] $m (apply)"
    ( cd "$MODS/$m" && ./run.sh )
  done
  marker_check
elif [[ $mode == verify-fast ]]; then
  marker_check
else
  echo "[bake] unknown mode: $1"; exit 1
fi
