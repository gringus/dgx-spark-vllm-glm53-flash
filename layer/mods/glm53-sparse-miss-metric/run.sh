# e6 re-derivation (upstream nightly 385dce36b): all anchors verified (stats.py
# PrefixCacheStats record block, kv_cache_manager record site, loggers counter+record,
# scheduler allocation-failure exit and admission-time call). Nightly natively carries
# Request.shared_prefix_boundary (kv_cache_manager.py:306) and returns
# num_uncached_common_prefix_tokens from coordinator find_longest_cache_hit, so the
# fork-side dependency is native now. Patcher copied verbatim from e4.
#!/bin/sh
# Reederey87 #87: observability-only - exposes shared-prefix tokens discarded during hybrid
# cache reconciliation as vllm:prefix_cache_sparse_retention_misses (backport of vLLM
# #52527). NOT in any A/B arm (zero behavior change); adopt at bake time. Targets stats.py,
# kv_cache_manager.py, loggers.py, scheduler.py. Fail-closed on anchor drift.
set -eu
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python3 "$DIR/patch_prefix_cache_sparse_miss_metric.py"
