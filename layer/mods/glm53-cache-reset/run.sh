#!/bin/sh
# e6 re-derivation (upstream nightly 385dce36b) of the kit patch_cache_reset.py (@aad6481).
# Adds a narrow elif at the dev-mode mount site: GLM53_EXPOSE_CACHE_RESET=1 attaches ONLY
# the cache-reset dev routes (/reset_prefix_cache, /reset_mm_cache, /reset_encoder_cache);
# VLLM_SERVER_DEV_MODE=1 keeps precedence (full dev surface, as upstream).
# Anchor moves: mount site retargeted from openai/api_server.py to
# vllm/entrypoints/launchers/api_server/routers.py:34-37 (nightly moved it; block
# byte-identical). routers.py lacks `import os`, which the elif arm needs, so the
# patcher inserts it (fail-closed count==1). The dev router itself is native on the
# nightly. Auth caveat unchanged: root-mounted routes bypass AuthenticationMiddleware,
# keep the gate at 0 on shared kits. Fail-closed on anchor drift; idempotent.
set -eu
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python3 "$DIR/patch_cache_reset.py"
