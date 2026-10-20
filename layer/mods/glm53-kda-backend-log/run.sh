#!/bin/sh
# glm53-kda-backend-log: one-line visibility mod - log the resolved KDA
# prefill backend (auto/triton/flashkda) at model build. The stock tree
# resolves it silently (kda.py:344); this makes the FlashKDA engagement
# visible in every boot log: "GLM-5.3 KDA prefill backend: flashkda".
# Fail-closed (count==1 anchors, ast.parse), idempotent (marker).
set -eu
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python3 "$DIR/patch_kda_backend_log.py"
