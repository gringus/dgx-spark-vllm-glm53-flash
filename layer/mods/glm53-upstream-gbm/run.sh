#!/bin/bash
# glm53-upstream-gbm: cherry-pick of vllm-project/vllm PR #49760 (commits
# 303527175f, 0c124a7d3a, 51e122a185) - respect gpu_memory_utilization on
# integrated/unified-memory GPUs; cap the allocator; reject exhausted budget.
# Fixes the startup profile overshoot that wedges GB10 hosts (issues #46307,
# #50780, #56824). Code files only (tests excluded). Fail-closed via --dry-run.
# Idempotent: on an already-baked tree (rebuild-from-baked path) the marker
# short-circuits; a CLEAN tree still goes through --dry-run fail-closed.
set -eu
SITE=/usr/local/lib/python3.12/dist-packages
cd "$SITE"
if grep -q "cap_unified_memory_budget" vllm/v1/worker/gpu_worker.py; then
  echo "[upstream-gbm] marker present - no-op"
  exit 0
fi
echo "[upstream-gbm] dry-run"
patch -p1 --dry-run < /opt/glm53/mods/glm53-upstream-gbm/gbm.patch
echo "[upstream-gbm] apply"
patch -p1 < /opt/glm53/mods/glm53-upstream-gbm/gbm.patch
python3 -c "
from pathlib import Path
s = Path('$SITE/vllm/v1/worker/gpu_worker.py').read_text()
assert 'cap_unified_memory_budget' in s, 'marker missing'
print('[upstream-gbm] verified')"
