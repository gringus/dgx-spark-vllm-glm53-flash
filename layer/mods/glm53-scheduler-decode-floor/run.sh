#!/bin/sh
# e6 re-derivation (upstream nightly 385dce36b) of the kit
# patch_scheduler_decode_floor.py v3.1 (@aad6481): mixed-prefill decode floor —
# a prefill does not share an engine step with a running decode (GLM53_MIXED_PREFILL_CHUNK=skip
# default), with warm-bypass (<= GLM53_MIXED_PREFILL_WARM_TOKENS), deadline
# (GLM53_MIXED_PREFILL_MAX_WAIT_MS -> LATE_CAP), aging escalation (v3) and split
# crawl wall/capped accounting (v3.1). Pristine -> v3.1 direct; v1/v2/v3 baked
# states upgrade in place via byte-exact anchors.
# All anchors verified byte-identical on nightly scheduler.py.
# IMPORT OWNERSHIP: this patcher owns the top-level `import os` insertion on the
# nightly; the adaptive-k patcher inserts its own (commented) only if none exists.
# ORDERING: MUST install before glm53-resident-tail (which rewrites the same LPTT
# cap blocks; a later decode-floor install fails loudly rather than silently).
# Fail-closed on anchor drift; idempotent.
set -eu
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python3 "$DIR/patch_scheduler_decode_floor.py"
