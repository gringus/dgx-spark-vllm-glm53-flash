#!/bin/sh
# e6 re-derivation (upstream nightly 385dce36b) of the e4 glm53-adaptive-k-b0 mod.
# Applies ONLY patch_adaptive_k.py (the e4 B0 form, not the kit margin/trim superset).
#
# align_floor RETIRED per the migration spec: nightly scheduler.py carries the same
# dormant disjunct natively, so patch_align_floor.py would be a pure drift canary.
# It is intentionally NOT shipped here. Do not resurrect without a nightly diff.
#
# Ordering note: e4 installed this after patch_scheduler_decode_floor.py; that patcher
# is kit-source and not shipped in e6/mods. On nightly the adaptive-k patcher no longer
# depends on it: nightly scheduler.py lacks a top-level import os (the fork's decode-floor
# patch used to add one), so the e6 patcher inserts `import os` itself (after
# `import itertools`, before `import time`).
#
# Runtime-gated: inert unless GLM53_ADAPTIVE_K (policy) or GLM53_ADAPTIVE_K_CAPTURE=1
# (B0 capture-only graphs) is set. Fail-closed on anchor drift; idempotent via the
# [glm53-adaptive-k] markers.
set -eu
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python3 "$DIR/patch_adaptive_k.py"
