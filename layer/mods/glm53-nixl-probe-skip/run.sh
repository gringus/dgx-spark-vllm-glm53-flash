#!/bin/sh
# glm53-nixl-probe-skip: silence the 4x-per-boot nixl_ep import-failure
# traceback (boot noise, user-reported). The 0.1.0 image ships a nixl_ep
# wheel whose compiled backend is tagged torch216 while the image runs torch
# 2.14 -> the import vllm.utils.import_utils.has_nixl_ep() attempts can never
# succeed. The mod adds a silent, import-free filesystem pre-probe to
# has_nixl_ep: no torch-matched backend .so -> False (no doomed import, no
# traceback); matching .so -> stock import path unchanged (fail-open on any
# probe error). Env GLM53_NIXL_PROBE_SKIP=0 restores stock (reboot only).
# Idempotent (marker), count==1 anchor, ast-verified before write.
set -eu
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python3 "$DIR/patch_nixl_probe_skip.py"
