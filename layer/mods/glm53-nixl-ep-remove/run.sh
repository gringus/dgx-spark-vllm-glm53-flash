#!/bin/sh
# glm53-nixl-ep-remove: delete the nixl_ep connector packages from the image.
# The wheel ships only a torch216-tagged backend .so (nixl_ep_cpp_torch216)
# vs image torch 2.14, so `import nixl_ep` can never succeed on this stack -
# but vllm's connector-plugin discovery probes it via _has_module on every
# boot, emitting a full traceback (and retry loops; the has_nixl_ep-level
# patch in glm53-nixl-probe-skip does NOT intercept this probe path).
# Removing the packages makes every probe return a clean not-found.
# Re-add only if the nixl wheel ever ships a torch-matching build.
# Self-checking: pre-check (present), post-check (absent, fail-closed).
set -eu
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python3 "$DIR/patch_remove.py"
for d in /usr/local/lib/python3.12/dist-packages/nixl_ep \
         /usr/local/lib/python3.12/dist-packages/nixl_ep_cu12 \
         /usr/local/lib/python3.12/dist-packages/nixl_ep_cu13; do
  rm -rf "$d"
done
python3 "$DIR/check_absent.py"
