#!/usr/bin/env bash
# Build the GLM-5.3-Flash vllm serving image from this repo.
#
# Requirements on the build host (aarch64, DGX Spark / GB10 class):
#   - docker with buildkit, >=100G free disk on /
#   - the ENGINE MUST BE DOWN while building (nvcc wants >10G host RAM; building
#     while the engine runs invites the GB10 UVM-livelock class)
#   - nothing else: the kit clone and the mod layer come from this repo
#
# Env:
#   IMAGE       ghcr target, e.g. ghcr.io/gringus/dgx-spark-vllm-glm53-flash (required)
#   REF         tag suffix for $IMAGE (default: latest)
#   LOCAL_TAG   local docker tag produced by the chain (default glm53-selfbuild:serving)
#   SHIP_OTHER  1 = ship to the peer spark node after build (default 1; 0 = skip)
#   KIT_BRANCH  kit branch (default e10-exl3163 = exl3 1.6.3 retarget)
#   BASE        the vllm nightly the serving lane requires (default: the pinned
#               77154ca cu134 digest; the lane REFUSEs on anything else)
set -euo pipefail
REPO_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
IMAGE=${IMAGE:-}
REF=${REF:-latest}
LOCAL_TAG=${LOCAL_TAG:-glm53-selfbuild:serving}
KIT_BRANCH=${KIT_BRANCH:-e10-exl3163}
# kit source: OUR fork carries the e8-exl3160 branch (exl3 1.6.0 retarget); upstream
# Reederey87 main is the 1.4.9-era kit and does not have the branch
KIT_REPO=${KIT_REPO:-https://github.com/gringus/glm53-flash-exl3-2x-dgx-spark.git}
BASE=${BASE:-vllm/vllm-openai@sha256:94ac05830c84fd3cfafc792d932e7ce1a5f42404b28f0d3b0dbcc63abdab30a8}
EXPECTED_BASE=${EXPECTED_BASE:-}
KIT=$HOME/glm53-kit

# 1. kit clone at the pinned branch
if [ -d "$KIT/.git" ]; then
  git -C "$KIT" fetch origin -q
else
  git clone --depth 50 "$KIT_REPO" "$KIT"
fi
git -C "$KIT" checkout -q "$KIT_BRANCH" 2>/dev/null || { git -C "$KIT" fetch origin "$KIT_BRANCH" -q && git -C "$KIT" checkout -q "$KIT_BRANCH"; }
echo "kit: $KIT_BRANCH at $(git -C "$KIT" rev-parse --short HEAD)"

# 2. stage this repo's mod layer into the kit build context
#    (file names match what scripts/glm53-selfbuild.sh expects)
cp "$REPO_DIR/layer/Dockerfile.serving-layer" "$KIT/"
cp "$REPO_DIR/layer/bake-serving.sh" "$KIT/bake-serving.sh"
cp "$REPO_DIR/layer/test_exl3_overlay.py" "$KIT/"
rm -rf "$KIT/mods-serving" && cp -r "$REPO_DIR/layer/mods" "$KIT/mods-serving"
( cd "$KIT" && ls Dockerfile.serving-layer bake-serving.sh mods-serving test_exl3_overlay.py >/dev/null && bash -n bake-serving.sh )

# 3. build (merged-Dockerfile generation + the 77154ca base swap live in the
#    vendored script; the bake inside the build is fail-closed, 31-marker check)
KIT="$KIT" TAG="$LOCAL_TAG" BASE="$BASE" EXPECTED_BASE="$EXPECTED_BASE" SHIP_OTHER="${SHIP_OTHER:-1}" bash "$REPO_DIR/scripts/glm53-selfbuild.sh"

# 4. publish (PUBLISH=0 for local experiment builds — no ghcr push)
IMG_ID=$(docker images --format '{{.ID}}' "$LOCAL_TAG" | head -1)
[ -n "$IMG_ID" ] || { echo "build produced no $LOCAL_TAG"; exit 1; }
if [ "${PUBLISH:-1}" = 1 ]; then
  [ -n "$IMAGE" ] || { echo "PUBLISH=1 needs IMAGE=ghcr.io/<owner>/<name>"; exit 1; }
  docker tag "$IMG_ID" "$IMAGE:$REF"
  docker push "$IMAGE:$REF"
  echo "PUBLISHED: $IMAGE:$REF ($IMG_ID)"
else
  echo "LOCAL BUILD DONE: $LOCAL_TAG ($IMG_ID) — publish skipped (PUBLISH=0)"
fi
