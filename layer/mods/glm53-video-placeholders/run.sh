#!/bin/sh
# e7 re-derivation (v0.31.0 db9527a468) of the kit patch_glm_video_placeholders.py
# (@aad6481).
#
# ONE change remains (the kpool half is retired):
# 1. Video placeholder alignment (runtime monkeypatch + .pth hook): the native
#    _construct_video_placeholder (glm4_1v.py:1388) still iterates raw
#    timestamps with no alignment to grid_t (a 1s clip = 3 timestamps vs T=1
#    -> 100 encoder tokens spread over 300 slots). Patch REQUIRED.
#    ANCHOR DRIFT: _get_video_frame_embed_token_id was removed upstream
#    (#57556); the native builder now reads
#    `frame_embed_token_id = hf_processor.image_token_id` (glm4_1v.py:1425)
#    and the hook's embed-id line is re-derived to match. Helpers verified
#    present with same semantics: _is_glmga_model (:1317),
#    _get_video_second_idx_glm46v (:1240); Glm4vProcessor imported from
#    transformers (glm4_1v.py:40) and still re-exported from the glm4_1v
#    namespace. "Glm5" name check routes Glm5NextProcessor to the glm46v
#    branch exactly as on the pin. main() fail-closed preflights 6
#    glm4_1v.py anchors (each count==1) before installing anything.
# 2. RETIRED (kpool persistent_topk disable): upstream deleted
#    vllm/model_executor/layers/sparse_attn_indexer_kpool.py in the v0.31.0
#    bump; selection moved to the `sparse_indexer_topk_backend` config knob
#    (vllm/config/kernel.py:278) — set it via recipe/kernel-config, not a
#    patch. All kpool code removed from this mod.
# DROPS: kpool half only. NEEDS-REVIEW: none.
# Opt-out: run.sh is the only trigger (sitecustomize import fires on every
# python start in the image); glm4_1v/patch/pth paths env-overridable via
# GLM53_SITEPACKAGES / GLM53_GLM4_1V_PY / GLM53_VIDEO_PATCH_PY /
# GLM53_VIDEO_PTH.
# Fail-closed on anchor drift; idempotent.
set -eu
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python3 "$DIR/patch_glm_video_placeholders.py"
