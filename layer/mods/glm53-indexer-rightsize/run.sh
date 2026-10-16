# e7 re-derivation (v0.31.0 db9527a468) of the e6 glm53-indexer-rightsize mod.
# Anchor moves: ANCHOR_IMPORT retargeted to "import numpy as np / import torch"
# (v0.31.0 inserts the numpy import between typing and torch); ANCHOR_SPLITTER
# re-indented one level deeper (renamed tensors, tuple assignment) with the
# fail-closed body re-indented to match. Path moves: GLM call site now
# vllm/models/glm5next/common/attention.py; kpool cap now
# vllm/models/glm5next/nvidia/sparse_indexer.py (module-level
# sparse_attn_indexer_kpool(...) takes total_seq_lens/index_kpool directly).
# All anchors verified count==1 on the v0.31.0 tree. Semantics unchanged.
#!/bin/sh
# Applies the kit indexer-workspace rightsize patch (staged next to this script).
# Runtime-gated by GLM53_INDEXER_WORKSPACE (stock default; the arm sets rightsize).
# GLM-only: the generic get_max_prefill_buffer_size stays stock so other models and
# the rank-0 DFlash2 drafter cannot enter the experiment. Fails closed (non-zero ->
# sparkrun aborts the deploy) on anchor drift; idempotent; atomic writes.
set -eu
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python3 "$DIR/patch_indexer_workspace.py"
