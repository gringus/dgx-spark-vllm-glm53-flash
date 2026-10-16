#!/bin/sh
# e6 mitigation (2026-10-08): build_grouped_fat_tables' ~125 small device ops
# serialize behind the in-flight MoE kernel in the live engine (queue delay per
# op) - the microbench on an idle GPU (0.17ms) structurally under-measures it.
# This mod replaces the op chain with 2 Triton kernels (prep + main), preserving
# the original as _build_grouped_fat_tables_reference for bit-exact validation
# and CPU fallback. Fail-closed on anchor drift; marker [glm53-fused-tables].
set -eu
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python3 "$DIR/patch_fused_tables.py"
