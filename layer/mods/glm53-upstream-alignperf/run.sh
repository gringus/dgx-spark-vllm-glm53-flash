#!/bin/sh
# glm53-upstream-alignperf: route Mamba2 metadata state-index fetches through
# the MRV2 one-launch aligned-index precompute (the low-risk core of upstream
# PR #58851, re-derived on this tree's mamba_attn.py — the PR's own diff bases
# on the post-#57169 packed-KV rewrite this tree predates).
# Kills the per-group mamba_get_block_table_tensor gathers per decode step
# (the ctx_tg c2 ITL spike 0.136-0.158 vs e5b-c 0.098-0.104).
# Gate: GLM53_MAMBA_ALIGNPERF (default '1'; '0' restores the per-group
# gathers — reboot only, no rebuild).
set -eu
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python3 "$DIR/patch_alignperf.py"
