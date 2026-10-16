#!/bin/sh
# glm53-upstream-grammar-resample: backport the core of upstream PR #60690
# (fixes #60451) to the 77154ca tree: keep the FIRST grammar-invalid draft so
# verification rejects it and resamples that position from the residual
# max(p - q, 0) instead of the full masked target p. Only drafts AFTER it are
# dropped/-1. Sync path = scheduler update_draft_token_ids truncation fix +
# num_invalid_spec_tokens propagation; the bitmask fill treats the kept draft
# as a cut slot (no grammar advance through a rejected token) and
# num_acceptable_drafts counts it so the runner does NOT -1 it.
# (The runner/_resample_kernel needs NO change: a kept, rejected draft takes
# the normal rejected-row residual path - exactly upstream's "scheduler only"
# fix. #61557's rework is not ported: draft, self-admitted sync-path perf
# regression and async grammar desync.)
# Bias scope: draft_sample_method=probabilistic (ours) + rejection != synthetic
# + grammar not terminated. Greedy drafts are one-hot -> dropping was already
# exact; at target temp 0 the correction is argmax over the same real-masked
# row either way, so this is output-neutral for the temp-0 TEB eval and
# distribution-correcting for temp>0 traffic (benchy samples at temp 1.0).
# Interplay with glm53-adaptive-k-b0 (must apply AFTER it): the kept draft is
# appended only when the valid prefix survived adaptive-k's uniform trim, so
# the trim invariant and cudagraph shapes are untouched; trimmed requests keep
# today's behavior.
# Gate: GLM53_GRAMMAR_RESAMPLE (default '1'; '0' restores upstream-77154ca
# behavior - reboot only, no rebuild).
set -eu
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python3 "$DIR/patch_grammar_resample.py"
