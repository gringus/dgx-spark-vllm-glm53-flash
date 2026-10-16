#!/bin/sh
# e6 re-derivation (upstream nightly 385dce36b) of the kit
# patch_suppress_stops_in_reasoning.py (@aad6481): keep client stop strings dormant
# while generation is inside <think> (think-in-prompt); guard arms only when the last
# prompt token is the reasoning start marker; spec-decode chunk that carries
# </think> advances stop_check_offset past the marker.
# All 4 anchors verified byte-identical on nightly v1/engine/detokenizer.py
# (import pair, detok factory, init fields, stop-check block). Native machinery
# confirmed present and NOT superseding this patch: stop_check_offset exists
# (~:120-142, used for new_char_count) and ReasoningConfig
# (vllm/config/reasoning.py) carries reasoning_start_str/reasoning_end_str which
# the helper already prefers — but the native reasoning code is thinking-budget
# forcing at sampling time, it does not gate detokenizer stop strings. No stages
# dropped. Opt-out GLM53_SUPPRESS_STOPS_IN_REASONING=0 / VLLM_...=0.
# Fail-closed on anchor drift; idempotent.
set -eu
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python3 "$DIR/patch_suppress_stops_in_reasoning.py"
