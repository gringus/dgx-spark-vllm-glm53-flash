#!/bin/sh
# glm53-strict-thinking: strict-thinking for grammar-active requests, per
# upstream RFC vllm-project/vllm#60722. On 0.32 a structured-output request
# whose reasoning ends IMPLICITLY (model jumps from reasoning straight into
# the tool-call header as plain BPE text - the THINK_END / TOOL_CALL_START
# special ids never commit) never latches reasoning_ended: the constraint
# bounds stay (len+1, len+1), grammar_bitmask leaves every row unconstrained,
# and the request free-runs tool-call tokens under an active JSON schema -
# non-terminating tool-call loops (TC-65/66/67/69, ~2 pts on tool-eval-bench
# hard) and, unbounded, a serving hang (observed +129,778 tokens at ~14 tok/s
# over 2h34m, spec acceptance 1.000/0.000x6 frozen, Running/Waiting/Deferred
# stuck). Fix: while the constraint has not started, clear the TOOL_CALL_START
# token's allow-bit in every bitmask row of grammar-active requests (xgrammar
# convention 1=allowed/0=banned; composes with _full_mask=-1 all-allowed rows
# and with real grammar rows). The explicit THINK_END stays allowed, so the
# model commits it and the stock constraint path engages. Non-grammar requests
# never enter grammar_bitmask - ordinary tool-calling turns keep implicit ends
# (agentic tool calling unaffected). RFC-accepted trade: the ban changes the
# reasoning-phase distribution for grammar-active requests - that IS the fix.
# Fail-closed (count==1 anchors, ast.parse), idempotent ([glm53-strict-thinking]
# marker), atomic write, stale pyc cleared. No runtime gate.
set -eu
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python3 "$DIR/patch_strict_thinking.py"
