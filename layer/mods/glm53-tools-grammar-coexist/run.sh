#!/bin/sh
# glm53-tools-grammar-coexist: with tools + response_format (structured
# outputs) and tool_choice=auto, keep the USER'S schema as the only grammar
# (pre-0.32 behavior). 0.32 composes OrFormat(tool_call_tag, schema) in
# Parser._apply_structural_tag (vllm/parser/abstract_parser.py) and nulls
# response_format whenever the tool parser supports structural tags - the
# tool-call branch is then LEGAL, so a model that prefers to answer a
# schema'd turn with another tool call bypasses the schema entirely
# (finish=tool_calls, schema unenforced; looped tool calls ran the token
# budget dry = the 2026-10-18/19 runaway stall). 0.31 never composed for GLM
# (no glm_4_7 tag builder; xgrammar 0.2.7 lacked the builtin) - the schema
# alone constrained output and the tool branch was illegal (passing e7
# behavior). required/named tool_choice still compose (forced calls are the
# user's intent). Tools stay in the prompt; the model simply cannot end a
# schema'd turn with a tool call. Evidence: minimal repro (tools+rf ->
# finish=tool_calls, content=None; same request sans tools -> schema
# enforced, finish=stop), TC-65/66/67/69 = ~2 pts on tool-eval-bench hard.
# Gate: GLM53_TOOLS_GRAMMAR_COEXIST (default '1'; '0' restores 0.32 composed
# behavior - reboot only, no rebuild).
# Fail-closed (count==1 anchors, ast.parse), idempotent (marker), atomic
# write, stale pyc cleared.
set -eu
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python3 "$DIR/patch_tools_grammar_coexist.py"
