#!/usr/bin/env python3
"""tools+response_format coexist: the user's schema wins on auto tool choice.

ROOT CAUSE (2026-10-19, TC-65/66/67/69, ~2 pts on tool-eval-bench hard):
0.32 (77154ca) composes a structural tag whenever tools are present and the
tool parser supports structural tags: ``Parser._apply_structural_tag``
(vllm/parser/abstract_parser.py) builds ``OrFormat([tool_call_tag,
output_format])`` from the user's response_format and REPLACES the request's
grammar (``request.structured_outputs = <tag>``; ``request.response_format =
None``). The tool-call branch of the Or is legal, so a model that prefers to
answer a schema'd request with another tool call (our eval turns: tool result
in context, response_format active) does exactly that - the schema is never
enforced, finish_reason=tool_calls, and with the turn budget removed the
request loops tool calls until the token budget dies (runaway generation).

On 0.31.0 (db9527a468) the same composition code existed but never engaged
for GLM: the vllm-side structural-tag registry had no glm_4_7 builder and
xgrammar 0.2.7 had no builtin for it, so ``tools_structural_tag`` was None
and the function returned with the user's response_format INTACT - the JSON
schema was the ONLY grammar, the tool-call branch was illegal, and the model
produced schema-valid JSON (the passing e7 behavior). 0.32 added the vllm
glm_4_7 builder (tool_parsers/structural_tag_registry.py, _glm_4_7_tool_tag)
plus xgrammar 0.2.8, which armed the composition.

THE FIX (0.31 effective semantics, per the 2026-10-19 decision): when the
request carries an explicit structured output (output_format is not None)
AND tool_choice is auto, skip the structural-tag composition entirely and
keep the user's schema as THE grammar - tool-call tokens are then illegal
while the schema-constrained content is produced, exactly like e7/0.31.
tool_choice=required/named keep the composed tag (forcing a tool call is the
user's stated intent there). Tools remain in the prompt; the model simply
cannot end its turn with a tool call when a response_format is set.

Gate: GLM53_TOOLS_GRAMMAR_COEXIST (default '1'; '0' restores 0.32 composed
behavior - reboot only, no rebuild).

Fails closed (count==1 anchors, ast.parse before write); idempotent via the
[glm53-tools-grammar-coexist] marker; atomic write; stale pyc cleared.
"""
from __future__ import annotations

import ast
import os
import stat
import sys
from pathlib import Path

D = os.environ.get(
    "GLM53_VLLM_DIR", "/usr/local/lib/python3.12/dist-packages/vllm"
)
AP = Path(os.environ.get("GLM53_ABSTRACT_PARSER_PY", D + "/parser/abstract_parser.py"))
MARKER = "[glm53-tools-grammar-coexist]"

ANCHOR_IMPORT = "from vllm.logger import init_logger\n"
PATCHED_IMPORT = "import os\n\nfrom vllm.logger import init_logger\n"

ANCHOR_COMPOSE = """        if resolved_tools is not None:
            tag_request = request
            if output_format is not None:
"""

PATCHED_COMPOSE = """        # [glm53-tools-grammar-coexist] With an explicit user schema and
        # tool_choice=auto, keep the schema as the ONLY grammar (pre-0.32
        # behavior): the composed OrFormat(tool_call_tag, schema) makes the
        # tool-call branch legal, and models that prefer tool-calling on
        # schema'd turns then bypass the schema entirely (finish=tool_calls,
        # schema never enforced; looped tool calls can run the token budget
        # dry). required/named tool choices still compose below - forcing a
        # call there is the user's stated intent.
        if (
            output_format is not None
            and is_auto
            and os.environ.get("GLM53_TOOLS_GRAMMAR_COEXIST", "1") != "0"
        ):
            logger.warning_once(
                "tools + structured outputs with tool_choice=auto: keeping the "
                "structured output schema as the only grammar (pre-0.32 "
                "behavior); tool calls are not constrained or possible on this "
                "request.",
                scope="local",
            )
            return request

        if resolved_tools is not None:
            tag_request = request
            if output_format is not None:
"""


def _read(p: Path) -> str:
    st = os.stat(p)
    if not stat.S_ISREG(st.st_mode):
        raise SystemExit(f"[tools-grammar-coexist] not a regular file: {p}")
    return p.read_text()


def _replace_file(p: Path, new: str) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp-coexist")
    tmp.write_text(new)
    os.replace(tmp, p)
    pyc_dir = p.parent / "__pycache__"
    if pyc_dir.is_dir():
        for f in pyc_dir.glob(p.stem + ".*.pyc"):
            f.unlink(missing_ok=True)


def main() -> int:
    if not AP.is_file():
        raise SystemExit(f"[tools-grammar-coexist] missing {AP}")
    src = _read(AP)
    if MARKER in src:
        if PATCHED_COMPOSE not in src:
            raise SystemExit(
                "[tools-grammar-coexist] marker present but patched body "
                "drifts; refusing to touch (fail closed)"
            )
        print(f"{AP.name}: tools-grammar-coexist no-op")
        return 0
    n = src.count(ANCHOR_COMPOSE)
    if n != 1:
        raise SystemExit(
            f"[tools-grammar-coexist] compose anchor count {n} != 1; "
            "refusing to patch (fail closed)"
        )
    patched = src
    if "import os\n" not in patched.split("\n\n", 1)[0]:
        n_imp = patched.count(ANCHOR_IMPORT)
        if n_imp != 1:
            raise SystemExit(
                f"[tools-grammar-coexist] import anchor count {n_imp} != 1; "
                "refusing to patch (fail closed)"
            )
        patched = patched.replace(ANCHOR_IMPORT, PATCHED_IMPORT, 1)
    patched = patched.replace(ANCHOR_COMPOSE, PATCHED_COMPOSE, 1)
    try:
        ast.parse(patched)
    except SyntaxError as e:
        raise SystemExit(f"[tools-grammar-coexist] patched tree does not parse: {e}")
    _replace_file(AP, patched)
    compile(patched, str(AP), "exec")
    env = os.environ.get("GLM53_TOOLS_GRAMMAR_COEXIST", "1")
    print(
        f"{AP.name}: tools-grammar-coexist patched (GLM53_TOOLS_GRAMMAR_COEXIST={env!r})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
