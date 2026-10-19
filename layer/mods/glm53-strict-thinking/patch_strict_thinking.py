#!/usr/bin/env python3
"""Strict thinking for grammar-active requests (upstream RFC vllm-project/vllm#60722).

On 0.32 (77154ca base), a structured-output request whose reasoning ends
IMPLICITLY - the model jumps from reasoning straight into a tool-call header
without committing the explicit THINK_END token id - never latches
``reasoning_ended``: the id-scan in ``StructuredOutputManager.accept_tokens``
sees neither the explicit end id nor the TOOL_CALL_START special id (the
header arrives as ordinary BPE text), ``_get_constraint_bounds`` keeps
returning ``(len+1, len+1)``, and ``grammar_bitmask`` leaves every row -
draft rows and the bonus row alike - unconstrained (``_full_mask``). The
model then free-runs tool-call tokens under an active JSON schema:
non-terminating tool-call loops (TC-65/66/67/69, ~2 points on tool-eval-bench
hard) and, with turn/token budgets removed, an unbounded-generation serving
hang (observed: +129,778 tokens at ~14 tok/s over 2h34m, spec acceptance
frozen at 1.000/0.000x6, Running/Waiting/Deferred stuck).

THE FIX (per RFC #60722's proposed direction, scoped): for requests with an
ACTIVE grammar whose constraint has NOT started (reasoning phase), the
per-position bitmask rows additionally CLEAR the bit of the implicit
reasoning-end marker (the parser's TOOL_CALL_START token-id terminal), so the
model cannot enter a tool call while reasoning and is pushed to commit the
explicit THINK_END - after which the stock constraint path engages normally.
The explicit end stays allowed; ordinary (non-grammar) requests are untouched
(tool-calling turns legitimately end reasoning implicitly); xgrammar bit
convention: 1 = allowed, 0 = banned, so banning = clearing the bit
(composes with _full_mask == -1 all-allowed rows and with real grammar rows).

This is the RFC's accepted trade: masking the implicit end changes the
distribution during reasoning for grammar-active requests - that IS the
point (restores the e7/0.31 effective semantics).

Fail-closed (count==1 anchors, ast.parse before write); idempotent via the
[glm53-strict-thinking] marker; atomic write; stale pyc cleared. No runtime
gate (the ban only fires where stock behavior leaves the grammar disengaged).
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
SO = Path(
    os.environ.get(
        "GLM53_STRUCTURED_OUTPUT_PY", D + "/v1/structured_output/__init__.py"
    )
)
MARKER = "[glm53-strict-thinking]"

ANCHOR_BOUNDS = """                req_tokens = scheduled_spec_decode_tokens.get(req_id, list())
                grammar_start, constraint_start = self._get_constraint_bounds(
                    request, strip_speculative_padding(req_tokens)
                )
                state_advancements = 0
"""

PATCHED_BOUNDS = """                req_tokens = scheduled_spec_decode_tokens.get(req_id, list())
                glm53_stripped = strip_speculative_padding(req_tokens)
                grammar_start, constraint_start = self._get_constraint_bounds(
                    request, glm53_stripped
                )
                # [glm53-strict-thinking] While the constraint has not started
                # (reasoning phase of a grammar-active request), ban the
                # implicit reasoning-end marker on every row of this request
                # (draft rows AND the bonus row): the model must commit the
                # explicit THINK_END instead of sliding into a tool call,
                # which would leave the grammar disengaged for the whole
                # request (unmasked tool-call emission, non-terminating loops,
                # runaway generation - see RFC vllm-project/vllm#60722).
                # Non-grammar requests never reach this loop.
                glm53_ban = (
                    self._glm53_implicit_end_ban(request)
                    if constraint_start > len(glm53_stripped)
                    else ()
                )
                state_advancements = 0
"""

ANCHOR_ROWFILL = """                    self._fill_bitmasks(((grammar, cumulative_index, apply_bitmask),))
"""

PATCHED_ROWFILL = """                    self._fill_bitmasks(((grammar, cumulative_index, apply_bitmask),))
                    # [glm53-strict-thinking] clear the marker's allow-bit
                    # (xgrammar bitmask: 1 = allowed, 0 = banned).
                    if glm53_ban:
                        glm53_row = bitmask[cumulative_index]
                        for glm53_tid in glm53_ban:
                            glm53_row[glm53_tid >> 5] &= ~(1 << (glm53_tid & 31))
"""

ANCHOR_BONUSFILL = """                        self._fill_bitmasks(((grammar, cumulative_index, bonus_apply),))
"""

PATCHED_BONUSFILL = """                        self._fill_bitmasks(((grammar, cumulative_index, bonus_apply),))
                        # [glm53-strict-thinking] same ban on the bonus row -
                        # this is the position where an unconstrained model
                        # would emit the tool-call header instead of ending
                        # reasoning explicitly.
                        if glm53_ban:
                            glm53_row = bitmask[cumulative_index]
                            for glm53_tid in glm53_ban:
                                glm53_row[glm53_tid >> 5] &= ~(1 << (glm53_tid & 31))
"""

ANCHOR_HELPER = """    def _fill_bitmasks(
"""

PATCHED_HELPER = '''    def _glm53_implicit_end_ban(self, request) -> tuple[int, ...]:
        """[glm53-strict-thinking] Implicit reasoning-end ids to ban.

        Resolves the parser's TOOL_CALL_START token-id terminal (the implicit
        reasoning end per RFC vllm-project/vllm#60722) from the request's
        engine-based reasoning parser and returns its vocab id. Empty unless
        an engine-based parser with a registered TOOL_START terminal is
        attached. Fail-open: any error keeps stock behavior.
        """
        try:
            structured_req = getattr(request, "structured_output_request", None)
            if structured_req is None:
                return ()
            reasoner = structured_req.reasoner or self._get_reasoner(request)
            if reasoner is None:
                return ()
            engine = getattr(reasoner, "_parser_engine", None)
            config = getattr(engine, "parser_engine_config", None)
            terminals = getattr(config, "token_id_terminals", None) or {}
            surface = terminals.get("TOOL_START")
            if not surface:
                return ()
            tokenizer = getattr(self, "tokenizer", None) or getattr(
                reasoner, "tokenizer", None
            )
            get_vocab = getattr(tokenizer, "get_vocab", None)
            tid = get_vocab().get(surface) if get_vocab is not None else None
            return (tid,) if tid is not None else ()
        except Exception:  # noqa: BLE001 - fail open to stock behavior
            return ()

    def _fill_bitmasks(
'''


def _read(p: Path) -> str:
    st = os.stat(p)
    if not stat.S_ISREG(st.st_mode):
        raise SystemExit(f"[strict-thinking] not a regular file: {p}")
    return p.read_text()


def _replace_file(p: Path, new: str) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp-strictthink")
    tmp.write_text(new)
    os.replace(tmp, p)
    pyc_dir = p.parent / "__pycache__"
    if pyc_dir.is_dir():
        for f in pyc_dir.glob(p.stem + ".*.pyc"):
            f.unlink(missing_ok=True)


def main() -> int:
    if not SO.is_file():
        raise SystemExit(f"[strict-thinking] missing {SO}")
    src = _read(SO)
    if MARKER in src:
        if PATCHED_BOUNDS not in src or "_glm53_implicit_end_ban" not in src:
            raise SystemExit(
                "[strict-thinking] marker present but patched body drifts; "
                "refusing to touch (fail closed)"
            )
        print(f"{SO.name}: strict-thinking no-op")
        return 0
    for name, anchor in (
        ("bounds", ANCHOR_BOUNDS),
        ("rowfill", ANCHOR_ROWFILL),
        ("bonusfill", ANCHOR_BONUSFILL),
        ("helper", ANCHOR_HELPER),
    ):
        n = src.count(anchor)
        if n != 1:
            raise SystemExit(
                f"[strict-thinking] {name} anchor count {n} != 1; "
                "refusing to patch (fail closed)"
            )
    patched = src.replace(ANCHOR_BOUNDS, PATCHED_BOUNDS, 1)
    patched = patched.replace(ANCHOR_ROWFILL, PATCHED_ROWFILL, 1)
    patched = patched.replace(ANCHOR_BONUSFILL, PATCHED_BONUSFILL, 1)
    patched = patched.replace(ANCHOR_HELPER, PATCHED_HELPER, 1)
    try:
        ast.parse(patched)
    except SyntaxError as e:
        raise SystemExit(f"[strict-thinking] patched tree does not parse: {e}")
    _replace_file(SO, patched)
    compile(patched, str(SO), "exec")
    print(f"{SO.name}: strict-thinking patched (bounds + rowfill + bonus + helper)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
