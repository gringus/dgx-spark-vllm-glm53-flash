#!/usr/bin/env python3
"""Backport the core of upstream PR #60690 (fixes #60451) to the 77154ca tree.

THE BUG (#60451, present since #54442 landed in this tree, absent at 0.31):
with spec decode + structured outputs + ``draft_sample_method="probabilistic"``,
the first grammar-invalid draft of a step was dropped from verification:
  - sync scheduling: ``update_draft_token_ids`` stored only the grammar-valid
    prefix (``validate_tokens`` truncates), so the invalid draft was never
    scheduled; when the valid prefix was fully accepted its position became
    the BONUS row and was drawn from the target distribution p.
  - async scheduling: ``num_acceptable_drafts`` excluded it, so
    ``grammar_invalid_drafts`` -1'd it too and ``_resample_kernel`` drew its
    row from p ("bonus" branch).
After a rejected draft, rejection sampling must draw from max(p - q, 0). A
draw from p is biased exactly when reaching that position depended on the
draft's value - which is the case here. e7/e8 (0.31) predate #54442 and verify
the draft normally, so they never had the bias.

THE FIX (upstream's, scheduler-only - no model-runner change): keep the FIRST
grammar-invalid draft for verification. Its bitmask row is filled with the
real grammar mask (state before the draft), so verification always rejects it
and the normal rejected-row residual path runs; only drafts AFTER it are
dropped/-1. Gated to probabilistic drafts + non-synthetic rejection + a grammar
that has not terminated (greedy drafts are one-hot: dropping was exact;
synthetic acceptance would accept the rejected token; a terminated grammar
masks nothing).

ADAPTIVE-K INTERPLAY (this mod must apply AFTER glm53-adaptive-k-b0): the
trimmed-uniform-length invariant is preserved - the kept draft is appended
only when the valid prefix survived the uniform trim; trimmed requests keep
today's behavior (their num_invalid_spec_tokens stays 0).

TEMP-0 SCOPE: at temp 0 _resample_kernel early-returns for non-bonus rows
("the target argmax is already in the sampled tensor") and the bonus branch
argmaxes the row's real-masked logits - the residual-vs-p distinction never
samples at temp 0, and both flows' correction rows carry the same grammar
state S_k'. The blanket "output-neutral at temp 0" claim is therefore
mechanism-specific, not a theorem: argmax(p) vs argmax(residual) CAN differ
in general, and the eval leg is the empirical arbiter. At temp > 0 the fix
restores the target distribution (upstream chi^2 p = 1.4e-09 FAIL -> 0.78
PASS); benchy samples at temp 1.0, so perf-bench accept rates read the fixed
path.

Fails closed (count==1 anchors, ast.parse before write); idempotent via the
[glm53-grammar-resample] marker. GLM53_GRAMMAR_RESAMPLE=0 restores
77154ca behavior (reboot only, no rebuild).
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

D = os.environ.get(
    "GLM53_VLLM_DIR", "/usr/local/lib/python3.12/dist-packages/vllm"
)
SCHED = Path(os.environ.get("GLM53_SCHEDULER_PY", D + "/v1/core/sched/scheduler.py"))
REQ = Path(os.environ.get("GLM53_REQUEST_PY", D + "/v1/request.py"))
MARKER = "[glm53-grammar-resample]"

# ---------------------------------------------------------------- request.py
REQ_ANCHOR = """\
        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
"""
REQ_NEW = """\
        self.spec_token_ids: list[int] = []
        # [glm53-grammar-resample] Trailing spec_token_ids the grammar
        # rejected (sync scheduling); propagated to SchedulerOutput so the
        # bitmask fill treats them as cut slots and the stats exclude them.
        self.num_invalid_spec_tokens = 0
        self.num_computed_tokens = 0
"""

# --------------------------------------------------------------- scheduler.py
# 1. schedule(): declare the per-step dict next to the spec-token dict.
INIT_ANCHOR = """\
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
"""
INIT_NEW = """\
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        num_invalid_spec_tokens: dict[str, int] = {}  # [glm53-grammar-resample]
"""

# 2. schedule(): a request preempted after it was scheduled in the same step
#    must drop its num_invalid entry (its drafts are no longer scheduled).
POP_ANCHOR = """\
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
"""
POP_NEW = """\
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            num_invalid_spec_tokens.pop(  # [glm53-grammar-resample]
                                preempted_req_id, None
                            )
"""

# 3. schedule(): propagate the kept-invalid count when the window was NOT
#    budget-trimmed (a trimmed window may have cut the kept draft off).
SCHED_ANCHOR = """\
                if num_scheduled_spec_tokens > 0:
                    spec_token_ids = request.spec_token_ids
                    if len(spec_token_ids) > num_scheduled_spec_tokens:
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids
"""
SCHED_NEW = """\
                if num_scheduled_spec_tokens > 0:
                    spec_token_ids = request.spec_token_ids
                    if len(spec_token_ids) > num_scheduled_spec_tokens:
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                    elif (  # [glm53-grammar-resample]
                        request.num_invalid_spec_tokens
                    ):
                        num_invalid_spec_tokens[request_id] = (
                            request.num_invalid_spec_tokens
                        )
                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids
"""

# 4. schedule(): hand the dict to SchedulerOutput (stats + bitmask fill read it).
OUT_ANCHOR = """\
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
"""
OUT_NEW = """\
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            num_invalid_spec_tokens=num_invalid_spec_tokens,  # [glm53-grammar-resample]
"""

# 5. get_grammar_bitmask(): fill invalid slots as cuts (no grammar advance
#    through a rejected token) and count the kept draft so the runner does
#    NOT -1 its row - it must reach verification.
GBIT_ANCHOR = """\
        bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduler_output.scheduled_spec_decode_tokens,
        )
        spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        num_acceptable_drafts = [
            len(strip_speculative_padding(spec_tokens.get(req_id, [])))
            for req_id in structured_output_request_ids
        ]
        return GrammarOutput(
            structured_output_request_ids, bitmask, num_acceptable_drafts
        )
"""
GBIT_NEW = """\
        # [glm53-grammar-resample] (#60451) Fill the rows of grammar-invalid
        # drafts as cut slots, without advancing the grammar through a draft
        # it rejected; count the first invalid draft in num_acceptable_drafts
        # so the runner verifies it (its row masks it out -> rejected ->
        # residual resample) instead of -1'ing it (bonus-row draw from p).
        spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        num_invalid = scheduler_output.num_invalid_spec_tokens or {}
        if num_invalid:
            spec_tokens = spec_tokens | {
                req_id: spec_tokens[req_id][:-n] + [-1] * n
                for req_id, n in num_invalid.items()
                if req_id in spec_tokens
            }
        bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            spec_tokens,
        )
        num_acceptable_drafts = [
            len(strip_speculative_padding(spec_tokens.get(req_id, [])))
            + (
                req_id in num_invalid
                and self._verifies_grammar_rejected_drafts(
                    self.requests.get(req_id)
                )
            )
            for req_id in structured_output_request_ids
        ]
        return GrammarOutput(
            structured_output_request_ids, bitmask, num_acceptable_drafts
        )

    def _verifies_grammar_rejected_drafts(  # [glm53-grammar-resample]
        self, request: Request | None
    ) -> bool:
        \"\"\"Whether the first draft the grammar rejected goes to verification.

        Its row masks it out, so verification rejects it and resamples that
        position from the residual. For a greedy draft that residual is the
        target distribution, so only sampled drafts need this. Synthetic
        acceptance ignores the row, so it must not see that draft. A
        terminated grammar rejects no draft, and its rows mask nothing.
        \"\"\"
        if os.environ.get("GLM53_GRAMMAR_RESAMPLE", "1") == "0":
            return False
        spec_config = self.vllm_config.speculative_config
        structured_req = getattr(request, "structured_output_request", None)
        grammar = getattr(structured_req, "grammar", None)
        return (
            request is not None
            and spec_config is not None
            and spec_config.draft_sample_method == "probabilistic"
            and spec_config.rejection_sample_method != "synthetic"
            and grammar is not None
            and not grammar.is_terminated()
        )
"""

# 6. update_draft_token_ids(): record the first grammar-rejected draft as a
#    candidate; it is appended AFTER adaptive-k's uniform trim so the trim
#    invariant holds (kept only when the valid prefix survived untrimmed).
UDTI_ANCHOR = """\
            # Add newly generated spec token ids to the request.
            request.spec_token_ids = self.structured_output_manager.validate_tokens(
                request, spec_token_ids
            )
            _ak_structured = bool(request.use_structured_output)  # [glm53-adaptive-k]
            if _GLM53_ADAPTIVE_K.enabled and request.spec_token_ids:  # [glm53-adaptive-k]
                _ak_reqs.append((request, _ak_structured))
        if _ak_reqs:  # [glm53-adaptive-k]
            _GLM53_ADAPTIVE_K.apply(_ak_reqs, self.requests)
"""
UDTI_NEW = """\
            # Add newly generated spec token ids to the request.
            valid_token_ids = self.structured_output_manager.validate_tokens(  # [glm53-grammar-resample]
                request, spec_token_ids
            )
            request.num_invalid_spec_tokens = 0  # [glm53-grammar-resample]
            _gr_candidate = None  # [glm53-grammar-resample]
            if (
                request.use_structured_output
                and len(valid_token_ids)
                < len(strip_speculative_padding(spec_token_ids))
            ):
                # First grammar-rejected draft (#60451); kept below only if it
                # survives adaptive-k's uniform trim.
                _gr_candidate = spec_token_ids[len(valid_token_ids)]
                _gr_valid_len = len(valid_token_ids)
            request.spec_token_ids = valid_token_ids
            _ak_structured = bool(request.use_structured_output)  # [glm53-adaptive-k]
            if _GLM53_ADAPTIVE_K.enabled and request.spec_token_ids:  # [glm53-adaptive-k]
                _ak_reqs.append((request, _ak_structured))
            if _gr_candidate is not None:  # [glm53-grammar-resample]
                _gr_reqs.append((request, _gr_candidate, _gr_valid_len))
        if _ak_reqs:  # [glm53-adaptive-k]
            _GLM53_ADAPTIVE_K.apply(_ak_reqs, self.requests)
        # [glm53-grammar-resample] (#60451) Keep the first grammar-rejected
        # draft as the trailing scheduled slot: its bitmask row masks it out,
        # verification rejects it and resamples the position from the residual
        # max(p - q, 0) instead of drawing it as the bonus row from p. Only
        # when the valid prefix survived adaptive-k's trim untrimmed, so the
        # uniform scheduled length per step is preserved.
        if _gr_reqs and (
            os.environ.get("GLM53_GRAMMAR_RESAMPLE", "1") != "0"
        ):
            for _gr_request, _gr_token, _gr_len in _gr_reqs:
                if (
                    len(_gr_request.spec_token_ids) == _gr_len
                    and self._verifies_grammar_rejected_drafts(_gr_request)
                ):
                    _gr_request.spec_token_ids.append(_gr_token)
                    _gr_request.num_invalid_spec_tokens = 1
"""

# schedule() head: the _gr_reqs accumulator next to _ak_reqs.
GRINIT_ANCHOR = """\
        _ak_reqs = []  # [glm53-adaptive-k]
"""
GRINIT_NEW = """\
        _ak_reqs = []  # [glm53-adaptive-k]
        _gr_reqs = []  # [glm53-grammar-resample]
"""


def patch_scheduler(text: str) -> str:
    for old, new, label in (
        (GRINIT_ANCHOR, GRINIT_NEW, "_gr_reqs accumulator"),
        (INIT_ANCHOR, INIT_NEW, "schedule() num_invalid dict"),
        (POP_ANCHOR, POP_NEW, "preemption-restore pop"),
        (SCHED_ANCHOR, SCHED_NEW, "schedule() propagation"),
        (OUT_ANCHOR, OUT_NEW, "SchedulerOutput kwarg"),
        (GBIT_ANCHOR, GBIT_NEW, "get_grammar_bitmask + helper"),
        (UDTI_ANCHOR, UDTI_NEW, "update_draft_token_ids keep-invalid"),
    ):
        n = text.count(old)
        if n != 1:
            raise SystemExit(
                f"scheduler.py {label}: expected exactly 1 anchor occurrence, "
                f"found {n}. The baked tree has drifted - re-derive the anchor."
            )
        text = text.replace(old, new, 1)
    return text


def patch_request(text: str) -> str:
    n = text.count(REQ_ANCHOR)
    if n != 1:
        raise SystemExit(
            f"request.py field anchor: expected exactly 1 occurrence, found {n}."
        )
    return text.replace(REQ_ANCHOR, REQ_NEW, 1)


def apply_file(path: Path, patch_fn) -> bool:
    text = path.read_text()
    if MARKER in text:
        print(f"{path.name}: {MARKER} already present - no-op")
        return False
    text = patch_fn(text)
    ast.parse(text, filename=str(path))
    tmp = path.with_name(f".{path.name}.glm53-gr.tmp")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(f"patched {path} ({MARKER})")
    return True


def main() -> int:
    changed = apply_file(REQ, patch_request)
    changed |= apply_file(SCHED, patch_scheduler)
    if changed:
        print(
            "glm53-grammar-resample: kept-first-grammar-invalid-draft backport "
            "applied (PR #60690 core on 77154ca; GLM53_GRAMMAR_RESAMPLE="
            f"{os.environ.get('GLM53_GRAMMAR_RESAMPLE', '1')})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
