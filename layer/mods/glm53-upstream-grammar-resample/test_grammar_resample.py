#!/usr/bin/env python3
"""Standalone logic tests for glm53-grammar-resample (no vllm import needed).

1. Resample math: drawing the grammar-rejected position from the full masked
   target p (the #60451 bug) vs the residual max(p - q, 0) (the fix) gives
   different distributions - and at temperature 0 both argmax to the SAME
   token, so the fix is output-neutral for the temp-0 TEB eval.
2. Sync-path keep/trim interplay with glm53-adaptive-k: the kept draft is
   appended only when the valid prefix survived the uniform trim.

Run: python3 test_grammar_resample.py   (exits non-zero on failure)
"""
import random

# ---------------------------------------------------------------- 1. math
# Row for the kept grammar-invalid draft: real grammar mask (draft token A is
# forbidden -> p_masked[A] = 0); draft proposal q liked A.
VOCAB = ["A", "B", "C", "D"]
p_masked = [0.0, 0.6, 0.3, 0.1]
q = [0.7, 0.1, 0.1, 0.1]

residual = [max(p - qv, 0.0) for p, qv in zip(p_masked, q)]
s = sum(residual)
assert s > 0, "residual must have mass"
residual_norm = [r / s for r in residual]

tv = 0.5 * sum(abs(a - b) for a, b in zip(p_masked, residual_norm))
assert tv > 1e-9, "biased (from-p) and residual draws must differ"
# The bias always removes draft mass: residual <= p elementwise, strictly
# somewhere (here C: 0.2 vs 0.3).
assert all(r <= p + 1e-12 for r, p in zip(residual, p_masked))
assert residual_norm[2] < p_masked[2], "draft-backed token C must lose mass"

# Temperature 0: both flows argmax the same row -> identical output token.
assert max(range(4), key=lambda i: p_masked[i]) == max(
    range(4), key=lambda i: residual_norm[i]
), "temp-0 correction token must be identical under bug and fix"

# Monte-Carlo sanity of the draw distributions.
rng = random.Random(42)
n = 200_000
draw_p = [0] * 4
draw_r = [0] * 4
for _ in range(n):
    x, acc = rng.random(), 0.0
    for i, p in enumerate(p_masked):
        acc += p
        if x <= acc:
            draw_p[i] += 1
            break
    x, acc = rng.random(), 0.0
    for i, p in enumerate(residual_norm):
        acc += p
        if x <= acc:
            draw_r[i] += 1
            break
for i in range(4):
    assert abs(draw_p[i] / n - p_masked[i]) < 0.01
    assert abs(draw_r[i] / n - residual_norm[i]) < 0.01

# ------------------------------------------------- 2. keep/trim interplay
class R:  # minimal Request stand-in
    def __init__(self, use_so=True):
        self.use_structured_output = use_so
        self.spec_token_ids = []
        self.num_invalid_spec_tokens = 0


def validate_tokens(spec):  # grammar validates the first two tokens here
    return spec[:2]


class SpecCfg:
    draft_sample_method = "probabilistic"
    rejection_sample_method = "standard"


def verifies(request, gate="1", terminated=False, cfg=SpecCfg()):
    if gate == "0":
        return False
    return (
        cfg.draft_sample_method == "probabilistic"
        and cfg.rejection_sample_method != "synthetic"
        and not terminated
    )


def update_draft_token_ids(request, spec, adaptive_k_n, gate="1", cfg=SpecCfg()):
    """Mirrors the patched update_draft_token_ids + adaptive-k trim."""
    valid = validate_tokens(spec)
    request.num_invalid_spec_tokens = 0
    candidate = None
    valid_len = None
    if request.use_structured_output and len(valid) < len(spec):
        candidate = spec[len(valid)]
        valid_len = len(valid)
    request.spec_token_ids = valid
    if adaptive_k_n is not None and request.spec_token_ids:
        request.spec_token_ids = request.spec_token_ids[:adaptive_k_n]
    kept = False
    if candidate is not None and gate == "1":
        if len(request.spec_token_ids) == valid_len and verifies(request, gate, cfg=cfg):
            request.spec_token_ids.append(candidate)
            request.num_invalid_spec_tokens = 1
            kept = True
    return kept


# (a) untrimmed: invalid draft kept as the trailing slot.
r = R()
assert update_draft_token_ids(r, [10, 11, 12, 13], adaptive_k_n=None)
assert r.spec_token_ids == [10, 11, 12] and r.num_invalid_spec_tokens == 1

# (b) adaptive-k "trims" to exactly the valid prefix length (no actual cut):
#     still kept.
r = R()
assert update_draft_token_ids(r, [10, 11, 12, 13], adaptive_k_n=2)
assert r.spec_token_ids == [10, 11, 12] and r.num_invalid_spec_tokens == 1

# (c) adaptive-k makes a REAL cut into the valid prefix: not kept (today's
#     behavior), count 0.
r = R()
assert not update_draft_token_ids(r, [10, 11, 12, 13], adaptive_k_n=1)
assert r.spec_token_ids == [10] and r.num_invalid_spec_tokens == 0

# (d) gate off: never kept.
r = R()
assert not update_draft_token_ids(r, [10, 11, 12, 13], adaptive_k_n=None, gate="0")
assert r.spec_token_ids == [10, 11] and r.num_invalid_spec_tokens == 0

# (d) greedy draft method: never kept (one-hot draft -> dropping was exact).
class GreedyCfg(SpecCfg):
    draft_sample_method = "greedy"


r = R()
assert not update_draft_token_ids(
    r, [10, 11, 12, 13], adaptive_k_n=None, cfg=GreedyCfg()
)
assert r.spec_token_ids == [10, 11] and r.num_invalid_spec_tokens == 0

# (e) synthetic rejection: never kept.
class SynthCfg(SpecCfg):
    rejection_sample_method = "synthetic"


assert verifies(R(), cfg=SynthCfg()) is False

# (f) terminated grammar: never kept.
assert verifies(R(), terminated=True) is False

print("test_grammar_resample: ALL OK")
