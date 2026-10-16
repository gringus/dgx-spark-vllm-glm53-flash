#!/usr/bin/env python3
"""Remove the eagle block back-off from get_mamba_prefill_checkpoint_position.

PR #60533 fixed the double cap (P-1 lookup limit plus the EAGLE drop) in the
LOOKUP limit, but the checkpoint position still subtracts one extra hash unit
when drop_eagle_block is set: checkpoint = proof_limit//U*U - U. On our hybrid
GLM-5.3-Flash geometry (P=18407, U=1152) that caps every mamba-keyed boundary
at 16128 (14x1152) instead of 17280 (15x1152), and the scheduler's block-
aligned split forces chunk ends exactly there, so the computed ~2279-token
tail can never register. The eagle drafter group is exempt from the hybrid
min-intersection by our [glm53-hybrid-apc] mod, so removing the back-off does
not let prefill lookahead pollute the drafter.

Fails closed (count==1 anchors); idempotent via marker.
"""
from pathlib import Path

P = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/kv_cache_interface.py")
s = P.read_text()
if "[glm53-eagle-backoff]" in s:
    print("[eagle-backoff] already present")
    raise SystemExit(0)

OLD = """    checkpoint_position = proof_limit // hash_block_size * hash_block_size
    if drop_eagle_block:
        checkpoint_position -= hash_block_size
"""
NEW = """    checkpoint_position = proof_limit // hash_block_size * hash_block_size
    # [glm53-eagle-backoff] The extra hash-unit back-off re-shortcaps every
    # mamba-keyed boundary (16128 = 14x1152 on our probes) after PR #60533
    # already fixed the lookup side; the eagle drafter group is exempt from
    # the hybrid min so the boundary is safe at proof_limit.
"""
assert s.count(OLD) == 1, s.count(OLD)
s = s.replace(OLD, NEW, 1)
P.write_text(s)
import ast
ast.parse(P.read_text())
print("[eagle-backoff] applied: eagle back-off removed")
