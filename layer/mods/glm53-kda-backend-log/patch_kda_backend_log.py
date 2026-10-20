#!/usr/bin/env python3
"""Log the resolved KDA prefill backend at model build (glm5next).

The 0.32 tree resolves kda_prefill_backend (auto/triton/flashkda) silently
(kda.py:344) - nothing in the boot log shows which chunked-prefill kernel
the KDA layers actually use. FlashKDA engages via auto on SM90/SM10x/SM12x
bf16 head_dim-128 bounded-gate configs (GB10 qualifies); this one-line log
makes the engagement visible at every boot:
  GLM-5.3 KDA prefill backend: flashkda
Fail-closed (count==1 anchors, ast.parse before write); idempotent via the
[glm53-kda-backend-log] marker.
"""
import ast
import os
import sys
from pathlib import Path

D = os.environ.get("GLM53_VLLM_DIR", "/usr/local/lib/python3.12/dist-packages/vllm")
SO = Path(os.environ.get("GLM53_KDA_PY", D + "/models/glm5next/common/kda.py"))
MARKER = "[glm53-kda-backend-log]"

ANCHOR_IMPORT = "from vllm.forward_context import get_forward_context\n"
PATCHED_IMPORT = (
    "from vllm.forward_context import get_forward_context\n"
    "\n"
    "from vllm.logger import init_logger\n"
    "\n"
    "logger = init_logger(__name__)\n"
)

ANCHOR_RESOLVE = (
    "        self.kda_prefill_backend = _resolve_kda_prefill_backend(\n"
    '            additional_config.get("kda_prefill_backend", "auto")\n'
    "            if isinstance(additional_config, dict)\n"
    '            else "auto",\n'
    "            self.head_dim,\n"
    "            vllm_config.model_config.dtype,\n"
    "            self.kda_lower_bound,\n"
    "        )\n"
)
PATCHED_RESOLVE = ANCHOR_RESOLVE + (
    "        # [glm53-kda-backend-log] make the resolved backend visible at boot\n"
    '        logger.info("GLM-5.3 KDA prefill backend: %s", self.kda_prefill_backend)\n'
)

def replace_once(source: str, old: str, new: str) -> str:
    n = source.count(old)
    if n != 1:
        raise SystemExit(f"[glm53-kda-backend-log] FAIL: anchor count={n} (expected 1): {old[:60]!r}")
    return source.replace(old, new)

def main() -> int:
    source = SO.read_text()
    if MARKER in source:
        print(f"{SO.name}: kda-backend-log no-op")
        return 0
    source = replace_once(source, ANCHOR_IMPORT, PATCHED_IMPORT)
    source = replace_once(source, ANCHOR_RESOLVE, PATCHED_RESOLVE)
    ast.parse(source)
    tmp = SO.with_suffix(".py.tmp")
    tmp.write_text(source)
    os.replace(tmp, SO)
    for pyc in SO.parent.rglob("__pycache__"):
        for f in pyc.glob("kda*.pyc"):
            f.unlink(missing_ok=True)
    print(f"{SO.name}: kda-backend-log patched (logger + info line)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
