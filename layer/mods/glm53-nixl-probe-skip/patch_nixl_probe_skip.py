#!/usr/bin/env python3
"""glm53-nixl-probe-skip: kill the 4x-per-boot nixl_ep import-failure traceback.

The 0.1.0 image (77154ca base) ships the nixl_ep wheel whose compiled backend
is tagged for torch 2.16 (nixl_ep_cpp_torch216.cpython-312-aarch64.so) while
the image runs torch 2.14 — `import nixl_ep` can NEVER succeed here, yet
vllm.utils.import_utils.has_nixl_ep() runs the doomed import once per process
(APIServer, EngineCore, each Worker), logging a full traceback through the
glm53_video_patch import hook every time (the user-visible boot noise).

Fix: a silent pre-probe in has_nixl_ep. Locate the backend package via
find_spec (no import), read torch/version.py from disk (no torch import),
and check for a backend .so matching THIS torch's major.minor. No match ->
return False without the doomed import (identical outcome, zero noise).
Match -> fall through to the stock _has_module("nixl_ep") import path so a
fixed/matched wheel keeps working.

Fails closed: any probe error falls through to the stock import attempt
(= today's behavior, including its log line). Idempotent via the
[glm53-nixl-probe-skip] marker. GLM53_NIXL_PROBE_SKIP=0 restores stock
(reboot only, no rebuild).

Anchors vs the 0.1.0 image tree; GLM53_VLLM_DIR overrides the vllm root for
sandbox validation (default /usr/local/lib/python3.12/dist-packages/vllm).
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

D = os.environ.get(
    "GLM53_VLLM_DIR", "/usr/local/lib/python3.12/dist-packages/vllm"
)
TARGET = Path(os.environ.get("GLM53_IMPORT_UTILS_PY", D + "/utils/import_utils.py"))
MARKER = "[glm53-nixl-probe-skip]"

# count==1 anchor, verified against the 0.1.0 tree (vllm 0.32.0rc1.dev100+g77154ca61)
ANCHOR = '''\
def has_nixl_ep() -> bool:
    """Whether the optional `nixl_ep` package is available."""
    return _has_module("nixl_ep")
'''

REPLACEMENT = '''\
def _nixl_backend_matches_torch() -> bool:
    # [glm53-nixl-probe-skip] Silent, import-free compatibility probe for the
    # nixl_ep compiled backend: True only when a nixl_ep_cu13 backend .so
    # tagged for the INSTALLED torch (major.minor read from torch/version.py
    # on disk) exists. Any error -> True (fail-open: stock import path runs).
    try:
        import importlib.util

        spec = importlib.util.find_spec("nixl_ep_cu13")
        if spec is None or not spec.submodule_search_locations:
            return False
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec is None or not torch_spec.origin:
            return True  # torch itself absent: let stock path decide
        version_py = (
            Path(torch_spec.origin).parent / "version.py"
        ).read_text()
        import re

        m = re.search(r'^__version__\\s*=\\s*"?(\\d+)\\.(\\d+)', version_py, re.M)
        if not m:
            return True
        want = f"nixl_ep_cpp_torch{m.group(1)}{m.group(2)}"
        pkg_dir = Path(spec.submodule_search_locations[0])
        return any(pkg_dir.glob(want + "*.so"))
    except Exception:
        return True  # fail-open: stock import path decides


def has_nixl_ep() -> bool:
    """Whether the optional `nixl_ep` package is available."""
    # [glm53-nixl-probe-skip] The wheel's backend is torch-version-tagged;
    # importing it on a mismatched torch can never succeed and logs a full
    # traceback per process at every boot. Probe the filesystem first: no
    # matching backend -> False without the doomed import.
    if os.environ.get("GLM53_NIXL_PROBE_SKIP", "1") != "0" and (
        not _nixl_backend_matches_torch()
    ):
        return False
    return _has_module("nixl_ep")
'''


def main() -> int:
    src = TARGET.read_text()
    if MARKER in src:
        print(f"{MARKER} already applied — nothing to do", flush=True)
        return 0
    count = src.count(ANCHOR)
    if count != 1:
        print(
            f"{MARKER} ANCHOR DRIFT: expected 1 occurrence of the "
            f"has_nixl_ep stock body, found {count} — NOT patching",
            file=sys.stderr,
            flush=True,
        )
        return 1
    patched = src.replace(ANCHOR, REPLACEMENT)
    ast.parse(patched)  # fail closed: refuse to write unparseable output
    TARGET.write_text(patched)
    print(f"{MARKER} applied to {TARGET} (silent backend pre-probe)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
