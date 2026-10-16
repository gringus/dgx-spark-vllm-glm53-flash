#!/usr/bin/env python3
"""Expose the cache-reset API routes on the :exl3 image (issue #31).

The nightly tree (385dce36b) ships ``/reset_prefix_cache``,
``/reset_mm_cache`` and ``/reset_encoder_cache`` in
``vllm/entrypoints/serve/dev/cache/api_router.py`` — but the mount site only
attaches them when ``VLLM_SERVER_DEV_MODE=1``, which would also expose the
whole dev surface (``/sleep``, ``/rlhf``, ``/rpc``, ``/server_info``).
Cold prefix-cache benchmarking then needs a full container restart, which
the README's prefix-cache ladder used to work around with per-session
salting only (issue #31, bench_prefix_cache.py docstring).

This patch adds a narrow ``elif`` to ``register_api_routers`` in
``vllm/entrypoints/launchers/api_server/routers.py`` (the nightly moved the
dev-mode mount out of ``openai/api_server.py`` into this launcher module):
when ``GLM53_EXPOSE_CACHE_RESET=1``, attach ONLY the cache-reset router.
The rest of the dev surface stays off, and ``VLLM_SERVER_DEV_MODE=1`` keeps
precedence (full dev mounts, as upstream). The flag is read at mount time,
so toggling it needs a container restart but not a re-patch.

Auth note (upstream quirk, documented in the README): the
``AuthenticationMiddleware`` only guards ``GUARDED_PREFIX = ("/v1", "/v2",
"/inference", "/cohere")``, so root-mounted routes — including the stock
``/tokenize`` and the cache-reset routes — answer without a bearer even
when ``VLLM_API_KEY`` is set. A cache reset can flush serving state, so
set ``GLM53_EXPOSE_CACHE_RESET=0`` on shared kits.

e6 re-derivation vs the e4 (fork) version, against upstream nightly
385dce36b:
* default target retargeted from ``vllm/entrypoints/openai/api_server.py``
  to ``vllm/entrypoints/launchers/api_server/routers.py`` (the dev-mode
  block moved there, intact at :34-37); ``GLM53_API_SERVER_PY`` env
  override kept for the preflight harness.
* routers.py has no top-level ``import os``; the ``elif`` arm calls
  ``os.getenv``, so this patcher inserts ``import os`` itself (fail-closed
  count==1 on the argparse import line).
* the dev router (``vllm/entrypoints/serve/dev/cache/api_router.py`` with
  ``attach_router`` and the three reset routes) is native on the nightly —
  verified, not patched.

Fail closed if the vLLM anchors drift.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

P = Path(
    os.environ.get(
        "GLM53_API_SERVER_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/launchers/"
        "api_server/routers.py",
    )
)
MARK = "# [glm53-cache-reset]"

IMPORT_OLD = "from argparse import Namespace\n"
# Plain (unmarked) import on purpose: the patch carries exactly one MARK, in
# the elif arm; a second marked line would break the post-patch verification.
IMPORT_NEW = "import os\nfrom argparse import Namespace\n"

DEV_OLD = """    if envs.VLLM_SERVER_DEV_MODE:
        from vllm.entrypoints.serve import register_vllm_dev_api_routers

        register_vllm_dev_api_routers(app)
"""

DEV_NEW = """    if envs.VLLM_SERVER_DEV_MODE:
        from vllm.entrypoints.serve import register_vllm_dev_api_routers

        register_vllm_dev_api_routers(app)
    elif os.getenv("GLM53_EXPOSE_CACHE_RESET", "0") == "1":
        # [glm53-cache-reset] Expose only the cache-reset dev routes (issue
        # #31): /reset_prefix_cache, /reset_mm_cache, /reset_encoder_cache.
        # The rest of the dev surface (sleep / rlhf / rpc / server_info)
        # stays off; VLLM_SERVER_DEV_MODE=1 keeps mounting everything.
        from vllm.entrypoints.serve.dev.cache.api_router import (
            attach_router as attach_cache_reset_router,
        )

        attach_cache_reset_router(app)
"""


def main() -> int:
    if not P.is_file():
        raise SystemExit(f"missing {P}")
    text = P.read_text()
    n_old = text.count(DEV_OLD)
    n_new = text.count(DEV_NEW)
    n_mark = text.count(MARK)
    if n_new == 1 and n_mark <= 1:
        print(f"{P.name}: cache-reset elif already present — skipping")
        return 0
    if n_mark or n_new:
        raise SystemExit(
            f"{P}: partial/inconsistent cache-reset patch "
            f"(old={n_old}, new={n_new}, marker={n_mark})"
        )
    if n_old != 1:
        raise SystemExit(
            f"{P}: expected exactly one pristine dev-mode block, found {n_old}"
        )
    # The elif arm calls os.getenv; the nightly routers.py has no import os.
    # Import block is at the top of the file; check the first 15 lines.
    if not any(
        line.strip() == "import os" for line in text.splitlines()[:15]
    ):
        n_import = text.count(IMPORT_OLD)
        if n_import != 1:
            raise SystemExit(
                f"{P}: expected exactly one import-os anchor, found {n_import}"
            )
        text = text.replace(IMPORT_OLD, IMPORT_NEW, 1)
    text = text.replace(DEV_OLD, DEV_NEW, 1)
    if text.count(DEV_NEW) != 1 or text.count(MARK) != 1:
        raise SystemExit(f"{P}: cache-reset post-patch verification failed")
    compile(text, str(P), "exec")
    P.write_text(text)
    print(f"patched {P.name} (GLM53_EXPOSE_CACHE_RESET=1 mounts only the "
          "cache-reset dev routes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
