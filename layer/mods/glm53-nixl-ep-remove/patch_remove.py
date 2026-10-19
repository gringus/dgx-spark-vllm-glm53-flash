#!/usr/bin/env python3
"""Sanity patcher: verify the nixl_ep packages exist to be removed (fail-closed
if the tree changed shape - e.g. a future nightly renames them), so the bake
never silently no-ops on a renamed package."""
import sys
from pathlib import Path

D = Path("/usr/local/lib/python3.12/dist-packages")
present = [d.name for d in D.glob("nixl_ep*") if d.is_dir()]
if not present:
    print("nixl-ep-remove: nothing to remove (already absent) - OK")
    sys.exit(0)
print("nixl-ep-remove: removing", present)
