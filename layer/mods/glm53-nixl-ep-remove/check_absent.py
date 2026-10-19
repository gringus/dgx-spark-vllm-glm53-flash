#!/usr/bin/env python3
"""Marker-equivalent check for a removal mod: the packages must be GONE."""
import sys
from pathlib import Path

D = Path("/usr/local/lib/python3.12/dist-packages")
left = [d.name for d in D.glob("nixl_ep*") if d.is_dir()]
if left:
    print("FAIL: nixl_ep packages still present:", left)
    sys.exit(1)
print("nixl-ep-remove: packages absent - OK")
