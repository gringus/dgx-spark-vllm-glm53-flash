# e6 re-derivation (upstream nightly 385dce36b): anchor "SHM_READER_RECHECK_INTERVAL_MS = 5000"
# verified intact at shm_broadcast.py:66; patcher copied verbatim from e4.
#!/bin/sh
# Applies Reederey87 #90: SHM lost-notify recheck 5000ms -> 50ms (a dropped ZMQ notify
# parked the reader for the whole cap; hot path unchanged). Apply-time gate
# GLM53_SHM_RECHECK_50MS=1; one-line constant rewrite, fail-closed on anchor drift.
set -eu
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
GLM53_SHM_RECHECK_50MS=1 python3 "$DIR/patch_shm_recheck.py"
