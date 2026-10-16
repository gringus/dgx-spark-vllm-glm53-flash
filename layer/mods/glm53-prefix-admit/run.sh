# e6 re-derivation (upstream nightly 385dce36b): both anchors verified on nightly
# scheduler.py (num_computed_tokens == 0 lookup site x1 at :911, logger/class-line
# helper insert point x1); patcher copied verbatim from e4.
#!/bin/sh
# Applies Reederey87 #92: hold admission of a second copy of a shared prefix until the
# running partner publishes it (otherwise the copy records partial pages and never looks
# again - the rest of the shared prefix computes twice). Scheduler deferral only; no KV
# pinning, no page-size change. Apply-time gate GLM53_PREFIX_ADMIT=1; flag off = the
# installer leaves scheduler.py byte-identical. Fail-closed on anchor drift.
set -eu
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
GLM53_PREFIX_ADMIT=1 python3 "$DIR/patch_prefix_admit.py"
