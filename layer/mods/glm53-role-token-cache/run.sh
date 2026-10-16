# e6 re-derivation (upstream nightly 385dce36b): all four renderer anchors verified on
# nightly renderers/base.py (tokenizer-init x1, encode site x1, tokenize_prompts sync x2
# async x2); patcher + role_token_cache module copied verbatim from e4.
#!/bin/sh
# Applies Reederey87 #88: token-exact chat-segment caching with request-salt isolation
# (skips re-tokenization of repeated rendered segments - our real workload resends the
# full conversation every turn). Installs the role_token_cache module beside the renderer,
# then the fail-closed installer. Apply-time gate GLM53_ROLE_TOKEN_CACHE=1.
set -eu
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
# The injected code inserts /opt/glm53 into sys.path and imports from there - the module
# belongs at /opt/glm53/role_token_cache.py (first boot failed with ModuleNotFoundError when
# staged inside renderers/; absolute import never searches the package dir).
cp "$DIR/role_token_cache.py" /opt/glm53/role_token_cache.py
GLM53_ROLE_TOKEN_CACHE=1 python3 "$DIR/patch_role_token_cache.py"
