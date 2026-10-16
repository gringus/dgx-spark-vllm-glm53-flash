#!/bin/sh
# e6 re-derivation (upstream nightly 385dce36b) of the kit patch_model_overrides.py
# (@aad6481): 1-line insertion of "exl3" at the head of ModelConfig's ordered
# quantization overrides list (vllm/config/model.py, overrides = [ ... "auto_gptq", ...]).
# Anchor verified byte-identical on the nightly; patcher copied verbatim.
#
# PLUGIN-HOOK NOTE (docs/07 prefers the plugin path): the nightly natively exposes
# vllm.model_executor.layers.quantization.register_quantization_config. A plugin
# registering "exl3" appends it to QUANTIZATION_METHODS, and because a registered
# custom name is NOT in get_args(QuantizationMethods), the override scan
# (config/model.py:~1300) permits it to override WITHOUT an overrides-list edit.
# That could retire this patch entirely IF (a) an entry-point plugin registers
# exl3 before ModelConfig is constructed (supported_quantization is read at
# init, config/model.py:1256) and (b) the real Exl3Config is importable in the
# nightly image. Until a plugin ships, this patch remains the reliable default.
# Fail-closed on anchor drift; idempotent.
set -eu
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python3 "$DIR/patch_model_overrides.py"
