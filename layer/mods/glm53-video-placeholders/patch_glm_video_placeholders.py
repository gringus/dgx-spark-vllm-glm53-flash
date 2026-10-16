# e7 re-derivation (v0.31.0 db9527a468) of the kit
# patch_glm_video_placeholders.py (@aad6481).
#
# Align GLM-5.3 video prompt placeholders to encoder grid_t.
# glm4_1v builds one image-block per timestamp; a 1s clip can yield 3 timestamps
# while video_grid_thw T=1, so vLLM assigns 100 encoder tokens to 300 slots.
#
# v0.31.0 status (verified against db9527a468 via git plumbing):
# * Glm5Next lives at vllm/models/glm5next/common/ (moved off
#   model_executor/models/ and off nvidia/ for the model file).
#   Glm5NextProcessingInfo subclasses Glm4vProcessingInfo and does NOT
#   override _construct_video_placeholder; prompt expansion is owned by
#   the inherited Glm4vMultiModalProcessor._get_prompt_updates. The native
#   _construct_video_placeholder (glm4_1v.py:1388) still iterates the raw
#   timestamp list with NO alignment to grid_t -> the crash gate remains.
# * RETIRED HALF: the kpool persistent_topk disable is gone from this mod.
#   Upstream deleted vllm/model_executor/layers/sparse_attn_indexer_kpool.py
#   in the v0.31.0 bump; selection now routes through the
#   ``sparse_indexer_topk_backend`` config knob
#   (vllm/config/kernel.py:278) — set it via recipe/kernel-config, not a
#   patch.
# * ANCHOR DRIFT: _get_video_frame_embed_token_id was removed upstream
#   (#57556); the native builder now reads the embed id directly:
#   ``frame_embed_token_id = hf_processor.image_token_id``
#   (glm4_1v.py:1425). The hook's embed-id line is re-derived to match, and
#   main() fail-closed preflights the glm4_1v.py anchors below (each must
#   count==1) before installing anything.
# * Helpers verified present with same semantics: _is_glmga_model
#   (glm4_1v.py:1317), _get_video_second_idx_glm4v (:1189),
#   _get_video_second_idx_glm46v (:1240), _get_video_second_idx_glmga
#   (:1325); Glm4vProcessor is imported from transformers (glm4_1v.py:40)
#   and still re-exported from the glm4_1v namespace, so
#   `from ...glm4_1v import Glm4vProcessor` keeps working.
# * The vllm-native Glm5NextProcessor carries "Glm5" in its name and its
#   Glm5NextVideoProcessor has no max_duration, so the patched dispatch
#   routes Glm5Next to the glm46v timestamp path, same as on the pin.
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path


SITEPACKAGES = Path(
    os.environ.get(
        "GLM53_SITEPACKAGES",
        "/usr/local/lib/python3.12/dist-packages",
    )
)
GLM41V = Path(
    os.environ.get(
        "GLM53_GLM4_1V_PY",
        str(SITEPACKAGES / "vllm/model_executor/models/glm4_1v.py"),
    )
)
HOOK_DST = Path(
    os.environ.get(
        "GLM53_VIDEO_PATCH_PY",
        str(SITEPACKAGES / "glm53_video_patch.py"),
    )
)
PTH_DST = Path(
    os.environ.get(
        "GLM53_VIDEO_PTH",
        str(SITEPACKAGES / "glm53_video.pth"),
    )
)
PTH_BODY = "import glm53_video_patch\n"

# Fail-closed anchors against v0.31.0's glm4_1v.py (each must count==1).
ANCHOR_EMBED_ID = "        frame_embed_token_id = hf_processor.image_token_id\n"
ANCHOR_GLM46V_DISPATCH = (
    "            timestamps = self._get_video_second_idx_glm46v("
    "metadata, len(video_array))\n"
)
ANCHOR_SIG = "    def _construct_video_placeholder(\n"
ANCHOR_GLM46V_HELPER = "    def _get_video_second_idx_glm46v(\n"
ANCHOR_GLMGA_HELPER = "    def _is_glmga_model(self, processor: object) -> bool:\n"
ANCHOR_INFO_CLASS = "class Glm4vProcessingInfo("
GLM41V_SITES = (
    ("embed-id line", ANCHOR_EMBED_ID),
    ("glm46v dispatch branch", ANCHOR_GLM46V_DISPATCH),
    ("_construct_video_placeholder signature", ANCHOR_SIG),
    ("_get_video_second_idx_glm46v helper", ANCHOR_GLM46V_HELPER),
    ("_is_glmga_model helper", ANCHOR_GLMGA_HELPER),
    ("Glm4vProcessingInfo class", ANCHOR_INFO_CLASS),
)


def _align_timestamps(timestamps: list, t_groups: int) -> list:
    if t_groups < 1:
        t_groups = 1
    if not timestamps:
        return [0] * t_groups
    n = len(timestamps)
    if n == t_groups:
        return timestamps
    if t_groups == 1:
        return [timestamps[0]]
    if n < t_groups:
        return timestamps + [timestamps[-1]] * (t_groups - n)
    last = n - 1
    return [timestamps[int(round(i * last / (t_groups - 1)))] for i in range(t_groups)]


def _use_glm4v_timestamps(hf_processor) -> bool:
    """True only for classic Glm4v video processors that expose max_duration."""
    from vllm.model_executor.models.glm4_1v import Glm4vProcessor

    name = type(hf_processor).__name__
    vp = getattr(hf_processor, "video_processor", None)
    vp_name = type(vp).__name__ if vp is not None else ""
    if "Glm5" in name or "Glm5" in vp_name:
        return False
    if vp is not None and not hasattr(vp, "max_duration"):
        return False
    return isinstance(hf_processor, Glm4vProcessor)


def apply() -> None:
    from vllm.model_executor.models.glm4_1v import Glm4vProcessingInfo

    if getattr(Glm4vProcessingInfo, "_glm53_video_t_aligned", False):
        return

    def _construct_video_placeholder(self, video_array, metadata, grid_thw):
        hf_processor = self.get_hf_processor()
        tokenizer = self.get_tokenizer()
        image_processor = hf_processor.image_processor
        hf_config = self.get_hf_config()
        merge_length = image_processor.merge_size**2
        t_hw = grid_thw.reshape(-1)
        t_groups, height, width = int(t_hw[0]), int(t_hw[1]), int(t_hw[2])
        n_per = int(height * width) // merge_length

        if _use_glm4v_timestamps(hf_processor):
            timestamps = self._get_video_second_idx_glm4v(metadata, len(video_array))
            ts_fmt = "{}"
        elif self._is_glmga_model(hf_processor):
            timestamps = self._get_video_second_idx_glmga(metadata, len(video_array))
            ts_fmt = "{:.1f} seconds"
        else:
            # Glm5NextProcessor / Glm5NextVideoProcessor: glm46v timestamps.
            timestamps = self._get_video_second_idx_glm46v(metadata, len(video_array))
            ts_fmt = "{:.1f} seconds"

        timestamps = _align_timestamps(list(timestamps), t_groups)
        # [glm53-video-placeholders] v0.31.0 removed
        # _get_video_frame_embed_token_id (#57556); the native builder reads
        # hf_processor.image_token_id directly (glm4_1v.py:1425).
        embed_id = hf_processor.image_token_id
        placeholder = [hf_config.video_start_token_id]
        for ts in timestamps:
            placeholder.append(hf_config.image_start_token_id)
            placeholder.extend([embed_id] * n_per)
            placeholder.append(hf_config.image_end_token_id)
            placeholder.extend(
                tokenizer.encode(ts_fmt.format(ts), add_special_tokens=False)
            )
        placeholder.append(hf_config.video_end_token_id)
        return placeholder

    Glm4vProcessingInfo._construct_video_placeholder = _construct_video_placeholder
    Glm4vProcessingInfo._glm53_video_t_aligned = True
    print(
        "glm53: video placeholders aligned to encoder grid_t (Glm5Next→glm46v)",
        file=sys.stderr,
    )


def _install_import_hook() -> None:
    import builtins
    import importlib

    if getattr(builtins, "_glm53_video_hook", False):
        return
    builtins._glm53_video_hook = True
    real_import = builtins.__import__
    applying = False

    def _maybe_apply(name: str) -> None:
        nonlocal applying
        if applying or "glm4_1v" not in name:
            return
        applying = True
        try:
            apply()
        except Exception as exc:
            print(f"glm53: video apply() failed: {exc!r}", file=sys.stderr)
        finally:
            applying = False

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        mod = real_import(name, globals, locals, fromlist, level)
        _maybe_apply(name)
        return mod

    builtins.__import__ = _import
    real_im = importlib.import_module

    def _im(name, package=None):
        mod = real_im(name, package)
        _maybe_apply(name)
        return mod

    importlib.import_module = _im


def verify_glm41v(source: str) -> None:
    """Fail closed if any v0.31.0 anchor drifted (each must count==1)."""
    drifted = [
        name for name, anchor in GLM41V_SITES if source.count(anchor) != 1
    ]
    if drifted:
        raise ValueError(
            "glm53: glm4_1v.py anchors drifted "
            + ", ".join(f"{name} (count!=1)" for name in drifted)
        )


def replace_file(target: Path, source: str, mode_from: Path | None = None) -> None:
    tmp = target.with_name(f".{target.name}.glm53-video.tmp")
    try:
        tmp.write_text(source)
        src_mode = target if target.exists() else mode_from
        if src_mode is not None and src_mode.exists():
            os.chmod(tmp, stat.S_IMODE(src_mode.stat().st_mode))
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


# Runtime sitecustomize path only. Overlay tests import this file for
# verify_glm41v() and must not install the process-wide import hook.
if Path(__file__).name == "glm53_video_patch.py" or __name__ == "glm53_video_patch":
    _install_import_hook()
    try:
        apply()
    except Exception:
        pass


def main() -> int:
    hook_src = Path(__file__).resolve()
    hook_text = hook_src.read_text()
    if not GLM41V.is_file():
        raise SystemExit(f"glm53: missing {GLM41V}")
    try:
        verify_glm41v(GLM41V.read_text())
    except ValueError as exc:
        raise SystemExit(f"glm53: video preflight failed: {exc}") from exc
    compile(hook_text, str(hook_src), "exec")
    HOOK_DST.parent.mkdir(parents=True, exist_ok=True)
    PTH_DST.parent.mkdir(parents=True, exist_ok=True)
    replace_file(HOOK_DST, hook_text, mode_from=hook_src)
    replace_file(PTH_DST, PTH_BODY)
    print("glm53: overlay install ok aligned=True", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
