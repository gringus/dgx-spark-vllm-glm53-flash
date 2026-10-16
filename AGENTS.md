# AGENTS.md — guidance for LLM coding agents working in this repo

## What this repo is

Build pipeline for a **vllm serving image of GLM-5.3-Flash on 2× NVIDIA DGX
Spark** (GB10, aarch64). The image is ~49 GB, aarch64-only, CUDA 13.x, and
**does not contain model weights** — weights load from Hugging Face at serving
time.

Key structural facts (do not break these):

- `scripts/glm53-selfbuild.sh` is the build orchestrator. It merges the
  upstream kit's staged Dockerfiles into one multi-stage build, swaps the vllm
  base to the pinned nightly, and applies the mod layer at build time. It is
  generation-aware via staged files and `TAG` (see its header for the lane
  logic).
- `layer/` is the mod layer: `bake-serving.sh` (ordered application + 31-marker
  fail-closed check), `mods/` (one directory per mod: `run.sh` + patcher),
  `Dockerfile.serving-layer`, `test_exl3_overlay.py` (GPU selfcheck used at runtime,
  not build time).
- `layer/mods/*/` patchers are **fail-closed and idempotent**: anchor count must
  be exactly 1 against the target tree, `ast.parse` before write, marker tags
  like `[glm53-<mod>]` make re-runs no-ops. A build with a missing marker
  cannot be tagged — keep it that way.
- `ci/build-image.sh` stages `layer/` into the kit clone (as `bake-serving.sh`,
  `mods-serving/`, `Dockerfile.serving-layer`) and invokes the orchestrator. File names
  are load-bearing for the orchestrator's staged-file detection.
- `recipe/glm-5.3-flash-vllm.yaml` is the serving recipe (sparkrun format).

## Rules of thumb

- **Never round-trip the recipe YAML through yaml.dump** — block scalars and
  comments do not survive. Text edits + read-only `yaml.safe_load` verification.
- **Never write raw apostrophes or special chars into recipe YAML strings** —
  the launch command is a shell string; reword instead.
- Comment lines inside the recipe's `command:` block scalar break the shell
  (`#` after a `\` continuation is a swallowed-arg bug). Rationale comments
  belong in the YAML comment layer only.
- Mods must keep the anchor/idempotency discipline described above; when
  upstream rebases, re-derive the anchor, don't loosen the check.
- The build must run **engine-down** (nvcc needs >10G host RAM; building while
  the engine is up invites the GB10 UVM-livelock class).
- Docker builds happen on the build node only; distribute to the peer node with
  `docker save | ssh <peer> docker load` (both ranks must run the identical
  image — the pipeline kernel fails closed on mismatch).
- The upstream kit is Reederey87/glm53-flash-exl3-2x-dgx-spark (Apache-2.0,
  branch `e8-exl3160`); keep `NOTICE` accurate when bumping any vendored
  component.

## Verification checklist for changes

1. `bash -n` every touched shell script.
2. `python3 -m py_compile` every touched patcher.
3. `yaml.safe_load` the recipe if touched (no dump round-trips).
4. Mod changes: run the mod's own test if present (e.g.
   `layer/mods/glm53-upstream-grammar-resample/test_grammar_resample.py`),
   then a build (`--check` mode of the orchestrator generates + lints without
   building) before a real build.
5. A real build ends with `--verify-fast`-equivalent marker output — all 31
   markers must pass; treat any marker failure as a true defect, not tooling
   noise.
