# GLM-5.3-Flash serving image for 2× NVIDIA DGX Spark (vllm 77154ca, exllamav3 1.6.0; 1.6.3 retarget on the `e10-exl3163` branch)

Production-serving container for **GLM-5.3-Flash** (320B MoE, 18B active) on a
pair of **NVIDIA DGX Spark** machines (GB10 Grace Blackwell, 121 GiB unified
memory each): tensor-parallel 2 over the 200 Gb QSFP link, DFlash2 speculative
decoding (k=7), 1M-token context, prefix-cache-friendly hybrid KDA(mamba)+MLA
serving.

The published image is **vllm at nightly `77154ca61`** (the #57169 native packed-KV line)
with **exllamav3 1.6.0** and **18 targeted mods** applied at build time
(fail-closed, marker-verified). Every mod is in
[[`layer/mods/`](layer/mods/) — nothing is hidden in the base image.

| cell (benchy, d=16384) | this image | 0.31-based predecessor |
|---|---|---|
| pp2048 c1 | 1205 tok/s | 1156 tok/s |
| tg128 c1 / c2 | 36.2 / 44.8 tok/s | 32.5 / 40.9 tok/s |
| ctx_tg c2 | 38.7 tok/s | 30.7 tok/s |
| ctx_pp c1 | 1600 tok/s | 1678 tok/s |

(tool-eval-bench hard-mode: 89-90% on v2.8.1, statistically tied with the
0.31-based predecessor; the one deterministic difference is documented below.)

## What this repo builds

`glm53-selfbuild:serving` (published as `ghcr.io/gringus/dgx-spark-vllm-glm53-flash`) — a ~49 GB aarch64 CUDA image. The model **weights are not
in the image**; they load from Hugging Face at start (see [Serving](#serving)).

The build chains from the
[Reederey87/glm53-flash-exl3-2x-dgx-spark](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark)
kit (branch `e8-exl3160`), swaps the vllm base to the `77154ca` nightly line,
and applies this repo's mod layer:

```
kit Dockerfile chain (Apache-2.0)  →  77154ca base swap (ci/build-image.sh)
                                   →  mod layer: 18 mods + marker check (fail-closed)
                                   →  ghcr.io/<owner>/dgx-spark-vllm-glm53-flash:<tag>
```

GitHub Actions cannot host this build (a 49 GB image needs more disk than
hosted runners provide), so the workflow runs on a **self-hosted runner** with
~120 GB free disk and Docker; everything else (trigger, auth, push) is plain
`workflow_dispatch` / tag-based CI.

## The mods (what and why)

| mod | purpose |
|---|---|
| `glm53-upstream-gbm` | GB10 boot-OOM guard (backport of vllm #49760: unified-memory budget cap) |
| `glm53-upstream-alignperf` | mamba prefill alignment (perf) |
| `glm53-eagle-backoff` | drafter checkpoint back-off fix at cache-hit resume |
| `glm53-cache-reset` | `/cache/reset` router endpoint |
| `glm53-model-overrides` | GLM config overrides |
| `glm53-shm-recheck` | shm_broadcast reader recheck 50 ms |
| `glm53-role-token-cache` | role-token cache in the render path |
| `glm53-video-placeholders` | video placeholder handling + persistent-topk guard |
| `glm53-indexer-rightsize` | indexer workspace sizing |
| `glm53-suppress-stops` | suppress stop-token bleed in reasoning |
| `glm53-prefix-admit` | prefix admission hold |
| `glm53-scheduler-decode-floor` | decode scheduling floor |
| `glm53-adaptive-k-b0` | EMA adaptive draft-k |
| `glm53-upstream-grammar-resample` | **backport of vllm #60690** (fixes #60451): grammar-invalid draft positions resampled from the residual `max(p−q,0)`, not the masked `p` — removes a real sampling-distribution bias under spec+structured-output at temp>0 |
| `glm53-resident-tail` | resident-tail scheduling |
| `glm53-sparse-miss-metric` | sparse-indexer miss metrics |
| `glm53-hybrid-prefix-hit` | **fine-grained prefix-cache hits for `SlidingWindowManager`** — the drafter group's hit was block-granular (4608), capping every group's reconciled hit; with `--prefix-match-unit 1152` this restores 1152-grid hits (resume TTFT 1.42 s → 0.29 s, parity with the 0.31-based predecessor) |
| `glm53-fused-tables` | fused embedding tables |

Every mod is idempotent, anchor-verified (count==1) against the target tree at
build time, and the bake ends with a 31-marker check — a build with a missing
marker cannot be tagged.

## Known deterministic eval signature

On tool-eval-bench hard-mode, this stack (the `77154ca` numerics) deterministically
scores 1/2 instead of 2/2 on four category-O structured-output scenarios
(`TC-65/66/67/69`) — sub-nat logit perturbations flipping top-2 near-ties on
brittle schema items. It is config-invariant (reproduces with bf16 KV), the
sampler/grammar/tokenizer channels are exonerated, and no upstream fix exists
(the SM120 sparse-MLA files are untouched since the base commit). Worth ~3-4
points on a bad draw; invisible in the suite mean.

## Serving

Requires 2× DGX Spark (or 2× GB10-class machines) with the 200 Gb link; the
serving recipe lives in [`recipe/`](recipe/) (sparkrun format):

```
sparkrun run recipe/glm-5.3-flash-vllm.yaml
```

Models pulled at start:
- [Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw) (target, EXL3 4bpw)
- [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) (drafter)

## Sources & references

- Base kit (image chain, EXL3 integration, DFlash2 serving):
  [Reederey87/glm53-flash-exl3-2x-dgx-spark](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark) (Apache-2.0)
- vllm: [vllm-project/vllm](https://github.com/vllm-project/vllm) (Apache-2.0), base commit `77154ca61`
  - #57169 — native packed-KV layout / `SlidingWindowSpec` drafter ring (the reason this generation exists)
  - #60690 / #60451 — grammar-invalid draft resampling bias (backported as `glm53-upstream-grammar-resample`)
  - #49760 — unified-memory budget cap (backported as `glm53-upstream-gbm`)
- exllamav3: [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3), v1.6.0 (`151539c`) for the published image; the kit fork's `e10-exl3163` branch retargets to v1.6.3 (`a3bfbfd`, experimental — quality-positive on the hard eval suite, ctx_tg c2 watch item rides the newer vllm nightly it was benched with)
- FlashInfer: [flashinfer-ci/flashinfer](https://github.com/flashinfer-ci/flashinfer) 0.7.0.post1
- NVIDIA DeepGEMM: [deep-gemm](https://github.com/deepseek-ai/DeepGEMM) (pin `6ce1c5ec66`)
- Upstream issue #53436 — spec-decode temp-0 run-to-run nondeterminism on SM120 (observed here too; bounds any single-run eval comparison)

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). This repo builds on
the kit and upstream projects listed above; their notices carry over into the
built image.
