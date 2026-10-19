# GLM-5.3-Flash serving image for 2× NVIDIA DGX Spark (vllm 77154ca, exllamav3 1.6.3)

Production-serving container for **GLM-5.3-Flash** (320B MoE, 18B active) on a
pair of **NVIDIA DGX Spark** machines (GB10 Grace Blackwell, 121 GiB unified
memory each): tensor-parallel 2 over the 200 Gb QSFP link, DFlash2 speculative
decoding (k=7), 1M-token context, prefix-cache-friendly hybrid KDA(mamba)+MLA
serving.

The image is **vllm at nightly `77154ca61`** (the #57169 native packed-KV line)
with **exllamav3 1.6.3** and **20 targeted mods** applied at build time
(fail-closed, marker-verified). Every mod is in
[[`layer/mods/`](layer/mods/) — nothing is hidden in the base image.

| cell (benchy, d=16384) | this image | 0.31-based predecessor |
|---|---|---|
| pp2048 c1 | 1186 tok/s | 1171 tok/s |
| tg128 c1 / c2 | 36.3 / 43.9 tok/s | 33.2 / 43.1 tok/s |
| ctx_tg c2 | 39.4 tok/s | 34.6 tok/s |
| ctx_pp c1 | 1600 tok/s | 1693 tok/s |

(tool-eval-bench hard-mode on v2.8.1: **91% (177/194) — parity with the
0.31-based reference**, after fixing the structured-output regression
documented below.)

## What this repo builds

`glm53-selfbuild:serving` (published as `ghcr.io/gringus/dgx-spark-vllm-glm53-flash`) — a ~49 GB aarch64 CUDA image. The model **weights are not
in the image**; they load from Hugging Face at start (see [Serving](#serving)).

The build chains from the
[Reederey87/glm53-flash-exl3-2x-dgx-spark](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark)
kit (our fork, `main` — the exllamav3 1.6.3 retarget), swaps the vllm base to the `77154ca` nightly line,
and applies this repo's mod layer:

```
kit Dockerfile chain (Apache-2.0)  →  77154ca base swap (ci/build-image.sh)
                                   →  mod layer: 20 mods + marker check (fail-closed)
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
| `glm53-tools-grammar-coexist` | **fixes the 0.32 structured-output regression**: with `tool_choice=auto`, keep the user's schema as the only grammar when tools are present (see the section above); `GLM53_TOOLS_GRAMMAR_COEXIST=0` restores 0.32 behavior |
| `glm53-strict-thinking` | **RFC #60722 defense-in-depth**: ban the implicit reasoning-end marker (`TOOL_START`) during reasoning sampling on grammar-active requests, forcing the explicit end; non-grammar turns unaffected |
| `glm53-resident-tail` | resident-tail scheduling |
| `glm53-sparse-miss-metric` | sparse-indexer miss metrics |
| `glm53-hybrid-prefix-hit` | **fine-grained prefix-cache hits for `SlidingWindowManager`** — the drafter group's hit was block-granular (4608), capping every group's reconciled hit; with `--prefix-match-unit 1152` this restores 1152-grid hits (resume TTFT 1.42 s → 0.29 s, parity with the 0.31-based predecessor) |
| `glm53-fused-tables` | fused embedding tables |

Every mod is idempotent, anchor-verified (count==1) against the target tree at
build time, and the bake ends with a 33-marker check — a build with a missing
marker cannot be tagged.

## Fixed upstream regression: structured outputs + tools

The 0.32 nightly introduced a regression for reasoning models with tool
definitions: `Parser._apply_structural_tag` composes
`OrFormat([tool_call_tag, output_format])` **from the request's own
`response_format`** and nulls it — the tool-call branch is legal under the
`Or`, so on schema'd turns the model can end in another tool call and the
user's schema is never enforced (deterministically on tool-bearing schema
requests; in agentic loops it compounds into non-terminating tool-call
generation). 0.31.0 was unaffected: the composition never armed for GLM (no
`glm_4_7` tag builder, xgrammar 0.2.7 lacked the builtin). Documented
upstream as the sampling-half of RFC #60722, with #61383 as a sibling
failure mode of the same spec×grammar coexistence.

`glm53-tools-grammar-coexist` restores the 0.31 semantics: with
`tool_choice=auto`, the user's schema is the only grammar (tool calls become
illegal on schema'd turns; `required`/named `tool_choice` still compose).
`glm53-strict-thinking` adds the RFC's proposed defense-in-depth: the
implicit reasoning-end marker (`TOOL_START`) is banned during reasoning
sampling on grammar-active requests, forcing the explicit end. With both,
tool-eval-bench hard-mode recovers to full parity with the 0.31 reference
(`TC-65/66/67/69` from 1/2 to 2/2).

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
- exllamav3: [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3), v1.6.3 (`a3bfbfd`)
- FlashInfer: [flashinfer-ci/flashinfer](https://github.com/flashinfer-ci/flashinfer) 0.7.0.post1
- NVIDIA DeepGEMM: [deep-gemm](https://github.com/deepseek-ai/DeepGEMM) (pin `6ce1c5ec66`)
- Upstream issue #53436 — spec-decode temp-0 run-to-run nondeterminism on SM120 (observed here too; bounds any single-run eval comparison)

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). This repo builds on
the kit and upstream projects listed above; their notices carry over into the
built image.
