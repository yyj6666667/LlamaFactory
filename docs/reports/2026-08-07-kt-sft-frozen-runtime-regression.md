# KT SFT frozen-runtime regression handoff — started 2026-08-07, accepted 2026-08-08

> Internal evidence record. Deliberately excluded from the Sphinx toctrees.

## Executive status

Status: **cleanup regression gate accepted**.

- The cleanup implementation is frozen and pushed. Exact-dependency static/unit regression passed on `qj5090` (`95 passed`, `2 warnings`).
- The documentation/example publication layer passed the same frozen frontend suite plus six user-surface schema/parser checks (`101 passed`, `2 warnings`); it changes no runtime code.
- All four Qwen3.5-397B text-only routed-BF16 cases passed end-to-end training, base-model immutability, adapter-content, resource, timing, and throughput gates.
- All three DeepSeek-V3.1 routed-INT8 cases passed the same fail-closed gates at the final implementation head.
- Q2 was repeated at the final implementation head and passed, closing the cross-family regression gate after the INT8-only loader fix.
- An earlier Q1 attempt exposed a harness-only nested-launch bug. The failed evidence was retained; the direct per-rank training entrypoint was then used for the passing Q1–Q4 runs.

The tested implementation head is `e3b55c7431880b5abef3456e363fc4a7d325c583`. It is not the docs-only commit that publishes this report. Q1, Q3, and Q4 were accepted at its direct parent, `c713916bf7ec801681c3885bbd5cb530c8546946`; Q2 and D1–D3 were accepted at the final implementation head. The child changes only the routed-INT8 cache-loading branch, and the repeated Q2 run verifies that it did not regress the Qwen path.

## Exact tested implementation identities

| Component | Branch / distribution | Exact identity |
| --- | --- | --- |
| LLaMA-Factory | `yyj/kt-sft-frozen-deps-cleanup` | `e3b55c7431880b5abef3456e363fc4a7d325c583` |
| KTransformers | `yyj/kt-sft-frozen-deps-compat` | `ebf3756a2ec8c26e89b801c4e9b6ad789a73a6f7` |
| Transformers-KT | `transformers-kt==5.6.0.post1` | `f7cf9750c33ca946f12203c1e8d15f8ae24b8628` |
| Accelerate-KT | `accelerate-kt==1.14.0.post1` | `300eebe175c44d41366a892f6be60f8e0cd43ef1` |
| Loaded KT extension | runtime binary SHA-256 | `c00d8def4043eecc25eb2635ed24c632bba0ce997e9316ab31064deb63dfafd2` |

Runtime platform: Ubuntu 22.04.5 / glibc 2.35; Python 3.11.15; PyTorch 2.9.1+cu128 with CXX11 ABI;
CUDA runtime 12.8; NVIDIA driver 580.173.02; 8×RTX 5090 32 GB installed (2/4 used per case); 2×AMD EPYC 9355,
128 logical CPUs, AVX512-BF16/VNNI. The accepted runtime selected `avx512_bf16` and `avx512-vnni` kernels.

Clean source clones and the runtime overlay are under:

`/mnt/sft_yyj_yyj/kt-frozen-deps-regression-20260807T055509Z/{src,runtime}`

The LLaMA-Factory cleanup stack on top of `2d0a41c69e39be54234032211c61cdb7568aa987` is:

1. `5cb2ecf62096a3c942ad9271f737cb2aa0986c47` — LF owns the frozen KT runtime configuration.
2. `f3545a3dfa475bcbf9a6a0d0446ccf3891929557` — LF validates and loads routed-INT8 caches.
3. `04a504d2671e11e388fff77fc02ecdf351b319e6` — LF publishes routed-INT8 adapter provenance atomically.
4. `d3b765695f4d2b75a8bcff58f356c3ab7098895e` — FSDP2 persistent-buffer and adapter-save hardening.
5. `c713916bf7ec801681c3885bbd5cb530c8546946` — frozen-stack attestation and regression tests.
6. `e3b55c7431880b5abef3456e363fc4a7d325c583` — preserve source generation metadata when routed-INT8 weights come from a weight-only cache.

## Cleanup and preserved boundaries

- LLaMA-Factory is the single user-facing configuration owner. Its YAML accepts a flat `kt_config` plus the explicit activation policy; the Accelerate YAML remains generic FSDP configuration.
- The historical KT integrations in Transformers-KT and Accelerate-KT are preserved at the exact commits above. Recent broad yyj-side forwarding/semantic changes to those repositories are not part of this delivery.
- LLaMA-Factory keeps a strong `HfTrainerKTConfig` reference and publishes the nested `KTransformersPlugin` configuration expected by the frozen Accelerate runtime.
- Supported activation policies are CPU/GPU `retain/recompute`, `retain/retain`, and `recompute/recompute`. CPU `recompute` with GPU `retain` fails validation because that execution order is not implemented. The seven-case hardware matrix covered the two GPU-recompute policies; `retain/retain` is configuration/unit tested but was not part of this matrix.
- The FSDP2 persistent-buffer compatibility hook is version-gated to the exact frozen Accelerate runtime and enabled only for CPU-RAM-efficient loading.
- Distributed placeholder validation exchanges rank-local failures before DCP entry, preventing asymmetric preflight hangs. Adapter gathering excludes only placeholder FQNs reported by the KT public ownership API.
- DeepSeek FP8 source weights are paired with pre-generated, validated BF16 non-expert weights and routed-INT8 expert caches. LLaMA-Factory validates and loads those artifacts; it does not create them. Final routed-INT8 adapter output includes an atomically written `kt_adapter_manifest.json`.
- Qwen3.5 cases use a maintainer-materialized `TEXTONLY` model directory whose weight index and shards resolve to the original checkpoint while its config omits the four vision token IDs and declares `kt_text_only: true`. The raw Hub config was not exercised by this matrix and is not release-qualified.
  The recorded source-config, derived-config, and shared weight-index SHA-256 values are `3ae7fa89c2f7d1354096418ddaf1331e9e0898a8ca11e804fb6dbaa087efb7da`, `818c2f2335f9b4dc187e577eb444fc13d2b1a536b594710328c42c8597f9d6d1`, and `407d6a184a29469034ad92bcfaa6b72b582f8ad6afcbfad18a38a1c59ed296f8`.
- Multi-rank optimizer resume, optimizer-bearing checkpoints, `load_best_model_at_end`, and native routed-expert FP8 training are intentionally outside this cleanup boundary.

## Seven-case smoke matrix

The gate is finite loss/gradients, unchanged base-model manifest, structurally valid and nonzero standard/router/fused LoRA output, no routed-expert duplication, no OOM/collective/cache-stack error, and throughput at least 70% of the recorded baseline.

| Case | Model / expert path | GPUs | Logical / physical S | GAS | CPU / GPU activation | Steps | Status | LF head | Rank-0 TPS / baseline | Peak GPU MiB | Peak cgroup bytes | Evidence |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| Q1 | Qwen3.5-397B text-only / routed BF16 | 2 | 1024 / 1096 | 8 | retain / recompute | 2 | **PASS** | `c713916b` | 114.285 / 121.53 | 24,096 | 1,119,331,893,248 | `results/Q1/validation.json` |
| Q2 | Qwen3.5-397B text-only / routed BF16 | 2 | 2048 / 2120 | 1 | retain / recompute | 3 | **PASS** | `e3b55c74` | 136.292 / 163.23 | 28,282 | 1,128,811,216,896 | `results/Q2/validation.json` |
| Q3 | Qwen3.5-397B text-only / routed BF16 | 2 | 2048 / 2120 | 1 | recompute / recompute | 3 | **PASS** | `c713916b` | 120.157 / 137.32 | 28,282 | 1,082,892,177,408 | `results/Q3/validation.json` |
| Q4 | Qwen3.5-397B text-only / routed BF16 | 4 | 1024 / 1096 | 2 | retain / recompute | 2 | **PASS** | `c713916b` | 142.101 / 160.06 | 18,922 | 1,087,983,841,280 | `results/Q4/validation.json` |
| D1 | DeepSeek-V3.1-671B / routed INT8 | 2 | 2048 / 2048 | 1 | retain / recompute | 3 | **PASS** | `e3b55c74` | 72.906 / 84.46 | 32,010 | 1,035,214,925,824 | `results/D1/validation.json` |
| D2 | DeepSeek-V3.1-671B / routed INT8 | 2 | 2048 / 2048 | 1 | recompute / recompute | 3 | **PASS** | `e3b55c74` | 58.849 / 64.16 | 32,010 | 947,224,743,936 | `results/D2/validation.json` |
| D3 | DeepSeek-V3.1-671B / routed INT8 | 4 | 1024 / 1024 | 2 | retain / recompute | 2 | **PASS** | `e3b55c74` | 74.461 / 84.99 | 21,210 | 950,425,665,536 | `results/D3/validation.json` |

All relative evidence paths in the table are rooted at:

`/mnt/sft_yyj_yyj/kt-frozen-deps-regression-20260807T055509Z`

A lightweight local mirror of final validation JSON, resource series, stack probes, and key logs is under:

`/home/yyj/Desktop/LoRA/artifacts/kt-frozen-deps-regression-20260807T055509Z`

## Verification evidence

- Frozen-stack attestation: the initial parent-head preflight is `results/stack-probe.preflight.json`; final-head Q2/D1–D3 attestations and parent-head Q1/Q3/Q4 attestations are in their respective `results/stack-probe.<CASE>.json` files.
- Targeted regression at the final implementation head: `logs/final-targeted-tests-e3b55c74.log` (`95 passed`, `2 warnings`).
- User-guide/example regression in the frozen frontend environment: `logs/final-user-guide-tests.xml` (`101 passed`, `2 warnings`).
- Per-case final decision: `results/<CASE>/validation.json`.
- Adapter inventory/nonzero checks: `results/<CASE>/adapter_validation.json`.
- Base-model immutability: `results/<CASE>/base_model.before.json` and `base_model.after.json`. Q1–Q4 share unchanged manifest SHA-256 `74de3068b97ff6b167a44d26c6dc1e602aab1b9a3b6227d3deacaf1d2abf4b89`; D1–D3 share unchanged SHA-256 `37acee01ae311a2b59d8db4a0460c894ffc85fcc6de983ec2267f79ffe1f89a4`.
- Resource accounting: `results/<CASE>/resource_summary.json`, `memory_samples.jsonl`, and `gpu_samples.csv`.
- Phase timing and loss/gradient trajectory: embedded in `validation.json`; raw training logs are `logs/<CASE>.train.log`.
- Preserved failed harness attempt: `results/failures/Q1-nested-launch-20260807T065754Z/`.
- Preserved D1 metadata-boundary failure: `results/failures/D1-generation-metadata-20260807T134034Z/`.
- Preserved parent-head Q2 acceptance evidence: `results/accepted-parent-head/Q2-c713916b/`.
- Clean source-tree and exact import-origin claims are machine-checked by the stack probe, not inferred from editable-install metadata.

## Performance and system insight framework

Use one timing boundary consistently:

`step = dataloader + data preparation + forward + backward + grad clip + optimizer + post-optimizer + logging/save/eval + unaccounted step wall time`

- `forward` is model forward plus loss computation.
- `backward` is `accelerator.backward`/autograd, including CPU expert backward and GPU autograd; it excludes grad clip and AdamW.
- `optimizer` is `optimizer.step`; `post-optimizer` is KT pointer synchronization, scheduler, and zero-grad.
- Report rank-0 end-to-end TPS from the full step wall, never kernel-only throughput. Compare CPU-retain and CPU-recompute only at the same model, GPU count, sequence shape, GAS, and step sampling rule.
- Interpret Q2 versus Q3 as a policy trade-off, not a model-speed claim: CPU retain reached 136.292 TPS while CPU recompute reached 120.157 TPS in these three-step S2048 runs, a 13.4% retain advantage. Their cgroup peaks were 1.129 TB and 1.083 TB respectively.
- The same trade-off is larger for DeepSeek INT8: D1 retain reached 72.906 TPS versus D2 recompute at 58.849 TPS, a 23.9% retain advantage; recompute reduced the cgroup peak from 1.035 TB to 0.947 TB.
- Across all seven cases, cgroup peaks were 0.947–1.129 TB (decimal bytes). A nominal 1 TB host cannot be declared safe from these runs; anonymous/file-cache composition and eviction behavior must be considered before selecting 1 TB rather than 1.5 TB.
- Every case remained above the 70% historical throughput gate and recorded zero cgroup OOM/OOM-kill events. This is regression evidence, not a claim of optimal performance.
- The first D1 run exposed a cache-boundary error: generation metadata was looked up in the weight-only cache. Passing source-derived generation metadata explicitly while retaining the cache as the sole weight source fixed the contract without changing Transformers-KT or Accelerate-KT.

## Known risks

1. Two or three optimizer steps validate execution and artifact semantics, not convergence, TTQ, or downstream generation quality.
2. Short-run means include first-step effects. Stable performance conclusions require a longer fixed-shape run with warm-up steps excluded.
3. The runtime is intentionally pinned to exact forks and one extension binary. Changing a source head, installed distribution, import origin, or extension SHA invalidates this attestation.
4. Host-memory headroom is workload- and cache-dependent. The observed cgroup peaks do not establish 1 TB deployability.
5. Optimizer-state resume, fused-expert adapter continuation, global-quality evaluation, native routed FP8, and unsupported activation-policy combinations remain explicit delivery gaps.
6. The seven cases attest a frozen source/runtime overlay, not an arbitrary source build or package-install order. Production must use an attested bundle until a standalone installer or image is separately qualified.
7. Qwen3.5 matrix evidence covers the supplied text-only materialization, not the unmodified Hub config or an operator-edited copy.
8. `attest_frozen_stack.py` attests source/import/runtime identities, not model/cache artifacts. The matrix records model immutability, but a release bundle still needs its own fail-closed artifact manifest.

## Next steps

1. Promote the 101-test frozen-stack and release-example suite to a fast CI gate and schedule the seven hardware cases separately on qj5090.
2. Add a longer fixed-shape performance run and a real held-out quality run before making throughput, TTQ, or user-delivery claims.
3. Qualify host-memory sizing explicitly at 1 TB and 1.5 TB; the present evidence favors 1.5 TB for retain policies.
4. Qualify native routed FP8, optimizer-state resume, and the unsupported CPU-recompute/GPU-retain policy as separate deliveries.
5. Qualify and publish one reproducible installer/container artifact, then keep the public guide and checked-in examples version-gated to that artifact and the reviewed LF-owned YAML contract.
6. Publish and attest the Qwen3.5 text-only materializer or distribute its immutable output and manifest as part of the release bundle.

## Evidence map

| Evidence | Remote path |
| --- | --- |
| Regression root | `/mnt/sft_yyj_yyj/kt-frozen-deps-regression-20260807T055509Z` |
| Clean LLaMA-Factory clone | `src/llama_factory` |
| Clean KTransformers clone | `src/ktransformers` |
| Clean Transformers-KT clone | `src/transformers` |
| Clean Accelerate-KT clone | `src/accelerate` |
| Runtime KT overlay | `runtime/kt-overlay` |
| Matrix drivers | Passing Q1: `logs/Q1.driver.log`; final DeepSeek: `logs/matrix-D1-D3-e3b55c74.driver.log`; final-head Q2: `logs/Q2-final-head.driver.log`; the partial `logs/matrix-Q2-D3.driver.log` retains the first failed D1 attempt |
| Targeted tests | `logs/final-targeted-tests-e3b55c74.log` |
| User-guide tests | `logs/final-user-guide-tests.xml` |
| Stack probes | `results/stack-probe*.json` |
| Case outcomes | `results/<CASE>/validation.json` |
| Case artifacts | `runs/<CASE>/output` |
| Resource series | `results/<CASE>/{resource_summary.json,memory_samples.jsonl,gpu_samples.csv}` |
