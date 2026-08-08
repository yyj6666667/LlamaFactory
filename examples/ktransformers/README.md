# KTransformers LoRA SFT examples

The Qwen3.5-397B text-only BF16 and DeepSeek-V3.1 routed-INT8 templates cover the release-qualified KTransformers SFT
contracts. The Qwen template requires the verified `TEXTONLY` model materialization shipped with the release bundle;
the original Hub config is not qualified for this runtime.
`accelerate/fsdp2_kt.yaml` is the only release launcher config. The current integration does not provide a
`ktransformers` inference backend in LLaMA-Factory. See the [Chinese production guide](../../docs/zh/advanced/ktransformers.md)
for prerequisites, validated models, artifacts, and failure handling.

## Launch

Install a matching KTransformers SFT stack, edit the model/data/output paths in a training YAML, then launch the worker
script directly with Accelerate:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --main_process_port 0 \
  --config_file examples/ktransformers/accelerate/fsdp2_kt.yaml \
  src/train.py \
  examples/ktransformers/train_lora/qwen3_5moe_lora_sft_kt.yaml
```

Do not wrap `llamafactory-cli train` inside `accelerate launch`; that can create a nested distributed launch. Accelerate
files configure only FSDP2 and process topology. Every KT setting belongs to the LLaMA-Factory training YAML:

- `use_kt`, `kt_cpu_activation`, and weight paths are top-level fields.
- `kt_config` is one flat kernel mapping without `enabled` or another nested `kt_config`.
- LoRA rank, alpha, dropout, activation policy, and train mode are derived by LLaMA-Factory.

## Activation policy

LLaMA-Factory's `disable_gradient_checkpointing` controls GPU activations. `kt_cpu_activation` independently chooses
whether CPU expert activations are retained while GPU checkpointing is enabled.

| `disable_gradient_checkpointing` | `kt_cpu_activation` | CPU / GPU behavior |
| --- | --- | --- |
| `false` (default) | omitted or `recompute` | recompute / recompute |
| `false` (default) | `retain` | retain / recompute |
| `true` | omitted or `retain` | retain / retain (not covered by the final hardware matrix) |
| `true` | `recompute` | unsupported; validation fails |

CPU `retain` can improve step time but requires more host RAM. It is maintained for CPU-expert AMX BF16 training and
the validated frozen routed-INT8 LoRA path. Do not also enable Trainer, Unsloth, or FSDP activation checkpointing.

## Weights and saves

- The BF16 examples keep `kt_backend: AMXBF16` in their training YAML.
- The DeepSeek-V3.1 INT8 example requires both a validated routed-INT8 directory and a validated BF16 non-expert cache.
  The training output also contains `kt_adapter_manifest.json` with their provenance.
- Qwen3.5 requires a maintainer-produced, verified `TEXTONLY` model directory. Do not point the example at the original
  Hub config or edit `config.json` by hand. Text-only preprocessing can add dummy-image tokens after `cutoff_len`; the
  S2048 example therefore reserves `kt_model_max_length: 2176`.
- Multi-rank KT saves adapters only. Optimizer-state resume, periodic optimizer checkpoints, and
  `load_best_model_at_end` are not supported; the examples use `save_only_model: true`.
- Every launch needs a fresh, empty `output_dir`. Adapter continuation is not supported: `adapter_name_or_path` does not
  restore KT fused expert LoRA in this release.

`pure_bf16: true` creates KT adapters in BF16. Existing FP32 adapters are not automatically downcast.
