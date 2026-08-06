# KTransformers SFT examples

KTransformers uses LLaMA-Factory's native `disable_gradient_checkpointing` setting for GPU activations. CPU expert
activations have one additional, optional setting:

```yaml
use_kt: true
disable_gradient_checkpointing: false
kt_cpu_activation: retain
```

The supported combinations are:

| GPU checkpointing | `kt_cpu_activation` | Behavior |
| --- | --- | --- |
| enabled (default) | omitted or `recompute` | Recompute GPU and CPU expert activations. |
| enabled (default) | `retain` | Recompute GPU activations and reuse retained CPU expert activations. |
| disabled | omitted or `retain` | Retain both GPU and CPU activations. |
| disabled | `recompute` | Unsupported. |

`retain` can improve step time but keeps KT CPU expert checkpoint state in host memory until backward. It currently
supports CPU-expert AMX BF16 training and supported frozen INT8/FP8 LoRA paths. Leave it unset for INT4, GPU expert,
LoRA-expert, and Hybrid paths. When GPU checkpointing is enabled, KTransformers uses non-reentrant checkpointing; do not
also enable Transformers Trainer, Unsloth, or FSDP activation checkpointing.

Some model preprocessors add tokens after applying `cutoff_len`. For example, Qwen3.5 text-only batches can include
dummy-image tokens. Set `kt_model_max_length` with enough headroom in the Accelerate KT config; LLaMA-Factory preserves
a configured capacity when it is larger than the computed text capacity. The Qwen3.5 S2048 example uses
`accelerate/fsdp2_kt_qwen3_5.yaml` with capacity 2176.

With `pure_bf16: true`, new and reloaded KT adapters remain BF16. Existing FP32 adapter checkpoints are not automatically
downcast.
