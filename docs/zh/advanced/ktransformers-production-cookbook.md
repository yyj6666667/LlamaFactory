# KTransformers LoRA SFT 生产 Cookbook

> 状态：发布候选已完成统一 HEAD 验收。Qwen3.5-397B-A17B、DeepSeek-V3.1 的七个训练 Case，以及 `save -> resume -> fresh-process generation` 闭环均为 PASS。

本文面向需要在消费级 GPU 与大内存 CPU 服务器上运行 MoE LoRA SFT 的用户。KTransformers（KT）负责 CPU routed experts、KT artifact 和 fused expert LoRA；LLaMA-Factory 负责训练参数、数据与单一 YAML 入口；Transformers 负责模型构造和 Trainer 生命周期；Accelerate 只负责 FSDP2 分布式编排。

## 1. 当前生产边界

当前发布候选范围：

- LoRA SFT，rank 8、alpha 16、dropout 0、target `all`；
- Qwen3.5-397B-A17B routed experts BF16；
- DeepSeek-V3.1 BF16 non-expert cache + routed experts INT8；
- CPU-only routed experts，2 卡与 4 卡 FSDP2；
- CPU activation `retain` / `recompute`，GPU activation `recompute`；
- standard PEFT、router LoRA、fused expert LoRA 的保存、同 world-size 续训和 fresh-process 生成。

当前不属于生产验收范围：Full-FT、INT4、原生 routed-FP8、GPU experts、DeepSpeed ZeRO-3、Unsloth、`predict_with_generate`、`compute_accuracy` 和 KT CLI `eval`。2–3 step smoke 只验证训练链路、梯度、artifact、资源边界与性能回归门槛，不证明任务收敛、泛化能力或下游质量；质量交付仍需真实数据集、足够训练步数和独立 held-out 评测。

## 2. 固定版本与运行前检查

正式交付必须记录四仓精确 HEAD，且运行 checkout 必须 clean。2026-08-08 冻结验收使用以下 source HEAD：

| 仓库 | 交付分支 | 冻结验收 HEAD |
| --- | --- | --- |
| LLaMA-Factory | `yyj/kt-thin-integration` | `d53fdc350142d55f3c2b050ebdf924998206d0d6` |
| KTransformers | `yyj/kt-sft-public-contracts` | `93daaedc0a20580b1f6f63be1d6c40f0d15f8ddc` |
| Transformers | `yyj/kt-trainer-lifecycle` | `e04dcff79d485d03b33b69efed40d3ee7748a923` |
| Accelerate | `yyj/kt-fsdp-public-contracts` | `6f6da79b4cb9f988f6a1d0bb5a6b3841e5ac81f8` |

启动前运行公共接口探针：

```bash
python - <<'PY'
import inspect

from accelerate import Accelerator
from kt_kernel.sft import resolve_kt_pretrained_artifacts
from transformers import TrainingArguments

assert hasattr(TrainingArguments, "update_kt_config")
assert "adapter_only" in inspect.signature(Accelerator.get_state_dict).parameters
print(resolve_kt_pretrained_artifacts)
PY
```

qj5090 的最终测试使用 AVX512-BF16 runtime variant（DeepSeek routed-INT8 expert 使用 AVX512-VNNI kernel）、64 个 KT 线程和 2 个 thread pools。最终报告必须记录 GPU、host/cgroup 与 swap 峰值；生产机器还要为模型文件缓存、操作系统和其他进程预留额外内存，不能把 smoke 的观测峰值直接当作有余量的容量规划。

## 3. 单一配置源与 activation 策略

训练 YAML 是 KT 的唯一用户配置源。不要在 Accelerate YAML 中加入 `kt_config`，也不要在 `kt_config` 重复 LoRA rank、alpha、dropout、activation policy 或 weight path。LLaMA-Factory 会从标准字段派生这些值。

activation 策略矩阵如下；前三行受支持，最后一行当前会在启动前拒绝：

| `disable_gradient_checkpointing` | `kt_cpu_activation` | CPU / GPU activation |
| --- | --- | --- |
| `false` | 省略或 `recompute` | recompute / recompute |
| `false` | `retain` | retain / recompute |
| `true` | 省略或 `retain` | retain / retain |
| `true` | `recompute` | **不支持**，启动前报错 |

CPU `recompute` + GPU `retain` 当前不支持。不要同时开启 FSDP activation checkpointing、Transformers 额外 checkpoint kwargs 或 Unsloth gradient checkpointing。

## 4. BF16 routed experts 训练 YAML

以下配置展示生产必需字段。数据集、模型路径、输出目录和容量必须按实际环境替换：

```yaml
### model
model_name_or_path: /abs/path/to/model
trust_remote_code: true
disable_gradient_checkpointing: false

### method
stage: sft
do_train: true
do_eval: false
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.0
lora_target: all
pure_bf16: true

### dataset
dataset: your_dataset
template: your_template
cutoff_len: 2120       # 真实 physical sequence length
packing: false         # 最终验收矩阵使用 false
preprocessing_num_workers: 1
dataloader_num_workers: 0

### output
output_dir: /abs/path/to/output
logging_steps: 1
save_strategy: steps
save_steps: 100
save_only_model: false # 需要续训时必须保留 optimizer/scheduler
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 1.0e-4
weight_decay: 0.0
optim: adamw_torch_fused
max_steps: 1000
lr_scheduler_type: cosine
warmup_ratio: 0.1
max_grad_norm: 1.0
bf16: true
seed: 42
data_seed: 42
include_num_input_tokens_seen: non_padding
ddp_timeout: 360000000

### eval
eval_strategy: "no"
load_best_model_at_end: false

### ktransformers
use_kt: true
kt_cpu_activation: retain
kt_config:
  kt_backend: AMXBF16
  kt_expert_weight_format: bf16
  kt_num_threads: 64
  kt_tp_enabled: true
  kt_threadpool_count: 2
  kt_max_cache_depth: 2
  kt_share_backward_bb: true
  kt_num_gpu_experts: 0
  kt_force_fused_expert_lora: true
  kt_model_max_length: 2176
```

`kt_model_max_length` 是 kernel capacity，不是随意填写的业务长度，必须不小于实际 physical sequence length。最终验收中，Qwen logical S1024 的 physical length 为 1096、capacity 为 1152；logical S2048 的 physical length 为 2120、capacity 为 2176。TPS 使用 logical token 数作为分子，因此性能报告必须同时写出 logical / physical 长度。

## 5. DeepSeek routed-INT8 配置

DeepSeek-V3.1 workload 保留源模型内置的 FP8 metadata，但运行时实际加载经过校验的 BF16 non-expert cache 与 routed-INT8 experts。相对 BF16 配置，仅替换以下字段：

```yaml
model_name_or_path: /abs/path/to/DeepSeek-V3.1
trust_remote_code: false
flash_attn: sdpa

kt_weight_path: /abs/path/to/routed-int8-experts
kt_non_expert_weight_path: /abs/path/to/bf16-non-expert-cache
kt_config:
  kt_backend: auto
  kt_expert_weight_format: int8
  kt_weight_lifecycle: persistent
  kt_num_threads: 64
  kt_tp_enabled: true
  kt_threadpool_count: 2
  kt_max_cache_depth: 2
  kt_share_backward_bb: true
  kt_num_gpu_experts: 0
  kt_force_fused_expert_lora: true
  kt_model_max_length: 2048
```

两个 cache 必须来自同一 source model，manifest 中的 source config/index SHA 必须匹配。不要设置 `quantization_bit`，也不要手工传入 `quantization_config`，不要删除或修改源模型中的 FP8 metadata。

这里必须区分“模型来源 metadata”和“用户显式量化请求”：当 KT non-expert cache 已配置时，LLaMA-Factory 保留源 FP8 metadata，但不把它提升成显式的 `from_pretrained(quantization_config=...)` 参数；KT/Transformers 随后校验 cache provenance 和 load plan，并阻止源 quantizer 参与 cache 加载。用户显式提供的 `quantization_bit` 或 `quantization_config` 仍与 KT cache 冲突，必须 fail-closed，不能为了兼容而放宽这条校验。

## 6. FSDP2 与启动方式

Accelerate YAML 只包含分布式配置。Qwen 和 DeepSeek 的 wrap class 不同：

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false
  fsdp_reshard_after_forward: true
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_transformer_layer_cls_to_wrap: Qwen3_5MoeDecoderLayer # DeepSeek: DeepseekV3DecoderLayer
  fsdp_version: 2
mixed_precision: bf16
num_machines: 1
num_processes: 2 # 必须等于可见 GPU 数
rdzv_backend: static
same_network: true
use_cpu: false
```

两卡标准启动命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
OMP_NUM_THREADS=64 \
KT_KERNEL_CPU_VARIANT=avx512_bf16 \
accelerate launch --main_process_port 0 \
  --config_file /abs/path/to/model-specific-2gpu-fsdp2.yaml \
  src/train.py /abs/path/to/train.yaml
```

四卡时同时修改 `CUDA_VISIBLE_DEVICES` 与 `num_processes: 4`。不要把裸 `llamafactory-cli train` 写成多卡 KT 的标准入口，也不要对混合 2/4 卡矩阵统一导出同一个 `CUDA_VISIBLE_DEVICES`。

## 7. 保存、续训与 fresh-process 生成

一个完整 KT adapter 目录至少应包含：

```text
adapter_config.json
adapter_model.safetensors
fused_expert_lora.safetensors
kt_adapter_manifest.json
```

需要精确续训时，checkpoint 还必须保存 rank-local optimizer、scheduler、RNG、Trainer state 和 FSDP adapter-only state。续训使用原训练 YAML，并增加：

```yaml
resume_from_checkpoint: /abs/path/to/output/checkpoint-N
```

当前分布式 optimizer checkpoint 要求保存和恢复时 world size 相同。artifact 缺失、hash 不匹配、来源模型不一致或 optimizer inventory 不一致时必须直接失败，不能静默回退。

fresh-process 生成应使用一个全新 Python 进程，并通过标准 `adapter_name_or_path` 指向本地完整目录：

```yaml
model_name_or_path: /abs/path/to/base-model
adapter_name_or_path: /abs/path/to/output/checkpoint-N
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.0

use_kt: true
kt_cpu_activation: retain
kt_config:
  kt_backend: AMXBF16
  kt_expert_weight_format: bf16
  kt_num_threads: 64
```

```bash
llamafactory-cli chat /abs/path/to/kt_adapter_infer.yaml
```

当前不要使用 `llamafactory-cli eval` 验证 KT adapter；该入口尚未纳入生产合同。自动验收应使用独立 fresh-generation 脚本，确认 standard 与 fused adapter 均已加载、active adapter 正确且生成文本非空。

## 8. 发布验收流程

每个 Case 都必须完成以下检查：

1. 配置静态校验和四仓 stack probe 通过，checkout clean、HEAD 精确匹配。
2. 训练进程 exit code 为 0，`validation.json.status == "PASS"`。
3. 每个 step 的 loss、grad norm 均 finite；global step 与预期相同。
4. `rank0_stable_tps` 不低于预设门槛；它恰好跳过第一个 cold optimizer step。另行报告包含 cold step 的全程端到端 `rank0_tps`，不能用稳定值替换或隐藏全程值；phase timing 使用与 gate 相同的稳定窗口。
5. PEFT artifact（其中包含 router）与 fused expert LoRA-B 均存在非零更新，manifest 为 `ready`，不存在 routed expert 重复项；router 计数是 PEFT artifact 的子集，不能与 PEFT 总数相加。
6. base model snapshot 前后逐字节一致。
7. cgroup `oom`、`oom_kill` 事件和训练 cgroup swap 均为 0；同时记录 `max` 事件以及 host/cgroup/GPU 峰值。host-wide swap 只作环境上下文，不与训练 cgroup swap 混为一谈。
8. 运行 `save -> 同 world-size resume -> fresh-process generation` 闭环；resume 必须在首个 optimizer step 前完成 exact state 校验。

冻结验收 artifact 的维护者命令为：

```bash
cd /path/to/frozen-delivery
python harness/validate_configs.py --root "$PWD" --all --static-only
harness/launch_checkpoint_smoke.sh save
harness/launch_checkpoint_smoke.sh resume
harness/launch_checkpoint_smoke.sh inference-preflight
harness/run_matrix.sh D1 D2 D3 Q1 Q2 Q3 Q4
```

这组 `harness/*` 是 release evidence 工具，不是普通用户训练 API。

## 9. 统一 HEAD 最终验收结果

远端冻结证据位于：

```text
ssh qj5090:/mnt/sft_yyj_yyj/kt-thin-delivery-20260808T120000Z
```

冻结四仓 manifest（`harness/reviewed_heads.json`）SHA256：`15c200888392e5919a13870bb5d824d000400d0b65bde542d0e85afe901ac677`。以下数值来自该 artifact 在第 2 节四仓 HEAD 上的一次完整、fresh、fail-stop 运行；旧 HEAD 的历史结果未拼入最终表。表中秒数、TPS 和 GiB 均由原始 JSON 值四舍五入到 3 位小数。

### 9.1 七 Case 状态

| Case | 模型 / expert 格式 | GPU | logical / physical S | GAS | CPU / GPU activation | steps | all-step mean (s) | `rank0_tps` | stable mean (s) | `rank0_stable_tps` | gate | 状态 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Q1 | Qwen3.5 BF16 | 2 | 1024 / 1096 | 8 | retain / recompute | 2 | 136.405 | 120.113 | 125.461 | 130.591 | 85.071 | PASS |
| Q2 | Qwen3.5 BF16 | 2 | 2048 / 2120 | 1 | retain / recompute | 3 | 33.283 | 123.066 | 28.329 | 144.586 | 114.261 | PASS |
| Q3 | Qwen3.5 BF16 | 2 | 2048 / 2120 | 1 | recompute / recompute | 3 | 36.884 | 111.050 | 32.963 | 124.262 | 96.124 | PASS |
| Q4 | Qwen3.5 BF16 | 4 | 1024 / 1096 | 2 | retain / recompute | 2 | 60.681 | 135.000 | 51.776 | 158.219 | 112.042 | PASS |
| D1 | DeepSeek-V3.1 routed-INT8 | 2 | 2048 / 2048 | 1 | retain / recompute | 3 | 70.227 | 58.325 | 62.277 | 65.771 | 59.122 | PASS |
| D2 | DeepSeek-V3.1 routed-INT8 | 2 | 2048 / 2048 | 1 | recompute / recompute | 3 | 82.669 | 49.547 | 77.843 | 52.618 | 44.912 | PASS |
| D3 | DeepSeek-V3.1 routed-INT8 | 4 | 1024 / 1024 | 2 | retain / recompute | 2 | 126.522 | 64.748 | 113.236 | 72.345 | 59.493 | PASS |

`rank0_tps = logical tokens_per_step / aggregate_all mean step`，包含 cold first step；`rank0_stable_tps = logical tokens_per_step / aggregate_stable mean step`，恰好跳过该 cold step，并且只有后者用于冻结 baseline regression gate。all-step 指标仍完整报告，不能用 stable 指标替换。Q1、Q4、D3 只有 2 step，因此所谓 stable window 只有 1 个 post-cold step：它只能作为本次 gate observation，不能解释为长期稳态估计。gate 不是对任意硬件、数据或长度的性能承诺。

### 9.2 Gate-window timing、数值与全程资源峰值

以下 phase timing 与 gate 使用同一个 stable window，均只跳过恰好一个 cold step；loss 与 grad norm 列仍列出全部 step。

| Case | loss | grad norm | forward (s) | backward (s) | clip (s) | AdamW (s) | post / other (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Q1 | 1.8349609375, 1.8349609375 | 0.371298999, 0.371292830 | 42.637 | 81.349 | 0.395 | 0.284 | 0.004 / 0.743 |
| Q2 | 1.75, 1.88671875, 1.90234375 | 0.339916110, 1.010792494, 0.548056602 | 8.412 | 14.601 | 4.951 | 0.254 | 0.005 / 0.097 |
| Q3 | 1.75, 1.88671875, 1.90625 | 0.338309467, 1.010657191, 0.510239661 | 7.701 | 20.325 | 4.572 | 0.247 | 0.005 / 0.100 |
| Q4 | 1.8349609375, 1.8349609375 | 0.371287555, 0.371308804 | 17.301 | 29.350 | 4.698 | 0.246 | 0.005 / 0.164 |
| D1 | 12.625, 12.625, 12.53125 | 1.174139023, 1.203364253, 1.088858843 | 19.962 | 26.310 | 14.032 | 1.932 | 0.005 / 0.033 |
| D2 | 12.625, 12.625, 12.53125 | 1.174139023, 1.203364253, 1.088900447 | 18.257 | 43.186 | 14.245 | 2.113 | 0.005 / 0.034 |
| D3 | 12.6640625, 12.6484375 | 1.205319524, 1.214640021 | 40.766 | 52.697 | 16.099 | 3.599 | 0.005 / 0.068 |

DeepSeek D1–D3 中，正确的全局 grad clip 实测为 14.032–16.099 秒，是稳定 step 的显著组成部分；不能通过关闭 clip 或遗漏 KT 参数来制造更高 TPS。若后续优化，应保持全局范数与 FSDP/KT 参数覆盖语义不变，并独立做数值与性能回归。

| Case | cgroup peak (GiB) | host used peak (GiB) | host available min (GiB) | GPU peaks (MiB) | cgroup swap (GiB) | host-wide swap used (GiB) | oom / kill / max |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| Q1 | 1053.812 | 1180.083 | 331.342 | 6:25738 / 7:25738 | 0 | 4.550 | 0 / 0 / 0 |
| Q2 | 1041.683 | 1211.725 | 299.700 | 6:29978 / 7:30012 | 0 | 4.572 | 0 / 0 / 0 |
| Q3 | 1022.388 | 1155.002 | 356.424 | 6:30046 / 7:30012 | 0 | 4.599 | 0 / 0 / 0 |
| Q4 | 1039.071 | 1222.222 | 289.203 | 4:20688 / 5:20724 / 6:20724 / 7:20688 | 0 | 4.642 | 0 / 0 / 0 |
| D1 | 979.801 | 1113.107 | 398.318 | 6:32012 / 7:31906 | 0 | 4.453 | 0 / 0 / 0 |
| D2 | 938.399 | 1035.233 | 476.192 | 6:32012 / 7:31906 | 0 | 4.486 | 0 / 0 / 0 |
| D3 | 924.510 | 1118.094 | 393.332 | 4:21210 / 5:21208 / 6:21208 / 7:21208 | 0 | 4.518 | 0 / 0 / 0 |

七个 Case 的 base snapshot 均不变；PEFT artifact（含 router）与 fused artifact validator 均 PASS；cgroup OOM 与 oom-kill 均为 0；cgroup swap peak 均为 0。表中的 `host-wide swap used` 是整机环境上下文，可能包含同机其他进程或运行前已有占用，不可归因给训练，也不能与对应训练 cgroup 的 swap 混为一谈。

### 9.3 Checkpoint 闭环

以下资源值来自当前 `results/QS/*/resource_summary.json`，而不是早期 checkpoint smoke：

| 阶段 | 关键结果 | adapter / state | 资源 |
| --- | --- | --- | --- |
| save | PASS；exit 0，global step 1；loss 1.8125，grad norm 0.4694406688；base snapshot `d88a293eb40f25e019c8dd960035c87cb5fc9cddb48a22428fb10d5b08f90c10` 不变 | manifest `ready`；FSDP adapter 1170 tensors / 34,330,080 values；PEFT artifact 1170 tensors、B-nonzero 585（其中 router A/B/B-nonzero 60/60/60）；fused 另计 360 tensors、B-nonzero 180；无 routed duplicate | cgroup peak 1042.260 GiB；host used peak / available min 1180.782/330.643 GiB；GPU 6/7 peak 25,602/25,602 MiB；cgroup swap 0；host-wide swap 4.329 GiB；OOM/oom-kill/max 0/0/0 |
| resume | PASS；exit 0，global step 2；loss 1.8125/1.83984375，grad norm 0.4694406688/0.4503556192；base snapshot 不变；router 120 tensors changed，其中 LoRA-B 60 | rank 0/1 均在首个 optimizer step 前 exact PASS，world size 2；FSDP2 adapter-only load 均 PASS，120 个 public KT 参数排除项一致；rank-local optimizer 4590/3510 tensors；PEFT artifact（含 router）与 fused 计数同 save | cgroup peak 1032.179 GiB；host used peak / available min 1180.621/330.805 GiB；GPU 6/7 peak 25,140/25,660 MiB；cgroup swap 0；host-wide swap 4.393 GiB；OOM/oom-kill/max 0/0/0 |
| fresh inference | validation PASS，`INFERENCE_PASS`；active adapter `default`；生成 16 个非空 token | PEFT / fused adapter 均已加载；manifest `ready`；PEFT artifact 1170 tensors / 34,330,080 values（其中 router A/B/B-nonzero 60/60/60），fused 另计 360 tensors / 3,774,873,600 values；四个必需 artifact 均完成 SHA256 校验 | cgroup peak 983.072 GiB；host used peak / available min 1082.758/428.667 GiB；GPU 6/7 peak 8,788/12,330 MiB；cgroup swap 0；host-wide swap 4.437 GiB；OOM/oom-kill/max 0/0/0 |

该 checkpoint smoke 只证明保存、同 world-size 恢复和 fresh-process 生成合同成立；它仍是短步系统验收，不是收敛或回答质量评测。

关键证据：

```text
results/QS/save/validation.json
results/QS/save/resource_summary.json
results/QS/resume/validation.json
results/QS/resume/resource_summary.json
results/QS/resume/resume_exact/rank0.json
results/QS/resume/resume_exact/rank1.json
results/QS/resume/fsdp_load/rank0.json
results/QS/resume/fsdp_load/rank1.json
results/QS/inference-preflight/validation.json
results/QS/inference-preflight/inference.json
results/QS/inference-preflight/resource_summary.json
```

七 Case 的汇总证据是 `results/final-matrix-summary.json`；逐项证据位于 `results/<CASE>/validation.json`、`results/<CASE>/timing/rank0/step_timing.json` 和 `results/<CASE>/resource_summary.json`。

## 10. 故障排查

| 现象 | 先检查 | 正确处理 |
| --- | --- | --- |
| `kt_non_expert_weight_path cannot be combined with an explicit quantization_config` | YAML 是否设置 `quantization_bit`；四仓 HEAD 是否为最终兼容组合 | 删除用户显式量化 override；如果仅存在模型内置 FP8 metadata，说明运行栈不兼容，应升级到已验最终栈，不能修改源 `config.json` 绕过 |
| cache manifest source/hash 不匹配 | routed cache、non-expert cache 和 base model 是否来自同一 source | 用正确 source 重新生成 cache；禁止关闭校验或混用 cache |
| C++ cache-stack underflow | activation 组合、checkpoint 开关、KT/Transformers/LF 版本 | 只使用第 3 节矩阵的前三种受支持策略；关闭 FSDP checkpointing、Unsloth GC 和额外 checkpoint kwargs；升级到同一冻结栈 |
| CPU RAM 不足或 cgroup OOM | physical length、`kt_model_max_length`、CPU policy、cgroup limit | 优先从 `retain` 改为 `recompute`；降低 physical length；提高 cgroup/host RAM，并保留文件缓存余量 |
| CUDA OOM | physical length、每卡 batch、FSDP process 数 | 降低 length/batch，确认 `num_processes` 等于可见 GPU 数；不要复用错误的 2/4 卡 Accelerate YAML |
| adapter 能加载但效果不变 | active adapter、router/fused 文件、LoRA-B 非零计数 | 检查 `adapter_model.safetensors`、`fused_expert_lora.safetensors` 和 ready manifest；fresh-process 验证 standard/router/fused 三类更新 |
| resume 在 optimizer load 失败 | world size、checkpoint 文件、optimizer inventory | 使用与保存时相同 world size 和完整 checkpoint；不要只复制 adapter 文件冒充可续训 checkpoint |
| `llamafactory-cli eval` 失败 | 当前 KT 评测入口未纳入合同 | 使用 `llamafactory-cli chat` 做人工验证，或使用 fresh-generation 验收脚本 |
| 训练日志长时间无输出 | GPU/CPU 利用率、日志 mtime、进程树 | GAS 较大时单步可长时间静默；只有日志、资源和进程连续 5 分钟均无进展时才按 stuck 处理 |

## 11. 交付前最终清单

- [x] 四仓冻结验收 HEAD 已记录，checkout 与安装包来源一致。
- [x] Q1–Q4、D1–D3 在统一最终 source HEAD 上全部 PASS。
- [x] 所有 Case 的 loss/grad finite、stable TPS 达门槛、base unchanged、OOM 事件为 0。
- [x] PEFT artifact（含 router）与 fused LoRA 更新及 ready manifest 已验证。
- [x] save、same-world-size resume、fresh-process generation 闭环通过。
- [x] 用户配置只存在于训练 YAML，Accelerate YAML 只负责 FSDP2。
- [x] 已明确本轮 smoke 不等同于模型质量或收敛验收。
