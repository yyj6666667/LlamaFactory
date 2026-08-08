# KTransformers LoRA SFT 交付使用说明（预发布）

本页只描述当前已验收的 KTransformers（KT）训练入口。它供 release bundle 的试用与交付验收使用；在可复现
镜像/离线安装包和模型/cache bundle 发布前，不进入公共文档目录。旧教程中的 `kt_optimize_rule`、
`requirements-ksft.txt`、`infer_backend: ktransformers` 和 KT 批量评测入口已经不适用。

## 1. 支持范围

当前交付面向 CPU routed experts + GPU 非专家模块的 MoE LoRA SFT：

| 路径 | 已验收配置 |
| --- | --- |
| Qwen3.5-397B-A17B text-only routed BF16 | 2/4 GPU，CPU retain/recompute，GPU recompute |
| DeepSeek-V3.1 routed INT8 + BF16 non-expert | 2/4 GPU，CPU retain/recompute，GPU recompute |

以下能力不在本次交付承诺内：LLaMA-Factory 内的 KT 对话/批量推理、GGUF adapter、native routed FP8、
多卡 optimizer-state resume、周期性 optimizer checkpoint、`load_best_model_at_end`，以及
CPU recompute + GPU retain。

两卡 S2048 验收中，每卡峰值约 27.6 GiB（Qwen BF16）或 31.3 GiB（DeepSeek INT8）。CPU retain
的 cgroup 峰值达到 1.04–1.13 TB，因此交付机建议先按 **1.5 TB RAM** 配置，并像验收任务一样在 job cgroup
设置 `MemorySwapMax=0`；1 TB 需要单独做容量验收。内部完整证据记录在
`docs/reports/2026-08-07-kt-sft-frozen-runtime-regression.md`。

七案运行在 Ubuntu 22.04.5 / glibc 2.35、Python 3.11.15、PyTorch 2.9.1+cu128（CXX11 ABI）、CUDA 12.8、
driver 580.173.02、双路 AMD EPYC 9355（128 logical CPUs）和 8×RTX 5090 32 GB 的主机上；每案只使用
2 或 4 张 GPU。虽然 YAML 的兼容后端名是 `AMXBF16`，该 AMD 主机实际 dispatch 为 `avx512_bf16`；其他
CPU ISA、NUMA 和内存带宽必须另做 smoke。

## 2. 安装与版本检查

训练必须使用团队发布并完成 attestation 的 Python 3.11 + CUDA/PyTorch + KT SFT release bundle。当前没有
验收过任意安装顺序的公开 pip 环境；bundle 必须同时提供镜像/离线包身份、模型/cache artifact 和 attestation
结果。不要在该环境中单独升级或重装 `transformers`、`accelerate`、`ktransformers`、`kt-kernel` 或
LLaMA-Factory。
这些项目的发行名不同但会安装同名 Python package，错误的安装顺序可能覆盖 KT integration，同时留下具有
迷惑性的旧 distribution metadata。

release implementation 与七案验收的核心身份是：

- LLaMA-Factory implementation `e3b55c7431880b5abef3456e363fc4a7d325c583`
- KTransformers source `ebf3756a2ec8c26e89b801c4e9b6ad789a73a6f7`
- `transformers-kt==5.6.0.post1`，source `f7cf9750c33ca946f12203c1e8d15f8ae24b8628`
- `accelerate-kt==1.14.0.post1`，source `300eebe175c44d41366a892f6be60f8e0cd43ef1`
- 验收 runtime/extension 标识为 `0.6.3.post1`，extension SHA-256 为
  `c00d8def4043eecc25eb2635ed24c632bba0ce997e9316ab31064deb63dfafd2`

Q1/Q3/Q4 在 implementation parent `c713916bf7ec801681c3885bbd5cb530c8546946` 通过，Q2/D1–D3 在上述
`e3b55c7` 通过；Q2 的 final-head 重跑关闭了跨模型回归门。本页所在 publication commit 只改文档、示例和测试，
不是 runtime identity。当前 KTransformers source tree 的开发版本与上述冻结 runtime 标识不同；自行执行 source build 会产生新的
二进制，必须重新跑 attestation 和至少一条硬件 smoke，不能沿用本报告结论。发布环境由维护者使用
`scripts/kt_sft/attest_frozen_stack.py` 做完整 source/import/extension 校验。用户启动前至少检查关键 integration
符号与导入来源：

```bash
python - <<'PY'
from importlib.metadata import version
import accelerate, kt_kernel, transformers
from accelerate.utils import KTransformersPlugin
from kt_kernel.sft.config import KTConfig
from kt_kernel.sft.weights import get_kt_expert_placeholders
from transformers.integrations.kt import HfTrainerKTConfig

print("transformers:", version("transformers-kt"), transformers.__file__)
print("accelerate:", version("accelerate-kt"), accelerate.__file__)
print("kt_kernel:", kt_kernel.__version__, kt_kernel.__file__)
print(HfTrainerKTConfig, KTransformersPlugin, KTConfig, get_kt_expert_placeholders)
PY
```

## 3. 准备模型与数据

1. 按常规 LLaMA-Factory 格式准备数据，并在 `data/dataset_info.json` 注册数据集。
2. Qwen3.5 BF16 必须填写随 KT SFT release bundle 提供并经过验证的 `TEXTONLY` 模型目录，例如
   `/models/Qwen3.5-397B-A17B-TEXTONLY`。本次验收没有使用原始 Hub config；在冻结的 PyTorch 2.9 runtime 中，原始
   multimodal config 会触发 Conv3D 保护并拒绝启动。`TEXTONLY` 目录复用原始权重，但 config 转换属于发布产物的
   一部分；不要手工删除视觉 token 字段，也不要把 `model_name_or_path` 改回 Hub ID。七案记录的 source config、
   derived config 和共享 weight index SHA-256 分别为 `3ae7fa89c2f7d1354096418ddaf1331e9e0898a8ca11e804fb6dbaa087efb7da`、
   `818c2f2335f9b4dc187e577eb444fc13d2b1a536b594710328c42c8597f9d6d1` 和
   `407d6a184a29469034ad92bcfaa6b72b582f8ad6afcbfad18a38a1c59ed296f8`；release bundle 必须通过自身 manifest
   固定这些身份。`config.json.bak` 不是 provenance，用户不应依赖它。
3. DeepSeek routed INT8 路径需要三份相互匹配的输入：原始模型目录、routed-INT8 expert 目录、BF16
   non-expert cache。后两者必须是绝对、非符号链接目录，并包含转换工具生成的 `ready` manifest；不要手写
   manifest。LLaMA-Factory 只验证和加载，不能在线生成；release bundle 未提供匹配 cache 时不能启动该路径。
4. `template` 必须与模型匹配。Qwen3.5 text-only 预处理会在 `cutoff_len` 后增加 dummy-image token；
   S2048 已验证使用 `kt_model_max_length: 2176`。

## 4. 配置训练

直接复制完整示例再修改路径和数据：

- BF16：`examples/ktransformers/train_lora/qwen3_5moe_lora_sft_kt.yaml`
- INT8：`examples/ktransformers/train_lora/deepseek_v3_1_int8_lora_sft_kt.yaml`

KT 只有一个配置源：**LLaMA-Factory 训练 YAML**。Accelerate YAML 只配置 FSDP2 和进程数。

BF16 的核心配置如下：

```yaml
disable_gradient_checkpointing: false

finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.0
lora_target: all
pure_bf16: true

use_kt: true
kt_cpu_activation: retain  # recompute 可能降低峰值，但 1 TB 仍未验收
kt_config:                 # 必须是平坦 mapping
  kt_backend: AMXBF16
  kt_num_threads: 64       # 按实际 CPU/NUMA 配额调节
  kt_tp_enabled: true
  kt_threadpool_count: 2    # INT8 时必须与 routed cache manifest 一致
  kt_max_cache_depth: 2
  kt_share_backward_bb: true
  kt_num_gpu_experts: 0
  kt_force_fused_expert_lora: true
  kt_model_max_length: 2176
```

DeepSeek routed INT8 额外使用：

```yaml
kt_weight_path: /abs/path/to/routed-int8
kt_non_expert_weight_path: /abs/path/to/bf16-nonexpert
kt_config:
  kt_backend: auto
  kt_expert_weight_format: int8
  kt_weight_lifecycle: persistent
  # 其余线程、TP、cache 和容量字段同上
```

不要在 `kt_config` 中重复填写 `enabled`、LoRA rank/alpha/dropout、activation policy、train mode 或 weight
paths；这些值由顶层字段派生。也不要启用 `gradient_checkpointing: true`、
`gradient_checkpointing_kwargs`、FSDP activation checkpointing、Unsloth GC 或旧的 `ACCELERATE_KT_*`
环境变量。

Activation 组合如下：

| `disable_gradient_checkpointing` | `kt_cpu_activation` | CPU / GPU | 取舍 |
| --- | --- | --- | --- |
| `false` | 省略或 `recompute` | recompute / recompute | 本次短 smoke 中峰值和 TPS 都较低 |
| `false` | `retain` | retain / recompute | 已验收的 RAM reuse；本次短 smoke 中 TPS 较高 |
| `true` | 省略或 `retain` | retain / retain | 代码/单测支持，本轮未做大模型硬件验收 |
| `true` | `recompute` | 不支持 | 启动前报错 |

多卡训练只保存 adapter。推荐 `save_only_model: true`；需要周期性保存时仍不能恢复 optimizer state。
每次启动必须使用新的空 `output_dir`。当前 `adapter_name_or_path` 不会恢复 KT fused expert LoRA，因此标准
adapter 续训也不在交付范围；每个训练任务必须从新的 LoRA 初始化开始。

## 5. 启动

多卡必须让 Accelerate 直接启动 `src/train.py`，不要在 Accelerate 外再套一层
`llamafactory-cli train`：

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
  --main_process_port 0 \
  --config_file examples/ktransformers/accelerate/fsdp2_kt.yaml \
  src/train.py \
  examples/ktransformers/train_lora/qwen3_5moe_lora_sft_kt.yaml
```

修改 GPU 数时，同时修改 Accelerate YAML 的 `num_processes` 和 `CUDA_VISIBLE_DEVICES`。本次只验收 2/4 GPU；
单卡和其他规模不得沿用本页性能/容量结论。启动前确认 `output_dir` 不存在或为空。

## 6. 交付验收

训练结束至少检查：

1. 每步 loss 和 grad norm 都是 finite；用 held-out 样本验证任务质量，不能只看两三个 smoke step。
2. 原始模型目录的文件清单和 SHA-256 未变化。
3. 输出目录包含 `adapter_config.json`、`adapter_model.safetensors` 和
   `fused_expert_lora.safetensors`；routed INT8 还必须包含 `kt_adapter_manifest.json`。
4. router LoRA-B 和 fused expert LoRA-B 均有非零张量。
5. 日志中没有 OOM、collective error、cache-stack underflow 或重复 distributed launch。

当前 LLaMA-Factory 不提供 KT inference backend。端到端生成质量必须使用另行验收、且能同时加载标准与
fused KT adapter 的推理链路；不要把 `infer_backend: ktransformers` 写入本分支配置。

## 7. 常见故障

| 现象 | 处理 |
| --- | --- |
| 提示从 Accelerate config 移除 `kt_config` | 把 KT 字段移到训练 YAML 的平坦 `kt_config` |
| 多卡启动后再次生成 torchrun 子进程 | 使用 `accelerate launch ... src/train.py train.yaml` |
| cache manifest/fingerprint 校验失败 | 重新生成匹配的 routed/non-expert cache，不要绕过校验 |
| cache-stack underflow 或容量不足 | 确认没有第二套 checkpoint wrapper，并增大 `kt_model_max_length` |
| CPU retain 导致主存不足 | 改用 `kt_cpu_activation: recompute` 或增加 RAM |
| resume/checkpoint/adapter 续训报错或 expert LoRA 重新初始化 | 从头启动；当前不支持恢复 optimizer 或 fused expert LoRA |
| `output_dir` 非空或自动 resume | 改用新的空目录，从新的 LoRA 初始化开始 |
| 原始 Qwen3.5 Hub config 触发 Conv3D 拒绝启动 | 改用 release bundle 中经过验证的 `TEXTONLY` 模型目录 |
| KT integration/ABI 缺失 | 恢复匹配的 `transformers-kt`、`accelerate-kt` 和 KT extension |
