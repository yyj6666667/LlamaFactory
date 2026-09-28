# Copyright 2026 the LlamaFactory team.
# Licensed under the Apache License, Version 2.0.

"""Validate the April Flash recipe before tokenizing data or loading weights."""

V4_LORA_TARGETS = frozenset(
    {
        "self_attn.q_a_proj",
        "self_attn.q_b_proj",
        "self_attn.kv_proj",
        "self_attn.o_b_proj",
        "self_attn.compressor.kv_proj",
        "self_attn.compressor.gate_proj",
        "mlp.shared_experts.gate_proj",
        "mlp.shared_experts.up_proj",
        "mlp.shared_experts.down_proj",
    }
)


def validate_kt_deepseek_v4_training(config, model_args, data_args, training_args, finetuning_args):
    if not model_args.use_kt or getattr(config, "model_type", None) != "deepseek_v4":
        return

    if data_args.template != "deepseek4_nothink":
        raise ValueError("KT April V4 Flash requires template: deepseek4_nothink for raw ordinary conversations.")
    if data_args.packing or data_args.neat_packing:
        raise ValueError("KT April V4 Flash does not support packing; set packing: false and neat_packing: false.")
    if data_args.enable_thinking is True or data_args.tool_format is not None:
        raise ValueError("KT April V4 Flash currently supports ordinary non-thinking conversations without tools.")
    if not 0 < data_args.cutoff_len <= 1024:
        raise ValueError("KT April V4 Flash currently supports cutoff_len up to 1024.")
    if not training_args.bf16 or finetuning_args.pure_bf16 or training_args.bf16_full_eval:
        raise ValueError(
            "KT April V4 Flash requires BF16 autocast: bf16: true, pure_bf16: false, bf16_full_eval: false."
        )
    if training_args.predict_with_generate:
        raise ValueError("Use static SGLang inference for April V4; Transformers generation is not supported here.")
    if model_args.flash_attn != "disabled":
        raise ValueError("KT April V4 Flash training requires flash_attn: disabled (eager attention).")
    if finetuning_args.stage != "sft" or finetuning_args.finetuning_type != "lora":
        raise ValueError("KT April V4 Flash supports stage: sft with finetuning_type: lora.")
    if (finetuning_args.lora_rank, finetuning_args.lora_alpha, finetuning_args.lora_dropout) != (8, 16, 0):
        raise ValueError("KT April V4 Flash requires lora_rank: 8, lora_alpha: 16, lora_dropout: 0.")
    targets = set(finetuning_args.lora_target)
    if not targets or not targets <= V4_LORA_TARGETS:
        raise ValueError(
            "KT April V4 Flash requires explicit LoRA targets from the supported set; "
            f"lora_target: all is not supported. Supported targets: {', '.join(sorted(V4_LORA_TARGETS))}"
        )
    if finetuning_args.additional_target or any(
        getattr(finetuning_args, name, False) for name in ("use_dora", "use_rslora", "pissa_init", "use_llama_pro")
    ):
        raise ValueError(
            "KT April V4 Flash supports standard LoRA on the explicit target set without additional_target."
        )
    if not model_args.kt_non_expert_weight_path:
        raise ValueError("Set kt_non_expert_weight_path; KT will automatically prepare and validate the V4 cache.")

    advanced = getattr(training_args, "kt_config", None) or {}
    if advanced.get("kt_expert_weight_format", "").lower() != "mxfp4":
        raise ValueError("KT April V4 Flash requires kt_config.kt_expert_weight_format: mxfp4.")
