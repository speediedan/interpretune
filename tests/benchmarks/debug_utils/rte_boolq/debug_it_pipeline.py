#!/usr/bin/env python
"""Debug batch collapse through the interpretune pipeline.

Loads an ITSession using internal APIs (not LightningCLI) and traces
exactly what happens in test_step with a batch of 2.
"""
from __future__ import annotations
import os
import sys
from datetime import datetime

import torch
import yaml


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_lines: list[str] = []

    def log(msg: str):
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"=== IT Pipeline Batch Collapse Debug ({timestamp}) ===\n")

    # Load the YAML config to understand what we're working with
    config_path = "src/it_examples/config/experiments/rte_boolq/gemma2/2b_chat_lightning_zs_test.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    log(f"Config: {config_path}")
    log(f"Module class: {cfg['session_cfg']['module_cls']}")
    log(f"Datamodule class: {cfg['session_cfg']['datamodule_cls']}")
    log(f"Adapter ctx: {cfg['session_cfg']['adapter_ctx']}")

    # Use ITSession directly instead of LightningCLI
    from interpretune.session import ITSession, ITSessionConfig
    from interpretune.config.datamodule import ITDataModuleConfig
    from interpretune.config.mixins import HFGenerationConfig
    from it_examples.experiments.rte_boolq import (
        RTEBoolqConfig,
        RTEBoolqModule,
        RTEBoolqDataModule,
        RTEBoolqGenerativeClassificationConfig,
    )
    from interpretune.config.module import HFFromPretrainedConfig

    log("\n1. Building configs...")

    # Datamodule config
    dm_cfg = ITDataModuleConfig(
        model_name_or_path="google/gemma-2-2b-it",
        task_name="rte",
        train_batch_size=2,
        eval_batch_size=2,
        cust_tokenization_pattern="gemma-chat",
        os_env_model_auth_key="HF_GATED_PUBLIC_REPO_AUTH_KEY",
        tokenizer_id_overrides={"eos_token_id": 1},
        enable_datasets_cache=True,
        prepare_data_map_cfg={"batched": True},
        data_collator_cfg={"collator_class": "transformers.DataCollatorWithPadding"},
        prompt_cfg={"class_path": "it_examples.example_prompt_configs.RTEBoolqGemma2PromptConfig"},
        tokenizers_parallelism=False,
        tokenizer_kwargs={
            "local_files_only": False,
            "add_bos_token": True,
            "padding_side": "left",
            "model_input_names": ["input_ids", "attention_mask"],
        },
    )

    # Module config
    gen_cfg = RTEBoolqGenerativeClassificationConfig(
        enabled=True,
        lm_generation_cfg=HFGenerationConfig(
            model_config={
                "eos_token_id": 1,
                "use_cache": False,
                "max_new_tokens": 5,
                "do_sample": True,
                "top_k": 50,
                "padding_side": "left",
                "output_logits": True,
                "return_dict_in_generate": True,
            }
        ),
    )

    module_cfg = RTEBoolqConfig(
        experiment_tag="debug_batch_collapse",
        model_cfg={"_attn_implementation": "eager"},
        generative_step_cfg=gen_cfg,
        hf_from_pretrained_cfg=HFFromPretrainedConfig(
            use_model_cache=False,
            model_head="transformers.Gemma2ForCausalLM",
            activation_checkpointing=False,
            pretrained_kwargs={"device_map": 0, "dtype": "bfloat16"},
        ),
    )

    log(f"   gen_cfg.lm_generation_cfg type: {type(gen_cfg.lm_generation_cfg).__name__}")
    log(f"   gen_cfg.lm_generation_cfg.model_config: {gen_cfg.lm_generation_cfg.model_config}")
    log(f"   gen_cfg.lm_generation_cfg.generate_kwargs: {gen_cfg.lm_generation_cfg.generate_kwargs}")

    log("\n2. Creating ITSession...")
    session_cfg = ITSessionConfig(
        adapter_ctx=["lightning"],
        datamodule_cfg=dm_cfg,
        module_cfg=module_cfg,
        datamodule_cls=RTEBoolqDataModule,
        module_cls=RTEBoolqModule,
    )
    session = ITSession(session_cfg)

    log("\n3. Composing interpretunable...")
    session.compose_interpretunable()

    log("\n4. Setting up datamodule...")
    dm = session.datamodule
    dm.prepare_data()
    dm.setup(stage="test")

    module = session.module
    log(f"   Module type: {type(module).__name__}")
    log(f"   Model type: {type(module.model).__name__}")

    # Check gen config on the model
    gen_config = getattr(module.model, "generation_config", None)
    if gen_config:
        log(f"\n5. Model generation_config:")
        log(f"   output_logits: {getattr(gen_config, 'output_logits', 'NOT SET')}")
        log(f"   return_dict_in_generate: {getattr(gen_config, 'return_dict_in_generate', 'NOT SET')}")
        log(f"   max_new_tokens: {getattr(gen_config, 'max_new_tokens', 'NOT SET')}")
        log(f"   do_sample: {getattr(gen_config, 'do_sample', 'NOT SET')}")
        log(f"   eos_token_id: {getattr(gen_config, 'eos_token_id', 'NOT SET')}")

    # Check _should_inspect_inputs
    log(f"\n6. Input inspection:")
    log(f"   _generate_accepts_kwargs: {module._generate_accepts_kwargs()}")
    log(f"   _generate_prepares_inputs: {module._generate_prepares_inputs()}")
    log(f"   _should_inspect_inputs: {module._should_inspect_inputs}")

    # Get a test batch
    log(f"\n7. Getting test batch...")
    test_dl = dm.test_dataloader()
    batch = next(iter(test_dl))
    log(f"   Batch keys: {list(batch.keys()) if hasattr(batch, 'keys') else type(batch)}")
    for k in batch.keys() if hasattr(batch, "keys") else []:
        if isinstance(batch[k], torch.Tensor):
            log(f"   {k}: shape={batch[k].shape}, dtype={batch[k].dtype}")
        else:
            log(f"   {k}: {type(batch[k]).__name__} = {str(batch[k])[:80]}")

    # Pop labels as test_step does
    labels = batch.pop("labels")
    log(f"   labels: {labels}")

    # Move batch to device
    device = module.device if hasattr(module, "device") else torch.device("cuda:0")
    log(f"   device: {device}")
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    log(f"\n8. Calling it_generate...")
    gen_kwargs = module.it_cfg.generative_step_cfg.lm_generation_cfg.generate_kwargs
    log(f"   generate_kwargs: {gen_kwargs}")
    log(f"   batch keys going to generate: {list(batch.keys())}")

    with torch.no_grad():
        outputs = module.it_generate(batch, **gen_kwargs)

    log(f"\n9. Generate outputs:")
    log(f"   type: {type(outputs).__name__}")
    if isinstance(outputs, torch.Tensor):
        log(f"   shape: {outputs.shape}")
    elif hasattr(outputs, "sequences"):
        log(f"   sequences shape: {outputs.sequences.shape}")
    if hasattr(outputs, "logits") and outputs.logits is not None:
        logits = outputs.logits
        log(f"   logits type: {type(logits)}")
        if isinstance(logits, tuple):
            log(f"   logits tuple length: {len(logits)}")
            for i, t in enumerate(logits):
                log(f"   logits[{i}] shape: {t.shape}")
        elif isinstance(logits, torch.Tensor):
            log(f"   logits tensor shape: {logits.shape}")
    else:
        log(f"   no logits attribute!")

    # Now trace through standardize_logits
    log(f"\n10. Tracing standardize_logits...")
    if hasattr(outputs, "logits") and outputs.logits is not None:
        logits = outputs.logits
    elif isinstance(outputs, torch.Tensor):
        logits = outputs
    else:
        log(f"   ERROR: No logits found!")
        logits = None

    if logits is not None:
        result = module.standardize_logits(logits)
        log(f"   standardized shape: {result.shape}")

        per_example_answers, _ = torch.max(result, dim=-2)
        log(f"   per_example_answers shape: {per_example_answers.shape}")
        preds = torch.argmax(per_example_answers, dim=-1)
        log(f"   preds shape: {preds.shape}")
        log(f"   preds: {preds}")
        log(f"   labels: {labels}")
        log(f"   preds count: {preds.shape[0]}, labels count: {labels.shape[0] if isinstance(labels, torch.Tensor) else len(labels)}")

    # Save log
    log_dir = "/home/speediedan/repos/distributed-insight/project_admin/interpretune/design/circuit-tracer-backend/benchmark_debugging"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"it_pipeline_debug_{timestamp}.log")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    log(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    main()
