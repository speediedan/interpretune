#!/usr/bin/env python
"""Debug batch collapse: why does generate() return 1 prediction for batch of 2?

Loads the baseline config model directly (no LightningCLI) and inspects
the generation pipeline step by step.
"""

from __future__ import annotations
import os
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def debug_batch_collapse():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_lines: list[str] = []

    def log(msg: str):
        print(msg)
        log_lines.append(msg)

    log(f"=== Batch Collapse Debug ({timestamp}) ===\n")

    # Model and tokenizer setup matching the baseline config
    model_name = "google/gemma-2-2b-it"
    access_token = os.environ.get("HF_GATED_PUBLIC_REPO_AUTH_KEY")

    log("1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=access_token,
        padding_side="left",
        add_bos_token=True,
    )
    # Match config: eos_token_id override
    tokenizer.eos_token_id = 1
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    log(f"   eos_token_id: {tokenizer.eos_token_id}")
    log(f"   pad_token_id: {tokenizer.pad_token_id}")
    log(f"   bos_token_id: {tokenizer.bos_token_id}")
    log(f"   padding_side: {tokenizer.padding_side}")

    log("\n2. Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=access_token,
        device_map=0,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.eval()

    # Apply generation config matching the YAML
    log("\n3. Setting generation config...")
    model.generation_config.eos_token_id = 1
    model.generation_config.use_cache = False
    model.generation_config.max_new_tokens = 5
    model.generation_config.do_sample = True
    model.generation_config.top_k = 50
    model.generation_config.padding_side = "left"
    model.generation_config.output_logits = True
    model.generation_config.return_dict_in_generate = True
    model.config.use_cache = False
    log(f"   generation_config: {model.generation_config}")

    # Check entailment mapping
    log("\n4. Entailment mapping...")
    mapping = ("Yes", "No")
    token_ids = tokenizer.convert_tokens_to_ids(mapping)
    log(f"   mapping: {mapping} -> token_ids: {token_ids}")

    # Build prompts matching gemma-chat format
    log("\n5. Building test prompts...")
    # RTE examples
    examples = [
        {
            "premise": "No Weapons of Mass Destruction Found in Iraq Yet.",
            "hypothesis": "Weapons of Mass Destruction Found in Iraq.",
        },
        {
            "premise": "A man is standing on the corner.",
            "hypothesis": "A person is standing on the corner.",
        },
    ]

    ctx_question_join = "\nQuestion: Does the above text entail that "
    question_suffix = "?\n"

    prompts = []
    for ex in examples:
        text = f"{ex['premise']}{ctx_question_join}{ex['hypothesis']}{question_suffix}"
        prompts.append(text)

    for i, p in enumerate(prompts):
        log(f"   Prompt {i}: {p[:100]}...")

    log("\n6. Tokenizing batch...")
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    log(f"   input_ids shape: {encoded['input_ids'].shape}")
    log(f"   attention_mask shape: {encoded['attention_mask'].shape}")

    # Move to device
    device = model.device
    encoded = {k: v.to(device) for k, v in encoded.items()}

    log("\n7. Running generate()...")
    with torch.no_grad():
        outputs = model.generate(**encoded)

    log(f"   type(outputs): {type(outputs).__name__}")
    if hasattr(outputs, "sequences"):
        log(f"   sequences shape: {outputs.sequences.shape}")
    else:
        log(f"   outputs shape: {outputs.shape}")

    if hasattr(outputs, "logits") and outputs.logits is not None:
        log(f"   logits type: {type(outputs.logits)}")
        if isinstance(outputs.logits, tuple):
            log(f"   logits tuple length: {len(outputs.logits)}")
            for i, t in enumerate(outputs.logits):
                log(f"   logits[{i}] shape: {t.shape}")
        elif isinstance(outputs.logits, torch.Tensor):
            log(f"   logits tensor shape: {outputs.logits.shape}")
    else:
        log("   logits: None or missing")

    # Now trace through standardize_logits logic
    log("\n8. Tracing standardize_logits logic...")
    if hasattr(outputs, "logits") and outputs.logits is not None:
        logits = outputs.logits
        if isinstance(logits, tuple):
            log(f"   logits is tuple of length {len(logits)}")
            stacked = torch.stack([out for out in logits], dim=1)
            log(f"   After stack: {stacked.shape}")
        else:
            stacked = logits
            log(f"   logits is tensor: {stacked.shape}")

        if stacked.ndim == 2:
            stacked = stacked.unsqueeze(1)
            log(f"   After unsqueeze: {stacked.shape}")

        # index_select with mapping indices
        map_indices = torch.tensor(token_ids, device=device)
        selected = torch.index_select(stacked, -1, map_indices)
        log(f"   After index_select: {selected.shape}")

        # collect_answers logic
        per_example, _ = torch.max(selected, dim=-2)
        log(f"   After max(dim=-2): {per_example.shape}")
        preds = torch.argmax(per_example, dim=-1)
        log(f"   preds shape: {preds.shape}")
        log(f"   preds: {preds}")

        # Decode the answers
        for i, pred_idx in enumerate(preds):
            log(f"   Example {i}: predicted '{mapping[pred_idx.item()]}'")

    # Also test with raw generate (return sequences only)
    log("\n9. Raw generate (sequences only)...")
    model.generation_config.output_logits = False
    model.generation_config.return_dict_in_generate = False
    with torch.no_grad():
        raw_outputs = model.generate(**encoded)
    log(f"   raw_outputs shape: {raw_outputs.shape}")
    for i in range(raw_outputs.shape[0]):
        decoded = tokenizer.decode(raw_outputs[i], skip_special_tokens=True)
        log(f"   Example {i} decoded: ...{decoded[-80:]}")

    # Save log
    log_dir = os.environ.get("IT_DEBUG_LOG_DIR", "/tmp/it_benchmark_debugging")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"batch_collapse_debug_{timestamp}.log")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    debug_batch_collapse()
