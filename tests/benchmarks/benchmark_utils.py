#!/usr/bin/env python
"""
Benchmark Utilities
===================

Shared diagnostic utilities for benchmark debugging across experiments.
Provides model loading, tokenizer inspection, dataset checks, and generation diagnostics.

Experiment-specific diagnostics live in ``debug_utils/<experiment_name>/dbg_<experiment_name>.py``.
"""

from __future__ import annotations

import json
import logging
import re
import traceback
from io import StringIO
from pathlib import Path

import torch

log = logging.getLogger(__name__)


def section(title: str) -> str:
    return f"\n{'=' * 70}\n  {title}\n{'=' * 70}\n"


def parse_accuracy(output: str) -> float | None:
    """Parse accuracy from CLI test output.

    Looks for patterns like:
        'accuracy': tensor(0.7906, ...)
        accuracy    0.7906
        {'accuracy': 0.7906}
        Test epoch end: {'accuracy': 0.7220}

    When multiple matches are found, the last match is used (epoch-end summary
    takes precedence over per-batch values).
    """
    patterns = [
        r"['\"]accuracy['\"]:\s*tensor\(([\d.]+)",
        r"accuracy\s+[\│|]?\s*([\d.]+)",
        r"['\"]accuracy['\"]:\s*([\d.]+)",
    ]
    last_match = None
    for pattern in patterns:
        for match in re.finditer(pattern, output):
            last_match = float(match.group(1))
    return last_match


def load_cli_session(config_path: str, cli_mode: str = "lightning"):
    """Load an ITSession via the CLI without running.

    Args:
        config_path: Path to the YAML config file.
        cli_mode: ``"lightning"`` for LightningCLI or ``"core"`` for core ITCLI.

    Returns:
        For lightning: (cli, trainer, model, datamodule) tuple.
        For core: (cli, None, model, datamodule) tuple (no trainer in core mode).
    """
    import sys as _sys

    orig_argv = _sys.argv
    if cli_mode == "lightning":
        # Note: l_cli_main(run_mode=False) uses a flat parser without subcommands,
        # so "test" should NOT be in argv (it would be an unrecognized argument).
        _sys.argv = ["diagnostics", "--config", config_path]
        try:
            from interpretune.base.components.cli import l_cli_main

            cli = l_cli_main(run_mode=False)
        finally:
            _sys.argv = orig_argv
        return cli, cli.trainer, cli.model, cli.datamodule
    else:
        _sys.argv = ["diagnostics", "--config", config_path]
        try:
            from interpretune.base.components.cli import core_cli_main

            cli = core_cli_main(run_mode=False)
        finally:
            _sys.argv = orig_argv
        return cli, None, cli.module, cli.datamodule


def check_model_info(model, buf: StringIO, results: dict) -> None:
    """Log model type, MRO, and config information."""
    buf.write(f"Model type: {type(model).__name__}\n")
    buf.write(f"Model MRO: {[c.__name__ for c in type(model).__mro__[:8]]}\n")
    if hasattr(model, "it_cfg"):
        buf.write(f"IT Config type: {type(model.it_cfg).__name__}\n")
        if hasattr(model.it_cfg, "generative_step_cfg"):
            gs = model.it_cfg.generative_step_cfg
            buf.write(f"Generative step enabled: {gs.enabled}\n")
            if hasattr(gs, "lm_generation_cfg"):
                gen_cfg = gs.lm_generation_cfg
                buf.write(f"Generation config type: {type(gen_cfg).__name__}\n")
                if hasattr(gen_cfg, "generate_kwargs"):
                    buf.write(f"Generate kwargs: {gen_cfg.generate_kwargs}\n")
        if hasattr(model.it_cfg, "num_labels"):
            buf.write(f"num_labels: {model.it_cfg.num_labels}\n")
    if hasattr(model, "model"):
        buf.write(f"Inner model type: {type(model.model).__name__}\n")
        if hasattr(model.model, "device"):
            buf.write(f"Model device: {model.model.device}\n")
    buf.write(f"Model device (module-level): {model.device}\n")
    results["model_loaded"] = True


def check_tokenizer(tokenizer, buf: StringIO) -> None:
    """Log tokenizer info (type, vocab, special tokens)."""
    buf.write(section("Tokenizer Check"))
    buf.write(f"Tokenizer type: {type(tokenizer).__name__}\n")
    buf.write(f"Vocab size: {tokenizer.vocab_size}\n")
    buf.write(f"Padding side: {tokenizer.padding_side}\n")
    buf.write(f"Model input names: {tokenizer.model_input_names}\n")
    buf.write(f"BOS token: {tokenizer.bos_token!r} (id={tokenizer.bos_token_id})\n")
    buf.write(f"EOS token: {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})\n")
    buf.write(f"PAD token: {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})\n")


def check_dataset(datamodule, trainer, tokenizer, buf: StringIO, results: dict) -> dict:
    """Run dataset/dataloader diagnostics.

    Returns the first batch.
    """
    buf.write(section("Dataset Check"))
    if trainer is not None:
        trainer.datamodule = datamodule
    datamodule.prepare_data()
    datamodule.setup(stage="test")

    test_dl = datamodule.test_dataloader()
    batch_iter = iter(test_dl)
    batch = next(batch_iter)

    buf.write(f"Batch keys: {list(batch.keys()) if hasattr(batch, 'keys') else type(batch)}\n")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            buf.write(f"  {k}: shape={v.shape}, dtype={v.dtype}\n")
        elif isinstance(v, list):
            buf.write(f"  {k}: list, len={len(v)}\n")
        else:
            buf.write(f"  {k}: {type(v).__name__}\n")

    if "input_ids" in batch:
        input_key = "input_ids"
    elif "input" in batch:
        input_key = "input"
    else:
        input_key = list(batch.keys())[0]

    first_input = batch[input_key][0]
    decoded_first = tokenizer.decode(first_input, skip_special_tokens=False)
    buf.write(f"\nFirst example ('{input_key}' decoded):\n")
    buf.write(f"  Length: {len(first_input)} tokens\n")
    buf.write(f"  Tokens: {first_input.tolist()[:20]}...\n")
    buf.write(f"  Decoded: {decoded_first[:500]}\n")
    results["first_example_decoded"] = decoded_first[:200]
    results["batch_size"] = len(batch[input_key])

    if "labels" in batch:
        buf.write(f"\nLabels: {batch['labels'].tolist()}\n")
        results["first_batch_labels"] = batch["labels"].tolist()

    return batch


def check_sanity_generation(model, tokenizer, buf: StringIO, results: dict) -> None:
    """Run a simple generation to verify the model can generate text."""
    buf.write(section("Model Sanity Check: Simple Generation"))
    simple_prompt = "The capital of France is"
    enc = tokenizer(simple_prompt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    buf.write(f"Prompt: '{simple_prompt}'\n")
    buf.write(f"Encoded shape: {enc['input_ids'].shape}\n")

    with torch.inference_mode():
        gen_out = model.model.generate(**enc, max_new_tokens=10, do_sample=False)
    gen_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
    buf.write(f"Generated: '{gen_text}'\n")
    results["sanity_generation"] = gen_text


def check_sanity_reasoning(model, tokenizer, datamodule, buf: StringIO, results: dict) -> None:
    """Run a multi-hop reasoning probe and display logit analysis for key tokens.

    Uses the Dallas→Austin reasoning prompt from the circuit-tracer analysis demo.
    Selects a chat-style prompt when the datamodule uses a chat tokenization pattern.
    Generation kwargs are sourced from the model's ``lm_generation_cfg`` when available,
    with sensible defaults for introspection (``output_logits`` and ``return_dict_in_generate``).
    Token encoding uses a leading space (" Austin") so SentencePiece-based tokenizers
    produce the correct "▁Austin" token.
    """
    buf.write(section("Model Sanity Check: Multi-Hop Reasoning"))

    is_chat = getattr(getattr(datamodule, "itdm_cfg", None), "cust_tokenization_pattern", None) is not None
    if is_chat:
        prompt = "Answer with only the missing city name. Fact: the capital of the state containing Dallas is"
    else:
        prompt = "Fact: the capital of the state containing Dallas is"

    buf.write(f"Prompt style: {'chat' if is_chat else 'plain'}\n")
    buf.write(f"Prompt: '{prompt}'\n")

    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}

    # Build generation kwargs from the model's config, falling back to safe defaults
    gen_kwargs: dict = {}
    gen_cfg = getattr(getattr(getattr(model, "it_cfg", None), "generative_step_cfg", None), "lm_generation_cfg", None)
    if gen_cfg is not None:
        if hasattr(gen_cfg, "generate_kwargs"):
            gen_kwargs.update(gen_cfg.generate_kwargs)
        elif hasattr(gen_cfg, "model_config"):
            gen_kwargs.update(gen_cfg.model_config)
    gen_kwargs.setdefault("max_new_tokens", 5)
    gen_kwargs.setdefault("do_sample", False)
    # Always request logits for introspection
    gen_kwargs["output_logits"] = True
    gen_kwargs["return_dict_in_generate"] = True
    buf.write(f"Generation kwargs: {gen_kwargs}\n")

    with torch.inference_mode():
        gen_out = model.model.generate(**enc, **gen_kwargs)

    gen_text = tokenizer.decode(gen_out.sequences[0], skip_special_tokens=True)
    buf.write(f"Generated: '{gen_text}'\n")

    # Analyse logits from the first generated token (the reasoning answer)
    first_logits = gen_out.logits[0][0]  # (vocab,) — first batch, first gen step
    probs = torch.softmax(first_logits.float(), dim=-1)

    # Key tokens: Austin (correct), Dallas, Texas
    # Use space-prefixed strings so SentencePiece tokenizers produce the ▁-prefixed tokens
    key_tokens = {"Austin": " Austin", "Dallas": " Dallas", "Texas": " Texas"}
    key_ids: dict[str, int | None] = {}
    for label, text in key_tokens.items():
        ids = tokenizer.encode(text, add_special_tokens=False)
        key_ids[label] = ids[0] if ids else None

    # Top-1 predicted token
    top_id = first_logits.argmax(dim=-1).item()
    top_token = tokenizer.decode([top_id])
    top_prob = probs[top_id].item()

    buf.write("\nFirst-token logit analysis:\n")
    buf.write(f"  {'Token':<12} {'ID':>8} {'Logit':>10} {'Prob':>10}\n")
    buf.write(f"  {'-' * 44}\n")
    for label, tid in key_ids.items():
        if tid is not None:
            logit_val = first_logits[tid].item()
            prob_val = probs[tid].item()
            buf.write(f"  {label:<12} {tid:>8} {logit_val:>10.4f} {prob_val:>10.6f}\n")
        else:
            buf.write(f"  {label:<12} {'N/A':>8} {'N/A':>10} {'N/A':>10}\n")
    buf.write(f"  {'-' * 44}\n")
    buf.write(f"  {'Top-1':<12} {top_id:>8} {first_logits[top_id].item():>10.4f} {top_prob:>10.6f}  ({top_token!r})\n")

    results["sanity_reasoning"] = gen_text
    results["reasoning_top1_token"] = top_token
    results["reasoning_key_probs"] = {label: probs[tid].item() for label, tid in key_ids.items() if tid is not None}


def check_generation(model, batch, tokenizer, buf: StringIO, results: dict) -> None:
    """Run the it_generate path and inspect output shape/content."""
    buf.write(section("Single-Example Generation via it_generate"))
    device_batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    device_batch.pop("labels")

    gen_kwargs = model.it_cfg.generative_step_cfg.lm_generation_cfg.generate_kwargs
    buf.write(f"Generate kwargs: {gen_kwargs}\n")

    try:
        with torch.inference_mode():
            outputs = model.it_generate(device_batch, **gen_kwargs)
        buf.write(f"Output type: {type(outputs).__name__}\n")
        logits = None
        if isinstance(outputs, torch.Tensor):
            buf.write(f"Output shape: {outputs.shape}\n")
            logits = outputs
        elif hasattr(outputs, "logits") and outputs.logits is not None:
            if isinstance(outputs.logits, tuple):
                buf.write(f"logits is tuple of len {len(outputs.logits)}\n")
                for i, ll in enumerate(outputs.logits):
                    buf.write(f"  logits[{i}] shape: {ll.shape}\n")
            else:
                buf.write(f"logits shape: {outputs.logits.shape}\n")
            logits = outputs.logits

        if logits is not None:
            buf.write("\nLogits analysis:\n")
            if isinstance(logits, tuple):
                stacked = torch.stack(list(logits), dim=1)
                buf.write(f"  Stacked logits shape: {stacked.shape}\n")
                logits_for_analysis = stacked
            else:
                logits_for_analysis = logits
                buf.write(f"  Logits shape: {logits.shape}\n")

            if logits_for_analysis.ndim == 2:
                logits_for_analysis = logits_for_analysis.unsqueeze(1)
                buf.write(f"  After unsqueeze: {logits_for_analysis.shape}\n")

        if hasattr(outputs, "sequences"):
            for i in range(min(outputs.sequences.shape[0], 4)):
                gen_seq = tokenizer.decode(outputs.sequences[i], skip_special_tokens=True)
                buf.write(f"\n  Generated seq[{i}]: ...{gen_seq[-100:]}\n")

        results["generation_success"] = True
    except Exception as e:
        buf.write(f"\nERROR in generation: {e}\n")
        buf.write(traceback.format_exc())
        results["generation_success"] = False
        results["generation_error"] = str(e)


def check_dataset_caching(datamodule, buf: StringIO) -> None:
    """Check dataset caching configuration."""
    buf.write(section("Dataset Caching Check"))
    if hasattr(datamodule, "itdm_cfg"):
        buf.write(f"enable_datasets_cache: {datamodule.itdm_cfg.enable_datasets_cache}\n")
        if hasattr(datamodule.itdm_cfg, "dataset_path"):
            dp = datamodule.itdm_cfg.dataset_path
            buf.write(f"dataset_path: {dp}\n")
            if dp and Path(dp).exists():
                buf.write("  Path exists: True\n")
                mtime = Path(dp).stat().st_mtime
                from datetime import datetime

                buf.write(f"  Last modified: {datetime.fromtimestamp(mtime)}\n")


def _detect_cli_mode_from_config(config_path: str) -> str:
    """Detect CLI mode from a YAML config's adapter_ctx field.

    Checks both top-level and ``session_cfg``-nested ``adapter_ctx``.
    Falls back to ``"lightning"`` when adapter_ctx cannot be determined.
    """
    try:
        with open(config_path) as fh:
            import yaml as _yaml

            cfg = _yaml.safe_load(fh)
        # adapter_ctx may be at top level or nested under session_cfg
        adapter_ctx = cfg.get("adapter_ctx") or cfg.get("session_cfg", {}).get("adapter_ctx", [])
        if isinstance(adapter_ctx, list) and "lightning" not in adapter_ctx:
            return "core"
    except Exception:
        pass
    return "lightning"


def run_shared_diagnostics(config_path: str, output_path: str | None = None, cli_mode: str | None = None) -> dict:
    """Run shared diagnostic checks (no experiment-specific logic).

    Args:
        config_path: Path to the YAML config file.
        output_path: Optional path to write diagnostic log.
        cli_mode: ``"lightning"`` or ``"core"``. Auto-detected from config if not provided.

    Returns:
        dict of diagnostic results.
    """
    if cli_mode is None:
        cli_mode = _detect_cli_mode_from_config(config_path)

    results = {}
    buf = StringIO()

    try:
        buf.write(section("Loading Config and Model"))
        buf.write(f"Config: {config_path}\n")
        buf.write(f"CLI mode: {cli_mode}\n")
        cli, trainer, model, datamodule = load_cli_session(config_path, cli_mode=cli_mode)
        tokenizer = datamodule.tokenizer

        check_model_info(model, buf, results)
        check_tokenizer(tokenizer, buf)
        batch = check_dataset(datamodule, trainer, tokenizer, buf, results)
        check_sanity_generation(model, tokenizer, buf, results)
        check_sanity_reasoning(model, tokenizer, datamodule, buf, results)
        check_generation(model, batch, tokenizer, buf, results)
        check_dataset_caching(datamodule, buf)
        results["diagnostics_complete"] = True
    except Exception as e:
        buf.write(f"\nFATAL ERROR: {e}\n")
        buf.write(traceback.format_exc())
        results["fatal_error"] = str(e)

    output = buf.getvalue()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(output)
            f.write(f"\n\n{'=' * 70}\nResults JSON:\n{'=' * 70}\n")
            f.write(json.dumps(results, indent=2, default=str))

    print(output)
    return results
