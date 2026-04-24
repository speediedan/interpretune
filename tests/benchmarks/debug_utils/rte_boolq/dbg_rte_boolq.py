#!/usr/bin/env python
"""
RTE/BoolQ Experiment Debug Utilities
=====================================

Experiment-specific diagnostics for the RTE/BoolQ benchmark.
Includes entailment mapping verification and label analysis.

Usage:
    python tests/benchmarks/debug_utils/rte_boolq/dbg_rte_boolq.py \
        --config src/it_examples/config/experiments/rte_boolq/gemma2/2b_chat_lightning_zs_test.yaml \
        --output /tmp/rte_boolq_diagnostics.log
"""

from __future__ import annotations

import argparse
import json
import logging
from io import StringIO
from pathlib import Path

import torch

from tests.benchmarks.benchmark_utils import (
    _detect_cli_mode_from_config,
    load_cli_session,
    run_shared_diagnostics,
    section,
)

log = logging.getLogger(__name__)


def check_entailment_mapping(model, tokenizer, buf: StringIO, results: dict) -> None:
    """Verify entailment token mapping (Yes/No -> token IDs)."""
    buf.write(section("Entailment Mapping"))
    if hasattr(model.it_cfg, "entailment_mapping"):
        em = model.it_cfg.entailment_mapping
        buf.write(f"Entailment mapping: {em}\n")
        token_ids = tokenizer.convert_tokens_to_ids(em)
        buf.write(f"Token IDs: {token_ids}\n")
        for tok, tid in zip(em, token_ids):
            decoded = tokenizer.decode([tid])
            buf.write(f"  '{tok}' -> id={tid} -> decoded='{decoded}'\n")
        results["entailment_token_ids"] = token_ids
    else:
        buf.write("No entailment_mapping on it_cfg\n")


def check_entailment_predictions(model, batch, buf: StringIO, results: dict) -> None:
    """Check entailment predictions on a sample batch."""
    buf.write(section("Entailment Prediction Check"))
    device_batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    labels = device_batch.pop("labels")

    gen_kwargs = model.it_cfg.generative_step_cfg.lm_generation_cfg.generate_kwargs

    try:
        with torch.inference_mode():
            outputs = model.it_generate(device_batch, **gen_kwargs)

        logits = None
        if isinstance(outputs, torch.Tensor):
            logits = outputs
        elif hasattr(outputs, "logits") and outputs.logits is not None:
            logits = outputs.logits

        if logits is not None:
            if isinstance(logits, tuple):
                logits_for_analysis = torch.stack(list(logits), dim=1)
            else:
                logits_for_analysis = logits
            if logits_for_analysis.ndim == 2:
                logits_for_analysis = logits_for_analysis.unsqueeze(1)

            if (
                hasattr(model.it_cfg, "entailment_mapping_indices")
                and model.it_cfg.entailment_mapping_indices is not None
            ):
                emi = model.it_cfg.entailment_mapping_indices
                buf.write(f"  entailment_mapping_indices: {emi}\n")
                selected = torch.index_select(logits_for_analysis, -1, emi)
                buf.write(f"  After index_select: {selected.shape}\n")
                for b in range(selected.shape[0]):
                    buf.write(f"    batch {b}: {selected[b].tolist()}\n")

                per_example, _ = torch.max(selected, dim=-2)
                preds = torch.argmax(per_example, dim=-1)
                buf.write(f"  Predictions: {preds.tolist()}\n")
                buf.write(f"  Labels: {labels.tolist()}\n")
                results["single_batch_preds"] = preds.tolist()
                results["single_batch_labels"] = labels.tolist()
    except Exception as e:
        buf.write(f"  ERROR: {e}\n")


def run_diagnostics(config_path: str, output_path: str | None = None) -> dict:
    """Run full diagnostics including RTE-specific checks."""
    cli_mode = _detect_cli_mode_from_config(config_path)
    # Run shared diagnostics first
    shared_results = run_shared_diagnostics(config_path, cli_mode=cli_mode)

    # Now run experiment-specific checks
    results = {**shared_results}
    buf = StringIO()

    try:
        cli, trainer, model, datamodule = load_cli_session(config_path, cli_mode=cli_mode)
        tokenizer = datamodule.tokenizer

        check_entailment_mapping(model, tokenizer, buf, results)

        # Get a batch for prediction check
        if trainer is not None:
            trainer.datamodule = datamodule
        datamodule.prepare_data()
        datamodule.setup(stage="test")
        batch = next(iter(datamodule.test_dataloader()))

        check_entailment_predictions(model, batch, buf, results)
    except Exception as e:
        buf.write(f"\nERROR in RTE-specific diagnostics: {e}\n")

    output = buf.getvalue()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "a") as f:
            f.write(f"\n\n{'=' * 70}\nRTE/BoolQ Specific Diagnostics\n{'=' * 70}\n")
            f.write(output)
            f.write("\n\nResults JSON:\n")
            f.write(json.dumps(results, indent=2, default=str))

    print(output)
    return results


def main():
    parser = argparse.ArgumentParser(description="RTE/BoolQ Benchmark Diagnostics")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--output", type=str, default=None, help="Output log file path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_diagnostics(args.config, args.output)


if __name__ == "__main__":
    main()
