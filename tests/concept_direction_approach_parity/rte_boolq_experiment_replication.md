# RTE BoolQ Experiment Replication

## Scope

This note records the current attempt to reproduce the established Gemma 2 IT RTE/BoolQ chat-path results from the existing Interpretune CLI configs.


## Target Configs

Primary configs examined:

1. [src/it_examples/config/experiments/rte_boolq/gemma2/2b_chat_lightning_zs_test.yaml](/home/speediedan/repos/interpretune/src/it_examples/config/experiments/rte_boolq/gemma2/2b_chat_lightning_zs_test.yaml)
2. [src/it_examples/config/experiments/rte_boolq/gemma2/2b_chat_lightning_sl_zs_test.yaml](/home/speediedan/repos/interpretune/src/it_examples/config/experiments/rte_boolq/gemma2/2b_chat_lightning_sl_zs_test.yaml)
3. [src/it_examples/experiments/rte_boolq.py](/home/speediedan/repos/interpretune/src/it_examples/experiments/rte_boolq.py)

Reference launch path:

1. [.vscode/launch.json](/home/speediedan/repos/interpretune/.vscode/launch.json)

## Historical Reference

The prior grounding note in the distributed-insight admin docs reported approximate Gemma 2 IT chat results around:

1. baseline: `79.1`
2. SAELens variant: `80.5`

Those values came from earlier `rte_boolq.py` experimentation and remain the baseline target for comparison.

## Replication Commands

Attempted command:

```bash
cd /home/speediedan/repos/interpretune
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
/mnt/cache/speediedan/.venvs/it_latest/bin/interpretune \
  --lightning_cli test \
  --config src/it_examples/config/experiments/rte_boolq/gemma2/2b_chat_lightning_zs_test.yaml \
  --config src/it_examples/config/global/base_debug.yaml
```

Current baseline command without the debug override:

```bash
cd /home/speediedan/repos/interpretune
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
/mnt/cache/speediedan/.venvs/it_latest/bin/interpretune \
  --lightning_cli test \
  --config src/it_examples/config/experiments/rte_boolq/gemma2/2b_chat_lightning_zs_test.yaml
```

Current SAE command:

```bash
cd /home/speediedan/repos/interpretune
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
/mnt/cache/speediedan/.venvs/it_latest/bin/interpretune \
  --lightning_cli test \
  --config src/it_examples/config/experiments/rte_boolq/gemma2/2b_chat_lightning_sl_zs_test.yaml
```

## Current Result

### Gemma 2 IT Baseline

The baseline run now finishes successfully in this working tree.

Observed result:

1. accuracy: `0.454873651266098`

This is still far below the historical reference of roughly `79.1`, so execution is unblocked but parity is not yet restored.

### Gemma 2 IT SAE

The SAE config still does not finish successfully.

Two separate issues were uncovered while trying to run it:

1. parser drift in the example config path
2. a current runtime CUDA indexing failure in the generative evaluation path once parsing is fixed

The current SAE failure ends with a device-side CUDA assert during evaluation:

```text
Assertion `srcIndex < srcSelectDimSize` failed
...
torch.AcceleratorError: CUDA error: device-side assert triggered
```

The traceback currently terminates through:

1. `test_step(...)`
2. `generative_classification_test_step(...)`
3. `_collect_generated_sequence_answers_serial(...)`
4. `_extract_preds_from_logits(...)`

A focused rerun with `CUDA_LAUNCH_BLOCKING=1 --trainer.limit_test_batches=1` narrows the assert further to:

1. `standardize_logits(...)`
2. `torch.index_select(logits, -1, entailment_mapping_indices)`

So the live SAE blocker is specifically that the current serial generative recovery path is still trying to remap logits with token ids that exceed the last-dimension width of the returned SAE generation logits.

### Previously Resolved Execution Failure

Before the current baseline fix, the run failed during evaluation in [src/it_examples/experiments/rte_boolq.py](/home/speediedan/repos/interpretune/src/it_examples/experiments/rte_boolq.py) with:

```text
ValueError: Mismatch in the number of predictions (1) and references (2)
```

That mismatch is now understood well enough to avoid for the baseline path.

The key runtime observation was:

1. generation input batch shape entered as `(2, 283)`
2. generated sequences came back as `(1, 6)`
3. generation logits likewise reflected only one example

So the original evaluation failure was caused by generation collapsing a 2-example batch down to a single returned sequence.

The current working tree now contains local recovery logic in [src/it_examples/experiments/rte_boolq.py](/home/speediedan/repos/interpretune/src/it_examples/experiments/rte_boolq.py) that:

1. first tries batch-preserving generated-sequence extraction
2. falls back to per-example serial generation when the returned generate output collapses the batch
3. only uses the legacy logits path when those recovery paths do not apply

## Interpretation

The current tree is no longer blocked on the original batch-size mismatch, but it still does not reproduce the earlier Gemma 2 IT results.

Current status by stage:

1. baseline executes but lands at `0.4549`, well below the historical `79.1`
2. SAE does not yet run to completion because of a separate CUDA-side evaluation failure
3. Gemma 2 circuit-tracer and Gemma 3 follow-on runs should stay blocked until Gemma 2 baseline and SAE behavior are understood

At minimum, one of these is still true:

1. the current Gemma 2 IT prompt / generation / scoring path no longer matches the historical setup,
2. the new baseline recovery logic restores execution but not the old answer-selection behavior,
3. the SAE path returns logits in a shape or vocabulary space the current example scoring code still mishandles.

## Relation To The Prompt-Parity Work

This failure is downstream of the prompt-parity work, not a contradiction of it.

What is already validated:

1. Gemma chat template parity between dataclass rendering and `apply_chat_template`
2. instruction-tuned Gemma models can be made to prefer `Austin` at the answer boundary when the assistant is constrained to output only the missing city name

What is not yet validated:

1. end-to-end `rte_boolq.py` replication for the current CLI configs
2. whether the older `79.1` / `80.5` numbers still reproduce in this checkout without additional fixes

## Next Debugging Steps

Recommended next steps for the RTE/BoolQ path:

1. Inspect one SAE serial-generation example under `CUDA_LAUNCH_BLOCKING=1` to determine the exact logits shape and vocabulary width before `_extract_preds_from_logits(...)`.
2. Confirm whether the SAE bridge generation path returns full-vocab logits, already mapped answer logits, or some other reduced output.
3. Compare the current Gemma 2 IT baseline prompt / decode path against the setup that produced the historical `79.1` reference, since execution now succeeds but accuracy is far lower.
4. Do not move on to Gemma 2 circuit-tracer or Gemma 3 until Gemma 2 baseline and SAE are both understood well enough to judge parity.
