# Memory-profiling deltas: interpretune against the upstream stack it wraps

Each sampled composition runs twice per measurement: an interpretune arm, and a matched upstream arm that reproduces
the same model, batches, optimizer, precision and step order with plain PyTorch, Hugging Face, Lightning or
finetuning-scheduler calls. The gate is on the difference (interpretune minus upstream), so upstream drift between
releases and between vendors cancels, and what remains is the memory interpretune's orchestration adds.

| Composition | Interpretune arm | Upstream arm |
| --- | --- | --- |
| `core_hf_train`, `core_hf_train_bf16` | core session `training_step` on gpt2 (RTE) | plain torch loop on the same HF model |
| `core_hf_test` | core session `test_step` (generative classification) | HF `generate` with the same kwargs |
| `lightning_train`, `lightning_test` | Lightning composition under `Trainer` | vanilla `LightningModule`, same `Trainer` |
| `core_hf_fts_train` | core loop with `FTSCoreCallback`, 3-phase schedule | plain loop driving the same `FTSController` |
| `tl_train`, `tl_test` | core session with the TransformerLens adapter | native `TransformerBridge.boot_transformers` |
| `sae_lens_train`, `sae_lens_test` | core session with the `sae_lens` adapter, SAE spliced at init | SAELens's `SAETransformerBridge` with the same SAE in `add_sae` |
| `nnsight_train`, `nnsight_test` | core session with the `nnsight` adapter | bare nnsight `LanguageModel` |
| `circuit_tracer_test` | core session with the `circuit_tracer` adapter attributing one prompt per step | bare circuit-tracer `ReplacementModel` on the same model and transcoders |
| `synthetic` | small MLP, optionally with a planted leak or overhead | the same MLP: the harness's own control |

## How a pair is measured

- One subprocess per arm (allocator state, hooks and caches cannot cross arms), with 5 repeats inside it.
- The first steps are warmup (2 by default, longer where an arm is still settling); peaks reset when warmup ends,
  and per phase when a schedule thaws.
- Per repeat: peak and end-of-window allocated bytes, reserved peak (recorded only), and a least-squares leak slope
  per phase that excludes each transition step and the settling step after it. Phased compositions also record
  exact structural counts (frozen parameters with gradients, optimizer state for frozen parameters), asserted
  without a threshold.
- Both arms record each step's input shape, and a pair whose arms consumed different batches fails.

## Bounds

`delta_bounds.yaml` holds, per vendor and composition, each gated metric's frozen baseline (the median delta) and
tolerance, `max(3 x p95(repeat spread), floor)` with floors of 8 MiB and 64 KiB per step. A metric fails only when
its median delta exceeds the baseline by more than the tolerance; shrinking never fails. A composition without an
entry for the running vendor is report-only, and so is every composition when `IT_PROFILING_DELTA_REPORT_ONLY` is
set (release and tag builds).

Re-freeze only from fresh measurements on both vendors:

```bash
IT_RUN_PROFILING_TESTS=2 IT_PROFILING_DELTA_DIR=/path/to/nvidia python -m pytest tests/profiling_delta/test_delta_pairs.py
# the same on the AMD host into /path/to/amd, then
python -m tests.profiling_delta.freeze --artifacts /path/to/nvidia /path/to/amd --out tests/profiling_delta/delta_bounds.yaml
```

To attribute a residual delta, set `IT_PROFILING_DELTA_INVENTORY=1`: each arm then records its live device tensors
at the end of the first repeat's window, so the two inventories can be diffed. A residual that the inventories do
not account for is allocator memory that Python cannot reach (workspaces, for one).
