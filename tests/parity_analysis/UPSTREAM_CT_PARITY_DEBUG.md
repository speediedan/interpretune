# Upstream Circuit-Tracer Parity Debugging

This note documents the manual sanity-check workflow for the semantic concept-direction intervention parity path.

The normal source of truth for regressions remains [tests/core/test_analysis_backend_parity.py](../core/test_analysis_backend_parity.py).
The manual upstream check exists for the narrower case where:

- the Interpretune native baseline and analysis-op baseline start to diverge unexpectedly
- the native baseline still passes, but you want to confirm that the underlying upstream circuit-tracer behavior has not shifted
- you want a quick three-way reference between upstream circuit-tracer, Interpretune native CT, and the Interpretune analysis-op path without hand-editing tests

## Manual Extractor

Use [tests/upstream_parity/extract_upstream_ct_semantic_reference.py](extract_upstream_ct_semantic_reference.py).

Example:

```bash
cd /home/speediedan/repos/interpretune && \
  /mnt/cache/speediedan/.venvs/it_latest/bin/python \
  tests/upstream_parity/extract_upstream_ct_semantic_reference.py \
  --output-json /tmp/ct_semantic_reference.json
```

What it does:

1. Replays the upstream circuit-tracer semantic-intervention logic against CPU-loaded ReplacementModels that are moved to CUDA one backend at a time.
2. Collects the current upstream NNsight and TransformerLens semantic-intervention outputs.
3. Builds the current Interpretune native CT NNsight baseline.
4. Builds the current Interpretune analysis-op baseline.
5. Writes a JSON payload with the raw values plus a markdown reference table.

This script is intentionally manual and should not be added to regular pytest or coverage flows.

## Investigation Findings

The earlier `~0.22` `gap_delta` difference was not caused by the analysis-op path. The native Interpretune path and the analysis-op path already matched exactly; the divergence was upstream-versus-Interpretune only.

The primary cause was dtype drift in the Interpretune test configuration:

- the upstream circuit-tracer reference test loads the ReplacementModel with `dtype=torch.float32`
- the Interpretune `CircuitTracerNNsightGemma2` test config was still inheriting `CircuitTracerConfig.dtype=torch.bfloat16`
- the Interpretune native and analysis-op activation values therefore showed bf16-style quantization steps such as `55.75`, `25.5`, and `16.375`, while the upstream values retained full float32 precision such as `55.7796516418457` and `25.557994842529297`

Evidence that ruled out the other main suspects:

- feature selection was not the issue: the top feature ids matched exactly across upstream, native, and analysis-op paths
- the op framework was not the issue: native and analysis-op matched exactly on pre-gap, post-gap, intervention values, and activation values
- attention implementation was not the issue: circuit-tracer's NNsight ReplacementModel already forces eager attention internally

Secondary factors considered:

- tokenizer configuration: not the dominant source here because token ids and top feature identities were unchanged
- graph-generation defaults such as `max_n_logits`, `desired_logit_prob`, and `offload`: these can affect memory/perf, but they did not explain identical feature identities with shifted activation magnitudes
- rounding in the table itself: not the source of the 7% discrepancy; the raw JSON payload showed the same drift before any markdown rounding

Action taken:

- the Interpretune `CircuitTracerNNsightGemma2` and `LightningCircuitTracerNNsightGemma2` test configs now set `CircuitTracerConfig.dtype=torch.float32` so the native path aligns with the upstream semantic test setup
- the extractor and this note were moved under `tests/upstream_parity/` to keep the manual sanity-check workflow alongside other test-only tooling

Current result after the float32 alignment:

- upstream CT NNsight, the Interpretune native CT path, and the Interpretune analysis-op path now match on `pre_gap`, `post_gap`, `gap_delta`, the top feature ids, and the recorded activation / intervention values for this semantic-intervention case
- the remaining upstream backend spread is only the small upstream NNsight versus upstream TransformerLens difference, with `post_gap_abs_diff=1.9073486328125e-06`

Acceptability guidance:

- native versus analysis-op parity remains the hard regression gate and should stay effectively exact
- upstream versus Interpretune should be kept as close as practical, but small residual drift is acceptable when it does not change the selected features or the sign of the intervention effect
- if upstream drift returns after the float32 alignment, investigate model init kwargs, tokenizer behavior, or upstream package changes before relaxing any Interpretune-native tolerances

## Current Reference Snapshot

Updated from `/tmp/ct_semantic_reference_after_fp32.json` after the float32 alignment change.

| Metric | Upstream CT NNsight | Interpretune Native CT | Interpretune Analysis Op | Expected Relation / Tolerance |
|---|---:|---:|---:|---|
| `pre_gap` | `2.573406219482422` | `2.573406219482422` | `2.573406219482422` | native/op should match at `rtol=1e-6`, `atol=1e-6` |
| `post_gap` | `5.546785354614258` | `5.546785354614258` | `5.546785354614258` | all three should widen the Austin-vs-Dallas gap |
| `gap_delta` | `2.973379135131836` | `2.973379135131836` | `2.973379135131836` | all three should remain positive |
| `top_feature_ids[:3]` | `[(21, 10, 5943), (24, 10, 6394), (23, 10, 12237)]` | `[(21, 10, 5943), (24, 10, 6394), (23, 10, 12237)]` | `[(21, 10, 5943), (24, 10, 6394), (23, 10, 12237)]` | native/op exact equality; upstream should remain a close sanity match |
| `concept_direction_cosine(native, op)` | `n/a` | `1.000000` | `1.000000` | must stay `> 0.999` |

Additional upstream sanity points to track:

- upstream CT NNsight vs upstream TransformerLens `post_gap` absolute difference
- Interpretune native vs analysis-op `pre_gap` absolute difference
- Interpretune native vs analysis-op `post_gap` absolute difference
- Interpretune native vs analysis-op max activation-value absolute difference
- Interpretune native vs analysis-op max intervention-value absolute difference

Latest values:

- upstream CT NNsight vs upstream TransformerLens `post_gap` absolute difference: `1.9073486328125e-06`
- Interpretune native vs analysis-op `pre_gap` absolute difference: `0.0`
- Interpretune native vs analysis-op `post_gap` absolute difference: `0.0`
- Interpretune native vs analysis-op max activation-value absolute difference: `0.0`
- Interpretune native vs analysis-op max intervention-value absolute difference: `0.0`

## When To Use This

Use the manual extractor when:

- `test_analysis_backend_parity_semantic_intervention_nnsight` starts failing and the failure suggests native vs op drift
- the native CT baseline still looks plausible but you need to verify that upstream circuit-tracer still produces the same semantic-intervention profile
- you are debugging an upstream package bump in `circuit-tracer`, `nnsight`, or `TransformerLens`

Do not use it as a substitute for the regular parity test. The normal test remains the faster and more maintainable regression gate.
