# Tuning: toward sparse attribution graphs (RTE)

[Tuning](tuning.md) describes how every composition declares and attaches its Fine-Tuning Scheduler view. This page
describes a research example built on it: the `speediedan/rte` component's `rte_sparse` variant, which tunes
gemma-3-1b-it toward sparse attribution graphs on RTE as one phased Fine-Tuning Scheduler run on the core loop. It is
research-class (a reserved GPU window rather than CI), and this page states what it has shown and what it has not.

```bash
interpretune --run_command train --config hub://speediedan/rte/runs/rte_sparse.gemma3.core.yaml
```

`rte_sparse.gpt2.core` runs the same schedule on a CPU with random transcoders, for the tests.

## The objective

Each RTE item is asked as a Yes/No question, and the target is the answer gap $y$, the Yes logit minus the No logit at
the last position. Graphs are circuit-tracer attribution graphs over the published Gemma Scope 2 transcoders, which the
composition holds as `instrument` tensors: never thawed or checkpointed.

The penalty acts on a per-item first-order attribution $s_i = a_i \partial y / \partial a_i$ over the transcoder
features of layers 17 to 25 at the answer position. Its sparsity term is the ratio
$r = \lVert s \rVert_1 / \lVert s \rVert_2$, which is 1 when one feature carries the whole attribution and grows as
it spreads, normalized by its value
$r_0$ on the untuned model. A second term penalizes the error share $e$ of the same attribution, so that moving
computation into the transcoders' reconstruction error is not a way to satisfy the first. With the task loss as
cross-entropy on the gap at temperature 4:

$$
\mathcal{L} = \mathrm{CE}(y / 4) + 0.3 \frac{r}{r_0} + 3 e
$$

Every phase is selected and judged on `val_task`, the task term alone, so the metric is the same across phases.

## The schedule

| phase | trains | constraint | starts from |
| --- | --- | --- | --- |
| 0, sparsity | final norm and blocks 14 to 25 | penalty | the untuned model |
| 1, recovery | final norm and blocks 22 to 25 (blocks 14 to 21 refrozen) | penalty, simplified graph enforced | phase 0's best |
| 2, recovery | final norm and blocks 14 to 25 | penalty, simplified graph enforced | phase 1's best (`restore: best:1`) |

At each boundary the ended phase's best checkpoint is restored and attribution graphs are built in-process:

- **At the sparsity boundary** the graphs of 100 items fit the simplified graph: the features default pruning keeps on
  correctly answered items, pooled by template region. Every other feature is held at its mean in every forward pass
  of both recovery phases, attention and norms live. The means and the mask are run-fitted state, saved with the
  checkpoints.
- **The gate** (`ChainGraphGate`) compares the median error share and replacement score of 40 gate items with a pinned
  untuned reference: the error share may rise by at most 5 points and the replacement score fall by at most 0.15
  (relaxed from 0.10 for this exploration). It judges phase 0 and the final phase. Phase 1 is graphed and recorded but
  cannot veto (`record_only_depths: [1]`), so the second step always runs. A veto ends the run on the previous phase's
  best.

Each recovery phase starts a fresh optimizer, and the plateau transition judges each phase against its own best
(`reset_best_on_phase`), so a recovery phase that starts below phase 0's best is not ended for that.

## What it has shown

Three seeds on one 24 GB GPU, about 30 minutes each (results on
[interpretune#899](https://github.com/speediedan/interpretune/issues/899)):

- **The chain's order holds on every seed.** Each stage continues from the one before, as the separate-process chain
  it ports did.
- **Sparsity falls only while blocks 14 to 21 train, and it falls on every seed.** The validation attribution ratio
  (untuned 5.26) drops by about 0.1 to 0.4 per epoch in phases 0 and 2, and barely moves in phase 1. The sparsest
  selected checkpoint reached 1.89 with 34 active features, at an error share 4.9 points above the reference.
- **Every seed passes the gate, one only under the relaxed bound,** and on the two seeds where phase 0's best was kept,
  no recovery checkpoint beat it on held-out accuracy.

## Limitations

- **The stopping and selection rules decide more than the schedule does.** `val_task` on 190 validation items swings by
  up to 0.2 between epochs at nearly constant accuracy, and a plateau of patience 1 ends a phase on one such swing, after
  which the restore discards the later, sparser weights. One seed ended two of its phases after two epochs and finished
  on its least sparse checkpoint. Final selection on `val_task` alone does not pursue sparsity.
- **Whether phasing helps is untested:** no run has had a step-matched single-phase control, and three seeds are too
  few to set the gate's bounds, which sit close to the recovery results at the current circuit-tracer pin.
- **The dictionary is fixed.** With the published transcoders held, part of any sparsity gain can be the decision
  moving off the dictionary, into error terms or attention routing, which a frozen-pattern graph does not see.
- **One model, one task, and graphs built at one circuit-tracer pin.** The reference is graphed at circuit-tracer
  `bd6bc784`; graphs from another pin are not comparable with it.

## Open questions and follow-up work

- [interpretune#988](https://github.com/speediedan/interpretune/issues/988): step-matched controls, a stopping rule
  sized to its metric, sparsity-aware selection and per-epoch instruments. The rest of this list builds on it.
- [interpretune#989](https://github.com/speediedan/interpretune/issues/989): how much of the decision moves into
  attention routing under the penalty, with attention-aware attribution.
- [interpretune#990](https://github.com/speediedan/interpretune/issues/990): stages by component, unit set, dictionary
  order and penalty strength, rather than by layer band.
- [interpretune#991](https://github.com/speediedan/interpretune/issues/991): graphs against J++ lens readouts at each
  layer.
- [interpretune#992](https://github.com/speediedan/interpretune/issues/992): training informed by the premise and
  hypothesis spans that agree or contradict.
- [interpretune#993](https://github.com/speediedan/interpretune/issues/993): dictionaries beyond fixed point
  transcoders.
