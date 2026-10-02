# Choosing Adapters

Interpretune composes a session from **adapters** — named capabilities such as model execution,
training orchestration, or feature attribution — rather than from a fixed model class. You name an
adapter context like `(core, transformer_lens, circuit_tracer)` and the registry produces the module
composition for it. This page says which adapter does what, what each one costs, and which
combinations are registered. It is linked from the {doc}`landing page <index>` and the
{doc}`core concepts <concepts>`; backend-specific detail lives with each adapter's own guide.

## The adapter set

| Adapter | What it is for | What it costs |
|---|---|---|
| `core` | Framework-agnostic execution: the default runner, hooks, and mixins with no training framework. | Nothing beyond the base install. Runs on CPU. |
| `lightning` | Full training orchestration through PyTorch Lightning's `Trainer` (callbacks, checkpointing, logging backends). | The `lightning` extra (`finetuning-scheduler`, `peft`). |
| `transformer_lens` | Model execution through TransformerLens (`TransformerBridge`), the substrate most analysis backends run on. | The `transformer-lens` package. A GPU for anything beyond tiny models. |
| `sae_lens` | Loading and running sparse autoencoders / transcoders, pretrained or custom. | The `sae-lens` package. |
| `circuit_tracer` | Attribution graphs and feature interventions via replacement models. | The `circuit-tracer` package (currently a git dependency). Needs an execution backend below it. |
| `nnsight` | Model execution through NNsight tracing, an alternative substrate to TransformerLens. | The `nnsight` package. |

`circuit_tracer` never stands alone: it needs `transformer_lens` or `nnsight` underneath to execute
the model it attributes. `sae_lens` composes with either execution backend too. `lightning`
replaces the `core` runner rather than combining with it — every registered combination contains
exactly one of the two.

## Decision path

- **Fine-tune or evaluate with hooks, no training framework** — `(core, ...)` with the execution
  backend matching your model path.
- **Train with callbacks, checkpointing, and logging integrations** — swap `core` for `lightning`.
- **Run a pretrained (or custom) SAE on activations** — add `sae_lens` to the execution backend.
- **Attribute predictions to features, or intervene on them** — add `circuit_tracer` on top of
  `transformer_lens` or `nnsight`.
- **Trace-based interventions with the HF model kept native** — prefer the `nnsight` substrate;
  for the TransformerLens hook ecosystem, prefer `transformer_lens`.

## Registered module combinations

These are the module compositions the registry produces today, in canonical (value-sorted) order.
A drift test pins this list against `CompositionRegistry`, so a newly registered combination fails
the suite until it is documented here:

- `(circuit_tracer, core)`
- `(circuit_tracer, core, nnsight)`
- `(circuit_tracer, core, nnsight, sae_lens)`
- `(circuit_tracer, core, transformer_lens)`
- `(circuit_tracer, lightning)`
- `(circuit_tracer, lightning, nnsight)`
- `(circuit_tracer, lightning, transformer_lens)`
- `(core,)`
- `(core, nnsight)`
- `(core, nnsight, sae_lens)`
- `(core, sae_lens)`
- `(core, sae_lens, transformer_lens)`
- `(core, transformer_lens)`
- `(lightning,)`
- `(lightning, nnsight)`
- `(lightning, nnsight, sae_lens)`
- `(lightning, sae_lens)`
- `(lightning, sae_lens, transformer_lens)`
- `(lightning, transformer_lens)`

Requesting anything else raises a miss naming the available combinations — and, when the miss is
really an absent backend rather than a nonexistent combination, names the missing extra too.

## Backend detail lives with the backends

Execution-backend specifics — model support, capability coverage, known refusals — are documented
once, next to the code they describe, and linked here rather than duplicated:

- {doc}`Circuit-tracer backend support <usage/circuit_tracer_backend_support>`, including the
  Backend Compatibility Matrix for the circuit-tracer replacement models.
- {doc}`Framework-level adapters <usage/framework_level_adapters>` for the core/lightning
  execution contexts.
