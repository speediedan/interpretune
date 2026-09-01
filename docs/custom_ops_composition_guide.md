# Custom Ops Composition Guide

**Status:** Draft guidance
**Audience:** contributors creating custom analysis ops, custom op collections, or hub-shareable analysis workflows

## Purpose

Interpretune's analysis system is most valuable when custom ops behave like the bundled ops
(the op families shipped with the package under `src/interpretune/analysis/ops/bundled/`):

- composable
- schema-aware
- backend-aware without being backend-entangled
- serializable through `AnalysisStore` where appropriate

This guide documents the current best practices for writing custom ops that compose cleanly across existing model and analysis backends.

## Core Design Rule

Write ops against the Interpretune protocol surface, not against one concrete backend unless the op is explicitly backend-specific.

That means:

- prefer generic batch and module inputs
- use backend capability validation
- route package-specific graph or intervention behavior through an analysis backend seam when one exists

## Anatomy of an op

An op should define:

- a name and description
- input schema
- output schema
- optional required capabilities
- optional `required_ops`: upstream ops whose declared fields your op consumes. Compilation merges
  their schemas into yours (schema inheritance, not execution), and it is the declared record of
  op-to-op dependency that composition and reviewers rely on.
- optional `importable_params`: callables injected into the implementation at instantiation time
  (e.g. a `logit_diff_fn`). References may target modules shipped in your own op collection or the
  sanctioned `interpretune.analysis.optools` namespace.
- optional `op_state`: declared cross-batch state (see "Cross-batch state" below). Declaring it is
  what authorizes an op to accumulate across batches.
- optional behavioral traits, which let your op ask the framework for what it needs instead of the
  framework special-casing op names: `requires_grad` (run the analysis loop with grad enabled),
  `uses_default_hooks` (install the default activation-cache forward/backward hooks), and
  `per_latent_preds` (predictions are emitted per latent model and must be joined across them before
  scoring). All default to `false`, and hub and local ops declare them exactly as bundled ops do.
- one implementation function

Relevant code:

- `src/interpretune/analysis/ops/base.py`
- `src/interpretune/analysis/ops/dispatcher.py`
- `src/interpretune/analysis/ops/bundled/` (the bundled op families, each a worked example of the
  op-collection shape: one YAML plus one implementation module)
- `src/interpretune/analysis/optools.py` (the op-authoring toolkit)

## What an op implementation may import

An op implementation module must be self-contained modulo the sanctioned surfaces below. This is
the same contract for every op source: bundled families, local collections, and hub repos.
Interpretune is installed wherever an op runs, so these imports are a declared, supported contract
rather than a reach into package internals:

- `interpretune.analysis.optools`: the op-authoring toolkit (tensor/logits utilities,
  tokenizer/embedding resolution, tokenization helpers, scoped-input conveniences)
- `interpretune.analysis.backends`: the capability seam for backend-specific behavior. Its
  `protocols`, `capabilities`, `interventions` and `feature_selection` modules are re-exported from
  the package, so **import from `interpretune.analysis.backends`** even though the API reference
  documents each module separately (the pages tell you where a name lives; the package is the stable
  import path). Importing a submodule directly is not rejected, it is just a path we may move.
  **One exception is rejected**: `interpretune.adapters` holds each adapter's concrete
  per-library backends (TransformerLens, nnsight, circuit-tracer), and an op may not import them.
  Ask for a capability, or take an `AnalysisBackend` through the protocols, so your op works on every
  backend that satisfies it rather than one. CI enforces this
  (`tests/core/test_bundled_op_publishability.py`). The modules are readable as worked examples if you
  are writing a backend of your own, which is a different job from writing an op.
- `interpretune.analysis.inputs`: the scoped-input execution contract (`AnalysisInputs`)
- the public op classes in `interpretune.analysis.ops.base` (`AnalysisBatch`, `OpSchema`,
  `ColCfg`, `get_batch_input`)
- `interpretune.protocol` types
- the bare public `interpretune` surface, function-level only, to invoke ops your YAML declares in
  `required_ops`
- your own collection's modules, plus the standard library and the op-level third-party
  dependencies (torch, transformers, jaxtyping, transformer_lens)

Anything else inside interpretune is private and may change without notice. The bundled families
are held to exactly this rule by CI (`tests/core/test_bundled_op_publishability.py`), so no bundled
family depends on interpretune internals that a hub op collection could not import.

Two mechanical gaps remain before a bundled family is literally publishable as-is, and both are
tracked in [#266](https://github.com/speediedan/interpretune/issues/266): implementation paths must
be rewritten to the hub loader's `module.function` form, and `required_ops` that cross family
boundaries need a declaration mechanism (today an unresolvable `required_ops` entry drops the op
with a warning rather than failing loudly).

## Preferred Notebook And Script Surface

For user-facing notebooks, examples, and ad hoc research scripts, prefer the top-level op wrappers on `interpretune` instead of reaching into the dispatcher directly.

Preferred pattern:

```python
import interpretune as it
import interpretune.analysis  # registers top-level op wrappers

analysis_batch = it.AnalysisBatch(prompts=[prompt])
analysis_batch = it.model_fwd_w_cache_latent_models(module=module, analysis_batch=analysis_batch, batch=batch, batch_idx=0)
analysis_batch = it.logit_diffs_cache(module=module, analysis_batch=analysis_batch, batch=batch, batch_idx=0)
```

Avoid this in notebook or experiment code unless you are extending dispatcher internals themselves:

```python
op = DISPATCHER.get_op("logit_diffs_cache")
analysis_batch = op(module, analysis_batch, batch, batch_idx)
```

Why this is preferred:

- it matches the public API surface we expect users to learn
- it keeps notebook code aligned with in-tree example usage
- it avoids local dispatcher plumbing in research harnesses that are not actually implementing new dispatch behavior

## Best Practices

### 1. Make schemas explicit

Your input and output schemas are part of the contract.

Do:

- declare every required upstream field explicitly
- prefer Arrow-native typed columns when practical
- use structured serialization patterns for richer objects that cannot be represented naturally as typed columns

Avoid:

- hiding large structured outputs in JSON strings unless there is no better short-term option
- marking a field `required` when your implementation does not consume it (see below)

#### `required` means "must be present", not "my source code names it"

Only `required: true` fields are enforced, by `AnalysisOp._validate_input_schema` before the
implementation runs. Fields merged in from a `required_ops` entry are compiled `required: false` and
skipped -- see NOTE [Inherited Inputs Are Not Obligations].

Enforcement checks **presence on the batch**, never that your code textually references the field.
That distinction matters, because implementations reach batch fields six different ways and only the
first is visible to a reader skimming the function:

```
analysis_batch.field                                   direct attribute
getattr(analysis_batch, key)                           dynamic key
resolve_aggregate_input(module, analysis_batch, "...")  scoped by string
backend.hydrate_graph_from_batch(analysis_batch)       whole batch to a backend
get_batch_input(batch)                                 helper reads the BatchEncoding
get_loss_preds_diffs(module, analysis_batch, ...)      helper reads the analysis batch
```

The last two are the common ones and the easiest to miss: `get_loss_preds_diffs` alone accounts for
every `label_ids` / `orig_labels` declaration in the bundled ops, and `get_batch_input` for most
`input` declarations. **Do not conclude a declaration is spurious because the field name does not
appear in the function body** -- follow anything the implementation hands the batch to.

A detector that scanned only for direct attribute access once reported 22 of ~40 bundled ops as
over-declaring; a full audit of all 41 `required: true` declarations found exactly one
(#299). Declare what your op needs present, and let the enforcement be about presence.

### 2. Keep implementation logic small and composable

An op should do one coherent piece of work.

Prefer:

- several small ops composed into a pipeline

Over:

- one large op that mixes caching, aggregation, intervention, logging, and formatting

### 2.1 Let `AnalysisOp` own scoped batch context by default

When an op runs through the normal dispatcher / `AnalysisOp` surface, scoped
`AnalysisBatch` lookup is already bound for that execution. That means op
implementations should prefer:

- `analysis_batch.get(...)`
- `analysis_batch.require(...)`
- shared execution helpers such as `execute_analysis_op(...)`

Avoid building new per-op context decorators as the default pattern.

(The former `with_analysis_batch_context(...)` compatibility shim has been removed — direct
`*_impl(...)` calls that intentionally bypass `AnalysisOp` no longer need a context wrapper.)

### 3. Use capability checks instead of backend-name checks

Prefer:

- required capabilities
- model backend interfaces
- analysis backend interfaces

Avoid:

- branching on adapter names or concrete class names inside generic ops

### 4. Keep backend-specific logic behind the backend seam

If an op needs package-specific graph or intervention behavior, prefer extending the analysis backend interface instead of importing a specific backend package into a generic op.

This is especially important for:

- circuit-tracer graph hydration and decomposition
- intervention spec construction
- package-specific prompt and target conversion

### 5. Design for persistence when the output has reuse value

Ask:

- should this output be reusable across sessions?
- should it be shareable through a hub workflow?
- should it be inspectable as dataset columns?

If the answer is yes, design the output schema accordingly.

## Composition Patterns

### Pattern 1: Producer op then consumer op

Example:

- produce `concept_direction`
- consume it in `compute_attribution_graph`

This is the normal composition pattern and should remain the default.

### Pattern 2: Composite op for stable workflows

If a sequence is reused often, define a composite op rather than duplicating notebook orchestration.

### Pattern 3: Aggregate workflow feeding later ops

If an analysis result is derived across multiple batches, declare `op_state` and accumulate through
it (see "Cross-batch state" below). Persist the *result* through `AnalysisStore` when it has reuse
value; do not thread accumulator state through notebook-local variables or write it onto the store
as ad-hoc attributes.

## Cross-Backend Guidance

### Model-level backends

Examples:

- TransformerLens
- NNsight

When your op depends on execution features such as hooks or gradients, rely on the model backend capability surface.

### Analysis-level backends

Examples:

- circuit-tracer

When your op depends on richer analysis object semantics, rely on the analysis backend surface.

## Testing Guidance

### Minimum expected tests

- schema validation
- required capability validation
- correct behavior on at least one supported backend path
- persistence or serialization behavior if the op produces reusable artifacts

### Develop against `IT_STRICT_OP_LOAD=1`

Op loading is fail-soft by design: a definition that fails to compile, an `importable_params`
reference that will not resolve, or a malformed `op_state` block is reported as a warning and the op
is dropped, so one bad collection cannot take down a session. While you are *developing* a
collection, that is the wrong default: a dropped op looks exactly like one you never wrote.

```bash
IT_STRICT_OP_LOAD=1 python -m pytest tests/my_op_tests.py
```

turns those paths into errors. Interpretune's own CI asserts every bundled op compiles and
instantiates under strict loading; the same check is worth having for your collection.

### Prefer focused tests over overly broad notebook-only validation

Notebook tests are useful, but they should not be the only correctness signal.

Good test targets include:

- `tests/core/test_analysis_ops_base.py`
- `tests/core/test_analysis_ops_dispatcher.py`
- `tests/core/test_analysis_ops_definitions.py`
- `tests/core/test_cross_backend_compat.py`
- `tests/core/test_bundled_op_publishability.py` (the sanctioned-imports contract the bundled
  families are held to; a useful template for linting your own collection)

### Add round-trip tests when serialization matters

If an op output is meant to survive storage and reload, add a round-trip test through `AnalysisStore`.

## Hub-Oriented Guidance

The long-term direction is for ops, stores, adapters, and configured modules to be more easily shareable.

Design custom ops so they are compatible with that future:

- keep config explicit
- avoid hidden runtime dependencies on notebook globals
- avoid implicit local-path assumptions
- prefer stable schema contracts

### Publishing an op collection

A well-formed ops repo carries three things: the op-definitions YAML, the Python module its
`implementation` references point at, and an `it_component.yaml` declaring which YAMLs are op
definitions.

```yaml
# it_component.yaml
it_schema_version: 1
kinds: [ops]
ops:
  files:
    - my_ops.yaml
```

Discovery is manifest-routed, so anything else the repo carries (a card, a config sample, a notebook)
is never fed to the op compiler. Publish with `interpretune.hub.publish.publish_op_collection`, or generate a
collection from an in-tree op family with `scripts/publish_op_collection.py`, which also handles the
one transformation publishing requires: bundled YAMLs address implementations by installed package
path, while the hub loader resolves a repo-relative `<module>.<function>` pair. A family published
verbatim is a repo whose every op fails to import.

### Versioning your op collection

Declare the collection's identity in a header at the top of its op YAML:

```yaml
collection:
  name: my_ops
  version: 0.3.0
  requires:
    interpretune: ">=0.1.0.dev0"    # optional; one window, checked at load
```

Four things to internalize before you pick a version or a window:

1. **The version versions the CONTRACT SET, not your package.** It describes the names, schemas and
   traits your ops present to callers. Bump it when a caller would have to change; leave it alone for
   an internal refactor that keeps every schema and trait identical.
2. **There is no solver.** One window per collection against the installed interpretune, and no
   cross-collection dependency resolution. An incompatible collection is skipped whole with a warning
   (or raises under `IT_STRICT_OP_LOAD=1`), because a partial load presents half a contract set.
3. **`>=0.1` does not mean what you want.** `setuptools_scm` produces `0.1.0.devN+g<sha>` between
   tags, and PEP 440 sorts a dev release *before* its release, so a bare floor silently skips your
   whole collection for anyone on a source install. Write `>=0.1.0.dev0`.
4. **The header lives in the op YAML, not in `it_component.yaml`.** It travels with the definitions,
   so bundled families and local collections — which have no manifest at all — declare it the same
   way, and there is no second place for it to disagree with itself.

Consumers can see exactly what they got:

```python
import interpretune as it

print(it.hub.op_info("my_op"))   # provenance, collection, version, cached revision, alternatives
```

### Naming an op-collection repo

Name ops repos **lowercase with underscores** (`concept_direction_ops`, not
`concept-direction-ops`), which differs from the hyphenated convention common on the Hub. The reason
is specific to this kind: an ops repo name is not just a label, it is an **identifier prefix**. It
prefixes every op's namespaced name (`speediedan.concept_direction_ops.concept_direction`) and feeds
the module path a collection's implementations resolve through.

Interpretune normalizes op names — case-folding, `-` to `_`, `/` to `.` — so a hyphenated repo does
work. But then the name you typed and the name the op answers to differ, and two repos whose names
differ only by that punctuation collide on one normalized namespace. Choosing the normalized form up
front makes the name you read the name you address.

## Current Open Gaps

### Prefer AnalysisBatch-scoped lookup for mixed batch, run, and store inputs

The current IG-7 execution path binds scoped input resolution directly onto `AnalysisBatch`.

Prefer:

- using `analysis_batch.field_name` as the primary access pattern for declared or required inputs
- using `analysis_batch.get("field_name")` when the value is genuinely optional
- using `analysis_batch.require("field_name")` when the value is mandatory
- overriding `scopes=` only when custom precedence is genuinely needed
- treating notebook variables and aggregate artifacts as `run` scope instead of relying on list indexing heuristics

Example:

```python
group_a = list(analysis_batch.concept_group_a)
target_ids = analysis_batch.require("logit_target_ids")
custom_value = analysis_batch.get("foo", scopes=("analysis_batch", "run", "store"))
```

Attribute-style access is execution-time resolution only. It uses the currently bound scope precedence:

- `analysis_batch`
- `batch`
- `run`
- `row`
- `store`

If the active op input schema declares a default value, attribute access will also use that default before raising.

Avoid:

- adding new direct `_value_for_batch(...)` style logic inside ops
- manually constructing resolver handles in op implementations unless you are extending framework internals
- assuming that every list-like value coming from an input store is row-scoped

The former `get_analysis_value(...)` / `get_input_store_value(...)` transition helpers have been
removed; op code resolves values through the `AnalysisBatch` access surface. For whole-column
aggregate inputs, `interpretune.analysis.optools.resolve_aggregate_input(...)` is the sanctioned
bridge. Note it solves a different problem from `op_state`: it *reads* an already-materialized
column, whereas `op_state` is where an op *accumulates* across batches.

### Serialization and formatter boundary

This lookup API is an execution-time convenience only.

It does not change how `AnalysisStore` persists data or how the custom datasets formatter materializes rows, batches, or columns. The existing serialization path still lives in:

- `src/interpretune/analysis/core.py`
- `src/interpretune/analysis/formatters.py`
- `src/interpretune/analysis/ops/auto_columns.py`

That means:

- `analysis_batch.field_name`, `analysis_batch.get(...)`, and `analysis_batch.require(...)` resolve against already-bound row, batch, run, and store objects
- `AnalysisStore` still owns dataset-backed column access, `set_format(...)`, and custom tensorization behavior
- op authors should treat scoped lookup as a read layer over already-prepared inputs, not as a new persistence mechanism

In particular, this does not change Hugging Face Dataset semantics:

- string access is still column access on `AnalysisStore`
- integer or slice access is still row or row-range access on the underlying dataset
- `_format_columns(...)` and the Interpretune dataset formatter still control how persisted columns are materialized back into tensors or lists

Keep the conceptual split clear:

- `analysis_batch` means the execution-time resolved input surface for one op call
- `batch` means the dataloader batch argument passed into the op
- `AnalysisStore` means the persisted dataset-backed artifact layer

### Cross-batch state

An op that accumulates across batches declares what it accumulates:

```yaml
my_streaming_op:
  implementation: my_ops.my_streaming_op_impl
  op_state:
    scope: run             # only `run` is supported today
    reset_each_epoch: false  # default: accumulate across epochs
    fields: [running_sum, running_weight]
```

The implementation reads and writes through `analysis_inputs.op_state`:

```python
def my_streaming_op_impl(module, analysis_batch, batch, batch_idx, **kwargs):
    state = kwargs["analysis_inputs"].op_state
    running = state.get("running_sum")            # None until first set
    state.set("running_sum", batch_value if running is None else running + batch_value)
```

What the framework guarantees:

- **A namespace.** Only declared field names are readable or writable; anything else raises, so a
  typo is an error rather than a silently-`None` read.
- **A lifecycle owner.** The container belongs to the `AnalysisCfg`, and the *analysis runner* drives
  it: cleared before the first batch, cleared at epoch boundaries only for ops that set
  `reset_each_epoch: true`, and released at the end of the run. Ops never decide when accumulation
  starts over, so nothing depends on `batch_idx == 0` (which restarts every epoch and therefore
  discards all but the last). Driving `execute_analysis_op` yourself in a loop, those callbacks do
  not fire: a fresh `AnalysisCfg` starts with fresh state, and `cfg.reset_op_state()` /
  `cfg.finalize_op_state()` are there if you reuse one across independent runs.
- **Isolation.** State is keyed per op, so the member ops of a composite each get their own.

Two consequences worth knowing:

- Accumulator state is *not* an input scope. `analysis_batch.get(...)` will not resolve it, and it
  is not persisted; emit converged results through your output schema.
- An op that declares `op_state` needs an owner. Run it through a runner or
  `interpretune.analysis.execution.execute_analysis_op` (both activate an `AnalysisCfg`). With no
  active cfg the framework has nothing to bind, so `analysis_inputs.op_state` is `None` and reaches
  your implementation that way, and it does not invent a container nobody can reset. **Raise on `None`
  rather than silently degrading**; the bundled concept ops do exactly that, and the message names
  what is missing. (Their previous behavior is the argument for it: the writes were swallowed and the
  failure surfaced several frames later as a data error.)

## Practical Rule of Thumb

If a custom op would be difficult to use from both a CLI runner and a notebook with only minor orchestration differences, its abstraction boundary is probably wrong.

Aim for ops that:

- operate on declared inputs
- expose declared outputs
- let the framework decide how those inputs are sourced and how those outputs are persisted
