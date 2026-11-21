# Interpretune Intervention APIs

## What an intervention is

An **intervention** applies an **op** to a **target** at a **site** during a pass: to the forward value of an
activation, or to its gradient. Causal-abstraction interventions (hard, soft and interchange), activation patching,
ablation, steering, and trainable interventions such as distributed alignment search and representation fine-tuning
are all interventions on a model's internal variables.

- **Ops** are names in venngram's op registry (`venngram.ops`), and that registry is federated: it starts from the
  core ops (`replace`, `add`, `scale`, `project`, `reject`, `patch`, `clamp` and `replace_with_error`), but you are
  not limited to them. Define your own op, register it in code, or publish it in a Hub op collection for others to
  use, and it is applied exactly as a core op is ([intervention ops](#intervention-ops)). Sharing ops this way is how
  one experiment's technique becomes the next one's starting point. Each core op's meaning is defined once, below.
- **Targets** are the whole activation, a set of its units, a latent model's features (venngram's `targets`), or a
  parameter tensor or its units ([parameter targets](#parameter-targets)). A subset of
  an attribution graph, distilled in one context and applied in another, is a planned target too
  ([#966](https://github.com/speediedan/interpretune/issues/966),
  [venngram#78](https://github.com/speediedan/venngram/issues/78)).
- **Value or gradient**: an address with `of: grad` names the gradient with respect to what it selects, so an op on a
  gradient target is a backward-pass intervention, applied by `model_bwd_intervention`
  ([gradient interventions](#gradient-interventions)). Paths between sites (stopping or routing the gradient from one
  set to another) are [venngram#67](https://github.com/speediedan/venngram/issues/67).

Kinds of intervention differ in the op and in **where the op's operand comes from**:

| operand source | examples | role of its tensors |
| --- | --- | --- |
| a constant or a dose | zero ablation, a fixed steering dose, a per-unit gain | none |
| computed from the model or the store | a concept direction, a J-lens patch pair, another run's activation (activation patching) | none |
| trainable parameters | a learned steering vector (`add`), a learned per-unit gain (`scale`) | `intervention:<name>` |
| a latent model's output | an SAE or transcoder spliced in (`replace` or `replace_with_error` with the reconstruction) | `latent:<name>` |

The first two rows are applied by the analysis ops on this page. The last two carry parameters a schedule can tune, so
they are attached through the tuning seams ([trainable interventions](tuning_interventions.md),
[latent models](tuning_latent_models.md)) and take the roles shown.

**An intervention's target is an activation or a parameter.** On an activation it runs in a hook, on every pass, and
may be scoped to some tokens. On a parameter (such as a weight patch, e.g. scaling or zeroing a column of `mlp.down_proj`)
it is materialized: applied to the tensor, so it acts on every token and persists into exports. Trained, a parameter
target's operand is an adapter (a LoRA is a trainable low-rank `add` on a weight, the `adapter` role). The ops,
addresses and roles are the same for both targets ([parameter targets](#parameter-targets)).

### Parameter targets

`interpretune.tuning.weights` materializes ops on parameters
([#960](https://github.com/speediedan/interpretune/issues/960)):

- `apply_weight_patch(module, patch)` applies a venngram weight patch (`kind: weight_patch`: a low-rank, sparse or dense
  delta, `add` or `replace`) to the one tensor its `target` addresses, after checking that the tensor hashes to the
  patch's `base_sha256`, so a patch never lands on a base it was not made for.
- `apply_recipe_weights(module, recipe)` applies a recipe's `weights` section (venngram recipe schema 3): `scale`,
  `add` and `replace` on a tensor or on its units, and `reject` and `project` of a direction on its output side. A
  recipe applies its weights before its stages, so `recipe_interventions` refuses a recipe with weights until they have
  been applied (`weights_applied=True`).
- `attach_unit_adapter(module, target, units, name=...)` tunes only some units of a tensor: a trainable delta over
  the selected units' slices along the tensor's unit axis, $W = W_0 + \Delta$, zero at first. Its tensor takes the
  role `adapter:<name>`, so a schedule thaws it by role (`role:adapter`), while the optimizer, a checkpoint and an
  export see an ordinary tensor. A gradient mask over the base weight would not do this: weight decay and optimizer
  state would still move the other units. The target may also be a site, `point:mlp.out.hook_in & layers:12`: neuron
  $n$ of the MLP's hidden layer is column $n$ of the down projection and row $n$ (and bias entry $n$) of the up and
  gate projections, so the adapter is placed on each of those slices. The units are the `units` argument, never part
  of the target's address ([#868](https://github.com/speediedan/interpretune/issues/868)).
- `weight_form(module, recipe)` materializes a per-neuron recipe (unit `scale` and `add` at `mlp.out`'s input, at
  token scope `all`) or an output rejection (a whole-activation `reject` or `project` at `mlp.out`'s output) as the
  weight edits it equals, below, and `check_weight_equivalence` compares the two forms of an
  edit on a batch.

Addresses resolve through the module's tunable view, so a target is written as a schedule's is
(`component:mlp.out & layers:13 & leaf:weight`). Addressing a tensor's **units** needs its unit axis, which venngram
measures per component map row (its `layout`, component map schema 5): with a view whose map records none, a unit
target is refused rather than guessed.

Every edit is reversible by default. It is a `torch.nn.utils.parametrize` parametrization, so the base tensor is kept,
the gradient reaches the edit's operand, and the returned handle's `remove()` restores the base exactly. With
`reversible=False` the edited value is written into the tensor instead, for an export. An edit's operand takes a
role, as an intervention's does: `adapter:<name>` when it is trained, `fitted` when it is computed in the run, and
`artifact` when it is loaded from a published patch (the default). Every interpretune module declares these roles
through `tunable_roles()`, so a schedule tunes a trainable patch by role.

At token scope `all`, three weight edits equal an activation op, because a matrix's input and output are activation
sites. For $y = W x + b$:

| weight-space edit | activation counterpart |
| --- | --- |
| scale column $j$ of $W$ by $k$ | `scale` unit $j$ of $x$ by $k$ |
| $b' = b + d W_{:,j}$ | `add` dose $d$ to unit $j$ of $x$ |
| $W' = (I - \alpha u u^{\top}) W$ and $b' = (I - \alpha u u^{\top}) b$ | `reject` $u$ with fraction $\alpha$ from $W x + b$ |
| $W' = W - \beta u v^{\top}$ | none at one site: $\Delta y = -\beta u (v^{\top} x)$ reads the input and writes the output |

Ops on one input compose in order, so a later `scale` scales an earlier `add` too: with the ops' combined effect
$x \mapsto s \odot x + a$, the weight form is $W \mathrm{diag}(s)$ with the bias patch $b + W a$, taken from the
unedited $W$. A whole-activation `reject` or `project` at `mlp.out`'s output (its stage's site is
`blocks.<layer>.mlp.out.hook_out`) acts after them, as in the forward, so it edits the output side of the edited weight
and bias alike; a span of several vectors takes the projector $V V^{+}$ in place of $u u^{\top}$. A token-scoped op has
no weight form, and a general low-rank or dense patch needs a parameter target.
venngram's parameter op vectors state these equivalences, and interpretune's edits are tested against them.

The `weight_equivalence_check` analysis op checks the equivalence on a model, batch by batch: it applies a recipe's
weight form and runs the forward, takes the form back, then runs the recipe's activation form through the model
backend's own intervention path (`fwd_w_intervention`), and records the greatest absolute logit difference
(`we_max_abs_diff`), the greatest logit magnitude (`we_max_abs`) and the verdict (`we_equivalent`). A recipe with no
exact weight form is refused with the reason before any forward runs. Unit ops need the unit axis of `mlp.out`'s weight
(an output span needs only its output side), so the weight form resolves through the module's view when that view places
units, else through venngram's registry map for the model the module carries, and a model whose map records no `layout`
is refused where the recipe has unit ops. A `TransformerBridge` renames the modules of the model it wraps but keeps
the modules themselves, so its view resolves the wrapped model's map through them: each tensor takes the layer,
component and unit axis of the same tensor in the wrapped model, and the op runs on a TransformerLens composition as on
the others ([#1055](https://github.com/speediedan/interpretune/issues/1055)). An nnsight composition registers the
model's own modules and is described by its map directly. The model-backend conformance suite runs the same comparison for a per-neuron
recipe on every target whose view places units, and for an output rejection on every target whose view resolves
`mlp.out`, each together with a check that taking the weight form back restores the base logits exactly.

## Surfaces

Interpretune has one intervention surface, `Capability.INTERVENTION`, and a composition's components provide it on
the targets they reach. Which component provides a target is a matter of declared targets, not of the component's
kind:

| Provider | Targets | Object acted on | Entry op | Executed by | Record |
|---|---|---|---|---|---|
| Model backend | `activation`, `units`, `features` | an activation at any vocabulary point (residual stream, attention, MLP, head-shaped), a set of its units, or a latent model's features at its sub-hook | `model_fwd_intervention` | `ModelBackend.fwd_w_intervention`, identical math on TransformerLens and NNsight | `InterventionSupport` (ops, targets, position scopes, value sources, bases, `of`) |
| Model backend, backward pass | `activation`, `units`, `features` (in a recipe, targets addressed `of: grad`) | the gradient with respect to an activation (or a latent model's features) at a hook point | `model_bwd_intervention` | `bwd_w_intervention` (`SupportsGradientIntervention`), through the same op executor as the forward pass: TransformerLens backward hooks, NNsight's `.grad` | `InterventionSupport` with `of: [value, grad]` |
| Analysis backend (e.g. circuit-tracer) | `features` | a replacement model's transcoder feature activations, as `(layer, position, feature_id) -> value` tuples | `feature_intervention_forward` | `ReplacementModel.feature_intervention` through `CircuitTracerAnalysisBackend` | `InterventionSupport` whose `features` target carries a `FeatureTargetSupport` (value sources, constrainable layers, returned activations, freezable norms) |

**Capabilities say what a component provides; support records say how.** A capability member names a surface and
answers only whether a component provides it. The configurations it honours are declared in a typed support record
whose axes mirror venngram's `Support`: a backend that implements `fwd_w_intervention` for `add` and `project` but
not `patch` declares the surface and a record saying so, and a gate refuses `patch` before anything runs.

> **The capability namespace is under review.** `Capability` is one flat enum whose members sit at different levels:
> surfaces (`INTERVENTION`, `CAPTURE`, `GRADIENTS`, `LATENT_MODELS`) beside techniques (`ATTRIBUTION_GRAPH`,
> `HEAD_ATTRIBUTION`). A member per technique does not scale to the range of experiments Interpretune composes, so
> the namespace is likely to change soon: to the core surfaces alone, with techniques as registered ops that declare
> what they require, or to a hierarchy that users and Hub collections can extend
> ([#984](https://github.com/speediedan/interpretune/issues/984)).

**Two providers of one target are legitimate, and are resolved by name.** In a circuit-tracer composition the model
backend reaches an SAE's features at `hook_sae_acts_post` and the replacement model reaches its transcoders' features.
`get_module_capabilities(module).intervention_for("features")` refuses that unqualified request, naming both providers,
rather than resolving it by precedence; an op names the path it uses (`intervention_for("features",
provider="analysis_backend")`, as `feature_intervention_forward` does). Surfaces that are not interventions work the
same way: `CaptureSupport` lists the vocabulary points a model backend can capture, and `AttributionGraphSupport` states
what graph construction requires of the model.

**Basis is a configuration, not a capability.** An intervention defined in a basis (for the Jacobian lens,
`jlens_unfolded` or `jlens_folded`, which are not interchangeable) states that basis on its spec, and the backend's
record says whether it honours it. An unstated or unhonoured basis is refused rather than defaulted.

## Current surfaces

- `ModelBackend.fwd_w_intervention(...)` is the model-level API for interventions on hook tensors,
  implemented with the same maths by both the TransformerLens and NNsight model backends at each declared position
  scope, `last_token`, `all_positions` or `positions` (including
  SAE/latent sub-hook targets via `use_latent_models`/`sae_handles`). Its record, `InterventionSupport`, declares
  the ops, targets and position scopes; an op declares the tensor ops it needs in `required_ops` (beside any
  composition ops it depends on) and the scopes in `required_position_scopes`, and the gate compares the two before
  the op runs.
- `bwd_w_intervention(...)` is its backward counterpart, on both model backends: the same payload, each spec applied to
  the gradient at its hook point during a backward pass ([gradient interventions](#gradient-interventions)). A backend
  offers it by declaring `grad` in its record's `of`, and `require_intervention_support(..., flow="grad")` refuses one
  that does not.
- `ReplacementModel.feature_intervention(...)` is the model-level primitive the analysis-level feature-intervention
  op executes on. Both circuit-tracer backend implementations consume canonical intervention tuples of the form
  `(layer, position, feature_idx, value)` and support passthrough controls such as constrained layers, sparse
  activation capture and optional activation return. `CircuitTracerAnalysisBackend` resolves the settings and
  refuses, through its `features` target's `FeatureTargetSupport`, a value source it does not honour, a constant
  source without a value, a layer constraint or an activation return it cannot provide.

- `interpretune.analysis.venngram` applies a [venngram](https://github.com/speediedan/venngram) recipe on either
  surface above, since venngram addresses each level of abstraction a surface acts on:
  - **Model backend:** `recipe_interventions` turns a recipe's treatments (ordered ops at a site, on the whole
    activation, on a set's units, or on a latent model's features at its `hook_sae_acts_post` sub-hook) into
    `InterventionSpec`s carrying the same ops.
  - **Analysis backend:** `recipe_feature_interventions` turns a recipe's feature ops into the surface's canonical
    `(layer, position, feature_idx, value)` tuples, which any component providing `Capability.INTERVENTION`
    on the `features` target takes (circuit-tracer's replacement model today). The values are computed by Interpretune
    and passed as constants, so a backend must honour the `constant` value source. A tuple sets a feature's value, so a
    feature `add` or `scale` is computed from the feature's clean activation: the backend captures it first
    (`capture_feature_activations`, declared by the `features` target's `captures_clean_activations`) and the caller
    passes it as `clean_activations`.
  - **Graph subsets, on the analysis backend:** an op that targets a venngram graph subset (`target: {graph: <name>}`,
    a pruned piece of an attribution graph as a `graph_subset` document) applies the reasoning it describes in
    another prompt. `recipe_feature_interventions(..., token_ids=, tokenizer=)` places each transcoder or CLT node at
    the stage's site by its anchor in the destination prompt (an absolute position, an offset from the end, a
    token-matched span, or every position), refusing the whole plan if any node does not resolve or two land on one
    coordinate. `replace` sets the node's column, or its delta from a baseline column; `add` sets the feature's clean
    activation plus it. `venngram_feature_subsets` declares what this path applies (venngram's `circuits.subsets`).
    A restriction (`outside`, ablating every feature of a universe the graph does not place) would be one tuple per
    feature and position, so it is refused here. Block-sparse nodes (a vector value per node rather than one feature's
    scalar) are not yet supported on this path and are refused by name
    ([#966](https://github.com/speediedan/interpretune/issues/966)). `export_graph_subset` writes such a document
    from a circuit-tracer attribution graph, pruned (`build_pruned_graph`) or whole: each selected feature is a node at
    its source layer, position and feature index with its activation, anchored `from_end` by default (or `absolute`,
    `token_match` with the source tokenizer, or `all`), the attributions between exported features are its edges, and
    the source prompt, attribution targets and pruning settings are its provenance. `.write(path)` writes it, its
    table in a sidecar when it is large.
  - **Graph subsets, on the model backend:** the same document applies where its space is the latent model attached at
    the stage's site. `recipe_interventions(..., token_ids=, tokenizer=)` places the nodes by the same anchors and
    writes them at that site's `hook_sae_acts_post` sub-hook, one `InterventionSpec` per layer and position that acts
    on that position's features alone (explicit `positions` and `unit_indices`), so every other feature and position
    is untouched. `replace` sets the node's value; `add` adds it to the activation the sub-hook sees, with no clean
    capture. SAE, transcoder and block-sparse nodes are placed; a block-sparse node's vector value is the block's
    contiguous columns. One latent model is attached at a hook, so a stage's nodes share one space, and cross-layer
    transcoder nodes are refused by name. The positions index the sequence the backend runs, so pass one unpadded
    prompt. `venngram_subsets` declares what this path applies, and `check_recipe` checks graph ops against it. A
    restriction has no native form at a latent sub-hook yet and is refused here too.

  What each recipe op is honoured as, per surface and target:

  | Op | Model backend: whole activation | Model backend: units, or features at the latent sub-hook | Analysis backend: features, at explicit positions |
  |---|---|---|---|
  | `replace` | the operand, or the dose broadcast | units: selected-coordinate replacement; latent features: refused here, except graph nodes at their positions | the dose $d$ |
  | `add` | the operand times the dose, or the dose | the dose per member, added to the activation the forward pass meets | $a + d$, for the clean activation $a$ |
  | `scale` | the dose, as a multiplier | the dose per member, ones elsewhere | $a \cdot d$, for the clean activation $a$ |
  | `project`, `reject`, `patch`, `clamp` | the operand (a vector, or a pair for `patch`) | refused | refused |

  The two `add` columns differ only where a recipe also intervenes upstream of the feature: the model backend adds to
  the activation the earlier intervention produces, while the analysis backend's value is fixed from the clean pass.

  A value is a dose, a table of doses, or a column of evidence the caller supplies, through venngram's transforms;
  Interpretune evaluates the transforms on a unit or feature target's members, checked against venngram's conformance
  vectors, and the `interpretune.analysis.venngram.values` module states what each one means. A statistic a transform
  names is supplied by the caller as one value, a value per layer, or a value per member.
  `venngram_support()` derives what Interpretune honours of a recipe from the backend's own records, and `check_recipe`
  / `require_supported` (with `analysis_backend=True` for the analysis-backend surface) refuse, before anything runs,
  whatever that record cannot express. `export_book` goes the other way, writing an intervention payload as a venngram
  book carrying it as a treatment, and `support_mapping` publishes the derived declaration as a `support_record`, so
  venngram's conformance checker reads both without Interpretune installed. Interventions on sets of units named by
  address, composed with set algebra, are the next section's subject
  ([unit-set interventions](#unit-set-interventions-by-address)); applying attribution-graph subsets from one context
  in another, on both surfaces, is tracked in [#966](https://github.com/speediedan/interpretune/issues/966) (a
  restriction to the subset is the open part).

At this layer Interpretune delegates to backend-native steering surfaces rather than inventing a second execution
mechanism. The closest analogs are TransformerLens hook-driven flows such as `run_with_hooks(...)` and NNsight
tracing that mutates activations inside a trace.

`feature_intervention_forward` is the analysis-level op. It consumes `top_feature_ids` plus value inputs from
`AnalysisStore`, constructs canonical intervention tuples, runs a clean forward pass and an intervened forward pass
through `module.replacement_model`, and stores Arrow-safe intervention summaries back into `AnalysisStore`.

`model_fwd_intervention` is the model-level op for direct tensor interventions. It accepts either explicit
`interventions` / `interventions_json` mappings or the shorthand `intervention_hook_pattern`, `intervention_op`,
`intervention_scale_factor` and `intervention_use_intervention_tensor_as_basis` fields; explicit mappings take
precedence. A payload key, or the shorthand's `intervention_hook_pattern`, may be a hook name, a pattern, or a venngram
site address, which names one activation the same way on every backend: `point:hook_resid_post & layers:13` rather
than one backend's module path. An address expands like a layer wildcard, to the sites it selects in layer order
(`point:hook_resid_post & layers:-4:` is the last four blocks' residual stream), each matched as its hook name would
be. A site the model does not expose is refused rather than mapped onto a nearby hook, and an address saying
`of: grad` is refused here: it names a gradient, which `model_bwd_intervention` edits. Concept-direction notebook experiments use this surface for direct-projection phases, including
configurations that inject the computed `concept_direction` as the `intervention_tensor` for a non-default hook
such as `blocks.0.hook_in` with the `project` op.

## Unit-set interventions by address

The two surfaces above say what a backend can do to an activation or to a latent model's features. A **unit-set
intervention** says which units it does it to: a set of neurons (or of a latent model's features) named once by a
venngram address, composed with other sets by set algebra, and acted on by a treatment's ops. Nothing in it is specific
to an architecture or to where a set came from: an address names a site and the units at it, and the same recipe runs
on every backend whose support record honours it.

### Direct unit payloads

`unit_interventions` builds a payload for `it.model_fwd_intervention` (its `interventions` input), without a recipe:

```python
from interpretune.analysis.venngram import unit_interventions

payload = unit_interventions(
    "point:mlp.out.hook_in & layers:12 & units:[431, 1007]",
    n_layers=32,
    site_widths={"mlp.out.hook_in": 14336},  # the checkpoint's MLP hidden width
    op="add",
    value=[0.5, -0.25],
    position_scope="all_positions",
)
```

`add`, `scale` and `replace` take a scalar or one value per selected unit, in ascending index order. A mapping from
layer index to those values supplies different doses at different layers. `clamp` takes `clamp_min` and/or
`clamp_max`, with no `value`, and bounds only the selected coordinates in the model's unit basis. All four modes
preserve every unselected coordinate. The token scope independently chooses the final real token or every position.

Addresses may use per-layer unit mappings, unit slices, negative indices and set algebra over book references
(`sets=book.sets()`). Widths are supplied by the model, so a slice resolves against the real width. An address
selecting nothing, an unavailable width or an unsupported field is refused. Guarded unit expressions are not
supported. A gradient address (`of:grad`) requires `flow="grad"` and the payload goes to
`it.model_bwd_intervention`; the forward translation refuses it.

The payload carries `InterventionSpec.unit_indices`. Its operand is in that selected coordinate order, and the shared
executor reads, edits and writes back only that slice. This differs from replacing the whole activation with a sparse
vector: zero is an explicit replacement value, while coordinates outside the selection retain their values and their
autograd path. A backend must declare the `units` target in its intervention support record. A direct unit payload
edits activations; editing the weights behind the same units is a unit adapter (`attach_unit_adapter`, under
[parameter targets](#parameter-targets)).

For normalized doses, `normalized_unit_doses` computes $d_u m / \lVert W_u \rVert_2$, where $m$ is the measured median
residual norm and $W_u$ is the down-projection column for unit $u$. Supply the stored weight and its measured
`unit_dim` (1 for `nn.Linear`, 0 for GPT-2 `Conv1D`), plus the selected units and reference median. The helper
refuses zero column norms and nonfinite inputs; it never guesses the weight layout from its shape. Its output is a
`value` for the direct payload, or a per-unit dose table in a recipe.

`export_book` writes selected `add`, `scale` and `replace` specs as unit-targeted recipe ops, including zero-valued
replacements. A selected clamp is refused on export: venngram's recipe registry currently declares `clamp` on a whole
activation with a vector or subspace operand, so it cannot represent this direct payload's selected unit basis.

### Naming and composing a set

A recipe's book names its sets. A set is a term (a site, the layers it spans and the units at each) or a composition of
other sets by reference: `@a & @b` (intersection), `@a | @b` (union) and `@a - @b` (difference). venngram resolves a
composition over `(layer, unit)` pairs when it reads the recipe, so Interpretune never evaluates set algebra itself and
a composed set means the same thing to every consumer.

```yaml
schema_version: 2
kind: recipe
name: steer-the-shared-units
model: {id: <model id>, revision: <revision>}
book:
  schema_version: 2
  name: shared-units
  portability: checkpoint
  applies_to: {models: [{id: <model id>, revision: <revision>}]}
  sets:
    a: {point: mlp.out.hook_in, layers: [12], units: {12: [1, 2, 3]}}
    b: {point: mlp.out.hook_in, layers: [12], units: {12: [3, 4]}}
    shared: "@a & @b"      # unit 3
    only_a: "@a - @b"      # units 1 and 2
stages:
  - site: mlp.out.hook_in
    tokens: last
    ops:
      - {op: add, target: {set: shared}, value: {dose: 0.5}}
      - {op: scale, target: {set: only_a}, value: {doses: {12: {1: 0.0, 2: 2.0}}}}
```

A unit target's stage names the layer-free site, since the set carries its own layers. The units are explicit indices,
as a flat list over the set's layers or per layer (above); a slice of units is refused, because a treatment needs to
know exactly which units it doses.

### How a treatment's ops reach the native modes

`recipe_interventions` translates each unit-targeted op into one `InterventionSpec` per layer the target spans, keyed
`blocks.<layer>.<site>` and appended in recipe order, so the model backend folds the ops in the order venngram gives
them:

| Op | Spec | Units the value names | Every other unit |
|---|---|---|---|
| `add` | `add`, with a dense vector at the site's width | the value | 0 |
| `scale` | `scale`, with a dense vector at the site's width | the value | 1 |
| `replace` | `replace`, with `unit_indices` and the selected values | the replacement | unchanged |

The specs use the shared executors that `fwd_w_intervention` applies on TransformerLens and NNsight. A replacement
selects its coordinates before execution, so it leaves every other unit untouched. The caller supplies each site's width
(`site_widths`) from the model's configuration, since the width is a fact of the checkpoint the recipe pins, and a
unit at or past it is refused rather than dropped. A feature set takes the same path one level down, at the latent
sub-hook (`blocks.<layer>.<site>.hook_sae_acts_post`), or becomes circuit-tracer's feature tuples on the analysis
backend (`recipe_feature_interventions`), as the [table above](#current-surfaces) describes.

A value is a `dose` (every member takes it), a `doses` table, or a column of evidence the caller supplies, through
venngram's transforms. On a composed target, a `dose` doses every member of the composition and a `doses` table only
the members it lists: a table entry for a unit outside the composition (unit 4 in `only_a` above) doses nothing.

### What bounds what runs

`venngram_support()` derives what Interpretune honours from the backend's own records, and a composed unit set is a
`units` target there like any other. `check_recipe` and `require_supported` list every refusal before anything runs,
and `recipe_interventions` refuses again what it cannot translate, so a caller that skipped the check still cannot get
a silently narrowed treatment. The refusals a unit-set treatment can meet:

| Refused | By |
|---|---|
| a reference to a set no set or import defines, or references that form a cycle | venngram, reading the book |
| a composition that names no units (`@a - @a`) | venngram, reading the recipe |
| a slice of units where explicit units are required | venngram for a composition, reading the recipe; `recipe_interventions` for a single set |
| a unit-targeted recipe op other than `add`, `scale` and `replace` | venngram's op registry or `op_refusal` |
| an operand on a unit-targeted op (it takes a value) | `op_refusal` |
| a site with no supplied width, or a unit at or past it | `recipe_interventions` |
| an evidence-sourced value on a composed target | `recipe_interventions` |
| a target addressing the gradient (`of: grad`), composed or not, in a translation for values | `op_refusal` |

`add` and `scale` use dense vectors with their identity values elsewhere; `replace` uses a selected-coordinate spec.
A gradient target is translated with `flow="grad"` for the backward surface
([gradient interventions](#gradient-interventions)).

### Not yet supported

- **Evidence-sourced values on a composed target.** venngram lists a composition's members only through an op's doses
  today, so a composed target takes a `dose` or a `doses` table.
- **Registered address books with evidence**, so an experiment cites a model's sets, their compositions and the
  evidence behind them by reference rather than re-deriving indices, with registered examples for supported models.

Both are tracked in [#905](https://github.com/speediedan/interpretune/issues/905).

## Latent addresses: blocks, frames, charts and linear spaces

`interpretune.analysis.venngram` resolves a latent address against an explicit catalog of loaded latent models
(`LatentCatalog`). This is the addressing and coordinate layer: it reads and edits coordinates of tensors it is
given. A frame or chart edit also runs natively inside a backend's forward pass (see "Native latent edits" below).

**One address form for every dictionary.** A `LatentBinding` declares a loaded model at one layer and activation point,
its decoder rows, and its block size $b$: $b = 1$ for an SAE or transcoder, and block $g$ of a block-sparse featurizer
is the latent range $[gb, (g+1)b)$. So `features:bsf_l17 & point:resid_post & layers:17 & units:[412]` selects one
block exactly as `features:sae & units:[1234]` selects one feature, and unit slices and negative units count blocks.
A named basis at a site is a `SubspaceBinding`, addressed as `subspace:<name>`, and is indivisible. External ids
(a Neuronpedia id, an SAE Lens pair, a circuit-tracer feature) resolve only through aliases the catalog is given to
canonical addresses; nothing is downloaded or inferred from an id's spelling.

**Frames.** A block's internal basis is arbitrary (any rotation in $O(b)$ preserves its span), so an address never
names a coordinate inside a block. Coordinate reads and edits name a frame: a `LatentFrame` is a named, exactly
versioned orthonormal basis for one target, checked against the target's current decoder rows for equal rank and
span each time it is used. Replacing framed coordinates changes only the target's span and keeps the orthogonal
complement and autograd.

**Charts.** A `LatentChart` is a named, versioned coordinate system fitted in one frame version: its kind, intrinsic
dimension, periodic dimensions with their periods, and fitted parameters. Each kind is a registered handler
(`register_chart_kind`); `affine` (a flat chart, no periodic dimension) and `pca_plane` (a plane read as a radius and
an angle of period $2\pi$) are built in, and any other kind, such as a spline or a flow, is refused by name until a
caller registers it. Edits are made in chart coordinates: `set`, `path` (piecewise linear through waypoints, the
shorter way round each periodic dimension), `rotate` (periodic dimensions only) and `scale_norm`. An edit moves the
frame coordinates by exactly the chart's change, so the part off the chart is kept. Charts can travel in an address
book's top-level `charts` key, which venngram keeps without changing what the book's sets select
(`charts_from_book`).

**Linear spaces.** Features trained on $J_\ell h$ rather than on $h$ have decoder rows in that space. A `LinearSpace`
carries the map $M$ and writes an edit back through its declared inverse or its pseudoinverse; the map composed with
the inverse must be the identity on the space, so a write lands the requested space coordinates exactly and leaves
the null space of $M$ as it was. A map whose rank is below its dimension is refused rather than written in least
squares. A binding names its space (`space="jlens"`), the address qualifier `space:jlens` selects by it, and the
catalog's `site_coordinates`, `replace_site_coordinates` and `apply_site_chart` read and write site activations
through it.

### Native latent edits

`latent_edit_spec(catalog, target, op, name=..., version=..., **args)` compiles one frame or chart edit of one
feature block into a hook name and an `InterventionSpec` for `model_fwd_intervention` or a backend's
`fwd_w_intervention`. The hook is the latent sub-hook of the model the block belongs to,
`blocks.<layer>.<site>.hook_sae_acts_post`, so the edit runs on both CPU backends with the latent model attached
(`use_latent_models`), and its token scope (`position_scope`) is independent of the block it selects.

| op | catalog edit | arguments |
| --- | --- | --- |
| `frame_replace` | `replace_coordinates` in frame `name` | `coordinates` |
| `chart_set` | `apply_chart(op="set")` in chart `name` | `value` |
| `chart_path` | `apply_chart(op="path")` | `waypoints`, `t` |
| `chart_rotate` | `apply_chart(op="rotate")` | `by` |
| `chart_scale_norm` | `apply_chart(op="scale_norm")` | `factor` |

At the latent sub-hook the activations are the block's latent coefficients $a_B$, and the values the catalog edits
are the block's decoded contribution $v = a_B D_B$, with $D_B$ the block's decoder rows. The executor computes the
catalog's own edit $v'$ and writes back the minimum-norm change of the coefficients,
$a_B' = a_B + d\left(v' - v\right) D_B^{+}$, where $d$ is the spec's operand, the dose (1 applies the edit, 0 is the
identity). The frame spans the rows of $D_B$, so the decoded contribution moves by exactly $d\left(v' - v\right)$, and
the other blocks and the latent model's error term are untouched.

The ops are tensor ops in the bundled `latent` op collection, each with its `OpSpec` and an executor registered by
name. Before any pass, both backends refuse an edit whose latent sub-hook has no attached model, whose attached model
is not the object the target resolved to, or whose decoder rows have changed since the catalog was built, and an edit
on the gradient flow. An unknown or stale frame or chart version is refused when the spec is compiled.

A frame-free edit of a whole block needs no new op: `block_edit_spec(catalog, target, op, value=...)` writes `set` (the
block's activations from a source run) and `ablate` as a core `replace` on the block's columns, and `scale_norm` as a
core `scale`, which scales the block's decoded contribution, and so its norm, by the same factor.

## Gradient interventions

`model_bwd_intervention` takes the payload `model_fwd_intervention` takes, and applies each spec to the gradient at its
hook point while a backward pass runs. The gradient at a point is the gradient of the readout (the scalar the backward
pass starts from) with respect to the activation there, summed over every consumer of that activation, so it is what
flows on to earlier points: an edit replaces it before it does, and every point upstream of the edit sees the effect.
The forward pass is not edited, so the readout is the same with and without the interventions. Specs at one point fold
in declaration order, as on the forward pass.

- **The readout** is `readout` when given (see [Readouts](#readouts)); otherwise the summed last-token logit of
  `logit_target_ids` (else the concept groups' token ids), or, with neither, of each row's own top prediction. A
  `readout` beside `logit_target_ids` is refused, since only the default reads the ids.
- **What is stored**: the gradient at every `grad_capture` site from a baseline backward pass and from the intervened
  one (`pre_intervention_grad_values`, `post_intervention_grad_values`, flattened beside `grad_shape` and
  `grad_point_names`), the readout's value (`bwd_intervention_metric`), and `intervention_grad_position_effect`, the
  largest change in any captured gradient at each position. Without `grad_capture`, the intervention sites are
  captured, each after its own edit.
- **Sites by address**: `grad_capture` takes hook names or patterns, or one venngram site address, which
  `interpretune.analysis.venngram.resolve_sites` evaluates over the model's block stack. A site address sets
  `point` (a spelling of the activation-point vocabulary, such as `hook_resid_post`), `layers` and `of`, and combines
  terms with `&`, `|`, `-` and `@<set>` references (`grad_capture_sets` supplies the book's sets). An address read for
  gradients must say `of: grad`. A payload key may be an address too, as on the forward surface, and must say
  `of: grad` for the same reason.
- **Recipes**: `recipe_interventions(recipe, ..., flow="grad")` translates a recipe's ops on `of: grad` targets into
  the same specs a value op becomes, for this op; the default `flow="value"` translates its value ops. A recipe's ops
  are translated for one flow at a time, and an op on the other flow is refused.

### Example

The model-level surface, runnable as written (gpt2, CPU). It zeroes the gradient that reaches block 6's input at the
last position, and reads the gradient at the output of blocks 0 to 4, picked with a venngram site address:

```python
import torch
from transformer_lens.model_bridge import TransformerBridge

from interpretune.adapters.transformer_lens.backends import TLModelBackend
from interpretune.analysis.venngram import resolve_sites

model = TransformerBridge.boot_transformers("gpt2", device="cpu")
tokens = model.to_tokens("The Eiffel Tower is in the city of")
paris = model.to_single_token(" Paris")


def metric(logits: torch.Tensor) -> torch.Tensor:
    return logits[:, -1, paris].sum()  # the logit of " Paris" at the last position


payload = {
    "blocks.6.hook_in": {
        "intervention_tensor": torch.zeros(model.cfg.d_model),
        "op": "replace",
        "position_scope": "last_token",
    }
}
capture = resolve_sites("point:hook_out & layers:0:5 & of:grad", n_layers=model.cfg.n_layers)

pre, post, value = TLModelBackend().bwd_w_intervention(
    model, {"input": tokens}, payload, metric, capture=list(capture.sites)
)
for site in capture.sites:  # gradient norms, baseline -> intervened
    last = f"{pre[site][:, -1].norm():.2f} -> {post[site][:, -1].norm():.2f}"
    earlier = f"{pre[site][:, :-1].norm():.2f} -> {post[site][:, :-1].norm():.2f}"
    print(f"{site}  last position {last}  earlier positions {earlier}")
```

```text
blocks.0.hook_out  last position 2.48 -> 0.00  earlier positions 4.06 -> 2.02
blocks.1.hook_out  last position 3.96 -> 0.00  earlier positions 7.14 -> 2.70
blocks.2.hook_out  last position 4.18 -> 0.00  earlier positions 4.56 -> 2.66
blocks.3.hook_out  last position 3.97 -> 0.00  earlier positions 3.59 -> 2.43
blocks.4.hook_out  last position 3.79 -> 0.00  earlier positions 2.97 -> 2.37
```

The last position's gradient is zero at every earlier block: below block 6 it comes only from the last position
itself, whose gradient the edit zeroed. The earlier positions keep what attention in blocks 6 to 11 had already moved to
them, and lose what attention below block 6 would have brought from the last position. An NNsight `LanguageModel` gives
the same gradients, to float32 rounding, through `NNsightModelBackend(HookNameResolver("GPT2LMHeadModel"))`, which takes
`{"input_ids": tokens}`.

In an analysis session, the same edit is the `model_bwd_intervention` op, run like any other
([Analysis Runner Usage](analysis_runner_usage.md)):

```python
import torch

import interpretune as it
from interpretune import AnalysisCfg

payload = {"blocks.6.hook_in": {"intervention_tensor": torch.zeros(768), "op": "replace", "position_scope": "last_token"}}
cfg = AnalysisCfg(
    target_op=it.model_bwd_intervention,
    run_inputs={
        "interventions": payload,
        "grad_capture": "point:hook_out & layers:0:5 & of:grad",
        "logit_target_ids": [6342],  # gpt2's " Paris"
    },
)
```

Each stored row holds both gradients at every captured site, flattened beside `grad_shape` (the shape code) and
`grad_point_names`, the readout's value, and `intervention_grad_position_effect`.

### Readouts

Which scalar a gradient is taken of decides what an edit to it means, since a vector added to a gradient is scaled by
it. A `Readout` names that scalar as `fn(logits, activations)`, where `activations` holds the activation at each site
in `sites` from the same forward pass the backward runs from, so the scalar can be a logit difference, an intermediate
activation's projection onto a probe or concept direction, or a latent model's feature activation
(`<hook>.hook_sae_acts_post`, with the latent model attached). It must return a scalar tensor, and use only standard
PyTorch ops, so a traced backend can apply it to its proxies. A plain callable of the logits is a readout reading no
sites, as `metric` is in the example above. Sites are hook names or `*` patterns: an exact name is handed back under
the name as written, and each match of a pattern under its concrete name.

```python
import torch
from transformer_lens.model_bridge import TransformerBridge

from interpretune.adapters.transformer_lens.backends import TLModelBackend
from interpretune.analysis.backends import Readout

model = TransformerBridge.boot_transformers("gpt2", device="cpu")
tokens = model.to_tokens("The Eiffel Tower is in the city of")
paris, london = model.to_single_token(" Paris"), model.to_single_token(" London")
# the unembedding direction that separates the two answers, read at block 8's output instead of at the logits
direction = (model.W_U[:, paris] - model.W_U[:, london]).detach()


def logit_difference(logits, activations):
    return (logits[:, -1, paris] - logits[:, -1, london]).sum()


def block_8_projection(logits, activations):
    return (activations["blocks.8.hook_out"][:, -1] @ direction).sum()


readouts = {
    "logit difference": Readout(logit_difference),
    "block 8 projection": Readout(block_8_projection, sites=("blocks.8.hook_out",)),
}
for name, readout in readouts.items():
    pre, _, value = TLModelBackend().bwd_w_intervention(
        model, {"input": tokens}, {}, readout, capture=["blocks.2.hook_out", "blocks.8.hook_out"]
    )
    norms = "  ".join(f"{site} {pre[site][:, -1].norm():.2f}" for site in pre)
    print(f"{name:<19} value {value.item():6.2f}  last-position gradient norm: {norms}")
```

```text
logit difference    value  -0.16  last-position gradient norm: blocks.2.hook_out 0.51  blocks.8.hook_out 0.37
block 8 projection  value  -9.79  last-position gradient norm: blocks.2.hook_out 3.51  blocks.8.hook_out 3.02
```

The projection's gradient at block 8's output is the direction itself (its norm is the direction's), since nothing
after block 8 reaches it; below block 8 the two readouts differ by everything blocks 9 to 11 and the final norm do to
the difference. A gradient captured at a site the readout reads includes the readout's own contribution, beside what
flows back from the logits.

In an analysis session, pass the readout as the op's `readout` input:

```python
cfg = AnalysisCfg(
    target_op=it.model_bwd_intervention,
    run_inputs={"interventions": payload, "grad_capture": ["blocks.2.hook_out"], "readout": readouts["block 8 projection"]},
)
```

### A readout declared in a treatment

A venngram treatment whose stages act on gradients (`of: grad`) declares the readout they are gradients of, as a
weighted sum of terms, each a registered reduction of the activations a book set addresses at a token scope; a stage
may declare its own in place of the treatment's. `recipe_readout` (`interpretune.analysis.venngram`) executes that
declaration as a `Readout`, which goes beside the payload `recipe_interventions(..., flow="grad")` translates:

```yaml
readout:                                              # a logit difference plus a block-8 projection
  terms:
    - {reduce: sum, set: answer_logit, tokens: last, weight: 1.0}     # point:unembed.hook_out & units:[1000]
    - {reduce: sum, set: foil_logit, tokens: last, weight: -1.0}      # point:unembed.hook_out & units:[2000]
    - {reduce: project, set: block8, tokens: last, operand: probe}    # point:hook_out & layers:8
```

```python
from interpretune.analysis.venngram import recipe_interventions, recipe_readout

payload = recipe_interventions(recipe, site_widths=widths, flow="grad")
readout = recipe_readout(recipe, n_layers=model.cfg.n_layers)  # None: the recipe predates readouts; the default applies
cfg = AnalysisCfg(target_op=it.model_bwd_intervention, run_inputs={"interventions": payload, "readout": readout})
```

- **What a term reads.** A block site is read from the forward pass the backward runs from, as a callable readout's
  sites are, and a set selecting several sites reads their addressed units side by side in layer order.
  `point:unembed.hook_out` is the logits, whose units are the vocabulary. Units are resolved against the activation's
  width when the readout runs, and a term without `units` reads every unit of its sites.
- **Token scopes:** `last`, `all`, and explicit `positions` (columns of the padded batch). `prompt`, `generated` and
  `slice` are refused.
- **Reductions:** `sum`, `project` (each token's values dotted with the operand, a vector over the addressed units in
  their order) and `norm`, each summed over the term's tokens; the readout is the weighted sum of the terms, summed over
  the batch's rows, as the default readout is. The executors are checked against venngram's conformance vectors, and
  `register_readout_executor` adds one for a reduction registered in venngram only if it passes them. The support record
  declares the reductions executed here (`readouts`), so `check_recipe` refuses a treatment using any other.
- **One scalar per backward pass.** Gradient stages declaring different readouts are refused rather than applied under
  one of them; apply the stages that share a readout together.

### Changing the readout, or editing its gradient

Two things move the gradient an upstream site receives, and they answer different questions:

- **Changing the readout** changes which scalar is differentiated: a loss, a logit difference in place of one logit, a
  probe's projection in place of the logits. Every gradient then follows from the new scalar through the unchanged
  graph. It answers "what drives this quantity?".
- **Editing the gradient at a site** keeps the scalar, and changes how its derivative propagates past that site, at
  the positions the edit's scope selects, with the forward pass untouched. It answers "what reaches the earlier sites
  through this one, and what changes if it does not?".

The two coincide in one case. An `add` edit of a vector $v$ at site $s$ gives every site upstream of $s$ the gradient
of $R + \langle v, h_s \rangle$, the readout $R$ plus a linear term in the activation $h_s$, with $v$ held constant: on
the example's prompt the two agree at `blocks.2.hook_out` to float32 rounding. A zeroing (`replace` with zeros),
`scale` or `project` edit has no such counterpart. Zeroing the gradient at $s$ for the last token is a stop-gradient on
that position's activation: it removes every path from the readout through it and keeps every other path. For one
input that equals the gradient of $R - \langle g_s, h_s \rangle$, where $g_s$ is that input's own gradient at $s$, so it
takes the backward pass to state; across inputs the edited gradient is not the gradient of any single function, so no
readout reproduces it. With the logit difference as the readout, zeroing the
gradient at `blocks.5.hook_in` for the last token takes the last position's gradient at `blocks.2.hook_out` from 0.51
to 0, while the earlier positions' falls only from 0.94 to 0.89: what attention at block 5 and above carried from them
to the last token still reaches them.

## Intervention ops

An intervention applies an op, and an op is a name in venngram's federated registry (`venngram.ops.OPS`), applied
through the executor registered for that name in `interpretune.analysis.backends.ops`. `InterventionSpec.op` names it,
the payload key is `op`, and the shorthand field is `intervention_op`. The core ops are `replace`, `add`, `project`, `patch`, `reject`, `clamp`, `scale`
and `replace_with_error`, registered from the shared op math, and a backend whose hook computes its edit during the
pass (both bundled model backends) declares every registered op, including one registered after it was built. A user
registers another op with venngram's `register_op` and its executor with `register_executor(name, fn)`, where
`fn(region, operand, spec)` returns the edited region; a hub op collection can carry one as a tensor op (see
[publishing an op collection](hub_op_collection_authoring.md)).

`model_fwd_intervention` dispatches the op the payload names through that registry, so a recipe or payload naming any
registered op (core, user or hub) is applied, or refused where the backend's record does not declare it. The `patch` op (the
J-space write) takes exactly two direction vectors stacked on a leading axis and swaps the activation's
coordinates along that pair, leaving the component orthogonal to the pair untouched. Mechanically it is
basis-agnostic: any pair of directions can serve as the patch pair, so embed-basis concept poles are as valid a
source as Jacobian lens rows; which basis produced the pair is the configuration the paragraph above requires the
op to state.

What each core op preserves and what it moves, in one place:

- `replace`: preserves nothing — the region becomes the target tensor (broadcast to shape).
- `add`: preserves everything except the shift — `h + target · scale_factor`, perturbing every
  component the added vector overlaps.
- `project`: preserves the component parallel to the basis and discards the rest (the opposite of
  `reject`; at full strength the two ops sum to the input).
- `patch`: preserves the component orthogonal to the pair and swaps the two in-pair coordinates
  (`V(σ(c) − c)` with `c = V⁺h`), exact by pseudoinverse rather than transpose.
- `reject`: preserves the component orthogonal to `span(V)` and removes the in-span part
  (`h − α·VV⁺h`); `scale_factor` is the removal fraction (1.0 removes, 0.0 is the identity),
  which is what makes graded ablation bands one op with a parameter instead of three.
- `clamp`: preserves in-range coordinates exactly and bounds the out-of-range ones into
  `[clamp_min, clamp_max]` measured in `V⁺` coordinates (`h + (c′ − c)V`); a clamp with
  neither bound is the identity and is refused rather than run. Bound-into-range and
  assign-to-value are different operations sharing an English word: no choice of bounds swaps
  two coordinates, so `patch` is a special case of assignment, never of clamping.
- `scale`: preserves every component's direction and multiplies it elementwise by the target
  tensor times `scale_factor` (a scalar, or a per-unit multiplier with ones where units are left
  alone), so a tensor of ones is the identity.
- `replace_with_error`: preserves the forward value exactly and moves only the gradient. For an operand $o$ (a latent
  model's reconstruction) at an activation $x$ it gives $o + \mathrm{sg}(x - o)$, where $\mathrm{sg}$
  stops the gradient: the forward value is $x$, and the gradient reaches $o$ rather than $x$. It is the straight-through
  substitution a spliced latent model trains with ([latent models](tuning_latent_models.md)); in an analysis pass,
  where nothing differentiates, it is the identity.

> **The pair and the model must share a residual basis, and TransformerLens weight processing changes it.**
> A `TransformerBridge` with `enable_compatibility_mode()` folds LayerNorm and centers weights by default, which
> rewrites the residual stream's geometry at every `hook_resid_*` point: a patch pair built from unprocessed weights
> (which is what lens artifacts and the circuit-tracer `ReplacementModel` path use) then swaps in the wrong plane.
> Measured on gpt2: the same pair's intervention delta disagrees between nnsight and TransformerLens by a **0.948
> relative gap** under default processing, against 0.000 with processing off
> (`enable_compatibility_mode(no_processing=True)`, or Interpretune's `ITLensFromPretrainedNoProcessingConfig`).
> Load unprocessed when applying patch pairs on TL, or build the pair from the processed model's own weights; never
> mix. (The unprocessed agreement is pinned by `tests/core/test_jlens_patch_validation.py`.)

> **A folded vector counts its fold exactly once, against the matrix whose processing state matches the scale.**
> The folded (`jlens_folded`) J-lens construction multiplies unembedding rows by the final norm's scale, and that scale
> lives on a specific side: `(1 + weight)` for HF gemma RMSNorms, plain `weight` elsewhere including TL
> models, whose gemma conversion folds the `+1` at load. Building folded vectors from a *processed* `W_U`
> while applying an HF-convention scale counts the norm twice — once baked into the weights, once in the
> fold — and the resulting directions are wrong in a way no magnitude adjustment repairs, for the same
> reason the pair basis above is not interchangeable. Go through `resolve_unembed_and_norm_scale` plus
> `jlens_direction_rows`, which pair each matrix with its own scale convention, rather than hand-rolling
> either side.

## Op-level entry points and composites

The registered analysis ops (all callable as `it.<name>(...)`):

- `concept_direction` (alias `semantic_direction`): builds a normalized direction in the basis named by the
  required `concept_basis`: `embed`, `store` (from store latents), `jlens_unfolded` or `jlens_folded`
  (modes `mean_difference` / `paired_rejection` / `single_group`; `streaming` or `in_memory` aggregation).
  There is no default basis and no fallback: `store` with no latent rows raises rather than returning an
  embedding direction, and the retired names `jlens_paper` / `jlens_norm_aware` are refused with their
  replacements.
- `jlens_read` (`jacobian_lens_read`), `jlens_concept_probe` (`jacobian_lens_concept_probe`) and
  `jlens_sparse_inventory` (`jacobian_lens_sparse_inventory`): the J-lens read path (ranked readout, concept
  cosine, and sparse nonnegative decomposition with its residual); see `jlens_usage.md`.
- `intervention_first_order_check`: checks an applied intervention against its first-order prediction per basis.
  The prediction is the linearization attribution patching uses to estimate an activation patch without applying it;
  this op applies the intervention and compares the measured change with that estimate.
- `weight_equivalence_check`: checks that a per-neuron recipe's or an output rejection's weight form gives the same logits as its activation
  form through the model backend ([parameter targets](#parameter-targets)).
- `compute_attribution_graph` (`ct_graph`), `graph_node_influence` (`ct_node_influence`),
  `extract_top_features` (`ct_top_features`), `feature_intervention_forward` (`ct_feature_intervention`),
  `model_fwd_intervention` (`direction_intervention` / `direct_concept_direction_intervention`).
- Composites: `attribution_from_concept` (direction -> graph -> influence -> top features),
  `intervention_from_features`, and `intervention_from_concept` (the full five-op pipeline).

## Feature selection (`FeatureSelectionSpec`)

`extract_top_features` accepts a structured `FeatureSelectionSpec` with OR-semantics filters (`layers`,
`positions`, `feature_ids`, `layer_slice`, `position_slice`, `triples`, `layer_feature_pairs`), ranking
controls (`score_source` — `influence` / `signed_influence`, `gradient`-family planned; `score_sign` —
`any` / `positive` / `negative`; `rank_by_abs`), and `activation_overrides` for pinned per-feature
activation values. This is the sign-aware selection surface exercised by the orange `fs_l10_n5` lineage.

## Active store-path scaling behavior

These are ACTIVE runtime controls (not just config candidates): `intervention_value_source`
(`top_feature_scores` | `top_feature_activation_values` | `constant`), `intervention_scale_factor`,
`intervention_max_influence_norm_scale` (per-feature `abs(score)/max(abs(score))` amplification), and
`intervention_sign_aware_scale` (default on). The pinned combined formula (verified by
`test_analysis_backend_parity_feature_intervention_wrapper_sign_aware_top5_any_scaling`) is
`value = sign(score) * abs(activation) * (scale_factor * abs(score)/max_abs_score)`.

## Config split: what exists and what is still planned

`FeatureTargetSupport`, the detail of circuit-tracer's `features` target, is now that provider's configuration space as
a record: value sources, whether layers can be constrained, whether activations can be returned. What is still planned
is the split of the `CircuitTracerConfig` fields themselves, which keep both levels' knobs at the top level behind
`intervention_` prefixes:

- a shared model-level intervention config dataclass reusable by `model_fwd_intervention`,
  `resolve_interventions(...)` and both model backends: `hook_pattern(s)`, `interventions`, `op`, `scale_factor`,
  `use_intervention_tensor_as_basis`, `intervention_tensor`, and the basis where one applies, with the
  non-prefixed spellings canonical and legacy prefixed spellings normalized at the op boundary;
- a circuit-tracer analysis-level intervention config dataclass for `value`, `value_source`, `sign_aware_scale`,
  `max_influence_norm_scale`, `constrained_layers`, `freeze_attention`, `apply_activation_function`, `sparse` and
  `return_activations`, the fields the record already describes.

Until that lands, `resolve_feature_intervention_settings` reads the prefixed fields and the record is the check.

## Hook pattern contract

Prefer canonical TransformerBridge-style hook names in new intervention configs, even when the current backend still
accepts older HookedTransformer aliases. The current portable subset and its legacy aliases are documented in
[intervention_hook_pattern_support.md](intervention_hook_pattern_support.md).

In practice this means new configs should prefer names such as `blocks.{i}.hook_in`, `blocks.{i}.hook_out`,
`blocks.{i}.attn.hook_out`, `blocks.{i}.attn.o.hook_in`, and `unembed.hook_in`. The intervention pattern expander now
tries supported canonical and legacy spellings in both directions before backend resolution.

## Multi-spec ordering at one hook point

A payload may name several specs at one hook point, given as a list in declaration order. They apply sequentially,
each seeing the previous spec's output: `[add(v), reject(v)]` differs from `[reject(v), add(v)]`, while two `add`
specs commute. Backends fold the spec tuple left to right, and adapters preserve the declaration order rather than
grouping by op.

## Direct projection config pattern

The concept-direction notebook configs now expose a notebook-facing `ANALYSIS.direct_projection` section. The most
expressive form mirrors `resolve_interventions(...)` by supplying an explicit intervention mapping whose values are
valid `InterventionSpec`-style payloads. The experiment wrapper injects the runtime `concept_direction` tensor when an
`intervention_tensor` is omitted.

Example:

```yaml
ANALYSIS:
    direct_projection:
        interventions:
            blocks.0.hook_in:
                op: project
                scale_factor: 10.0
                use_intervention_tensor_as_basis: true
```

When an explicit `interventions` mapping is not supplied, the notebook wrapper falls back to the shorthand fields and
lets `resolve_interventions(...)` derive the final payload using its standard precedence rules.

## Current storage contract

The analysis op stores both a compact JSON payload and primitive Arrow-safe columns:

- `intervention_config`: JSON-serialized config summary
- `intervention_specs_json`: JSON-serialized canonical tuple payload
- `intervention_layers`, `intervention_positions`, `intervention_feature_ids`, `intervention_values`: primitive summary columns for downstream filtering and inspection
- `feature_intervention_dict` / `feature_intervention_dict_json`: hydrate-ready intervention mapping payloads
- `intervention_base_values`, `intervention_scale_factors`, `intervention_score_values`: per-feature scaling provenance
- `pre_intervention_logits`, `post_intervention_logits`, `logit_diff`: forward-only comparison outputs
- `intervention_activation_cache` (optional): captured activations when `return_activations` is enabled

The circuit-tracer analysis backend hydrates `intervention_specs_json` back into canonical tuple lists when an
`AnalysisStore` row or batch is formatted with `analysis_backend=DEFAULT_CT_ANALYSIS_BACKEND`.

## Missing-feature constrained selection

`extract_top_features` now supports constrained selections that refer to `(layer, feature_id)` pairs not present in
the original attribution graph rows. When a requested feature is missing, the op synthesizes candidate rows across the
observed positions for that layer, or across all observed positions when the layer has no active rows. Synthetic rows
inherit score baselines from same-layer rows when available and fall back to global means otherwise.

For activation-derived interventions, the op also carries forward optional activation overrides keyed by
`(layer, feature_id)`. When an override is not provided, synthetic rows use the mean activation of same-layer active
rows when available and the global mean activation otherwise. This lets `feature_intervention_forward` intervene on
requested features that were absent from the original graph without degrading to a zero activation heuristic.

When constrained feature selection is active, top-feature ranking now preserves at least one highest-scoring row per
requested `(layer, feature_id)` pair before returning the final top-feature payload. This prevents repeated positions
for one requested feature from crowding out a second requested feature before downstream intervention tuple
construction.

## Scope boundary

Interpretune currently exposes forward-only intervention analysis at the analysis-op layer.

`ReplacementModel.feature_intervention_generate(...)` remains a model-level API and is not yet exposed as a first-class
Interpretune analysis op. That follow-up should land as a separate generation-oriented intervention task once the
forward-only storage, config, and validation story is stable.

## Testing guidance

Cross-backend parity for intervention behavior should follow the same resource-aware testing direction already used for other expensive analysis flows:

- prefer shared extraction/caching helpers from `tests/analysis_resource_utils.py`
- prefer focused parity comparisons over blanket standalone isolation
- reserve standalone-heavy GPU coverage for cases where backend/runtime constraints actually require it


## Explicit position sets

A hook payload can set `position_scope: "positions"` and `positions: [9, 10]` to edit selected columns of the
activation tensor. Indices are distinct, non-negative and zero-based in the **padded batch tensor**. Left padding
is included in that coordinate system: an index never shifts according to an example's attention mask. To refer
to real tokens in differently padded examples, prepare batches whose selected columns have the intended meaning.

The intervention tensor has one leading operand row for each listed position, in the same order. For example:

```python
payload = {
    "blocks.18.hook_resid_post": {
        "op": "add",
        "position_scope": "positions",
        "positions": [9, 10],
        "intervention_tensor": torch.stack([city_delta, answer_delta]),
    }
}
it.model_fwd_intervention(..., interventions=payload)
```

The remaining operand axes have their usual meaning. A patch therefore supplies one pair of lens vectors per
position; a rejection supplies one complete span per position. Even a shared operand is stacked explicitly, so a
position axis cannot be mistaken for a basis axis. Position counts, operand shapes and bounds are checked before
any position is changed. The same payload can edit gradients through `model_bwd_intervention` on a backend that
declares that flow. A backend without `positions` in its support record refuses the payload.

Venngram recipes with `tokens: {positions: [...]}` translate to this scope. A recipe's operand is shared over its
token set, so its translator repeats the operand rows explicitly; different operands can be expressed as separate
stages. A raw hook payload can carry different rows in one spec. The op shorthand names the indices through
`intervention_positions` alongside `intervention_position_scope`.
