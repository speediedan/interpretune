# Tuning: interventions

[Tuning](tuning.md) describes how every composition declares and attaches its finetuning-scheduler view.

An *intervention* applies an op (`add`, `scale`, `reject`, `patch`, ...: venngram's op registry) to a site's activations, as [Interpretune Intervention APIs](interpretune_intervention_apis.md) describes. Kinds of intervention differ in the op and in where its operand comes from: a constant or a dose (a zero ablation), a value computed from the model (a concept direction, another run's activation), a [latent model](tuning_latent_models.md)'s reconstruction, or parameters of its own. `interpretune.tuning.interventions` is the seam for the last kind, *trainable* interventions, whose operand a schedule tunes. A steering vector is the `add` op with a learned vector; a learned per-unit gain is `scale`. Any trainable intervention can be attached at a site, on any backend but circuit-tracer (below), and scheduled by role.

## The seam

- **`OpIntervention(op, operand, scale_factor=1.0)`**: applies `op` at every position with the operand's tensor (times `scale_factor`), through the same op math the analysis ops use (`apply_op`, its out-of-place form, so gradients reach the operand). `operand` is a module returning the tensor, or a shape (an `int` for one vector over the site's last axis, or a tuple such as `(n_heads, d_head)` for a per-head operand), for a `LearnedOperand` that starts where the op is the identity (`0` for `add`, `1 / scale_factor` for `scale`), so attaching it changes nothing until it trains. The ops applied so far are `add` and `scale`; other registered ops are refused.
- **A steering vector** is `OpIntervention("add", d_model)`: its learned direction is the operand's tensor, `operand.value`.
- **`attach_intervention(root, name, intervention, site, frozen=False)`**: registers the intervention as `interventions.<name>` on the module (so it is in the module's state dict and reachable by its view) and splices it in with a forward hook on the module at `site`, whose output (the first element of a tuple output) becomes `intervention(output)`. `detach_intervention(root, name)` removes it.
- **Sites** are venngram site addresses, which name a site once for every backend: `point:hook_resid_post & layers:12` attaches at block 12's residual output on each of them. A module name under the interpretune module also works, but names the site on one composition only: `model.model.layers.12` on a core module, `hf_model.model.layers.12` on an nnsight module, or a TransformerLens bridge's hook point such as `model.blocks.12.hook_out`. Interpretune resolves an address on the composition (`interpretune.tuning.sites.resolve_site`): a TransformerLens composition through the hook points it exposes, in any spelling the activation-point vocabulary knows, and a core/HF or nnsight composition through its architecture's component map, under the model PEFT wraps when `lora_cfg` wraps it. An address that resolves to a module's input (`point:hook_resid_pre & layers:12`, block 12's residual input) is rewritten before that module runs. An address must select exactly one site of an activation's value, and a point the composition cannot place is refused, naming it. An address may reference a named set of an address book (`@resid13`), so one shared book names the site on every backend: pass the book's sets (`interpretune.tuning.sites.book_sets(book)`) as `attach_intervention(..., sets=)`, or give a configured intervention its book (`InterventionCfg(site="@resid13", book="steering.yaml")`, a local path or a venngram store reference). A book entry that names units (`steer: "point:mlp.out.hook_in & layers:0 & units:[1, 6]"`) gives the splice its units too: `@steer` configures the same two-unit `UnitIntervention` as the literal address. The shipped configurations, examples and notebooks still to move to addresses are tracked in [#876](https://github.com/interpretune/interpretune/issues/876).
- **Roles:** an attached intervention's tensors take venngram's role `intervention:<name>`, as a PEFT adapter's take `adapter:<name>`. A schedule selects one intervention with `role:intervention:<name>` and every intervention with `role:intervention`. Every interpretune module declares these roles in its cooperative `tunable_roles()`, which its venngram view reads, so no mixin is needed. A frozen intervention takes a role that says whether its value is saved: `frozen="fitted"` for one computed in the run (a mean-difference direction, say), which Fine-Tuning Scheduler checkpoints and never optimizes, so a resumed run keeps it; `frozen="artifact"` for one loaded from a published or saved source, which is reloaded from that source rather than checkpointed, and which a later stage may promote to `latent:<name>`. Both are applied in every forward pass and never trained. `frozen=True` is refused, because it does not say which.
- **Placement:** an attached intervention takes the layer and component of the module at its site, so `role:intervention & layers:12:` selects the interventions spliced at layers 12 and above. Interpretune modules declare these placements with those of their [latent models](tuning_latent_models.md), in their cooperative `tunable_placements()`.

Interventions without parameters (an ablation, a patch, a dose of a concept direction) need no role and no seam: the analysis ops apply them during a pass.

**On a circuit-tracer composition** an intervention's role is merged with the transcoders' (only a conflicting declaration is refused), its site must be a module of the model circuit-tracer runs, and attribution sees it in the forward pass but never reaches its gradient: [Tuning: circuit-tracer](tuning_circuit_tracer.md#interventions-and-other-latent-models-beside-the-transcoders). circuit-tracer's own interventions (`feature_intervention_forward`, and `model_fwd_intervention` on concept directions; [circuit-tracer backend support](circuit_tracer_backend_support.md)) carry no parameters, so they need no role.

## From a configuration

Every interpretune module attaches the interventions its configuration declares (`tuning_interventions`, a list of `InterventionCfg`) once its model is built and every module a site may name is registered, before its view is attached. So a steered run needs no module code, on any backend and on either training-loop interface:

```yaml
module_cfg:
  class_path: interpretune.config.module.ITConfig
  init_args:
    tuning_interventions:
      - name: steer
        site: "point:hook_resid_post & layers:12"  # block 12's residual output, on every backend
```

`op` is `add` (the default, a steering vector) or `scale`, `shape` defaults to the model's hidden size (`config.hidden_size`, or a TransformerLens model's `cfg.d_model`), and `scale_factor` is a fixed multiplier on the operand. A configured intervention is always tunable: one attached frozen would start at the identity and never change. To hold a trained operand fixed, refreeze it in a later stage (below). To apply a precomputed one, attach it in code with `frozen="fitted"` for a value computed in the run or `frozen="artifact"` for one loaded from a saved source.

## Selected units

A configured site address may select explicit units as well as one site:

```yaml
tuning_interventions:
  - name: selected_neurons
    site: "point:mlp.out.hook_in & layers:12 & units:[431, 1007]"
    op: add
```

This attaches a `UnitIntervention` with two learned values. Its operand starts at the identity, trains under
`role:intervention:selected_neurons`, and changes only those two units. It does not allocate or optimize a full-width
vector. `scale` learns gains the same way. A configured unit splice names one layer and explicit nonnegative indices;
use one entry per layer. The actual activation width is checked during the forward pass, rather than guessed from the
largest selected index. A `shape`, if supplied, must agree with the selected count.

In code, `UnitIntervention(op, units, operand=...)` also accepts a fixed tensor or a module returning its operand.
`replace` needs an explicit operand. `clamp` takes bounds and no operand, and bounds the selected model coordinates.
Attach fixed interventions with the appropriate frozen role. A unit address on `attach_intervention` must name the
same units as its `UnitIntervention`; attaching a full-width intervention to such an address is refused.

All four modes share the analysis surface's selected-coordinate executor and preserve the complement's autograd path.
See [direct unit payloads](interpretune_intervention_apis.md#direct-unit-payloads) for analysis and normalized doses.

A `UnitIntervention` acts on the selected units' activations at its site. To tune the parameters that produce those
units instead, use a unit adapter (`attach_unit_adapter`, under
[parameter targets](interpretune_intervention_apis.md#parameter-targets)): a trainable delta over the units' weight
slices, which applies at every token and is part of the tensor an export writes.

## Attaching them in code

Attach interventions in `register_tunable_modules()`, which runs once the model exists and before the view is built:

```python
from interpretune.tuning.interventions import OpIntervention, attach_intervention


class MySteeredModule(MyModule):
    def register_tunable_modules(self) -> None:
        super().register_tunable_modules()
        d_model = self.model.config.hidden_size
        attach_intervention(self, "steer", OpIntervention("add", d_model), "point:hook_resid_post & layers:12")
```

## Schedules: phases as stages

finetuning-scheduler's per-phase `refreeze` and `restore` keys let each phase be a stage of the experiment, rather than a step of gradually thawing more. Here the steering vector trains alone. Then a second stage holds it fixed and tunes the last four blocks, starting from where the steering stage ended:

```yaml
0:
  params: ["role:intervention:steer"]
  max_transition_epoch: 2
1:
  params: ["role:base & layers:-4:"]
  refreeze: ["role:intervention:steer"]
  restore: none
```

`layers:-4:` needs the layer structure a venngram component map gives the module's view. Where no published map describes the model, the view addresses tensors by name and role only, and a stage names its blocks by regex instead (for example `model\.model\.layers\.(2[4-7])\..*`). `refreeze` and `restore` run on the core loop; the Lightning callback refuses both keys.
