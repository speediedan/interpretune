# The Module Contract, and Its Limits

**Audience:** users writing their own `ITModule` subclasses or composing modules with adapters

Interpretune deliberately does not require your module to inherit from a single abstract base class.
Instead, the contract a module must satisfy is *structural*: it is expressed as runtime-checkable
protocols in `src/interpretune/protocol.py` and validated when an `ITSession` composes your module
with its adapters. This page states that contract explicitly, describes which `BaseITModule` methods
are required, overridable, or cooperative, and is honest about what the framework does *not* check,
so you know what to expect before something surprises you.

## What a module must provide

A composed module satisfies `ITModuleProtocol` when it exposes **all** of the invariants and **at
least one** step method:

**Invariants** (from `ModuleInvariants`):

- `it_cfg`: the module's `ITConfig`
- `analysis_backend` (property): the attached analysis backend, or `None`
- `setup(*args, **kwargs)`
- `configure_optimizers()`

**At least one step method** (from `ModuleSteppable`):

- `training_step`, `validation_step`, `test_step`, or `predict_step`

`BaseITModule` supplies the invariants; it deliberately does **not** supply default step
implementations. A module with no step methods has no defined execution semantics, and a silently
inherited no-op step would turn a missing implementation into a run that "succeeds" while doing
nothing. You (or a mixin such as `ClassificationMixin`, or an adapter) must define the step methods
for the phases you intend to run.

The datamodule contract is the same shape: `ITDataModuleProtocol` requires the datamodule invariants
plus at least one of `train_dataloader`, `val_dataloader`, `test_dataloader`, or
`predict_dataloader`.

### How the "at least one" semantics are generated

The protocol variants are generated rather than hand-enumerated: `gen_protocol_variants()` composes
the invariant protocol with each step protocol (`TrainSteppable`, `ValidationSteppable`, ...) and
returns their union, so `isinstance(module, ITModuleProtocol)` is true exactly when the invariants
hold and *any* step protocol is satisfied. This keeps "which phases exist" defined in one place
(`AllPhases` and the step protocols) instead of in a combinatorial explosion of ABCs. The same
mechanism produces `ITDataModuleProtocol` from the loader protocols.

## The `BaseITModule` lifecycle, and which methods are yours

`BaseITModule.__init__` runs a fixed initialization sequence:

1. `_before_it_cfg_init(it_cfg)` — last chance to adjust configuration before it is bound
2. `_make_init_dirs()` / `_connect_extensions()` — internal: directories and extensions
3. `_dispatch_model_init()` — internal dispatcher, which calls in order:
   - `auto_model_init()` — configuration-driven model init (e.g. `hf_from_pretrained_cfg`);
     adapters override this to construct their own model types
   - `post_auto_model_init()` — if `auto_model_init` produced a model, adjust it here
   - `model_init()` — if it did **not**, load a custom model here
   - `_capture_hyperparameters()` then `load_metric()`

Methods by contract role:

| Role | Methods | Rule |
| --- | --- | --- |
| **Required by protocol** | at least one `*_step` | define it yourself or inherit it from a mixin/adapter; there is no default |
| **Template hooks** (override freely, defaults are no-ops or config-driven) | `_before_it_cfg_init`, `auto_model_init`, `post_auto_model_init`, `model_init`, `load_metric` | no `super()` call needed unless you want the default behavior too |
| **Cooperative hooks** (override, but call `super()`) | `setup`, `configure_optimizers`, `on_session_end`, `on_train_end`, `on_test_end`, `on_predict_end` | the base implementations do real work (datamodule wiring, optimizer instantiation from `it_cfg`, memprofiler dump, session-completion state); skipping `super()` silently drops it |
| **Internal** (do not override) | `_dispatch_model_init`, `_capture_hyperparameters`, `_make_init_dirs`, `_connect_extensions` | these define the lifecycle other components rely on |

`forward()` delegates to `self.model(*args, **kwargs, **it_cfg.cust_fwd_kwargs)`; override it only
when your module genuinely needs different call semantics than its wrapped model.

Session-end hooks fire once per session: `on_train_end` / `on_test_end` / `on_predict_end` each call
`on_session_end` only if the session has not already completed, because most training sessions run
both a fit and an evaluation loop and the hook must not run twice.

## Limitations: what the framework does not do for you

These are deliberate, documented behaviors of the current design, not bugs. Several stem from the
pre-MVP flexibility bias recorded in the
[protocol architecture working design](../design/protocol_architecture_working_design.md).

1. **Protocol checks validate presence, not signatures or behavior.** `isinstance` against a
   runtime-checkable protocol confirms an attribute or method *exists*; it cannot confirm the
   signature, return type, or semantics. A `test_step` with the wrong signature passes composition
   and fails at execution time.

2. **Session validation warns; it does not raise.** When `ITSession` composes your module and the
   result still does not satisfy `ITModuleProtocol` (or the datamodule its protocol), you get a
   warning naming the original class and the adapters composed, not an exception. Erring toward
   flexibility is intentional at this stage: an object that fails the structural check may still be
   perfectly runnable for the phase you care about. Treat the warning as a contract violation to
   fix, not as noise.

3. **Framework-agnostic module definitions should stay framework-agnostic.** Task-step definitions
   (e.g. `RTEBoolqSteps`) should not contain framework-specific hooks or accumulation logic; use
   `ClassificationMixin` for prediction accumulation and metric reporting, and rely on
   `_call_itmodule_hook(..., optional=True)` dispatch to tolerate absent hooks. Logging is uniform
   by design: `self.log()` / `self.log_dict()` work in both core and Lightning contexts, but the
   *destination* differs (core accumulates in `_logged_metrics` and prints averages at test epoch
   end; Lightning routes through its own logger connectors).

4. **Adapter composition is MRO-based, and order is canonicalized.** Adapters enrich your class by
   method resolution order, not by wrapping. The practical consequences: `adapter_ctx` combinations
   are canonicalized so `(sae_lens, core)` and `(core, sae_lens)` name the same composition; an
   adapter can shadow a method you defined; and cooperative `super()` calls are what keep the chain
   intact. See the [adapter development guide](adapter_development_guide.md) for the
   composition rules.

5. **`configure_optimizers` is optional in core context.** Core interpretune sessions only need it
   when a phase actually optimizes; some adapter contexts (Lightning) require it. The base
   implementation instantiates optimizer and scheduler from `it_cfg` if configured, so most modules
   never override it.

6. **Sessions are still biased toward a paired datamodule + module.** Module-only analysis and
   notebook-first workflows work but are not yet first-class; the open design questions are tracked
   in the working design document rather than silently absorbed here.

Pre-MVP caveat: interpretune is pre-MVP, and this contract can change without deprecation shims.
When it does, this page and the protocol definitions change together.
