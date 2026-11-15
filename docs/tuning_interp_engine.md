# Tuning: interp-engine (hub adapter)

[Tuning](tuning.md) describes how every composition declares and attaches its Fine-Tuning Scheduler view.

The interp-engine hub adapter declares no view of its own, and needs none: its compositions receive Interpretune's
default view over the Interpretune module. For `(core, interp_engine)` that is the identity view, which fits with no
override, because the HF model stays at `module.model` and the adapter's `EagerModel` wraps it in place without
registering anything. Its base weights are frozen by default, which does not matter: the view takes roles from
`tunable_roles()` and never infers them from `requires_grad`.

The adapter measures this with Interpretune's tuning conformance. Every `TuningConformance` case that applies to the
view passes on CPU (gpt2): mapping completeness, the default schedule compiling, exact and deterministic thaws, a step
changing only thawed tensors, a checkpoint round trip, and an unmatched rule refusing. The cases are in its
[`tests/test_tuning_conformance.py`](https://github.com/speediedan/it-interp-engine-adapter/blob/main/tests/test_tuning_conformance.py),
with phased runs in
[`tests/test_tuning_phased.py`](https://github.com/speediedan/it-interp-engine-adapter/blob/main/tests/test_tuning_phased.py)
and its run configurations checked by
[`tests/test_cli_run_config.py`](https://github.com/speediedan/it-interp-engine-adapter/blob/main/tests/test_cli_run_config.py).

Its circuit-tracer composition places `BaseCircuitTracerModule` ahead of its own base, so it resolves to Interpretune's
[circuit-tracer view](tuning_circuit_tracer.md): the transcoders are `instrument` until promoted, a base-model
tensor is refused unless `CircuitTracerConfig.tune_base` is set, and with no replacement model loaded the
composition is refused. The replacement model is registered on the module once.

Parity with the packaged adapters (views, conformance and examples) is tracked in
[interpretune#815](https://github.com/speediedan/interpretune/issues/815).

The adapter ships an RTE tuning example as run configurations alone, in its
[`examples/rte_ft/`](https://github.com/speediedan/it-interp-engine-adapter/tree/main/examples/rte_ft) (gpt2 and
Qwen3-0.6B, on the core loop and on Lightning). Tuning examples for every adapter are tracked in
[interpretune#812](https://github.com/speediedan/interpretune/issues/812).
