# Tuning: interp-engine (hub adapter)

[Tuning](tuning.md) describes how every composition declares and attaches its Fine-Tuning Scheduler view.

The interp-engine hub adapter declares no view of its own yet, so its compositions receive Interpretune's default view
over the Interpretune module. That fits only where its model is reachable there under its own names, which
is still to be checked per composition. Its base weights are frozen by default, so roles must be declared through
`tunable_roles()`, not inferred from `requires_grad`.

Its circuit-tracer composition places `BaseCircuitTracerModule` ahead of its own base, so it resolves to Interpretune's
[circuit-tracer view](fts_tuning_circuit_tracer.md): the transcoders are `instrument` until promoted, a base-model
tensor is refused by name unless `CircuitTracerConfig.tune_base` is set, and with no replacement model loaded the
composition is refused. The replacement model is registered on the module once.

Parity with the packaged adapters (views, conformance and examples) is tracked in
[interpretune#815](https://github.com/speediedan/interpretune/issues/815).

Concrete example usage for this adapter is to come; it is tracked in [interpretune#812](https://github.com/speediedan/interpretune/issues/812).
