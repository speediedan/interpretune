# Tuning: interp-engine (hub adapter)

[Tuning](fts_tunable_views.md) describes how every composition declares and attaches its finetuning-scheduler view.

The interp-engine hub adapter declares no view of its own yet, so its compositions receive interpretune's default
identity view over the interpretune module. That fits only where its model is reachable there under its own names, which
is still to be checked per composition. Its base weights are frozen by default, so roles must be declared through
`tunable_roles()`, not inferred from `requires_grad`. Its circuit-tracer composition already takes interpretune's
circuit-tracer refusal, because it places `BaseCircuitTracerModule` ahead of its own base; it also registers the
replacement model twice on the module, which circuit-tracer tuning will have to resolve. Parity with the packaged adapters (views,
conformance and examples) is tracked in [interpretune#815](https://github.com/speediedan/interpretune/issues/815).

Concrete example usage for this adapter is to come; it is tracked in [interpretune#812](https://github.com/speediedan/interpretune/issues/812).
