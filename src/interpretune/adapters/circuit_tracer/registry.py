"""The circuit-tracer backend rendezvous, deliberately free of the circuit-tracer import chain.

These two registries are the seam a third-party backend enters. They are pure data -- a name to a class
NAME, and a name to a callable -- so nothing here needs circuit-tracer, transformer_lens or nnsight, and
this module imports none of them.

**That is the whole point of the module existing.** `adapter.py` imports `circuit_tracer` eagerly at
module level, which is correct for a module whose classes subclass circuit-tracer's, but it means anything
living beside them inherits a hard dependency on an OPTIONAL package. Two things then follow that should
not:

* a hub-delivered adapter cannot register its backend name unless the consumer has circuit-tracer
  installed, even though registration itself needs nothing from it;
* :class:`~interpretune.adapters.circuit_tracer.config.CircuitTracerConfig` could not VALIDATE without
  circuit-tracer, because validation reads the registry.

Splitting the registries out fixes both by construction rather than by a guard, and it keeps the rule
easy to state: **if it needs circuit-tracer, it belongs in `adapter.py`; if it is the rendezvous, it
belongs here.**

Registering a backend NAME is not the same as registering a COMPOSITION. A composition that includes the
circuit-tracer adapter needs that adapter's class, so it genuinely requires circuit-tracer installed;
nothing here changes that, and it should not.
"""

from __future__ import annotations

from typing import Any, Callable

#: Registry mapping a circuit-tracer backend NAME to the ReplacementModel class name it must produce.
#:
#: This is the rendezvous, and it is extensible by assignment: a third-party adapter registers its
#: backend here rather than `adapter.py` growing a branch for it. Nothing compares a backend name to a
#: literal -- the registry answers "is this valid" and "what should it have produced".
CT_BACKEND_REGISTRY: dict[str, str] = {
    "transformerlens": "TransformerLensReplacementModel",
    "nnsight": "NNSightReplacementModel",
}

#: Backend NAME -> a callable returning the ``ModelBackend`` to attach for it, given the module.
#:
#: Separate from the validation registry above because the two answer different questions, and a backend
#: may legitimately register for one and not the other: a component whose module composition already
#: attaches its own model backend registers no factory here at all, and the attach in `adapter.py` leaves
#: it alone (see the "attach, do not override" note there).
#:
#: Each factory imports its framework LOCALLY, which is what keeps this module free of eager
#: transformer_lens / nnsight imports. circuit-tracer owns the two bundled pairings because it arrived
#: after both; a later adapter owns its own pairing and registers it from its own package.
CT_MODEL_BACKEND_FACTORIES: dict[str, Callable[[Any], Any]] = {}


def _tl_model_backend(_module: Any) -> Any:
    from interpretune.adapters.transformer_lens.backends import TLModelBackend

    return TLModelBackend()


def _nnsight_model_backend(module: Any) -> Any:
    from interpretune.adapters.nnsight.backends import NNsightModelBackend, get_default_configs_per_pass
    from interpretune.analysis.backends.hook_mapping import HookNameResolver

    hf_model = NNsightModelBackend._get_hf_model(module.model)
    hf_config = getattr(hf_model, "config", None)
    architectures = getattr(hf_config, "architectures", None) if hf_config else None
    model_arch = architectures[0] if architectures else type(hf_model).__name__
    backend = NNsightModelBackend(HookNameResolver(model_arch), configs_per_pass=get_default_configs_per_pass())
    backend.register_model_hooks(module.model)
    return backend


CT_MODEL_BACKEND_FACTORIES["transformerlens"] = _tl_model_backend
CT_MODEL_BACKEND_FACTORIES["nnsight"] = _nnsight_model_backend
