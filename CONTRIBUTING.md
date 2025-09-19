# Contributing to Interpretune

> **Note — pre‑MVP:** Interpretune is currently in pre‑MVP stage. We expect to reach an MVP soon and begin accepting external PRs. If you're preparing contributions, please open issues or draft PRs now; maintainers will review them as we transition to the public MVP.

Thank you for contributing to Interpretune.

## Import-time policy

Policy summary
- The top-level package import (`import interpretune`) should only import small, essential core modules and must not eagerly import heavy adapter/extension packages (for example: `transformer_lens`, `lightning`, `neuronpedia`, `circuit_tracer`, etc.) that would otherwise dominate cold startup time.

How to follow this policy
1. Do not add top-level imports of heavy third-party adapter/extension packages inside `src/interpretune/__init__.py` or other package-level modules.
2. Prefer centralized helpers for minimal registration and deferred import logic. This repository includes `src/interpretune/adapters/_light_register.py` as a small, centralized place for minimal, low-overhead registration logic. If you need more general deferred import behavior, implement a single deferred-import helper (import-proxy) rather than sprinkling localized imports across many files.


Profiling and diagnosing import-time issues
- We have profiling and fixture-analysis tooling to make it easy to diagnose testing hotspots (e.g. including import-time analysis):
  - `tests/PROFILING.md` documents capturing CPU profiles with `py-spy` and viewing results in `speedscope`.
  - `scripts/speedscope_top_packages.py` helps summarize CPU time by package from speedscope JSON files.
  - `tests/dynamic_fixture_benchmark.py` enumerates pytest fixtures and can be used to micro-benchmark fixture setup costs.
- Use these tools if you think an import is unexpectedly expensive. A typical workflow is:
  1. Reproduce the import path or fixture that looks slow.
  2. Run `py-spy` to capture a short CPU profile while the import runs.
  3. Open the resulting speedscope JSON (or use the helper script) to find the heavy package call stacks.

- We also provide an automated test that guards against accidental eager imports and enforces a fast package import time: `tests/core/test_import_time_and_adapters.py`.

Recommended process for larger deferred-import work
1. Prototype a centralized deferred-importer (import-proxy) in a feature branch.
2. Add unit and integration tests that confirm both lazy and eager code paths behave as expected.
3. Run fixture benchmarks and a py-spy capture to ensure the change actually improves import time and does not introduce ordering regressions.
4. Keep the implementation small and well-documented; prefer explicitness over surprising implicit side-effects.

Contact
- If you're unsure whether an import should be deferred, or how to centralize it safely, open an issue or a draft PR and tag a maintainer for guidance.

Thanks for helping keep Interpretune fast and maintainable!

## Object summary helpers for __repr__

Motivation
- We want compact, safe, and informative `__repr__` output for important Interpretune objects (for example `ITState`, modules, and sessions) without accidentally materializing large objects (tensors, datasets, large configuration classes) or invoking heavy initialization. A small centralized helper ensures a consistent policy and reduces duplication.

Where to look
- The centralized helpers live in `src/interpretune/utils/repr_helpers.py`.

What it provides
- `summarize_obj(obj)` — a small, non‑materializing summary for common value types (tensors summarized by shape/dtype/device, dicts by keys/length, lists/tuples/sets by length, strings truncated). Custom objects are rendered as `ClassName(...)` to indicate a stateful object.
- `state_to_dict(obj, custom_key_transforms=...)` — reads `obj._obj_summ_map` (a dict mapping attribute name -> label) and returns a JSON‑serializable dict of summaries. This is the canonical way to build a safe inspectable dict for an object.
- `state_to_summary(state_dict, obj)` — builds a compact parenthesized single‑line summary using the ordering and labels defined in `obj._obj_summ_map`.
- `state_repr(summary, class_name)` — formats the final `ClassName(...)` string given the parenthesized summary.

How to use it (recommended)
1. Define a per‑class `_obj_summ_map` dictionary mapping attribute names to labels. Keep keys as actual attribute names and labels as the human‑facing name to display in summaries. Example:

```python
_obj_summ_map = {
  "_device": "device",
  "_current_epoch": "epoch",
  "_global_step": "step",
  "_init_hparams": "init_cfg",
}
```

2. Use `state_to_dict(self, custom_key_transforms={...})` in your `to_dict()` implementation. Provide `custom_key_transforms` for keys that need bespoke handling (for example, summarizing nested dicts or representing extension state):

```python
def to_dict(self):
  return state_to_dict(self, custom_key_transforms={"_init_hparams": self._init_hparams_transform})
```

3. In `__repr__()`, build the compact summary via `state_to_summary(d, self)` and format with `state_repr(...)`:

```python
# in __repr__()
inner = state_to_summary(self.to_dict(), self)
return state_repr(inner, self.__class__.__name__)
```

Custom transforms
- `state_to_dict` supports a `custom_key_transforms` mapping from attribute name (the dict key in `_obj_summ_map`) to a callable that takes the raw attribute value and returns a safe serializable summary. Typical use cases:
  - Truncate long strings
  - Represent nested dictionaries as `"{...}"` to avoid printing huge nested content
  - Represent custom config objects with a short string like `MyCfg(...)`

Example transform:

```python
# Custom transform for summarizing the `_init_hparams` mapping in ITState
@staticmethod
def _init_hparams_transform(v):
  if not v:
    return {}
  out = {}
  for k, val in v.items():
    if isinstance(val, dict):
      out[k] = "{...}" if val else {}
    else:
      out[k] = summarize_obj(val)
  return out
```

Notes
- Keep the transforms cheap and side‑effect free. Never call into expensive setup code or materialize tensors — prefer `summarize_obj` for safety.
- This mechanism is intentionally conservative: it is meant for developer‑facing summaries and debugging, not for full serialization of state.

## Interpretune Core Class-level Internal Metadata (`_it_cls_metadata`)

To maximize compositional flexibility while reducing noise when inspecting/debugging core Interpretune classes, we make the set of internal, implementation-focused class attributes explicit and consolidate related class-level constants into a single protected container named `_it_cls_metadata` on core classes (for example `ITSession`, `PropertyDispatcher` implementations, and adapter classes).

The container is a typed, frozen dataclass (`ITClassMetadata`) held as a protected class attribute. It centralizes values like attribute-mapping tables, composition target names, and small tuples of internal method names.

### Utility and benefits
- Cleaner debugging: developers inspecting a class see a single protected attribute instead of many scattered constants.
- Safer extension: the dataclass is frozen to discourage accidental reassignment, while nested mutable containers (dicts) may be mutated in-place by adapters or tests when needed.
- Better discoverability: tooling and docs can point to a single symbol to explain how to customize class-level behavior.

### How to use
- The container is intended to be overridden on subclasses when custom behavior is required. Prefer replacing the entire `_it_cls_metadata` with a new `ITClassMetadata` instance in subclass definitions rather than mutating at import time from unrelated modules.

Example:

```py
from interpretune.base.metadata import ITClassMetadata
from interpretune.protocol import Adapter

# Define the adapter and its metadata directly, following the same
# pattern used by built-in adapters (for example `LightningAdapter`).
class MyFrameworkAdapter:
  _it_cls_metadata = ITClassMetadata(
    core_to_framework_attrs_map={
      # Map internal core names to framework attribute paths/defaults/messages
      "_it_optimizers": ("trainer.optimizers", None, "No optimizers have been set yet."),
      "_current_epoch": ("trainer.current_epoch", 0, "No epoch available"),
      # Add framework-specific helpers or custom mappings here
      "_custom_hook": ("trainer.custom_hook", None, "Custom hook not registered"),
    },
    property_composition={
      # Example: expose a `device` property that dispatches to a framework mixin
      "device": {"enabled": True, "target": object, "dispatch": lambda self: "device"},
    },
  )

  @classmethod
  def register_adapter_ctx(cls, adapter_ctx_registry):
    adapter_ctx_registry.register(
      Adapter.some_framework,
      component_key="module",
      adapter_combination=(Adapter.some_framework,),
      composition_classes=(MyFrameworkAdapter,),
      description="Adapter for SomeFramework",
    )

  # Instances of classes composed with MyFrameworkAdapter will see the
  # adapter-supplied mappings via `type(obj)._it_cls_metadata.core_to_framework_attrs_map`.
```

Note: when customizing an existing base adapter (instead of defining a new
adapter from scratch), prefer using `dataclasses.replace()` on the base
adapter's `_it_cls_metadata` to preserve unrelated fields on the frozen
dataclass. Constructing metadata directly (as shown above) is the common
pattern for adding new adapters; customizing an existing adapter is a
secondary workflow and `replace()` makes it safe and explicit.

Notes and recommendations
- Prefer overriding `_it_cls_metadata` at the class definition site. This makes the intent explicit and avoids cross-module side effects.
- If you must patch behavior in tests, mutate nested containers (for example `type(obj)._it_cls_metadata.core_to_framework_attrs_map[...] = ...`) rather than replacing the dataclass instance — the latter is permitted but can make test isolation harder if not reverted.
- Document any non-obvious overrides in your module's docstring so users can easily understand the intent.
