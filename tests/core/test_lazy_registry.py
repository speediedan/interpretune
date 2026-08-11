"""LazyModuleRegistry facade behavior (builder + hydrator modes) — core machinery, fixture-tree data."""

from pathlib import Path

import it_examples
from interpretune.registry import LazyModuleRegistry
from tests.module_registry import make_component_tree_registry

EXAMPLES_ROOT = Path(it_examples.__file__).parent / "examples"


class DummyRegistry(dict):
    def available_keys(self):
        return list(self.keys())


def test_lazy_initialization_and_single_call():
    calls = []

    def builder():
        calls.append(1)
        # return a simple dict-like registry
        return DummyRegistry({"a": 1, "b": 2})

    lm = LazyModuleRegistry(builder=builder)

    # Before access, the builder shouldn't be called
    assert calls == []

    # Access a method that triggers initialization
    keys = lm.keys()
    assert set(keys) == {"a", "b"}
    assert calls == [1]

    # Subsequent access should not call the builder again
    _ = lm.available_keys()
    assert calls == [1]


def test_hydrator_mode_defers_and_indexes_without_construction():
    lm = make_component_tree_registry(EXAMPLES_ROOT)

    # instantiating the facade must not parse manifests or construct entries
    assert lm._hydrator is not None
    assert lm._hydrator._index is None
    assert lm._hydrator._hydrated == set()

    # membership via the manifest index constructs nothing
    assert "rte_demo.gemma2.circuit_tracer" in lm
    assert lm._hydrator._hydrated == set()
