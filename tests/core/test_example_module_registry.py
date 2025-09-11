from it_examples.example_module_registry import LazyModuleRegistry


class DummyRegistry(dict):
    def available_keys(self):
        return list(self.keys())


def test_lazy_initialization_and_single_call(monkeypatch):
    calls = []

    def fake_create_registry(self):
        calls.append(1)
        # return a simple dict-like registry
        return DummyRegistry({"a": 1, "b": 2})

    monkeypatch.setattr(LazyModuleRegistry, "_create_registry", fake_create_registry)

    lm = LazyModuleRegistry()

    # Before access, create_registry shouldn't be called
    assert calls == []

    # Access a method that triggers initialization
    keys = lm.keys()
    assert set(keys) == {"a", "b"}
    assert calls == [1]

    # Subsequent access should not call _create_registry again
    _ = lm.available_keys()
    assert calls == [1]
