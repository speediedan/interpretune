"""Unit tests for Hub-runnable model-spec resolution (interpretune#498 slice 1)."""

from __future__ import annotations

import pytest

from interpretune.harness.sessions import ExperimentDataModule, HubModelSpec, resolve_model_spec


def test_known_spec_resolves_with_composition():
    spec = resolve_model_spec("gemma3", "1b_it")
    assert isinstance(spec, HubModelSpec)
    assert spec.model_name == "google/gemma-3-1b-it"
    assert spec.composition == ("core", "nnsight", "circuit_tracer")
    assert spec.use_chat_template is True


def test_unknown_selection_refused_by_name():
    with pytest.raises(ValueError, match="Unsupported model selection nope.nothing"):
        resolve_model_spec("nope", "nothing")


def test_overrides_replace_base_values():
    spec = resolve_model_spec("gemma3", "1b_it", model_name_override="org/other", use_chat_template_override=False)
    assert spec.model_name == "org/other"
    assert spec.use_chat_template is False
    assert spec.transcoder_set == "mwhanna/gemma-scope-2-1b-it/transcoder_all/width_16k_l0_small_affine"


def test_every_table_entry_carries_composition():
    from interpretune.harness.sessions import _hub_model_spec_registry

    registry = _hub_model_spec_registry()
    assert len(registry) >= 8
    for (family, variant), spec in registry.items():
        assert spec.composition, f"{family}.{variant} has no composition"
        assert all(isinstance(name, str) and name for name in spec.composition)


def test_experiment_datamodule_closure_over_base_gap():
    """The base declares no dataloader methods (the session protocol gap); the subclass adds refusing ones."""
    import types

    from interpretune.base.datamodules import ITDataModule

    for phase in ("train", "val", "test", "predict"):
        assert not hasattr(ITDataModule, f"{phase}_dataloader")
        method = getattr(ExperimentDataModule, f"{phase}_dataloader")
        assert callable(method)
        with pytest.raises(NotImplementedError, match=f"do not serve {phase} dataloaders"):
            method(types.SimpleNamespace())
