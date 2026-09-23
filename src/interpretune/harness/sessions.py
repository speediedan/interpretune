"""Hub-runnable model-spec resolution and session construction for notebook experiments.

The notebook-harness session layer historically lived in the wheel-excluded test
tree; Hub-published experiments cannot reach it. This module carries the portable
half of that layer: resolving a model family/variant to its spec (model ids,
transcoder set, Neuronpedia coordinates, adapter composition) from the core
``harness/configs/model_specs.yaml``, plus a minimal datamodule for sessions
whose pipelines build their own batches. Full public session assembly (module
side) is the follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from interpretune.base.datamodules import ITDataModule


@dataclass(frozen=True)
class HubModelSpec:
    """Test-free model spec for Hub experiment sessions.

    Mirrors the fields experiment pipelines consume, with the test-only config alias replaced by the adapter composition
    the public registry resolves.
    """

    family: str
    variant: str
    model_name: str
    transcoder_set: str
    neuronpedia_model: str
    neuronpedia_set: str
    use_chat_template: bool
    composition: tuple[str, ...]
    hf_model_head: str | None = None
    nnsight_overrides: dict[str, Any] | None = None
    circuit_tracer_overrides: dict[str, Any] | None = None


def _specs_path() -> Path:
    return Path(__file__).resolve().parent / "configs" / "model_specs.yaml"


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping config at {path}")
    return payload


@lru_cache(maxsize=1)
def _hub_model_spec_registry() -> dict[tuple[str, str], HubModelSpec]:
    payload = _load_yaml_mapping(_specs_path())
    raw_specs = payload.get("model_specs")
    if not isinstance(raw_specs, dict):
        raise ValueError(f"Expected 'model_specs' mapping in {_specs_path()}")
    registry: dict[tuple[str, str], HubModelSpec] = {}
    for raw_key, raw_value in raw_specs.items():
        if not isinstance(raw_value, dict):
            raise ValueError(f"Model spec '{raw_key}' must map to a dictionary")
        family, _, variant = str(raw_key).partition(".")
        if not family or not variant:
            raise ValueError(f"Model spec key '{raw_key}' must use 'family.variant' format")
        composition = raw_value.get("composition")
        if (
            not isinstance(composition, list)
            or not composition
            or not all(isinstance(name, str) and name for name in composition)
        ):
            raise ValueError(
                f"Model spec '{raw_key}' must declare a non-empty `composition` list of "
                "adapter names (the public-registry counterpart of the test config alias)."
            )
        registry[(family, variant)] = HubModelSpec(
            family=family,
            variant=variant,
            model_name=str(raw_value["model_name"]),
            transcoder_set=str(raw_value["transcoder_set"]),
            neuronpedia_model=str(raw_value["neuronpedia_model"]),
            neuronpedia_set=str(raw_value["neuronpedia_set"]),
            use_chat_template=bool(raw_value["use_chat_template"]),
            composition=tuple(composition),
            hf_model_head=raw_value.get("hf_model_head"),
            nnsight_overrides=raw_value.get("nnsight_overrides"),
            circuit_tracer_overrides=raw_value.get("circuit_tracer_overrides"),
        )
    return registry


def resolve_model_spec(
    model_family: str,
    model_variant: str,
    *,
    model_name_override: str | None = None,
    transcoder_set_override: str | None = None,
    neuronpedia_model_override: str | None = None,
    neuronpedia_set_override: str | None = None,
    use_chat_template_override: bool | None = None,
) -> HubModelSpec:
    """Resolve a model family/variant to its Hub-runnable spec, refusing unknown selections by name."""
    try:
        spec = _hub_model_spec_registry()[(model_family, model_variant)]
    except KeyError as exc:
        supported = ", ".join(f"{family}.{variant}" for family, variant in sorted(_hub_model_spec_registry()))
        raise ValueError(f"Unsupported model selection {model_family}.{model_variant}. Supported: {supported}") from exc
    return HubModelSpec(
        family=spec.family,
        variant=spec.variant,
        model_name=model_name_override or spec.model_name,
        transcoder_set=transcoder_set_override or spec.transcoder_set,
        neuronpedia_model=neuronpedia_model_override or spec.neuronpedia_model,
        neuronpedia_set=neuronpedia_set_override or spec.neuronpedia_set,
        use_chat_template=spec.use_chat_template if use_chat_template_override is None else use_chat_template_override,
        composition=spec.composition,
        hf_model_head=spec.hf_model_head,
        nnsight_overrides=spec.nnsight_overrides,
        circuit_tracer_overrides=spec.circuit_tracer_overrides,
    )


def _refusing_dataloader(phase: str):
    """Build a dataloader method that refuses by name: experiment pipelines build batches manually."""

    def _loader(self):
        raise NotImplementedError(
            f"Hub experiment sessions do not serve {phase} dataloaders: pipelines build "
            "prompt batches manually and only need the datamodule's tokenizer handle. "
            f"Calling {phase}_dataloader here is a programming error."
        )

    _loader.__name__ = f"{phase}_dataloader"
    return _loader


class ExperimentDataModule(ITDataModule):
    """Minimal datamodule for Hub experiment sessions: tokenizer handle, no dataloaders.

    The base class configures the tokenizer but declares no dataloader methods, so a
    bare instance fails the session's protocol check. Experiment pipelines build
    their own prompt batches and never iterate dataloaders; the methods exist only
    to satisfy the structural check and refuse loudly if ever called.
    """

    train_dataloader = _refusing_dataloader("train")
    val_dataloader = _refusing_dataloader("val")
    test_dataloader = _refusing_dataloader("test")
    predict_dataloader = _refusing_dataloader("predict")
