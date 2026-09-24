"""Hub-runnable model-spec resolution and session construction for notebook experiments.

The notebook-harness session layer historically lived in the wheel-excluded test
tree; Hub-published experiments cannot reach it. This module carries the portable
half of that layer: resolving a model family/variant to its spec (model ids,
transcoder set, Neuronpedia coordinates, adapter composition) from the core
``harness/configs/model_specs.yaml``, a minimal datamodule for sessions whose
pipelines build their own batches, and public-API session assembly for the
circuit-tracer/nnsight composition the concept-direction experiments run.
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
    module_cls: str
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
        module_cls = raw_value.get("module_cls")
        if not isinstance(module_cls, str) or not module_cls:
            raise ValueError(
                f"Model spec '{raw_key}' must declare `module_cls`: the importable module "
                "class sessions compose for its composition."
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
            module_cls=module_cls,
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
        module_cls=spec.module_cls,
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

    def setup(self, stage: str | None = None, module: Any | None = None, *args: Any, **kwargs: Any) -> None:
        """Attach the module handle without loading any dataset.

        Overrides the base implementation, which asserts a configured dataset
        path and loads it from disk: experiment pipelines build their own prompt
        batches, so there is nothing to load. Anything reaching for
        ``self.dataset`` afterwards fails with a plain ``AttributeError``.
        """
        if module is not None:
            self._module = module


class NeutralSessionMixin:
    """Task-neutral auto-composition trigger for Hub experiment sessions.

    A plain marker class, deliberately not a dataclass: it must serve as a base for the synthesized (non-frozen) config
    dataclass, which cannot inherit from a frozen one. It carries no fields; it exists so config auto-composition
    materializes the adapter kwargs (circuit-tracer/nnsight sections) as fields on the synthesized class, instead of
    refusing them as unexpected keyword arguments. Task components pass their own mapping mixin; Hub experiments have no
    task.
    """


# Session kwargs the harness assembly consumes; anything else (debug presets,
# Neuronpedia upload toggles) stays caller-side and is named in debug output
# rather than silently dropped.
_CONSUMED_SESSION_KWARGS = frozenset(
    {
        "model_family",
        "model_variant",
        "model_name",
        "transcoder_set",
        "hf_model_head",
        "force_device",
        "batch_size",
        "max_feature_nodes",
    }
)


def build_session_body(
    spec: HubModelSpec,
    *,
    force_device: str | None = None,
    batch_size: int | None = None,
    max_feature_nodes: int | None = None,
    session_name: str = "HubExperimentSession",
) -> dict[str, Any]:
    """Build the declarative session body for a resolved spec (pure; no weights).

    The body follows the component shape the loader understands: registry info
    with the spec's adapter composition, a module config whose auto-composition
    synthesizes the adapter sections, and a default datamodule config (the
    datamodule instance itself is attached separately — see
    :class:`ExperimentDataModule`).
    """
    import torch

    device_type = force_device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Config-covering adapters: the framework entries never provide config
    # sections, so only the rest scope the subclass search.
    config_adapters = [name for name in spec.composition if name not in ("core", "lightning")]
    return {
        "reg_info": {"adapter_combinations": [list(spec.composition)]},
        "module_cls": spec.module_cls,
        "module_cfg": {
            "class_path": "interpretune.config.module.ITConfig",
            "init_args": {
                "model_name_or_path": spec.model_name,
                "auto_comp_cfg": {
                    "class_path": "interpretune.config.shared.AutoCompConfig",
                    "init_args": {
                        "module_cfg_name": session_name,
                        "module_cfg_mixin": {
                            "class_path": "interpretune.harness.sessions.NeutralSessionMixin",
                            "import_only": True,
                        },
                        "target_adapters": config_adapters,
                    },
                },
                "circuit_tracer_cfg": {
                    "class_path": "interpretune.adapters.circuit_tracer.config.CircuitTracerConfig",
                    "init_args": {
                        "backend": "nnsight",
                        "model_name": spec.model_name,
                        "transcoder_set": spec.transcoder_set,
                        **({"batch_size": batch_size} if batch_size is not None else {}),
                        **({"max_feature_nodes": max_feature_nodes} if max_feature_nodes is not None else {}),
                    },
                },
                "nnsight_cfg": {
                    "class_path": "interpretune.adapters.nnsight.config.NNsightConfig",
                    "init_args": {"model_name": spec.model_name, "device_map": device_type},
                },
            },
        },
        "datamodule_cfg": {
            "class_path": "interpretune.config.datamodule.ITDataModuleConfig",
            "init_args": {"model_name_or_path": spec.model_name},
        },
    }


def experiment_session(
    work_root: str | Path,
    run_name: str,
    *,
    model_family: str,
    model_variant: str,
    model_name: str | None = None,
    transcoder_set: str | None = None,
    force_device: str | None = None,
    batch_size: int | None = None,
    max_feature_nodes: int | None = None,
    **kwargs: Any,
):
    """Build a Hub-runnable experiment session through public construction.

    Signature-compatible with the test-tree helper it replaces: yields
    ``(session, module, tokenizer)``. Session kwargs beyond the consumed model
    surface (debug presets, Neuronpedia upload toggles) are caller-side and
    named in debug output rather than silently dropped.
    """
    import logging
    from contextlib import contextmanager

    from interpretune.config.datamodule import ITDataModuleConfig
    from interpretune.config.loading import load_session_cfg
    from interpretune.session import ITSession

    ignored = sorted(set(kwargs) - _CONSUMED_SESSION_KWARGS)
    if ignored:
        logging.getLogger("interpretune.harness").debug(
            "experiment_session ignoring caller-side session kwargs: %s", ignored
        )
    spec = resolve_model_spec(
        model_family,
        model_variant,
        model_name_override=model_name,
        transcoder_set_override=transcoder_set,
    )
    body = build_session_body(
        spec,
        force_device=force_device,
        batch_size=batch_size,
        max_feature_nodes=max_feature_nodes,
    )
    session_cfg = load_session_cfg(body)
    session_cfg.datamodule = ExperimentDataModule(
        ITDataModuleConfig(model_name_or_path=spec.model_name),
    )
    session = ITSession(session_cfg)

    @contextmanager
    def _session():
        from interpretune.base.call import it_init

        session_dir = Path(work_root) / run_name
        session_dir.mkdir(parents=True, exist_ok=True)
        it_init(**session)
        module = session.module
        datamodule = session.datamodule
        assert module is not None, "experiment session built no module"
        assert datamodule is not None, "experiment session built no datamodule"
        tokenizer = getattr(datamodule, "tokenizer", None)
        assert tokenizer is not None, "experiment session built no tokenizer"
        try:
            yield session, module, tokenizer
        finally:
            from interpretune.utils.resource_mgmt import cleanup_python_cuda

            cleanup_python_cuda()

    return _session()


def _session_surface_presets_path() -> Path:
    return Path(__file__).resolve().parent / "configs" / "session_surface_presets.yaml"


def resolve_session_surface_preset_config_defaults(preset: str) -> dict[str, Any]:
    """Resolve a debug session-surface preset to config-section defaults.

    Pure mapping over the core presets file; unknown presets are refused by
    name. The ``notebook_default`` preset is the empty mapping by definition.
    """
    if preset == "notebook_default":
        return {}
    payload = _load_yaml_mapping(_session_surface_presets_path())
    raw_presets = payload.get("session_surface_presets")
    if not isinstance(raw_presets, dict):
        raise ValueError(f"Expected 'session_surface_presets' mapping in {_session_surface_presets_path()}")
    preset_payload = raw_presets.get(preset)
    if preset_payload is None:
        supported = ", ".join(sorted(str(key) for key in raw_presets))
        raise ValueError(f"Unsupported debug session surface preset: {preset}. Supported: {supported}")
    config_defaults = preset_payload.get("config_defaults", {})
    if not isinstance(config_defaults, dict):
        raise ValueError(f"Session surface preset '{preset}' config_defaults must map to a dictionary")
    return {str(key): value for key, value in config_defaults.items()}
