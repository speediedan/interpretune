"""What an AnalysisStore was computed on: the ``subject`` block of its ``it_artifact.json`` envelope.

A store's columns are only meaningful against the model, tokenizer, component map and latent dictionaries that
produced them, and none of those can be recovered from the columns. So the subject is recorded at write time, from
the module the analysis ran on, and published with the store.

Each field records what interpretune can actually read and states ``None`` for what it cannot (a model loaded
without a resolved commit, a store built without a datamodule): an unknown is recorded as unknown, never filled with
a plausible guess. Only the model id is required, since a store with no stated model cannot be matched to anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class StoreSubjectError(ValueError):
    """A store's subject is missing, or a subject record is malformed."""


@dataclass(frozen=True)
class StoreSubject:
    """The model, tokenizer, component map and latent dictionaries an AnalysisStore was computed on.

    Args:
        model_id: The Hub model id or local path the module loaded (required).
        model_revision: The resolved commit of the model's weights, when the loader reports one.
        architecture: The model's architecture class name (``config.architectures[0]``).
        tokenizer_id: The tokenizer's ``name_or_path``.
        component_map: The component map that resolved the store's sites: its ``architecture``, ``source`` and
            ``schema_version``.
        dictionaries: The latent dictionaries attached to the model, each a mapping with a ``kind``
            (``"sae_lens"`` or ``"circuit_tracer"``) and the identifiers that kind publishes under.
        n_layers: The model's depth (its decoder layer count), which a layer-bound column needs to carry over.
    """

    model_id: str
    model_revision: str | None = None
    architecture: str | None = None
    tokenizer_id: str | None = None
    component_map: dict[str, Any] | None = None
    dictionaries: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    n_layers: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise StoreSubjectError(f"A store subject needs a model id, got {self.model_id!r}.")

    def to_dict(self) -> dict[str, Any]:
        """The envelope form: JSON-native, with every field present so an unknown reads as ``None``."""
        return {
            "model": {"id": self.model_id, "revision": self.model_revision},
            "architecture": self.architecture,
            "tokenizer": {"id": self.tokenizer_id},
            "component_map": dict(self.component_map) if self.component_map else None,
            "dictionaries": [dict(d) for d in self.dictionaries],
            "n_layers": self.n_layers,
        }

    @classmethod
    def from_dict(cls, record: Any, source: str = "<subject>") -> StoreSubject:
        """Read an envelope's ``subject`` block, refusing one that does not name a model."""
        model = record.get("model") if isinstance(record, dict) else None
        if not isinstance(model, dict) or not model.get("id"):
            raise StoreSubjectError(f"{source}: the `subject` block must name the model as `model.id`.")
        tokenizer = record.get("tokenizer") or {}
        return cls(
            model_id=model["id"],
            model_revision=model.get("revision"),
            architecture=record.get("architecture"),
            tokenizer_id=tokenizer.get("id") if isinstance(tokenizer, dict) else None,
            component_map=record.get("component_map"),
            dictionaries=tuple(record.get("dictionaries") or ()),
            n_layers=n_layers if isinstance(n_layers := record.get("n_layers"), int) else None,
        )


def _str(value: Any) -> str | None:
    """A non-empty string as itself, anything else as unknown: a stand-in object is not an identifier."""
    return value if isinstance(value, str) and value else None


def _hf_config(module) -> Any:
    """The Hugging Face config of the module's model, wherever the composition keeps it, or ``None``."""
    model = getattr(module, "model", None)
    for candidate in (model, getattr(model, "_module", None)):
        config = getattr(candidate, "config", None)
        if config is not None and hasattr(config, "to_dict"):
            return config
    return None


def _depth(config) -> int | None:
    """The model's decoder layer count from its config (a text sub-config first, for a multimodal model)."""
    for candidate in (getattr(config, "text_config", None), config):
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            value = getattr(candidate, attr, None)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    return None


def _model_id(it_cfg, config) -> str | None:
    """The configured model id, with an adapter's own model name taking precedence where one is set."""
    for adapter_cfg in ("circuit_tracer_cfg", "nnsight_cfg"):
        if name := _str(getattr(getattr(it_cfg, adapter_cfg, None), "model_name", None)):
            return name
    return _str(getattr(it_cfg, "model_name_or_path", None)) or _str(getattr(config, "_name_or_path", None))


def _component_map_record(architecture: str | None) -> dict[str, Any] | None:
    if not architecture:
        return None
    from interpretune.analysis.points.component_map import component_map_for

    try:
        component_map = component_map_for(architecture)
    except KeyError:  # no registered map for this architecture: the store states none rather than a near match
        return None
    return {
        "architecture": component_map.architecture,
        "source": str(component_map.source),
        "schema_version": component_map.schema_version,
    }


def _dictionaries(it_cfg) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    sae_cfgs = getattr(it_cfg, "sae_cfgs", None)
    for sae_cfg in sae_cfgs if isinstance(sae_cfgs, (list, tuple)) else ():
        release, sae_id = _str(getattr(sae_cfg, "release", None)), _str(getattr(sae_cfg, "sae_id", None))
        if release or sae_id:
            records.append({"kind": "sae_lens", "release": release, "sae_id": sae_id})
    transcoder_set = _str(getattr(getattr(it_cfg, "circuit_tracer_cfg", None), "transcoder_set", None))
    if transcoder_set:
        records.append({"kind": "circuit_tracer", "transcoder_set": transcoder_set})
    return tuple(records)


def subject_from_module(module) -> StoreSubject:
    """Read the subject of a store computed on ``module``.

    Raises:
        StoreSubjectError: The module states no model id, so there is nothing to record a store against.
    """
    it_cfg = getattr(module, "it_cfg", None)
    config = _hf_config(module)
    model_id = _model_id(it_cfg, config)
    if not model_id:
        raise StoreSubjectError(
            f"{type(module).__name__} states no model id (`it_cfg.model_name_or_path`), so a store computed on it"
            " has no subject. Set the model id, or pass `subject=` when publishing."
        )
    architectures = getattr(config, "architectures", None)
    architecture = _str(architectures[0]) if isinstance(architectures, (list, tuple)) and architectures else None
    datamodule = getattr(module, "datamodule", None)
    tokenizer = getattr(datamodule, "tokenizer", None) or getattr(it_cfg, "tokenizer", None)
    return StoreSubject(
        model_id=model_id,
        model_revision=_str(getattr(config, "_commit_hash", None)),
        architecture=architecture,
        tokenizer_id=_str(getattr(tokenizer, "name_or_path", None)),
        component_map=_component_map_record(architecture),
        dictionaries=_dictionaries(it_cfg),
        n_layers=_depth(config),
    )
