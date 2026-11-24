"""What an AnalysisStore column's values name in the model that produced them, and whether they carry over.

A column can hold values that mean something only against the model, tokenizer or dictionaries it was computed on: a
layer index, a hook name used as a key, a token id, a latent's index. An op schema declares each as a
:class:`SiteBinding` on the column's ``ColCfg.site``, which is published with the store's ``col_cfg``. From the
declarations alone, :func:`check_bindings` decides whether a store's columns still mean the same thing on another
model (``AnalysisStore.resolve_on``), before any row is read. A column whose meaning cannot be established is refused
by name: nothing is inferred from a column's name or its values.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from interpretune.hub.subject import StoreSubject

#: What a binding says a value names, and so what the target must share with the source.
BINDING_KINDS: dict[str, str] = {
    "layer": "a layer index: the target must have the source's depth",
    "hook_key": "the column's keys are hook names: each must resolve on the target's component map",
    "tokens": "token ids or positions in a tokenized prompt: the target must use the source's tokenizer",
    "latent_feature": "indices into a latent dictionary: the target must have the source's dictionaries attached",
    "frame": "values in the model's activation space: the target must be the same model at the same revision",
}


class SiteBindingError(ValueError):
    """A site binding is malformed."""


class StoreResolutionError(ValueError):
    """A store's columns do not carry over to the target, each named with the reason."""


@dataclass(frozen=True)
class SiteBinding:
    """One thing a column's values name.

    Args:
        binds: The kind of thing, one of :data:`BINDING_KINDS`.
        slot: The index along the column's last axis the binding describes (``0`` is the layer of a
            ``[layer, position, feature]`` triple), or ``None`` when it describes the whole value.
        point: For a ``layer`` binding, the activation point the layers index, in the vocabulary's layer-free spelling
            (``hook_mlp_out``). The target's component map must resolve it.
    """

    binds: str
    slot: int | None = None
    point: str | None = None

    def __post_init__(self) -> None:
        if self.binds not in BINDING_KINDS:
            raise SiteBindingError(
                f"binds={self.binds!r} is not a binding kind: expected one of {sorted(BINDING_KINDS)}"
            )
        if self.slot is not None and (type(self.slot) is not int or self.slot < 0):
            raise SiteBindingError(f"slot={self.slot!r} must be a non-negative integer index, or None")
        if self.point is not None and self.binds != "layer":
            raise SiteBindingError(f"point={self.point!r} belongs to a `layer` binding, not {self.binds!r}")


def parse_site(raw: Any) -> tuple[SiteBinding, ...] | None:
    """A ``ColCfg.site`` declaration (a binding, a mapping, or a list of either) as a tuple of bindings, or
    ``None``."""
    if raw is None:
        return None
    entries = raw if isinstance(raw, (list, tuple)) else [raw]
    if not entries:
        raise SiteBindingError("`site` is empty: leave it out for a column that names no site")
    bindings: list[SiteBinding] = []
    for entry in entries:
        if isinstance(entry, SiteBinding):
            bindings.append(entry)
            continue
        if not isinstance(entry, Mapping):
            raise SiteBindingError(f"a site binding is a mapping with `binds`, got {entry!r}")
        unknown = sorted(set(entry) - {"binds", "slot", "point"})
        if unknown or "binds" not in entry:
            raise SiteBindingError(f"a site binding takes `binds`, `slot` and `point`, got {dict(entry)!r}")
        bindings.append(SiteBinding(**entry))
    return tuple(bindings)


def site_to_list(site: tuple[SiteBinding, ...] | None) -> list[dict[str, Any]] | None:
    """The JSON form of a parsed ``site``."""
    return None if site is None else [asdict(b) for b in site]


def declares_bindings(col_cfg: Mapping[str, Any] | None) -> bool:
    """Whether a store's ``col_cfg`` was written by a schema that declares bindings.

    Every ``ColCfg`` written since the declaration exists carries a ``site`` key, ``None`` for a column that names no
    site, so an entry without the key predates it.
    """
    return bool(col_cfg) and all(isinstance(cfg, Mapping) and "site" in cfg for cfg in col_cfg.values())


def _component_map(architecture: str | None) -> tuple[Any, str | None]:
    if not architecture:
        return None, "its architecture is not recorded"
    from interpretune.analysis.points.component_map import component_map_for

    try:
        return component_map_for(architecture), None
    except KeyError:
        return None, f"no component map is registered for {architecture}"


def _depth_reason(source: StoreSubject, target: StoreSubject) -> str | None:
    if source.n_layers is None or target.n_layers is None:
        side = "store" if source.n_layers is None else "target"
        return f"the {side}'s layer count is not recorded, so a layer index cannot be shown to mean the same block"
    if source.n_layers != target.n_layers:
        return (
            f"the store's model has {source.n_layers} layers and the target {target.n_layers} (depth is never rescaled)"
        )
    return None


def _point_reason(name: str, subject: StoreSubject, side: str) -> str | None:
    from interpretune.analysis.points.resolution import Unresolvable, describe_unresolvable, resolve
    from interpretune.analysis.points.vocabulary import UnknownPointError, parse

    cmap, missing = _component_map(subject.architecture)
    if cmap is None:
        return f"{name!r} cannot be addressed on the {side}: {missing}"
    try:
        point = parse(name)
    except UnknownPointError as e:
        return f"{name!r} cannot be addressed: {e}"
    resolution = resolve(point, cmap)
    if isinstance(resolution, Unresolvable):
        return f"on the {side}, {describe_unresolvable(point, resolution)}"
    return None


def _tokenizer_reason(source: StoreSubject, target: StoreSubject) -> str | None:
    if source.tokenizer_id is None or target.tokenizer_id is None:
        side = "store" if source.tokenizer_id is None else "target"
        return f"the {side}'s tokenizer is not recorded"
    if source.tokenizer_id != target.tokenizer_id:
        return f"the store was tokenized with {source.tokenizer_id!r} and the target uses {target.tokenizer_id!r}"
    return None


def _dictionary_reason(source: StoreSubject, target: StoreSubject) -> str | None:
    if not source.dictionaries:
        return "the store records no latent dictionaries"
    missing = [d for d in source.dictionaries if d not in target.dictionaries]
    return f"the target lacks the store's dictionaries {missing}" if missing else None


def _frame_reason(source: StoreSubject, target: StoreSubject) -> str | None:
    if source.model_id != target.model_id:
        return f"its values are in {source.model_id}'s activation space and the target is {target.model_id}"
    if source.model_revision is None or source.model_revision != target.model_revision:
        return (
            f"its values are in {source.model_id}'s activation space, and the revisions"
            f" ({source.model_revision} and {target.model_revision}) are not shown to be the same"
        )
    return None


def _binding_reasons(
    binding: SiteBinding, keys: Iterable[str], source: StoreSubject, target: StoreSubject
) -> list[str]:
    """Why ``binding`` does not carry over, empty when it does."""
    if binding.binds == "tokens":
        reasons = [_tokenizer_reason(source, target)]
    elif binding.binds == "latent_feature":
        reasons = [_dictionary_reason(source, target)]
    elif binding.binds == "frame":
        reasons = [_frame_reason(source, target)]
    else:  # layer and hook_key both name blocks
        reasons = [_depth_reason(source, target)]
        if binding.binds == "layer" and binding.point is not None:
            reasons.append(_point_reason(f"blocks.0.{binding.point}", target, "target"))
        if binding.binds == "hook_key":
            for key in keys:
                reasons += [_point_reason(key, source, "store's model"), _point_reason(key, target, "target")]
    return [r for r in reasons if r is not None]


def check_bindings(
    col_cfg: Mapping[str, Mapping[str, Any]],
    columns: Iterable[str],
    keys: Mapping[str, Iterable[str]],
    source: StoreSubject,
    target: StoreSubject,
) -> None:
    """Refuse, naming each, the columns whose declared bindings do not carry over from ``source`` to ``target``.

    Args:
        col_cfg: The store's serialized column configurations.
        columns: The columns the store holds.
        keys: For each column keyed by hook name, its keys, read from the column's features.
        source: What the store was computed on.
        target: What it is to be read against.

    Raises:
        StoreResolutionError: a column has no configuration, or a binding does not carry over.
    """
    problems: list[str] = []
    for column in columns:
        cfg = col_cfg.get(column)
        if cfg is None:
            problems.append(f"{column!r}: the store records no column configuration for it")
            continue
        seen: set[str] = set()
        for binding in parse_site(cfg.get("site")) or ():
            for reason in _binding_reasons(binding, keys.get(column, ()), source, target):
                if reason not in seen:
                    seen.add(reason)
                    problems.append(f"{column!r} ({binding.binds}): {reason}")
    if problems:
        raise StoreResolutionError(
            f"This store was computed on {source.model_id}; these columns do not carry over to {target.model_id}:\n  - "
            + "\n  - ".join(problems)
            + "\nSelect the columns that do with `select_columns`, or read the store against the model it was computed"
            " on."
        )
