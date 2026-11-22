"""Carrying an AnalysisStore's site-bound columns over to another model, or refusing by name what does not carry.

A store's columns can name things of the model they were computed on: layers, hook names, token ids and positions,
features of an attached latent dictionary. Each op declares which in its schema (``ColCfg.site``), and the store
carries those declarations in its column configs. :func:`resolve_store_on` checks every declared binding against a
target model before a row is read:

- **tokens** carry over only under the same tokenizer;
- **latent features** only under the same dictionaries;
- **layers** only at the same depth: cross-depth use is always refused, never rescaled;
- **hook keys** only at the same depth, and only where the target's component map resolves the point each key names.

A store published before bindings existed declares none, and is refused rather than read by guessing from column
names. Everything that passes these checks names the same thing on the target, so the values carry over unchanged.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from interpretune.hub.subject import StoreSubject


class StoreResolutionError(ValueError):
    """A store's site-bound columns cannot carry over to the target model."""


def _subject_of(target: Any) -> StoreSubject:
    if isinstance(target, StoreSubject):
        return target
    from interpretune.hub.subject import subject_from_module

    return subject_from_module(target)


def _column_cfgs(store) -> dict[str, Any]:
    from interpretune.analysis.ops.base import ColCfg

    raw = (store.it_format_kwargs or {}).get("col_cfg") or {}
    return {name: cfg if isinstance(cfg, ColCfg) else ColCfg.from_dict(dict(cfg)) for name, cfg in raw.items()}


def _store_features(store) -> Any:
    source = store.stream if getattr(store, "stream", None) is not None else store.dataset
    return getattr(source, "features", None) or {}


def _hook_problems(column: str, keys: list[str], target: StoreSubject) -> list[str]:
    """Why some of a per-hook column's keys do not name a resolvable point on the target, one line per key."""
    from interpretune.analysis.points.component_map import component_map_for
    from interpretune.analysis.points.resolution import Unresolvable, resolve
    from interpretune.analysis.points.vocabulary import UnknownPointError, parse

    cmap = None
    if target.architecture:
        try:
            cmap = component_map_for(target.architecture)
        except KeyError:
            cmap = None
    problems = []
    for key in keys:
        try:
            point = parse(key)
        except UnknownPointError as err:
            problems.append(f"{column}: hook key {key!r} names no activation point ({err})")
            continue
        if cmap is None:
            problems.append(
                f"{column}: hook key {key!r} cannot be checked, since the target ({target.architecture!r}) has no"
                " registered component map"
            )
        elif isinstance(res := resolve(point, cmap), Unresolvable):
            problems.append(f"{column}: hook key {key!r} does not resolve on {target.architecture}: {res.reason}")
    return problems


def _dictionaries_key(subject: StoreSubject) -> str:
    return json.dumps(sorted(json.dumps(d, sort_keys=True) for d in subject.dictionaries))


def resolve_store_on(store, target: Any):
    """Check a store's declared site bindings against ``target`` and return the store as read on it.

    Args:
        store: An ``AnalysisStore`` carrying a ``subject`` and its column configs.
        target: The model to read the store on: a module, or the ``StoreSubject`` describing one.

    Returns:
        A shallow copy of the store, sharing its data, whose ``resolved_on`` is the target's subject.

    Raises:
        StoreResolutionError: The store states no subject or declares no bindings, or a declared binding does not
            carry over; the message names every column that fails and why.
    """
    source = getattr(store, "subject", None)
    if source is None:
        raise StoreResolutionError(
            "This store states no subject (what it was computed on), so nothing can be checked against the target."
            " Pull it from a current envelope, or set `store.subject`."
        )
    target_subject = _subject_of(target)
    cfgs = _column_cfgs(store)
    features = _store_features(store)
    bound = {name: cfg.site for name, cfg in cfgs.items() if cfg.site and name in features}
    if not bound:
        raise StoreResolutionError(
            "This store declares no site bindings, so which of its columns name layers, hook keys, tokens or latent"
            " features is unknown and none can be checked. Re-push it with a current interpretune."
        )
    problems: list[str] = []
    for column, sites in sorted(bound.items()):
        kinds = {b.binds for b in sites}
        if "tokens" in kinds and not (source.tokenizer_id and source.tokenizer_id == target_subject.tokenizer_id):
            problems.append(
                f"{column}: tokens and positions carry over only under the same tokenizer (store"
                f" {source.tokenizer_id!r}, target {target_subject.tokenizer_id!r})"
            )
        if "latent_feature" in kinds and not (
            source.dictionaries and _dictionaries_key(source) == _dictionaries_key(target_subject)
        ):
            problems.append(
                f"{column}: latent features carry over only under the same dictionaries (store"
                f" {list(source.dictionaries)}, target {list(target_subject.dictionaries)})"
            )
        if kinds & {"layer", "hook_key"} and not (source.n_layers and source.n_layers == target_subject.n_layers):
            problems.append(
                f"{column}: layers carry over only at the same depth, and cross-depth use is refused (store"
                f" {source.n_layers} layers, target {target_subject.n_layers})"
            )
        elif "hook_key" in kinds:
            keys = list(features[column].keys()) if hasattr(features[column], "keys") else []
            problems.extend(_hook_problems(column, keys, target_subject))
    if problems:
        raise StoreResolutionError(
            f"This store ({source.model_id}) does not carry over to {target_subject.model_id}:\n  "
            + "\n  ".join(problems)
        )
    resolved = copy.copy(store)
    resolved.resolved_on = target_subject
    return resolved
