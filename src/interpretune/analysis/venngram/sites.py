r"""
Resolving venngram Site Addresses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

venngram resolves parameter addresses itself and leaves site addresses to a consumer: a term that sets ``point``
names an activation point, and turning a point at a layer into something a backend can hook is interpretune's to do.
:func:`resolve_sites` evaluates a site address into the concrete hook names it selects, ``blocks.<layer>.<point>``
in the vocabulary's spelling, together with what the address acts on there: ``value`` or, under ``of: grad``, the
gradient with respect to the activation at each site.

The fields a site term may set are ``point``, ``layers`` and ``of``; ``&``, ``|`` and ``-`` combine terms and
``@<set>`` follows a reference into the address book's sets. A ``point`` is a spelling of the activation-point
vocabulary (``hook_resid_post``, ``attn.hook_out``, ``mlp.out.hook_in``), checked by
:func:`~interpretune.analysis.points.parse`, so a spelling the vocabulary does not know is refused rather than mapped
onto a guess. A term without a ``point`` (``layers:-4:``) narrows the points other terms name, and an address that
names no point at all selects no site and is refused. Every other field (``role``, ``component``, ``units``,
``features``, ...) has no site meaning here yet and is refused by name, as is a ``stack`` other than the primary one.

The flow is venngram's own reading (``venngram.resolve.gradient_of``), so an address whose terms disagree on ``of`` is
refused as venngram refuses it.

>>> resolve_sites("point:hook_resid_post & layers:8:12 & of:grad", n_layers=12).sites
('blocks.8.hook_resid_post', 'blocks.9.hook_resid_post', 'blocks.10.hook_resid_post', 'blocks.11.hook_resid_post')

.. warning:: This is an experimental API and may change in a future release.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from interpretune.analysis.points import UnknownPointError, parse
from interpretune.utils import MisconfigurationException

#: The term fields a site address may set here.
SITE_FIELDS = frozenset({"point", "layers", "of"})


@dataclass(frozen=True)
class SiteSelection:
    """The sites an address selects, as concrete hook names in layer order, and what it acts on there."""

    sites: tuple[str, ...]
    flow: str
    """``value``, or ``grad`` under ``of: grad``: the gradient with respect to the activation at each site."""


def _venngram(module: str) -> Any:
    try:
        return importlib.import_module(f"venngram.{module}")
    except ImportError as e:
        raise MisconfigurationException(
            "Resolving a venngram address needs venngram, an interpretune dependency: reinstall interpretune."
        ) from e


def resolve_sites(address: Any, *, n_layers: int, sets: Mapping[str, Any] | None = None) -> SiteSelection:
    """Evaluate a venngram site address into the hook names it selects (see the module docstring).

    Args:
        address: The address, in its string or mapping form, or a parsed expression.
        n_layers: The depth of the block stack, which negative indices and slices count from.
        sets: The address book's sets (``AddressBook.sets()``), for ``@<set>`` references.

    Returns:
        A :class:`SiteSelection`.

    Raises:
        MisconfigurationException: If the address sets a field with no site meaning here, names a point the
            vocabulary does not know, references a set it was not given, selects nothing, or its terms disagree on
            ``of``.
    """
    address_mod, errors = _venngram("address"), _venngram("errors")
    sets = dict(sets or {})
    try:
        expr = address_mod.parse(address)
        flow = _venngram("resolve").gradient_of(expr, sets)
    except (errors.AddressError, errors.AddressUnresolvable) as e:
        raise MisconfigurationException(f"cannot resolve the site address {address!r}: {e}") from e
    evaluator = _SiteEvaluator(address_mod, sets, n_layers, where=str(address))
    points = evaluator.points(expr, ())
    if not points:
        raise MisconfigurationException(
            f"the site address {address!r} names no `point`, so it selects no site; a site term sets `point`"
        )
    universe = frozenset((layer, point) for point in points for layer in range(n_layers))
    selected = evaluator.evaluate(expr, universe, ())
    if not selected:
        raise MisconfigurationException(f"the site address {address!r} selects no site of a {n_layers}-layer stack")
    ordered = sorted(selected, key=lambda pair: (pair[0], sorted(points).index(pair[1])))
    return SiteSelection(sites=tuple(f"blocks.{layer}.{point}" for layer, point in ordered), flow=flow)


class _SiteEvaluator:
    """Set algebra over ``(layer, point)`` pairs, the universe being every point the address names at every
    layer."""

    def __init__(self, address_mod: Any, sets: Mapping[str, Any], n_layers: int, *, where: str) -> None:
        self._a, self._sets, self._n_layers, self._where = address_mod, sets, n_layers, where

    def _refuse(self, reason: str) -> MisconfigurationException:
        return MisconfigurationException(f"cannot resolve the site address {self._where!r}: {reason}")

    def _deref(self, name: str, stack: tuple[str, ...]) -> Any:
        if name in stack:
            raise self._refuse(f"`@{name}` refers to itself through {' -> '.join(stack)}")
        if name not in self._sets:
            raise self._refuse(f"`@{name}` names no set of the book given (sets: {sorted(self._sets)})")
        return self._a.parse(self._sets[name])

    def points(self, expr: Any, stack: tuple[str, ...]) -> set[str]:
        """Every ``point`` the expression names, following references; each checked against the vocabulary."""
        a = self._a
        if isinstance(expr, a.Term):
            if expr.point is None:
                return set()
            try:
                parse(expr.point)
            except UnknownPointError as e:
                raise self._refuse(f"`point:{expr.point}` is not an activation point: {e}") from e
            return {expr.point}
        if isinstance(expr, a.Ref):
            return self.points(self._deref(expr.name, stack), (*stack, expr.name))
        if isinstance(expr, (a.And, a.Or)):
            return set().union(*(self.points(item, stack) for item in expr.items))
        if isinstance(expr, a.Minus):
            return self.points(expr.left, stack) | self.points(expr.right, stack)
        if isinstance(expr, a.Guarded):
            return self.points(expr.expr, stack)
        raise self._refuse(f"an expression of kind {type(expr).__name__} has no site reading")

    def evaluate(self, expr: Any, universe: frozenset, stack: tuple[str, ...]) -> frozenset:
        a = self._a
        if isinstance(expr, a.Term):
            return self._term(expr, universe)
        if isinstance(expr, a.Ref):
            return self.evaluate(self._deref(expr.name, stack), universe, (*stack, expr.name))
        if isinstance(expr, a.And):
            out = universe
            for item in expr.items:
                out = out & self.evaluate(item, universe, stack)
            return out
        if isinstance(expr, a.Or):
            return frozenset().union(*(self.evaluate(item, universe, stack) for item in expr.items))
        if isinstance(expr, a.Minus):
            return self.evaluate(expr.left, universe, stack) - self.evaluate(expr.right, universe, stack)
        if isinstance(expr, a.Guarded):
            out = self.evaluate(expr.expr, universe, stack)
            if not out and not expr.optional:
                raise self._refuse(f"`{a.to_string(expr.expr)}` selects no site and is not optional")
            if expr.expect is not None and len(out) != expr.expect:
                raise self._refuse(f"`{a.to_string(expr.expr)}` selects {len(out)} sites, expected {expr.expect}")
            return out
        raise self._refuse(f"an expression of kind {type(expr).__name__} has no site reading")

    def _term(self, term: Any, universe: frozenset) -> frozenset:
        fields = set(term.set_fields())
        if fields - SITE_FIELDS - {"stack"}:
            raise self._refuse(
                f"{sorted(fields - SITE_FIELDS - {'stack'})} have no site meaning here; a site term sets "
                f"{sorted(SITE_FIELDS)}"
            )
        if term.stack is not None:
            raise self._refuse(f"`stack:{term.stack}`: only the primary block stack is resolved")
        layers = None if term.layers is None else set(self._layers(term.layers))
        return frozenset(
            (layer, point)
            for layer, point in universe
            if (term.point is None or point == term.point) and (layers is None or layer in layers)
        )

    def _layers(self, layer_set: Any) -> tuple[int, ...]:
        try:
            return layer_set.select(self._n_layers)
        except _venngram("errors").AddressError as e:
            raise self._refuse(str(e)) from e
