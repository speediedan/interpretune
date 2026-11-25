r"""
Resolving venngram Site Addresses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

venngram resolves parameter addresses itself and leaves site addresses to a consumer: a term that sets ``point``
names an activation point, and turning a point at a layer into something a backend can hook is interpretune's to do.
:func:`resolve_sites` evaluates a site address into the concrete hook names it selects, ``blocks.<layer>.<point>``
in the vocabulary's spelling (or the bare point for a global one, such as ``unembed.hook_in``, which belongs to no
block), together with what the address acts on there: ``value`` or, under ``of: grad``, the gradient with respect to the
activation at each site. A global point has no layer, so a ``layers`` term never selects it.

The fields a site term may set are ``point``, ``layers`` and ``of``; ``&``, ``|`` and ``-`` combine terms and
``@<set>`` follows a reference into the address book's sets. A ``point`` is a spelling of the activation-point
vocabulary (``hook_resid_post``, ``attn.hook_out``, ``mlp.out.hook_in``), checked by
:func:`~interpretune.analysis.points.parse`, so a spelling the vocabulary does not know is refused rather than mapped
onto a guess. A term without a ``point`` (``layers:-4:``) narrows the points other terms name, and an address that
names no point at all selects no site and is refused. Every other field (``role``, ``component``, ``units``,
``features``, ...) has no site meaning here yet and is refused, as is a ``stack`` other than the primary one.

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

from interpretune.analysis.points import UnknownPointError, component_map_for, known_architectures, parse
from interpretune.utils import MisconfigurationException

#: The term fields a site address may set here.
SITE_FIELDS = frozenset({"point", "layers", "of"})

#: Where a block layer would be in a ``(layer, point)`` pair: a point that belongs to no block.
GLOBAL = -1


def _global_components() -> frozenset[str]:
    """The components outside every block stack in the bundled and registered maps, which no block shares a name
    with: the names that make a point global whatever model it is later resolved on."""
    maps = [component_map_for(name) for name in known_architectures()]
    stacked = {c for m in maps for stack in m.stack_names() for c in m.block_components(stack)}
    return frozenset(c for m in maps for c in m.global_components()) - stacked


@dataclass(frozen=True)
class SiteSelection:
    """The sites an address selects, as concrete hook names in layer order with global points last, and what it
    acts on there."""

    sites: tuple[str, ...]
    flow: str
    """``value``, or ``grad`` under ``of: grad``: the gradient with respect to the activation at each site."""


@dataclass(frozen=True)
class UnitSiteSelection:
    """The selected last-axis units at each concrete hook, in ascending index order, and their flow."""

    units: Mapping[str, tuple[int, ...]]
    flow: str


def resolve_unit_sites(
    address: Any, *, n_layers: int, site_widths: Mapping[str, int], sets: Mapping[str, Any] | None = None
) -> UnitSiteSelection:
    """Resolve unit addresses with the site's set algebra and venngram's unit-index grammar.

    Args:
        address: An address naming a point and units, including book references and set algebra.
        n_layers: The model's block count.
        site_widths: Last-axis widths, keyed by the layer-free point spelling. Width is supplied by the model,
            never inferred from the largest selected index. Slices and negative unit indices use that width.
        sets: Book sets for references.

    Raises:
        MisconfigurationException: The address has no point or units, selects nothing, mixes flows, lacks a width,
            or sets a field without a unit-site meaning (including latent ``features``).
    """
    a, errors = _venngram("address"), _venngram("errors")
    sets = dict(sets or {})
    try:
        expr = a.parse(address)
        flow = _venngram("resolve").gradient_of(expr, sets)
        evaluator = _UnitEvaluator(a, sets, n_layers, where=str(address))
        points = evaluator.points(expr, ())
        selected = {}
        if outside := sorted(point for point in points if evaluator.is_global(point)):
            raise evaluator._refuse(f"{outside} are global points; a unit address names points inside the block stack")
        for point in sorted(points):
            width = site_widths.get(point)
            if type(width) is not int or width <= 0:
                raise evaluator._refuse(f"point:{point} needs its positive last-axis width (supplied: {site_widths})")
            universe = frozenset(range(width))
            for layer in range(n_layers):
                evaluator.layer, evaluator.point, evaluator.width = layer, point, width
                units = evaluator.evaluate(expr, universe, ())
                if units:
                    selected[f"blocks.{layer}.{point}"] = tuple(sorted(units))
        if not evaluator.saw_units:
            raise evaluator._refuse("a unit address must name `units` and a `point`")
        if not selected:
            raise evaluator._refuse("the unit address selects nothing")
        return UnitSiteSelection(selected, flow)
    except (errors.AddressError, errors.AddressUnresolvable) as err:
        raise MisconfigurationException(f"cannot resolve the unit address {address!r}: {err}") from err


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
    outside = {point for point in points if evaluator.is_global(point)}
    universe = frozenset((layer, point) for point in points - outside for layer in range(n_layers))
    selected = evaluator.evaluate(expr, universe | {(GLOBAL, point) for point in outside}, ())
    if not selected:
        note = f" ({sorted(outside)} are global points, which have no layer)" if outside else ""
        raise MisconfigurationException(
            f"the site address {address!r} selects no site of a {n_layers}-layer stack{note}"
        )
    ordered = sorted(selected, key=lambda pair: (pair[0] == GLOBAL, pair[0], sorted(points).index(pair[1])))
    return SiteSelection(
        sites=tuple(point if layer == GLOBAL else f"blocks.{layer}.{point}" for layer, point in ordered), flow=flow
    )


class _SiteEvaluator:
    """Set algebra over ``(layer, point)`` pairs, the universe being every point the address names at every
    layer."""

    def __init__(self, address_mod: Any, sets: Mapping[str, Any], n_layers: int, *, where: str) -> None:
        self._a, self._sets, self._n_layers, self._where = address_mod, sets, n_layers, where
        self._globals = _global_components()

    def is_global(self, point: str) -> bool:
        """Whether a point spelling names a component outside the block stack."""
        return parse(point).component in self._globals

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


class _UnitEvaluator(_SiteEvaluator):
    """The same expression algebra, evaluated on the unit axis of one point at one layer."""

    layer: int
    point: str
    width: int
    saw_units = False

    def evaluate(self, expr: Any, universe: frozenset, stack: tuple[str, ...]) -> frozenset:
        if isinstance(expr, self._a.Guarded):
            raise self._refuse("guarded unit-site expressions are not supported; use explicit set algebra")
        return super().evaluate(expr, universe, stack)

    def _term(self, term: Any, universe: frozenset) -> frozenset:
        fields = set(term.set_fields())
        if unknown := fields - SITE_FIELDS - {"units"}:
            raise self._refuse(f"{sorted(unknown)} have no unit-site meaning here")
        if term.units is not None:
            self.saw_units = True
        if term.point is not None and term.point != self.point:
            return frozenset()
        if term.layers is not None and self.layer not in self._layers(term.layers):
            return frozenset()
        return universe if term.units is None else universe & frozenset(term.units.for_layer(self.layer, self.width))
