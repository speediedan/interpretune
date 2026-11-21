"""Resolve latent addresses against an explicit catalog of loaded models and named bases.

Feature units are blocks in the model's declared decoder layout. The returned identity keeps a block intact; its decoder
rows expose its span without inventing coordinates inside it. External identifiers are explicit aliases for canonical
feature addresses, so resolving an identifier never downloads or guesses a model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import torch

from interpretune.analysis.points import parse
from interpretune.analysis.venngram.sites import _SiteEvaluator, _venngram
from interpretune.utils import MisconfigurationException

if TYPE_CHECKING:
    from interpretune.analysis.venngram.charts import LatentChart
    from interpretune.analysis.venngram.spaces import LinearSpace


def _point(name: str) -> str:
    try:
        point = parse(name)
    except ValueError as err:
        raise MisconfigurationException(f"latent point {name!r} is unsupported: {err}") from err
    if point.layer is not None:
        raise MisconfigurationException(f"latent point {name!r} includes a layer; declare the layer separately")
    if point.contribution is not None:
        return f"hook_{point.contribution}_out"
    return point.canonical


def _basis(value: torch.Tensor, where: str) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or min(value.shape) < 1
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise MisconfigurationException(f"{where} needs a nonempty, finite floating-point matrix of basis rows")


@dataclass(frozen=True, eq=False)
class LatentBinding:
    """One loaded latent model at one site, with its decoder layout declared explicitly.

    Args:
        name: The canonical model identifier used by ``features`` addresses.
        layer: The model block where this model is attached.
        point: A layer-free activation-point spelling.
        model: The loaded model, kept as the caller supplied it.
        decoder_rows: Decoder directions, one row per scalar latent coordinate.
        block_size: Scalar coordinates per addressable feature block, one for an SAE or transcoder.
        space: The space where the decoder rows are expressed, or ``None`` for the site's activation space.
    """

    name: str
    layer: int
    point: str
    model: Any
    decoder_rows: torch.Tensor
    block_size: int
    space: str | None = None

    def __post_init__(self) -> None:
        _validate_location(self.name, self.layer, self.point, self.space)
        if self.model is None:
            raise MisconfigurationException(f"latent binding {self.name!r} needs its loaded model")
        _basis(self.decoder_rows, f"latent model {self.name!r}")
        if type(self.block_size) is not int or self.block_size < 1:
            raise MisconfigurationException(f"latent model {self.name!r} needs a positive integer block_size")
        if self.decoder_rows.shape[0] % self.block_size:
            raise MisconfigurationException(
                f"latent model {self.name!r}: {self.decoder_rows.shape[0]} decoder rows do not divide into"
                f" blocks of {self.block_size}"
            )
        object.__setattr__(self, "point", _point(self.point))

    @property
    def n_blocks(self) -> int:
        """The number of addressable blocks, independent of the decoder's activation width."""
        return self.decoder_rows.shape[0] // self.block_size


@dataclass(frozen=True, eq=False)
class SubspaceBinding:
    """A named basis at one site, with rows expressed in its declared space."""

    name: str
    layer: int
    point: str
    basis: torch.Tensor
    space: str | None = None

    def __post_init__(self) -> None:
        _validate_location(self.name, self.layer, self.point, self.space)
        _basis(self.basis, f"subspace {self.name!r}")
        object.__setattr__(self, "point", _point(self.point))


def _validate_location(name: str, layer: int, point: str, space: str | None) -> None:
    if not isinstance(name, str) or not name:
        raise MisconfigurationException("a latent binding needs a nonempty name")
    if type(layer) is not int or layer < 0:
        raise MisconfigurationException(f"latent binding {name!r} needs a nonnegative integer layer, got {layer!r}")
    if not isinstance(point, str) or not point:
        raise MisconfigurationException(f"latent binding {name!r} needs an activation point")
    if space is not None and (not isinstance(space, str) or not space):
        raise MisconfigurationException(f"latent binding {name!r} needs a nonempty linear-space name")


@dataclass(frozen=True, order=True)
class LatentTarget:
    """A stable latent identity: a whole feature block or a whole named subspace."""

    kind: Literal["feature", "subspace"]
    name: str
    layer: int
    point: str
    unit: int

    def __post_init__(self) -> None:
        _validate_location(self.name, self.layer, self.point, None)
        if self.kind not in ("feature", "subspace"):
            raise MisconfigurationException(f"unsupported latent target kind {self.kind!r}")
        if type(self.unit) is not int or self.unit < 0 or (self.kind == "subspace" and self.unit != 0):
            raise MisconfigurationException(f"invalid unit {self.unit!r} for latent target kind {self.kind!r}")
        object.__setattr__(self, "point", _point(self.point))

    @property
    def site(self) -> str:
        """The concrete site in the activation-point vocabulary."""
        return f"blocks.{self.layer}.{self.point}"


@dataclass(frozen=True)
class LatentSelection:
    """Canonical targets in deterministic order, with the address's value or gradient flow."""

    targets: tuple[LatentTarget, ...]
    flow: str


@dataclass(frozen=True, eq=False)
class LatentFrame:
    """A named, versioned orthonormal frame for one canonical latent target.

    Rows are coordinates in the binding's declared space. The frame must span the entire target, and does not
    change its identity. Tolerances apply to validation in double precision; ``rank_rtol`` determines which
    singular values of the decoder rows count toward the target's rank.
    """

    name: str
    version: str
    target: LatentTarget
    rows: torch.Tensor
    atol: float = 1e-6
    rank_rtol: float = 1e-6

    def __post_init__(self) -> None:
        for field in ("name", "version"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise MisconfigurationException(f"latent frame needs a nonempty {field}, got {value!r}")
        if not isinstance(self.target, LatentTarget):
            raise MisconfigurationException(f"frame {self.name!r} needs a canonical LatentTarget")
        for field in ("atol", "rank_rtol"):
            value = getattr(self, field)
            if type(value) not in (int, float) or not 0 < value < 1:
                raise MisconfigurationException(f"frame {self.name!r} needs {field} strictly between zero and one")
        _basis(self.rows, f"frame {self.name!r} version {self.version!r}")
        object.__setattr__(self, "rows", self.rows.detach().clone())
        self._check_orthonormal()

    def _check_orthonormal(self) -> torch.Tensor:
        _basis(self.rows, f"frame {self.name!r} version {self.version!r}")
        rows = self.rows.detach().to(device="cpu", dtype=torch.float64)
        gram = rows @ rows.T
        identity = torch.eye(len(rows), dtype=rows.dtype, device=rows.device)
        if not torch.allclose(gram, identity, atol=self.atol, rtol=0):
            error = float((gram - identity).abs().max())
            raise MisconfigurationException(f"frame {self.name!r} is not orthonormal (maximum Gram error {error:g})")
        return rows

    def validate(self, basis: torch.Tensor) -> None:
        """Require equal rank and span against the target's current decoder rows or subspace basis."""
        rows = self._check_orthonormal()
        _basis(basis, f"target of frame {self.name!r}")
        basis = basis.detach().to(device="cpu", dtype=torch.float64)
        if rows.shape[1] != basis.shape[1]:
            raise MisconfigurationException(
                f"frame {self.name!r} width {rows.shape[1]} differs from target width {basis.shape[1]}"
            )
        _, singular_values, directions = torch.linalg.svd(basis, full_matrices=False)
        rank = int((singular_values > self.rank_rtol * singular_values.max()).sum())
        if len(rows) != rank:
            raise MisconfigurationException(f"frame {self.name!r} has {len(rows)} rows; target rank is {rank}")
        directions = directions[:rank]
        residual = rows - (rows @ directions.T) @ directions
        error = float(residual.abs().max())
        if error > self.atol:
            raise MisconfigurationException(
                f"frame {self.name!r} does not span its target (maximum residual {error:g})"
            )


class LatentCatalog:
    """The loaded latent models and subspaces an address may select.

    External ids map to explicit canonical feature addresses. A Neuronpedia feature id, an SAE Lens pair encoded
    by the caller, or a circuit-tracer feature id can therefore resolve to the same identity without string-prefix
    heuristics. Duplicate bindings at the same name, layer and point are refused, including ones in different
    spaces: a canonical identity must refer to one loaded object.
    """

    def __init__(
        self,
        features: Sequence[LatentBinding] = (),
        subspaces: Sequence[SubspaceBinding] = (),
        *,
        external_ids: Mapping[str, Any] | None = None,
        frames: Sequence[LatentFrame] = (),
        charts: Sequence[LatentChart] = (),
        spaces: Sequence[LinearSpace] = (),
    ) -> None:
        self._features = self._index(features)
        self._subspaces = self._index(subspaces)
        self.external_ids = dict(external_ids or {})
        names = {name for name, _, _ in self._features}
        if conflicts := names & self.external_ids.keys():
            raise MisconfigurationException(f"external ids shadow canonical latent models: {sorted(conflicts)}")
        if any(not isinstance(name, str) or not name for name in self.external_ids):
            raise MisconfigurationException("external latent ids must be nonempty strings")
        self._frames: dict[tuple[str, str], LatentFrame] = {}
        for frame in frames:
            key = (frame.name, frame.version)
            if key in self._frames:
                raise MisconfigurationException(f"duplicate latent frame {key}")
            frame.validate(self.basis(frame.target))
            self._frames[key] = frame
        self._charts: dict[tuple[str, str], LatentChart] = {}
        for chart in charts:
            key = (chart.name, chart.version)
            if key in self._charts:
                raise MisconfigurationException(f"duplicate latent chart {key}")
            chart.validate(self)
            self._charts[key] = chart
        self._spaces: dict[str, LinearSpace] = {}
        for space in spaces:
            if space.name in self._spaces:
                raise MisconfigurationException(f"duplicate linear space {space.name!r}")
            self._spaces[space.name] = space
        for binding in (*self._features.values(), *self._subspaces.values()):
            rows = binding.decoder_rows if isinstance(binding, LatentBinding) else binding.basis
            if binding.space in self._spaces and rows.shape[1] != self._spaces[binding.space].dim:
                raise MisconfigurationException(
                    f"latent binding {binding.name!r} has rows of width {rows.shape[1]}, but linear space"
                    f" {binding.space!r} has dimension {self._spaces[binding.space].dim}"
                )

    @staticmethod
    def _index(bindings: Sequence[Any]) -> dict[tuple[str, int, str], Any]:
        indexed = {}
        for binding in bindings:
            key = (binding.name, binding.layer, binding.point)
            if key in indexed:
                raise MisconfigurationException(f"duplicate latent binding for {key}")
            indexed[key] = binding
        return indexed

    def binding(self, target: LatentTarget) -> LatentBinding | SubspaceBinding:
        """The loaded object or basis that owns a canonical target."""
        table = self._features if target.kind == "feature" else self._subspaces
        key = (target.name, target.layer, target.point)
        if key not in table:
            raise MisconfigurationException(f"target {target} has no binding in this catalog")
        binding = table[key]
        width = binding.n_blocks if target.kind == "feature" else 1
        if not 0 <= target.unit < width:
            raise MisconfigurationException(f"target {target} is outside its binding's {width} units")
        return binding

    def basis(self, target: LatentTarget) -> torch.Tensor:
        """The selected block's decoder rows or the named subspace's basis, in its declared space."""
        binding = self.binding(target)
        if isinstance(binding, SubspaceBinding):
            return binding.basis
        start = target.unit * binding.block_size
        return binding.decoder_rows[start : start + binding.block_size]

    def frame(self, target: LatentTarget, *, name: str, version: str) -> LatentFrame:
        """Find the exact frame version and revalidate it against the target's current basis."""
        key = (name, version)
        if key not in self._frames:
            raise MisconfigurationException(f"unknown latent frame {name!r} version {version!r}")
        frame = self._frames[key]
        if frame.target != target:
            raise MisconfigurationException(
                f"frame {name!r} version {version!r} belongs to {frame.target}, not {target}"
            )
        frame.validate(self.basis(target))
        return frame

    def coordinates(self, target: LatentTarget, values: torch.Tensor, *, frame: str, version: str) -> torch.Tensor:
        """Read explicitly framed coordinates from values in the binding's declared space."""
        rows = self._frame_rows(target, values, frame=frame, version=version)
        return values @ rows.T

    def replace_coordinates(
        self,
        target: LatentTarget,
        values: torch.Tensor,
        coordinates: torch.Tensor,
        *,
        frame: str,
        version: str,
    ) -> torch.Tensor:
        """Replace framed coordinates while preserving the target's orthogonal complement.

        Values and coordinates must have the same batch dimensions, dtype and device. These are coordinates in a
        declared linear space, not encoder activations or a model's nonlinear feature firing values.
        """
        rows = self._frame_rows(target, values, frame=frame, version=version)
        expected = (*values.shape[:-1], rows.shape[0])
        if (
            not isinstance(coordinates, torch.Tensor)
            or coordinates.shape != expected
            or coordinates.dtype != values.dtype
            or coordinates.device != values.device
            or not bool(torch.isfinite(coordinates).all())
        ):
            raise MisconfigurationException(
                f"frame {frame!r} needs finite coordinates with shape {expected}, dtype {values.dtype}"
                f" and device {values.device}"
            )
        return values + (coordinates - values @ rows.T) @ rows

    def space(self, name: str) -> LinearSpace:
        """The registered linear space ``name``; an unregistered one is refused, naming the registered ones."""
        if name not in self._spaces:
            raise MisconfigurationException(
                f"linear space {name!r} is not registered in this catalog (registered: {sorted(self._spaces)})"
            )
        return self._spaces[name]

    def to_space(self, target: LatentTarget, h: torch.Tensor) -> torch.Tensor:
        """Site activations in the space the target's binding is expressed in (unchanged for the site's own)."""
        name = self.binding(target).space
        return h if name is None else self.space(name).read(h)

    def _from_space(self, target: LatentTarget, h: torch.Tensor, edited: torch.Tensor) -> torch.Tensor:
        name = self.binding(target).space
        return edited if name is None else self.space(name).write(h, edited)

    def site_coordinates(self, target: LatentTarget, h: torch.Tensor, *, frame: str, version: str) -> torch.Tensor:
        """Framed coordinates read from site activations, through the binding's linear space."""
        return self.coordinates(target, self.to_space(target, h), frame=frame, version=version)

    def replace_site_coordinates(
        self, target: LatentTarget, h: torch.Tensor, coordinates: torch.Tensor, *, frame: str, version: str
    ) -> torch.Tensor:
        """Replace framed coordinates in site activations: edited in the binding's space, written back exactly."""
        values = self.to_space(target, h)
        edited = self.replace_coordinates(target, values, coordinates, frame=frame, version=version)
        return self._from_space(target, h, edited)

    def apply_site_chart(
        self, target: LatentTarget, h: torch.Tensor, *, chart: str, version: str, op: str, **args: Any
    ) -> torch.Tensor:
        """:meth:`apply_chart` on site activations, through the binding's linear space."""
        values = self.to_space(target, h)
        edited = self.apply_chart(target, values, chart=chart, version=version, op=op, **args)
        return self._from_space(target, h, edited)

    def chart(self, target: LatentTarget, *, name: str, version: str) -> LatentChart:
        """Find the exact chart version and revalidate it against its frame."""
        key = (name, version)
        if key not in self._charts:
            raise MisconfigurationException(f"unknown latent chart {name!r} version {version!r}")
        chart = self._charts[key]
        if chart.target != target:
            raise MisconfigurationException(
                f"chart {name!r} version {version!r} belongs to {chart.target}, not {target}"
            )
        chart.validate(self)
        return chart

    def chart_coordinates(
        self, target: LatentTarget, values: torch.Tensor, *, chart: str, version: str
    ) -> torch.Tensor:
        """Read chart coordinates from values in the binding's declared space, through the chart's frame."""
        payload = self.chart(target, name=chart, version=version)
        framed = self.coordinates(target, values, frame=payload.frame, version=payload.frame_version)
        return payload.handler.to_chart(payload, framed)

    def apply_chart(
        self, target: LatentTarget, values: torch.Tensor, *, chart: str, version: str, op: str, **args: Any
    ) -> torch.Tensor:
        """Edit values in chart coordinates with one of ``set``, ``path``, ``rotate`` or ``scale_norm``.

        The frame coordinates move by exactly the chart's change, so the part of the frame coordinates off the chart
        and the target's orthogonal complement are both preserved. The op's arguments are described by
        :func:`~interpretune.analysis.venngram.charts.apply_chart_op`.
        """
        from interpretune.analysis.venngram.charts import apply_chart_op

        payload = self.chart(target, name=chart, version=version)
        handler = payload.handler
        framed = self.coordinates(target, values, frame=payload.frame, version=payload.frame_version)
        point = handler.to_chart(payload, framed)
        moved = apply_chart_op(payload, point, op, args)
        edited = framed + handler.from_chart(payload, moved) - handler.from_chart(payload, point)
        return self.replace_coordinates(target, values, edited, frame=payload.frame, version=payload.frame_version)

    def _frame_rows(self, target: LatentTarget, values: torch.Tensor, *, frame: str, version: str) -> torch.Tensor:
        payload = self.frame(target, name=frame, version=version)
        if (
            not isinstance(values, torch.Tensor)
            or values.ndim < 1
            or values.shape[-1] != payload.rows.shape[1]
            or not values.is_floating_point()
            or not bool(torch.isfinite(values).all())
        ):
            raise MisconfigurationException(
                f"frame {frame!r} needs finite floating-point values with final width {payload.rows.shape[1]}"
            )
        return payload.rows.to(device=values.device, dtype=values.dtype)

    def resolve(self, address: Any, *, n_layers: int, sets: Mapping[str, Any] | None = None) -> LatentSelection:
        """Resolve set algebra over canonical feature blocks and named subspaces.

        Slices and negative units count feature blocks, and per-layer unit sets use the block count of the model at that
        layer. A named subspace is indivisible. Unsupported qualifiers, unknown models and aliases, missing book
        references, alias cycles, mixed flows and an empty final selection are refused.
        """
        if type(n_layers) is not int or n_layers < 1:
            raise MisconfigurationException(f"latent resolution needs a positive stack depth, got {n_layers!r}")
        a, errors = _venngram("address"), _venngram("errors")
        try:
            expr = a.parse(address)
            flow = _venngram("resolve").gradient_of(expr, sets or {})
            evaluator = _LatentEvaluator(a, sets or {}, n_layers, catalog=self, where=str(address))
            selected = evaluator.evaluate(expr, self._universe(n_layers), ())
            if not selected:
                raise MisconfigurationException(f"latent address {address!r} selects nothing in the supplied catalog")
            return LatentSelection(tuple(sorted(selected)), flow)
        except (errors.AddressError, errors.AddressUnresolvable) as err:
            raise MisconfigurationException(f"cannot resolve latent address {address!r}: {err}") from err

    def _universe(self, n_layers: int) -> frozenset[LatentTarget]:
        result = []
        for kind, table in (("feature", self._features), ("subspace", self._subspaces)):
            for (name, layer, point), binding in table.items():
                if layer >= n_layers:
                    raise MisconfigurationException(f"latent model {name!r} at layer {layer} exceeds depth {n_layers}")
                count = binding.n_blocks if isinstance(binding, LatentBinding) else 1
                result.extend(
                    LatentTarget(cast(Literal["feature", "subspace"], kind), name, layer, point, unit)
                    for unit in range(count)
                )
        return frozenset(result)


class _LatentEvaluator(_SiteEvaluator):
    def __init__(self, *args: Any, catalog: LatentCatalog, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.catalog = catalog
        self._alias_stack: list[str] = []

    def _refuse(self, reason: str) -> MisconfigurationException:
        return MisconfigurationException(f"cannot resolve latent address {self._where!r}: {reason}")

    def evaluate(self, expr: Any, universe: frozenset, stack: tuple[str, ...]) -> frozenset:
        if isinstance(expr, self._a.Guarded):
            selected = self.evaluate(expr.expr, universe, stack)
            if not selected and not expr.optional:
                raise self._refuse("a non-optional expression selects no latent target")
            if expr.expect is not None and len(selected) != expr.expect:
                raise self._refuse(f"selected {len(selected)} latent targets, expected {expr.expect}")
            return selected
        return super().evaluate(expr, universe, stack)

    def _term(self, term: Any, universe: frozenset) -> frozenset:
        allowed = {"features", "subspace", "point", "layers", "units", "space", "of"}
        if unknown := set(term.set_fields()) - allowed:
            raise self._refuse(f"{sorted(unknown)} have no latent-address meaning here")
        selected = universe
        if term.features is not None:
            selected &= self._features(term.features, universe)
        if term.subspace is not None:
            selected &= self._named(term.subspace, "subspace", universe)
        if term.point is not None:
            point = _point(term.point)
            selected = frozenset(t for t in selected if t.point == point)
        if term.layers is not None:
            layers = self._layers(term.layers)
            selected = frozenset(t for t in selected if t.layer in layers)
        if term.space is not None:
            selected = frozenset(t for t in selected if self.catalog.binding(t).space == term.space)
        if term.units is not None:
            if term.subspace is not None:
                raise self._refuse("a named subspace is indivisible; units select feature blocks")
            selected = frozenset(t for t in selected if t.kind == "feature")
            by_binding = {}
            for t in selected:
                binding = cast(LatentBinding, self.catalog.binding(t))
                key = (t.name, t.layer, t.point)
                if key not in by_binding:
                    by_binding[key] = frozenset(term.units.for_layer(t.layer, binding.n_blocks))
            selected = frozenset(t for t in selected if t.unit in by_binding[t.name, t.layer, t.point])
        return selected

    def _named(self, name: str, kind: str, universe: frozenset) -> frozenset:
        selected = frozenset(t for t in universe if t.kind == kind and t.name == name)
        if not selected:
            raise self._refuse(f"{kind} {name!r} has no loaded binding")
        return selected

    def _features(self, name: str, universe: frozenset) -> frozenset:
        if name not in self.catalog.external_ids:
            return self._named(name, "feature", universe)
        if name in self._alias_stack:
            raise self._refuse(f"external id {name!r} refers to itself through {self._alias_stack}")
        self._alias_stack.append(name)
        try:
            expr = self._a.parse(self.catalog.external_ids[name])
            if _venngram("resolve").gradient_of(expr, self._sets) != "value":
                raise self._refuse(f"external id {name!r} must identify features without declaring gradient flow")
            selected = self.evaluate(expr, universe, ())
            if not selected or any(t.kind != "feature" for t in selected):
                raise self._refuse(f"external id {name!r} must select loaded feature blocks")
            return selected
        finally:
            self._alias_stack.pop()
