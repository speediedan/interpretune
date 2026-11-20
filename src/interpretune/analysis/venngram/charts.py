"""Charts: named, versioned coordinate systems fitted on a latent target's frame.

A chart lives in an explicit frame (:class:`~interpretune.analysis.venngram.latents.LatentFrame`) and maps that frame's
coordinates to chart coordinates of a declared intrinsic dimension, some of which may be periodic. Interventions
edit chart coordinates (``set``, ``path``, ``rotate``, ``scale_norm``) and write the change back through the frame, so
the part of the activation outside the chart, and outside the target, is preserved.

Each chart kind is a registered handler. Two are built in: ``affine`` (a flat chart over an orthonormal set of frame
directions) and ``pca_plane`` (a plane read in polar coordinates, its angle periodic). A spline or a flow is a kind a
caller registers with :func:`register_chart_kind`; an unregistered kind is refused by name rather than approximated.
Charts can travel in an address book's top-level ``charts`` key, which venngram keeps in ``AddressBook.extra``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from interpretune.analysis.venngram.latents import LatentTarget
from interpretune.utils import MisconfigurationException

if TYPE_CHECKING:
    from interpretune.analysis.venngram.latents import LatentCatalog

CHART_OPS = ("set", "path", "rotate", "scale_norm")


class ChartKind:
    """A chart handler: validates a chart's parameters and maps between frame and chart coordinates.

    Subclasses set ``name`` and ``ops`` and implement :meth:`check`, :meth:`to_chart` and :meth:`from_chart`.
    ``scale_norm`` scales the chart coordinates named by :meth:`radial`, the point's distance from its origin.
    """

    name: ClassVar[str]
    ops: ClassVar[frozenset[str]]

    def check(self, chart: LatentChart, rank: int) -> None:
        """Refuse parameters, dimensions or periods this kind cannot interpret for a frame of ``rank`` rows."""
        raise NotImplementedError

    def to_chart(self, chart: LatentChart, coordinates: torch.Tensor) -> torch.Tensor:
        """Frame coordinates ``(..., rank)`` to chart coordinates ``(..., intrinsic_dim)``."""
        raise NotImplementedError

    def from_chart(self, chart: LatentChart, point: torch.Tensor) -> torch.Tensor:
        """Chart coordinates ``(..., intrinsic_dim)`` to frame coordinates ``(..., rank)`` on the chart."""
        raise NotImplementedError

    def radial(self, chart: LatentChart) -> tuple[int, ...]:
        """The chart dimensions ``scale_norm`` scales."""
        raise NotImplementedError


_CHART_KINDS: dict[str, ChartKind] = {}


def register_chart_kind(kind: ChartKind) -> None:
    """Register a chart handler under its ``name``; a second handler for a registered name is refused."""
    if not isinstance(kind, ChartKind) or not isinstance(getattr(kind, "name", None), str) or not kind.name:
        raise MisconfigurationException("a chart kind is a named ChartKind instance")
    if unknown := set(kind.ops) - set(CHART_OPS):
        raise MisconfigurationException(f"chart kind {kind.name!r} declares unknown ops {sorted(unknown)}")
    if kind.name in _CHART_KINDS:
        raise MisconfigurationException(f"chart kind {kind.name!r} is already registered")
    _CHART_KINDS[kind.name] = kind


def chart_kind(name: str) -> ChartKind:
    """The registered handler for ``name``; an unregistered kind is refused, naming the registered ones."""
    if name not in _CHART_KINDS:
        raise MisconfigurationException(
            f"chart kind {name!r} is not registered (registered: {sorted(_CHART_KINDS)}); register a handler with"
            " register_chart_kind"
        )
    return _CHART_KINDS[name]


@dataclass(frozen=True, eq=False)
class LatentChart:
    """A named, versioned chart on one latent target, fitted in one exact frame version.

    Args:
        name: The chart's name, which an intervention references.
        version: The chart's exact version; lookups never fall back to another.
        kind: A registered chart kind.
        target: The canonical latent target the chart describes.
        frame: The frame the chart's parameters are expressed in.
        frame_version: That frame's exact version.
        intrinsic_dim: The number of chart coordinates.
        periodic: Periodic chart dimensions and their periods.
        params: The fitted parameters, as the kind defines them.
    """

    name: str
    version: str
    kind: str
    target: LatentTarget
    frame: str
    frame_version: str
    intrinsic_dim: int
    periodic: Mapping[int, float] = field(default_factory=dict)
    params: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("name", "version", "kind", "frame", "frame_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise MisconfigurationException(f"latent chart needs a nonempty {name}, got {value!r}")
        if not isinstance(self.target, LatentTarget):
            raise MisconfigurationException(f"chart {self.name!r} needs a canonical LatentTarget")
        if type(self.intrinsic_dim) is not int or self.intrinsic_dim < 1:
            raise MisconfigurationException(f"chart {self.name!r} needs a positive integer intrinsic_dim")
        periodic = {}
        for dim, period in dict(self.periodic).items():
            if type(dim) is not int or not 0 <= dim < self.intrinsic_dim:
                raise MisconfigurationException(
                    f"chart {self.name!r}: periodic dimension {dim!r} is outside its {self.intrinsic_dim} dimensions"
                )
            if type(period) not in (int, float) or not math.isfinite(period) or period <= 0:
                raise MisconfigurationException(f"chart {self.name!r}: dimension {dim} needs a positive finite period")
            periodic[dim] = float(period)
        params = {}
        for key, value in dict(self.params).items():
            if not isinstance(value, torch.Tensor) or not value.is_floating_point() or not bool(value.isfinite().all()):
                raise MisconfigurationException(f"chart {self.name!r}: parameter {key!r} must be a finite float tensor")
            params[key] = value.detach().clone()
        object.__setattr__(self, "periodic", periodic)
        object.__setattr__(self, "params", params)
        chart_kind(self.kind)

    @property
    def handler(self) -> ChartKind:
        """The registered handler for this chart's kind."""
        return chart_kind(self.kind)

    def validate(self, catalog: LatentCatalog) -> None:
        """Check the chart against its frame as the catalog currently holds it."""
        frame = catalog.frame(self.target, name=self.frame, version=self.frame_version)
        self.handler.check(self, frame.rows.shape[0])

    def params_like(self, reference: torch.Tensor) -> dict[str, torch.Tensor]:
        """The parameters on ``reference``'s device and dtype."""
        return {k: v.to(device=reference.device, dtype=reference.dtype) for k, v in self.params.items()}

    @classmethod
    def from_payload(cls, name: str, payload: Mapping[str, Any]) -> LatentChart:
        """Build a chart from its address-book form (plain mappings and nested lists)."""
        allowed = {"version", "kind", "target", "frame", "intrinsic_dim", "periodic", "params"}
        if not isinstance(payload, Mapping):
            raise MisconfigurationException(f"chart {name!r} must be a mapping")
        if unknown := set(payload) - allowed:
            raise MisconfigurationException(f"chart {name!r} has unknown keys {sorted(unknown)}")
        if missing := {"version", "kind", "target", "frame", "intrinsic_dim"} - set(payload):
            raise MisconfigurationException(f"chart {name!r} is missing {sorted(missing)}")
        target, frame = payload["target"], payload["frame"]
        if not isinstance(target, Mapping) or set(target) != {"kind", "name", "layer", "point", "unit"}:
            raise MisconfigurationException(f"chart {name!r} needs a target with kind, name, layer, point and unit")
        if not isinstance(frame, Mapping) or set(frame) != {"name", "version"}:
            raise MisconfigurationException(f"chart {name!r} needs a frame with name and version")
        params = {k: torch.tensor(v, dtype=torch.float64) for k, v in (payload.get("params") or {}).items()}
        return cls(
            name=name,
            version=str(payload["version"]),
            kind=payload["kind"],
            target=LatentTarget(**target),
            frame=frame["name"],
            frame_version=str(frame["version"]),
            intrinsic_dim=payload["intrinsic_dim"],
            periodic=dict(payload.get("periodic") or {}),
            params=params,
        )


def charts_from_book(book: Any) -> tuple[LatentChart, ...]:
    """The charts an address book carries in its top-level ``charts`` key, in name order."""
    charts = getattr(book, "extra", {}).get("charts") or {}
    if not isinstance(charts, Mapping):
        raise MisconfigurationException(f"address book {getattr(book, 'name', book)!r}: `charts` must be a mapping")
    return tuple(LatentChart.from_payload(name, charts[name]) for name in sorted(charts))


def apply_chart_op(chart: LatentChart, point: torch.Tensor, op: str, args: Mapping[str, Any]) -> torch.Tensor:
    """Edit chart coordinates ``(..., intrinsic_dim)`` with one op the chart's kind supports.

    ``set`` takes ``value`` (a chart point, broadcast over the batch); ``rotate`` takes ``by``, a mapping from periodic
    dimensions to the amount added, wrapped into the period; ``scale_norm`` takes a positive ``factor`` for the
    kind's radial dimensions; ``path`` takes ``waypoints`` (two or more chart points) and ``t`` in ``[0, 1]``, and
    moves along the piecewise-linear path through them, taking the shorter way around each periodic dimension.
    """
    handler = chart.handler
    if op not in handler.ops:
        raise MisconfigurationException(
            f"chart {chart.name!r} (kind {chart.kind!r}) does not support {op!r}; it supports {sorted(handler.ops)}"
        )
    expected = {"set": {"value"}, "rotate": {"by"}, "scale_norm": {"factor"}, "path": {"waypoints", "t"}}[op]
    if set(args) != expected:
        raise MisconfigurationException(f"chart op {op!r} takes {sorted(expected)}, got {sorted(args)}")
    if op == "set":
        value = _chart_points(chart, args["value"], point, "value")
        return _wrap(chart, value.expand_as(point).clone())
    if op == "rotate":
        by = args["by"]
        if not isinstance(by, Mapping) or not by or not set(by) <= set(chart.periodic):
            raise MisconfigurationException(
                f"chart {chart.name!r} rotates only its periodic dimensions {sorted(chart.periodic)}, got {by!r}"
            )
        delta = torch.zeros(chart.intrinsic_dim, dtype=point.dtype, device=point.device)
        for dim, amount in by.items():
            if type(amount) not in (int, float) or not math.isfinite(amount):
                raise MisconfigurationException(f"chart op 'rotate' needs a finite amount for dimension {dim}")
            delta[dim] = float(amount)
        return _wrap(chart, point + delta)
    if op == "scale_norm":
        factor = args["factor"]
        if type(factor) not in (int, float) or not math.isfinite(factor) or factor <= 0:
            raise MisconfigurationException(f"chart op 'scale_norm' needs a positive finite factor, got {factor!r}")
        scale = torch.ones(chart.intrinsic_dim, dtype=point.dtype, device=point.device)
        scale[list(handler.radial(chart))] = float(factor)
        return point * scale
    return _path(chart, _chart_points(chart, args["waypoints"], point, "waypoints"), args["t"], point)


def _chart_points(chart: LatentChart, value: Any, reference: torch.Tensor, what: str) -> torch.Tensor:
    value = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if value.shape[-1:] != (chart.intrinsic_dim,) or not bool(value.isfinite().all()):
        raise MisconfigurationException(
            f"chart {chart.name!r}: {what} needs finite points of {chart.intrinsic_dim} coordinates"
        )
    return value


def _wrap(chart: LatentChart, point: torch.Tensor) -> torch.Tensor:
    if not chart.periodic:
        return point
    columns = list(point.unbind(-1))
    for dim, period in chart.periodic.items():
        columns[dim] = torch.remainder(columns[dim] + period / 2, period) - period / 2
    return torch.stack(columns, -1)


def _path(chart: LatentChart, waypoints: torch.Tensor, t: Any, point: torch.Tensor) -> torch.Tensor:
    if waypoints.ndim != 2 or len(waypoints) < 2:
        raise MisconfigurationException(f"chart {chart.name!r}: a path needs two or more waypoints")
    t = torch.as_tensor(t, dtype=point.dtype, device=point.device)
    if not bool(((t >= 0) & (t <= 1)).all()):
        raise MisconfigurationException(f"chart {chart.name!r}: a path position t must lie in [0, 1]")
    steps = waypoints[1:] - waypoints[:-1]
    for dim, period in chart.periodic.items():
        steps[:, dim] = torch.remainder(steps[:, dim] + period / 2, period) - period / 2
    segments = len(steps)
    where = t * segments
    index = torch.clamp(where.floor().long(), max=segments - 1)
    local = (where - index.to(where.dtype)).unsqueeze(-1)
    moved = waypoints[index] + local * steps[index]
    return _wrap(chart, moved.expand_as(point).clone())


def _orthonormal_rows(chart: LatentChart, key: str, rows: int, rank: int) -> None:
    value = chart.params.get(key)
    if value is None or value.shape != (rows, rank):
        raise MisconfigurationException(f"chart {chart.name!r} needs {key!r} with shape ({rows}, {rank})")
    value = value.to(torch.float64)
    error = float((value @ value.T - torch.eye(rows, dtype=value.dtype)).abs().max())
    if error > 1e-6:
        raise MisconfigurationException(f"chart {chart.name!r}: {key!r} rows are not orthonormal (error {error:g})")


def _origin(chart: LatentChart, rank: int) -> None:
    value = chart.params.get("origin")
    if value is None or value.shape != (rank,):
        raise MisconfigurationException(f"chart {chart.name!r} needs 'origin' with shape ({rank},)")


def _expect_params(chart: LatentChart, keys: Sequence[str]) -> None:
    if set(chart.params) != set(keys):
        raise MisconfigurationException(f"chart {chart.name!r} of kind {chart.kind!r} takes params {sorted(keys)}")


class AffineChart(ChartKind):
    """A flat chart: coordinates along orthonormal frame directions ``axes`` from ``origin``, none periodic."""

    name = "affine"
    ops = frozenset({"set", "path", "scale_norm"})

    def check(self, chart: LatentChart, rank: int) -> None:
        """Refuse parameters that are not an origin and orthonormal axes in a frame of ``rank`` rows."""
        _expect_params(chart, ("origin", "axes"))
        if chart.intrinsic_dim > rank:
            raise MisconfigurationException(f"chart {chart.name!r} has more dimensions than its frame's {rank}")
        if chart.periodic:
            raise MisconfigurationException(f"chart {chart.name!r}: an affine chart has no periodic dimension")
        _origin(chart, rank)
        _orthonormal_rows(chart, "axes", chart.intrinsic_dim, rank)

    def to_chart(self, chart: LatentChart, coordinates: torch.Tensor) -> torch.Tensor:
        """Project frame coordinates onto the chart."""
        p = chart.params_like(coordinates)
        return (coordinates - p["origin"]) @ p["axes"].T

    def from_chart(self, chart: LatentChart, point: torch.Tensor) -> torch.Tensor:
        """Place a chart point in frame coordinates."""
        p = chart.params_like(point)
        return p["origin"] + point @ p["axes"]

    def radial(self, chart: LatentChart) -> tuple[int, ...]:
        """The dimensions measuring distance from the origin."""
        return tuple(range(chart.intrinsic_dim))


class PlaneChart(ChartKind):
    """A plane read in polar coordinates: radius, then an angle with period $2\\pi$, from ``origin`` along ``axes``."""

    name = "pca_plane"
    ops = frozenset(CHART_OPS)

    def check(self, chart: LatentChart, rank: int) -> None:
        """Refuse parameters that are not an origin and orthonormal axes in a frame of ``rank`` rows."""
        _expect_params(chart, ("origin", "axes"))
        if chart.intrinsic_dim != 2 or set(chart.periodic) != {1} or not math.isclose(chart.periodic[1], 2 * math.pi):
            raise MisconfigurationException(
                f"chart {chart.name!r}: a pca_plane chart has two dimensions, the second periodic with period 2*pi"
            )
        _origin(chart, rank)
        _orthonormal_rows(chart, "axes", 2, rank)

    def to_chart(self, chart: LatentChart, coordinates: torch.Tensor) -> torch.Tensor:
        """Project frame coordinates onto the chart."""
        p = chart.params_like(coordinates)
        planar = (coordinates - p["origin"]) @ p["axes"].T
        radius = torch.linalg.vector_norm(planar, dim=-1)
        return torch.stack((radius, torch.atan2(planar[..., 1], planar[..., 0])), -1)

    def from_chart(self, chart: LatentChart, point: torch.Tensor) -> torch.Tensor:
        """Place a chart point in frame coordinates."""
        p = chart.params_like(point)
        radius, angle = point[..., 0], point[..., 1]
        planar = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), -1)
        return p["origin"] + planar @ p["axes"]

    def radial(self, chart: LatentChart) -> tuple[int, ...]:
        """The dimensions measuring distance from the origin."""
        return (0,)


register_chart_kind(AffineChart())
register_chart_kind(PlaneChart())
