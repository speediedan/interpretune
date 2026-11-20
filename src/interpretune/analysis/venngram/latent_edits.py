r"""
Native Latent Frame and Chart Edits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A frame or chart edit of one feature block, run inside a forward pass at the latent sub-hook of the loaded latent model
the block belongs to, ``blocks.<layer>.<site>.hook_sae_acts_post``, on any backend whose hooks compute their edit during
the pass.

:class:`~interpretune.analysis.venngram.latents.LatentCatalog` edits values in a binding's declared linear space. At the
latent sub-hook the hook sees the latent activations $a$, so the values a block edit acts on are the block's decoded
contribution $v = a_B D_B$, where $a_B$ are the block's activations and $D_B$ its decoder rows. The edit computes
$v'$ with the catalog's own function (:meth:`~interpretune.analysis.venngram.latents.LatentCatalog.replace_coordinates`
or :meth:`~interpretune.analysis.venngram.latents.LatentCatalog.apply_chart`) and writes back the minimum-norm change of
the block's activations,

$$
a_B' = a_B + d\left(v' - v\right) D_B^{+}
$$

where $d$ is the edit's dose. The frame spans the rows of $D_B$ (the catalog checks rank and span on every use), so
$v' - v$ lies in that row space and the decoded contribution moves by exactly $d\left(v' - v\right)$. The other blocks,
the latent model's error term and every position outside the token scope are untouched.

The ops (their ``OpSpec`` declared as data in the bundled ``latent`` op collection) are ``frame_replace``, which sets a
block's coordinates in a named frame version, and ``chart_set``, ``chart_path``, ``chart_rotate`` and
``chart_scale_norm``, which edit them in a named chart version
(:func:`~interpretune.analysis.venngram.charts.apply_chart_op`). Each spec's operand is the dose, a scalar: 1 applies
the edit, 0 is the identity. Frame-free block edits need no new op: :func:`block_edit_spec` writes them as core ops on
the block's columns.

:func:`latent_edit_spec` compiles an edit into the hook name and the spec a backend's intervention payload takes.
Before a backend applies it, :func:`~interpretune.analysis.backends.interventions.require_attached_latent_models`
checks that the latent model attached at that hook is the loaded object the address resolved to and that its decoder
rows are still the catalog's.

.. warning:: This is an experimental API and may change in a future release.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import types
from typing import Any

import torch

from interpretune.analysis.backends.capabilities import PositionScope
from interpretune.analysis.backends.interventions import LATENT_SUBHOOK, InterventionSpec
from interpretune.analysis.points.vocabulary import parse
from interpretune.analysis.venngram.latents import LatentBinding, LatentCatalog, LatentFrame, LatentTarget
from interpretune.utils import MisconfigurationException

#: Each native latent edit op and the catalog edit it runs: a frame coordinate replacement, or a chart op.
LATENT_EDIT_OPS: Mapping[str, str] = types.MappingProxyType(
    {
        "frame_replace": "replace",
        "chart_set": "set",
        "chart_path": "path",
        "chart_rotate": "rotate",
        "chart_scale_norm": "scale_norm",
    }
)

#: The key a spec's ``params`` carries its compiled edit under.
LATENT_EDIT_PARAM = "latent_edit"


@dataclass(frozen=True, eq=False)
class LatentEdit:
    """One frame or chart edit of one feature block, compiled against a catalog.

    Args:
        catalog: The catalog the target was resolved in; its frames and charts are looked up at their exact versions.
        target: A canonical feature target (one block of one loaded latent model).
        op: One of :data:`LATENT_EDIT_OPS`.
        name: The frame (``frame_replace``) or chart (the chart ops) the edit is expressed in.
        version: That frame's or chart's exact version.
        args: The edit's arguments: ``coordinates`` for ``frame_replace``; for a chart op, those
            :func:`~interpretune.analysis.venngram.charts.apply_chart_op` takes.
    """

    catalog: LatentCatalog
    target: LatentTarget
    op: str
    name: str
    version: str
    args: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.op not in LATENT_EDIT_OPS:
            raise MisconfigurationException(f"unknown latent edit op {self.op!r}; known: {sorted(LATENT_EDIT_OPS)}")
        if not isinstance(self.target, LatentTarget) or self.target.kind != "feature":
            raise MisconfigurationException(
                f"a native latent edit acts on a feature block of a loaded latent model, not {self.target!r}"
            )
        args = dict(self.args)
        if self.op == "frame_replace" and set(args) != {"coordinates"}:
            raise MisconfigurationException(f"latent edit 'frame_replace' takes ['coordinates'], got {sorted(args)}")
        object.__setattr__(self, "args", types.MappingProxyType(args))
        self.frame()  # refuses an unknown or stale frame or chart version by name now, not inside the forward pass

    @property
    def binding(self) -> LatentBinding:
        """The binding of the loaded latent model the target belongs to."""
        binding = self.catalog.binding(self.target)
        assert isinstance(binding, LatentBinding)
        return binding

    @property
    def model(self) -> Any:
        """The loaded latent model the target resolved to."""
        return self.binding.model

    @property
    def columns(self) -> tuple[int, ...]:
        """The block's columns among the latent model's feature activations."""
        size = self.binding.block_size
        return tuple(range(self.target.unit * size, (self.target.unit + 1) * size))

    def decoder_rows(self) -> torch.Tensor:
        """The block's decoder rows, in the binding's declared space."""
        return self.catalog.basis(self.target)

    def frame(self) -> LatentFrame:
        """The frame the edit writes coordinates in, revalidated against the block's current decoder rows."""
        if self.op == "frame_replace":
            return self.catalog.frame(self.target, name=self.name, version=self.version)
        chart = self.catalog.chart(self.target, name=self.name, version=self.version)
        return self.catalog.frame(self.target, name=chart.frame, version=chart.frame_version)

    def edit_values(self, values: torch.Tensor) -> torch.Tensor:
        """The catalog's edit of the block's decoded values ``(..., width)``."""
        if self.op == "frame_replace":
            coordinates = torch.as_tensor(self.args["coordinates"], dtype=values.dtype, device=values.device)
            coordinates = coordinates.expand(*values.shape[:-1], coordinates.shape[-1])
            return self.catalog.replace_coordinates(
                self.target, values, coordinates, frame=self.name, version=self.version
            )
        return self.catalog.apply_chart(
            self.target, values, chart=self.name, version=self.version, op=LATENT_EDIT_OPS[self.op], **self.args
        )


def latent_edit_of(spec: InterventionSpec) -> LatentEdit:
    """The compiled edit a latent edit op's spec carries, refusing a spec without one by name."""
    edit = (spec.params or {}).get(LATENT_EDIT_PARAM)
    if not isinstance(edit, LatentEdit):
        raise MisconfigurationException(
            f"op {spec.op!r} runs a compiled latent edit, and this spec carries none in params[{LATENT_EDIT_PARAM!r}];"
            " build it with interpretune.analysis.venngram.latent_edit_spec"
        )
    return edit


def latent_edit_executor(region: torch.Tensor, operand: torch.Tensor, spec: InterventionSpec) -> torch.Tensor:
    """Every latent edit op's executor: the block's activations ``(rows, block_size)`` moved so their decoded
    values take the catalog's edit, scaled by the dose ``operand``."""
    edit = latent_edit_of(spec)
    rows = edit.decoder_rows().detach()
    if region.shape[-1] != rows.shape[0]:
        raise MisconfigurationException(
            f"latent edit {spec.op!r} on {edit.target} needs the block's {rows.shape[0]} columns, got a region of"
            f" width {region.shape[-1]}; compile the spec with latent_edit_spec, which selects them"
        )
    if operand.numel() != 1:
        raise MisconfigurationException(
            f"latent edit {spec.op!r} takes its dose, a scalar, as its operand; got shape {tuple(operand.shape)}"
        )
    inverse = torch.linalg.pinv(rows.to(torch.float64), rtol=edit.frame().rank_rtol)
    rows, inverse = rows.to(region), inverse.to(region)
    values = region @ rows
    return region + operand.reshape(()) * ((edit.edit_values(values) - values) @ inverse)


def latent_hook_name(binding: LatentBinding) -> str:
    """The latent sub-hook of a binding's loaded model, refusing a model whose declared hook is not the binding's
    layer and point."""
    hook = getattr(getattr(getattr(binding.model, "cfg", None), "metadata", None), "hook_name", None)
    if not isinstance(hook, str) or not hook:
        raise MisconfigurationException(
            f"latent model {binding.name!r} declares no hook (cfg.metadata.hook_name); a native latent edit runs at"
            " the hook its attached model is spliced at"
        )
    point = parse(hook)
    spelled = point.base if point.contribution is None else f"hook_{point.contribution}_out"
    if point.layer != binding.layer or spelled != binding.point:
        raise MisconfigurationException(
            f"latent model {binding.name!r} is attached at {hook!r}, but its binding declares layer {binding.layer}"
            f" point {binding.point!r}"
        )
    return f"{hook}.{LATENT_SUBHOOK}"


def _ensure_registered(op: str) -> None:
    """Load the op definitions once, which registers the bundled latent edit ops, if ``op`` is not yet known."""
    from venngram.ops import OPS

    if op not in OPS:
        from interpretune.analysis.ops.dispatcher import DISPATCHER

        DISPATCHER.load_definitions()


def latent_edit_spec(
    catalog: LatentCatalog,
    target: LatentTarget,
    op: str,
    *,
    name: str,
    version: str,
    dose: float = 1.0,
    position_scope: PositionScope | str = PositionScope.LAST_TOKEN,
    **args: Any,
) -> tuple[str, InterventionSpec]:
    """Compile a frame or chart edit of one feature block into ``(hook name, spec)`` for a backend's payload.

    Args:
        catalog: The catalog ``target`` was resolved in.
        target: A canonical feature target.
        op: One of :data:`LATENT_EDIT_OPS`.
        name: The frame or chart, by name.
        version: Its exact version.
        dose: The fraction of the edit applied, 1 for all of it.
        position_scope: The token scope, independent of the block selected.
        args: The edit's arguments (see :class:`LatentEdit`).

    Raises:
        MisconfigurationException: For an unknown op, a target that is not a feature block, an unknown or stale frame
            or chart version, or a latent model whose declared hook is not its binding's.
    """
    edit = LatentEdit(catalog, target, op, name, version, args)
    hook = latent_hook_name(edit.binding)
    _ensure_registered(op)
    spec = InterventionSpec(
        torch.tensor(float(dose)),
        op=op,
        position_scope=position_scope,
        unit_indices=edit.columns,
        params={LATENT_EDIT_PARAM: edit},
    )
    return hook, spec


def block_edit_spec(
    catalog: LatentCatalog,
    target: LatentTarget,
    op: str,
    *,
    value: torch.Tensor | float | None = None,
    position_scope: PositionScope | str = PositionScope.LAST_TOKEN,
) -> tuple[str, InterventionSpec]:
    """A frame-free edit of a whole feature block, as a core op on the block's columns at its latent sub-hook.

    ``set`` writes ``value``, the block's activations from a source run (one per block coordinate); ``scale_norm``
    multiplies the block by ``value``, which scales its decoded contribution, and so its norm, by the same factor;
    ``ablate`` zeroes it.

    Raises:
        MisconfigurationException: For another op, a missing or misshapen value, or a target that is not a feature
            block of a model attached at its binding's hook.
    """
    if not isinstance(target, LatentTarget) or target.kind != "feature":
        raise MisconfigurationException(
            f"a block edit acts on a feature block of a loaded latent model, not {target!r}"
        )
    binding = catalog.binding(target)
    assert isinstance(binding, LatentBinding)
    size = binding.block_size
    columns = tuple(range(target.unit * size, (target.unit + 1) * size))
    if op == "ablate":
        if value is not None:
            raise MisconfigurationException("block edit 'ablate' takes no value")
        tensor, core = torch.zeros(size), "replace"
    elif op == "set":
        if value is None or torch.as_tensor(value).shape != (size,):
            raise MisconfigurationException(f"block edit 'set' needs a value of {size} activations, got {value!r}")
        tensor, core = torch.as_tensor(value, dtype=torch.float32), "replace"
    elif op == "scale_norm":
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not value > 0:
            raise MisconfigurationException(f"block edit 'scale_norm' needs a positive factor, got {value!r}")
        tensor, core = torch.tensor(float(value)), "scale"
    else:
        raise MisconfigurationException(f"unknown block edit {op!r}; known: ['ablate', 'scale_norm', 'set']")
    spec = InterventionSpec(tensor, op=core, position_scope=position_scope, unit_indices=columns)
    return latent_hook_name(binding), spec


# --------------------------------------------------------------------------------------------------------------------
# Contract probes: a catalog small enough to build at load, for each op's executor
# --------------------------------------------------------------------------------------------------------------------

_PROBE_BLOCK, _PROBE_WIDTH, _PROBE_ROWS = 3, 5, 4


def _probe_catalog(generator: torch.Generator) -> tuple[LatentCatalog, LatentTarget]:
    from interpretune.analysis.venngram.charts import LatentChart

    decoder = torch.randn(2 * _PROBE_BLOCK, _PROBE_WIDTH, generator=generator, dtype=torch.float64)
    model = types.SimpleNamespace(
        cfg=types.SimpleNamespace(metadata=types.SimpleNamespace(hook_name="blocks.0.hook_out"))
    )
    binding = LatentBinding("probe", 0, "hook_out", model, decoder, _PROBE_BLOCK)
    target = LatentTarget("feature", "probe", 0, binding.point, 0)
    rows = torch.linalg.qr(decoder[:_PROBE_BLOCK].T).Q.T.contiguous()
    frame = LatentFrame("probe", "1", target, rows)
    axes = torch.eye(_PROBE_BLOCK, dtype=torch.float64)
    origin = torch.zeros(_PROBE_BLOCK, dtype=torch.float64)
    plane = LatentChart(
        "plane", "1", "pca_plane", target, "probe", "1", 2, {1: 2 * torch.pi}, {"origin": origin, "axes": axes[:2]}
    )
    return LatentCatalog([binding], frames=[frame], charts=[plane]), target


def _probe(
    op: str, name: str, generator: torch.Generator, **args: Any
) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]:
    catalog, target = _probe_catalog(generator)
    region = torch.randn(_PROBE_ROWS, _PROBE_BLOCK, generator=generator)
    edit = LatentEdit(catalog, target, op, name, "1", args)
    return region, torch.tensor(0.75), {LATENT_EDIT_PARAM: edit}


def probe_frame_replace(generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]:
    """The contract probe for ``frame_replace``."""
    return _probe("frame_replace", "probe", generator, coordinates=[0.5, -1.0, 2.0])


def probe_chart_set(generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]:
    """The contract probe for ``chart_set``."""
    return _probe("chart_set", "plane", generator, value=[1.5, 0.5])


def probe_chart_path(generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]:
    """The contract probe for ``chart_path``."""
    return _probe("chart_path", "plane", generator, waypoints=[[1.0, 0.0], [2.0, 1.0]], t=0.25)


def probe_chart_rotate(generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]:
    """The contract probe for ``chart_rotate``."""
    return _probe("chart_rotate", "plane", generator, by={1: 0.5})


def probe_chart_scale_norm(generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]:
    """The contract probe for ``chart_scale_norm``."""
    return _probe("chart_scale_norm", "plane", generator, factor=1.5)


__all__ = [
    "LATENT_EDIT_OPS",
    "LATENT_EDIT_PARAM",
    "LatentEdit",
    "block_edit_spec",
    "latent_edit_executor",
    "latent_edit_of",
    "latent_edit_spec",
    "latent_hook_name",
]
