# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trainable interventions spliced into a module's forward pass, as tunable ``intervention:<name>`` tensors (#812,
#950).

An *intervention* applies an op (``add``, ``scale``, ``reject``, ...: venngram's registry, :data:`venngram.ops.OPS`) to
a site's activations. What distinguishes kinds of intervention is the op and where its operand comes from: a constant
or a dose (zero ablation), a value computed from the model or the store (a concept direction, another run's
activation), the output of a latent model (:mod:`interpretune.tuning.latents`), or parameters of its own. This module is
the seam for the last kind, *trainable* interventions, whose operand is tuned by a schedule:

- :class:`OpIntervention`, an op applied at every position with an operand module's output, through the op math every
  model backend shares (:func:`~interpretune.analysis.backends.interventions.apply_op`). A steering vector is the
  ``add`` op with a learned vector (``OpIntervention("add", d_model)``); a learned per-unit gain is ``scale``;
- :class:`LearnedOperand`, a trainable operand, initialized to its op's identity so attaching it changes nothing until
  it trains;
- :func:`attach_intervention` and :func:`detach_intervention`, which register an intervention under the root's
  ``interventions`` and splice it in with a forward hook on the module whose output it rewrites;
- :func:`intervention_roles`, the roles to declare: ``intervention:<name>`` (venngram's role vocabulary, as PEFT
  adapters are ``adapter:<name>``) for a tunable intervention; for a frozen one, ``fitted`` when its value is computed
  in the run (checkpointed, never optimized) or ``artifact`` when it is loaded from a published or saved source
  (reloaded, never saved; finetuning-scheduler never thaws an artifact, and promotes artifacts only to latent roles).
  Every interpretune module declares them through its cooperative ``tunable_roles()``, which its venngram view reads,
  so a schedule selects one intervention with ``role:intervention:<name>`` and all of them with ``role:intervention``;
- :class:`InterventionCfg`, an intervention declared in a module's configuration (``tuning_interventions``), which
  every interpretune module attaches once its model is built (:func:`attach_configured_interventions`), so an example
  needs no module code to steer.

Interventions without parameters (an ablation, a patch, a dose of a concept direction) need no role and no seam: they
are applied by the analysis ops (``model_fwd_intervention``, ``feature_intervention_forward``) during a pass.

Sites are module paths under the root, as latent models' are, so the same intervention attaches to a core module's
``model.model.layers.N``, an nnsight module's ``hf_model.model.layers.N`` or a TransformerLens bridge's
``model.blocks.N.hook_out``. A venngram site address (``point:hook_resid_post & layers:13``) names the same site
once for every backend: :func:`~interpretune.tuning.sites.resolve_site` resolves it on the composition the
intervention attaches to.

A trained intervention is held fixed in a later stage with finetuning-scheduler's per-phase ``refreeze`` key, rather
than by attaching it frozen.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import torch
from torch import nn

from interpretune.tuning.latents import _first, _require_reachable, _with_first
from interpretune.tuning.sites import book_sets, is_site_address, resolve_site

# the `nn.ModuleDict` on the root that holds attached interventions, by name
INTERVENTIONS_ATTR = "interventions"

#: The ops a trainable intervention applies. Other registered ops are refused until a test covers them.
TRAINABLE_OPS = frozenset({"add", "scale"})

_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")
_HOOKS_ATTR = "_intervention_hooks"
_SITES_ATTR = "_intervention_sites"
# each attached intervention's resolved module, by name: where its view places it (`intervention_placements`)
_PLACED_ATTR = "_intervention_placed"


def _require_trainable_op(op: str) -> None:
    from venngram.ops import OPS

    if op not in OPS:
        raise ValueError(f"{op!r} is not a registered op; venngram's registry holds {', '.join(sorted(OPS))}")
    if op not in TRAINABLE_OPS:
        raise ValueError(
            f"A trainable intervention applies {', '.join(sorted(TRAINABLE_OPS))}; {op!r} is a registered op that this"
            " seam does not yet apply with a trainable operand"
        )


class LearnedOperand(nn.Module):
    """A trainable operand: one tensor, starting at ``init`` everywhere.

    Args:
        shape: The operand's shape: an ``int`` for one vector over a site's last axis (a residual stream's
            ``d_model``), or a tuple (``(n_heads, d_head)`` for a per-head operand).
        init: The starting value, by default ``0.0``; :class:`OpIntervention` passes the value at which its op is the
            identity.
    """

    def __init__(self, shape: int | Sequence[int], init: float = 0.0) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.full((shape,) if isinstance(shape, int) else tuple(shape), float(init)))

    def forward(self) -> torch.Tensor:
        """The operand."""
        return self.value


class OpIntervention(nn.Module):
    """An op applied at every position of a site's activations with a trainable operand.

    The op is a name from venngram's registry (:data:`TRAINABLE_OPS` lists those applied here); the operand is a
    module whose output is the op's tensor, so its parameters are what a schedule tunes. The op math is the one every
    model backend shares, so ``add`` here is the same ``add`` an analysis op applies.

    Args:
        op: The op: ``add`` (``x + scale_factor * operand``) or ``scale`` (``x * (scale_factor * operand)``).
        operand: A module returning the operand, or the operand's shape (an ``int`` or a tuple, as
            :class:`LearnedOperand` takes) for a :class:`LearnedOperand` starting where the op is the identity (``0``
            for ``add``, ``1 / scale_factor`` for ``scale``).
        scale_factor: A fixed multiplier on the operand (nonzero).
    """

    def __init__(self, op: str, operand: nn.Module | int | Sequence[int], scale_factor: float = 1.0) -> None:
        super().__init__()
        _require_trainable_op(op)
        if not scale_factor:
            raise ValueError(
                "A trainable intervention's scale_factor must be nonzero: at zero its operand has no effect"
            )
        self.op = op
        self.scale_factor = scale_factor
        if not isinstance(operand, nn.Module):
            operand = LearnedOperand(operand, 0.0 if op == "add" else 1.0 / scale_factor)
        self.operand = operand

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` with the op applied."""
        # imported here: this module is imported with the module configuration, and the op seam imports the analysis
        # package, which a configuration import must not pull in
        from interpretune.analysis.backends.interventions import InterventionSpec, PositionScope, apply_op

        spec = InterventionSpec(
            self.operand(), op=self.op, scale_factor=self.scale_factor, position_scope=PositionScope.ALL_POSITIONS
        )
        return apply_op(x, spec)


class _ConstantOperand(nn.Module):
    value: torch.Tensor

    def __init__(self, value: torch.Tensor | float) -> None:
        super().__init__()
        self.register_buffer("value", torch.as_tensor(value).detach().clone())

    def forward(self) -> torch.Tensor:
        return self.value


def _unit_clamp_operand(count: int, operand: Any, lower: float | None, upper: float | None) -> nn.Module:
    if operand is not None or (lower is None and upper is None):
        raise ValueError("a unit clamp needs at least one bound and takes no operand")
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("unit clamp bounds are reversed")
    return _ConstantOperand(torch.eye(count))


def _unit_operand(
    op: str, count: int, operand: Any, scale_factor: float, lower: float | None, upper: float | None
) -> nn.Module:
    if op == "clamp":
        return _unit_clamp_operand(count, operand, lower, upper)
    if lower is not None or upper is not None:
        raise ValueError(f"unit op {op!r} takes an operand, not clamp bounds")
    if operand is None:
        _require_trainable_op(op)
        if not scale_factor:
            raise ValueError("a trainable unit operand needs a nonzero scale_factor")
        return LearnedOperand(count, 0.0 if op == "add" else 1.0 / scale_factor)
    return operand if isinstance(operand, nn.Module) else _ConstantOperand(operand)


class UnitIntervention(nn.Module):
    """Apply an op only to selected last-axis units, preserving their complement and its gradient.

    Args:
        op: ``add``, ``scale``, ``replace`` or ``clamp``.
        units: Explicit nonnegative indices, in the operand's coordinate order.
        operand: A module returning the operand, or a fixed tensor/scalar. Without one, ``add`` and ``scale``
            allocate a trainable vector at their identity. ``replace`` needs an operand. ``clamp`` takes no
            operand: its basis is the selected model units.
        scale_factor: Multiplier for an add or scale operand.
        clamp_min: Optional lower unit-coordinate bound.
        clamp_max: Optional upper unit-coordinate bound.

    Attach with :func:`attach_intervention`, using ``frozen="fitted"`` or ``"artifact"`` for a fixed operand or
    clamp. A trainable operand has one value per selected unit; no full-width tensor is optimized.
    """

    def __init__(
        self,
        op: str,
        units: Sequence[int],
        operand: nn.Module | torch.Tensor | float | None = None,
        *,
        scale_factor: float = 1.0,
        clamp_min: float | None = None,
        clamp_max: float | None = None,
    ) -> None:
        super().__init__()
        self.unit_indices = tuple(units)
        if (
            not self.unit_indices
            or any(type(unit) is not int or unit < 0 for unit in self.unit_indices)
            or len(set(self.unit_indices)) != len(self.unit_indices)
        ):
            raise ValueError(f"units {self.unit_indices!r} must be nonempty, distinct nonnegative integer indices")
        if op not in {"add", "scale", "replace", "clamp"}:
            raise ValueError(f"UnitIntervention applies add, scale, replace or clamp, not {op!r}")
        self.operand = _unit_operand(op, len(self.unit_indices), operand, scale_factor, clamp_min, clamp_max)
        self.op, self.scale_factor = op, scale_factor
        self.clamp_min, self.clamp_max = clamp_min, clamp_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The op applied at every position, only on the selected units."""
        from interpretune.analysis.backends.interventions import InterventionSpec, PositionScope, apply_op

        return apply_op(
            x,
            InterventionSpec(
                self.operand(),
                op=self.op,
                scale_factor=self.scale_factor,
                unit_indices=self.unit_indices,
                clamp_min=self.clamp_min,
                clamp_max=self.clamp_max,
                position_scope=PositionScope.ALL_POSITIONS,
            ),
        )


def _unit_splice(
    root: nn.Module, address: str, sets: Mapping[str, Any] | None = None
) -> tuple[str, tuple[int, ...] | None]:
    """A one-site unit address split into its site address and explicit unit indices.

    A reference to an address book's named set (``@steer_units``) is followed through ``sets`` first, so a book entry
    gives the splice both its site and its units.
    """
    from venngram.address import Ref, Term, parse, to_string

    expr = parse(address)
    seen: set[str] = set()
    while isinstance(expr, Ref) and sets and expr.name in sets and expr.name not in seen:
        seen.add(expr.name)  # the book refuses reference cycles; this only keeps the walk finite
        expr = sets[expr.name]
    if not isinstance(expr, Term) or expr.units is None:
        return address, None
    if expr.point is None:
        raise ValueError(f"unit splice {address!r} must name a point")
    site = to_string(replace(expr, units=None))
    resolved = resolve_site(root, site, sets)
    layer = int(resolved.hook.split(".")[1])
    indices = expr.units.flat
    if expr.units.per_layer is not None:
        if set(expr.units.layers or ()) != {layer}:
            raise ValueError(f"unit splice {address!r} must name units of its one layer, {layer}")
        indices = dict(expr.units.per_layer)[layer]
    if indices is None or indices.indices is None or any(index < 0 for index in indices.indices):
        raise ValueError(f"unit splice {address!r} needs explicit nonnegative indices; its width is checked at forward")
    return site, tuple(indices.indices)


def _interventions(root: nn.Module, create: bool = False) -> nn.ModuleDict | None:
    held = root._modules.get(INTERVENTIONS_ATTR)
    if isinstance(held, nn.ModuleDict):
        return held
    if held is not None:
        raise AttributeError(f"{type(root).__name__}.{INTERVENTIONS_ATTR} is not an nn.ModuleDict")
    if not create:
        return None
    root.add_module(INTERVENTIONS_ATTR, held := nn.ModuleDict())
    return held


def _site_device(root: nn.Module, target: nn.Module) -> torch.device | None:
    """The device of the site's tensors, or of ``root``'s when the site holds none (a TransformerLens hook
    point)."""
    for module in (target, root):
        for t in (*module.parameters(), *module.buffers()):
            return t.device
    return None


# the role of a frozen intervention's tensors: `fitted` (computed in the run) or `artifact` (loaded from a source)
FrozenKind = Literal["fitted", "artifact"]

FROZEN_KINDS: tuple[FrozenKind, ...] = ("fitted", "artifact")


def attach_intervention(
    root: nn.Module,
    name: str,
    intervention: nn.Module,
    site: str,
    *,
    frozen: Literal[False] | FrozenKind = False,
    sets: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Register ``intervention`` on ``root`` as ``interventions.<name>`` and splice it in at ``site``.

    The intervention rewrites the output of the module at ``site`` (the first element of a tuple output, as HF decoder
    layers return): the output becomes ``intervention(output)``. A site address that resolves to a module's input
    (a block's residual input, say) rewrites that input instead, before the module runs. Its parameters become
    ``root``'s (``interventions.<name>.<...>``), so they are saved in its state dict and reachable by its tunable view.
    :func:`intervention_roles` declares them ``intervention:<name>``, or the ``frozen`` role kind. The
    intervention moves to the site's device (a model placed by ``device_map`` is not moved again by the core loop), and
    keeps its dtype.

    Args:
        root: The module the intervention is registered on (an interpretune module is its view's root).
        name: The intervention's name: letters, digits, ``_`` and ``-``.
        intervention: An ``nn.Module`` mapping the site's activations to their rewritten value.
        site: Path, under ``root``, of the module whose output is rewritten, or a venngram site address selecting one
            site (``point:hook_resid_post & layers:13``), resolved on ``root``'s composition by
            :func:`~interpretune.tuning.sites.resolve_site`.
        frozen: Declare its parameters with a frozen role rather than ``intervention:<name>``, for an intervention
            applied but never trained. ``"fitted"`` for a value computed in this run (a mean difference, a fitted
            direction): finetuning-scheduler checkpoints it and never optimizes it, so a resumed run keeps it.
            ``"artifact"`` for a value loaded from a published or saved source: never checkpointed, because it is
            reloaded from that source, and promotable to ``latent:<name>``. ``False`` (the default) is tunable.
        sets: An address book's named sets (:func:`~interpretune.tuning.sites.book_sets`), when ``site`` references
            one (``@steer_site``).

    Returns:
        ``intervention``.

    Raises:
        ValueError: ``name`` is not a valid name or is already attached, ``frozen`` is neither ``False`` nor a frozen
            role kind (``True`` included: it does not say whether the value is saved), or ``root`` is a circuit-tracer
            composition whose forward never runs through ``site``, or a site address does not resolve to one site on
            this composition.
        AttributeError: the site's module does not exist under ``root``.
    """
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ValueError(f"Intervention name {name!r} must be letters, digits, '_' and '-' only")
    if not isinstance(intervention, nn.Module):
        raise ValueError(f"{type(intervention).__name__} is not an nn.Module")
    if frozen is not False and frozen not in FROZEN_KINDS:
        raise ValueError(
            f"frozen={frozen!r} must be False, 'fitted' (a value computed in this run: checkpointed, never optimized)"
            " or 'artifact' (a value loaded from a published or saved source: reloaded, never checkpointed)"
        )
    held = _interventions(root, create=True)
    assert held is not None
    if name in held:
        raise ValueError(f"An intervention named {name!r} is already attached; detach it first")
    path, io = site, "output"
    if is_site_address(site):
        site_address, units = _unit_splice(root, site, sets)
        if units is not None and (not isinstance(intervention, UnitIntervention) or intervention.unit_indices != units):
            raise ValueError(f"unit site {site!r} requires a UnitIntervention with units {units}")
        resolved = resolve_site(root, site_address, sets)
        path, io = resolved.path, resolved.io
    target = root.get_submodule(path)
    _require_reachable(root, "interventions", target)
    if (device := _site_device(root, target)) is not None:
        intervention.to(device=device)

    def rewrite(module: nn.Module, args: tuple[Any, ...], output: Any) -> Any:
        return _with_first(output, intervention(_first(output)))

    def rewrite_input(module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
        return (intervention(args[0]), *args[1:])

    if io == "input":
        handle = target.register_forward_pre_hook(rewrite_input)
    else:
        handle = target.register_forward_hook(rewrite)
    held[name] = intervention
    hooks: dict[str, Any] = root.__dict__.setdefault(_HOOKS_ATTR, {})
    sites: dict[str, tuple[str, Literal[False] | FrozenKind]] = root.__dict__.setdefault(_SITES_ATTR, {})
    hooks[name] = handle
    sites[name] = (site, frozen)
    root.__dict__.setdefault(_PLACED_ATTR, {})[name] = path
    return intervention


def detach_intervention(root: nn.Module, name: str) -> nn.Module:
    """Remove the intervention ``name`` from ``root`` and its site's forward pass.

    Returns:
        The detached intervention.

    Raises:
        KeyError: no intervention of that name is attached.
    """
    held = _interventions(root)
    if held is None or name not in held:
        raise KeyError(f"No intervention named {name!r} is attached")
    root.__dict__.get(_HOOKS_ATTR, {}).pop(name).remove()
    root.__dict__.get(_SITES_ATTR, {}).pop(name, None)
    root.__dict__.get(_PLACED_ATTR, {}).pop(name, None)
    intervention = held[name]
    del held[name]
    return intervention


def intervention_sites(root: nn.Module) -> Mapping[str, str]:
    """The site of each intervention attached to ``root``, by name."""
    return {name: site for name, (site, _) in root.__dict__.get(_SITES_ATTR, {}).items()}


def intervention_placements(root: nn.Module) -> dict[str, str]:
    """Where ``root``'s view places each attached intervention: its module (``interventions.<name>``) by the module
    path, under ``root``, of the module whose input or output it rewrites (its resolved site).

    Interventions are registered beside the model, so venngram's view takes this mapping (``attached``) to give their
    tensors the layer and component of the module they are spliced at.
    """
    return {f"{INTERVENTIONS_ATTR}.{name}": path for name, path in root.__dict__.get(_PLACED_ATTR, {}).items()}


def intervention_roles(root: nn.Module) -> dict[str, str]:
    """The role of every attached intervention's parameters and buffers, by their names under ``root``."""
    held = _interventions(root)
    if held is None:
        return {}
    sites = root.__dict__.get(_SITES_ATTR, {})
    roles: dict[str, str] = {}
    for name, intervention in held.items():
        _, frozen = sites[name]
        role = frozen or f"intervention:{name}"
        tensors = [*intervention.named_parameters(), *intervention.named_buffers()]
        roles.update({f"{INTERVENTIONS_ATTR}.{name}.{n}": role for n, _ in tensors})
    return roles


def intervention_role_addresses(root: nn.Module) -> dict[str, str]:
    """The roles of :func:`intervention_roles`, declared by address: each role mapped to the interventions it
    covers (``{"intervention:steer": "module:interventions.steer"}``), as venngram views take them (``roles=``)."""
    from interpretune.tuning.views import merge_roles, module_address

    held = _interventions(root)
    if held is None:
        return {}
    sites = root.__dict__.get(_SITES_ATTR, {})
    declared = []
    for name, intervention in held.items():
        if [*intervention.parameters(), *intervention.buffers()]:  # venngram refuses an address that selects nothing
            declared.append({sites[name][1] or f"intervention:{name}": module_address(f"{INTERVENTIONS_ATTR}.{name}")})
    return merge_roles(*declared)


@dataclass(frozen=True, kw_only=True)
class InterventionCfg:
    """A trainable intervention declared in a module's configuration, attached once the module's model is built.

    Args:
        name: The intervention's name (``interventions.<name>``).
        site: Path, under the module, of the module whose output it rewrites, or a venngram site address
            (``point:hook_resid_post & layers:13``), which may reference a named set of ``book`` (``@steer_site``).
            Explicit ``units`` on an address allocate one learned value per selected unit, as a
            :class:`UnitIntervention`.
        op: The op it applies (:data:`TRAINABLE_OPS`): ``add`` (a steering vector, the default) or ``scale``.
        shape: The operand's shape (an ``int``, or a list or tuple of them). Defaults to the model's hidden size
            (``config.hidden_size`` of an HF or nnsight model, ``cfg.d_model`` of a TransformerLens one): one vector
            over a residual-stream site.
        scale_factor: A fixed multiplier on the operand.
        book: The address book whose named sets ``site`` references: a local path to a book document, or a venngram
            store reference to a published one, so one shared book names the site on every backend.

    A configured intervention's operand starts where its op is the identity and is tunable (``intervention:<name>``).
    A frozen one would be a no-op, so there is no ``frozen`` setting: a later stage holds a trained operand fixed with
    ``refreeze``, and a precomputed one is attached in code (``attach_intervention(..., frozen="artifact")``, or
    ``"fitted"`` for a value computed in the run).
    """

    name: str
    site: str
    op: str = "add"
    shape: int | tuple[int, ...] | None = None
    scale_factor: float = 1.0
    book: str | None = None

    def __post_init__(self) -> None:
        # a YAML list arrives as a list; the operand's shape is a tuple
        if isinstance(self.shape, list):
            object.__setattr__(self, "shape", tuple(self.shape))


def as_intervention_cfg(entry: InterventionCfg | Mapping[str, Any]) -> InterventionCfg:
    """An :class:`InterventionCfg` from itself or from the mapping a YAML configuration gives.

    jsonargparse types ``tuning_interventions`` when it parses ``ITConfig`` itself, but a configuration synthesized by
    auto-composition (a TransformerLens or SAE Lens composition) is built from mappings, which arrive untyped.
    """
    if isinstance(entry, InterventionCfg):
        return entry
    if isinstance(entry, Mapping):
        return InterventionCfg(**entry)
    raise TypeError(f"An intervention configuration must be an InterventionCfg or a mapping, got {entry!r}")


def _model_hidden_size(root: nn.Module) -> int | None:
    model = getattr(root, "model", None)
    for cfg_attr, size_attr in (("config", "hidden_size"), ("cfg", "d_model")):
        size = getattr(getattr(model, cfg_attr, None), size_attr, None)
        if isinstance(size, int):
            return size
    return None


def attach_configured_interventions(
    root: nn.Module, cfgs: Sequence[InterventionCfg | Mapping[str, Any]]
) -> list[nn.Module]:
    """Build and attach each configured intervention (an :class:`InterventionCfg` or its mapping) in order.

    Returns:
        The attached interventions.

    Raises:
        ValueError: an op is not one a trainable intervention applies, or no shape is given and the model's hidden size
            cannot be read.
        TypeError: an entry is neither an :class:`InterventionCfg` nor a mapping.
    """
    attached = []
    books: dict[str, dict[str, Any]] = {}  # each book is read once, however many interventions it serves
    for cfg in map(as_intervention_cfg, cfgs):
        sets = None
        if cfg.book is not None:
            if cfg.book not in books:
                books[cfg.book] = book_sets(cfg.book)
            sets = books[cfg.book]
        units = _unit_splice(root, cfg.site, sets)[1] if is_site_address(cfg.site) else None
        if units is not None:
            if cfg.shape is not None and cfg.shape not in (len(units), (len(units),)):
                raise ValueError(f"unit intervention {cfg.name!r} has {len(units)} units, not shape {cfg.shape!r}")
            intervention = UnitIntervention(cfg.op, units, scale_factor=cfg.scale_factor)
            attached.append(attach_intervention(root, cfg.name, intervention, cfg.site, sets=sets))
            continue
        shape = cfg.shape if cfg.shape is not None else _model_hidden_size(root)
        if shape is None:
            raise ValueError(f"Intervention {cfg.name!r} needs a `shape`: the model's hidden size is not readable")
        intervention = OpIntervention(cfg.op, shape, scale_factor=cfg.scale_factor)
        attached.append(attach_intervention(root, cfg.name, intervention, cfg.site, sets=sets))
    return attached
