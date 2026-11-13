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
  adapters are ``adapter:<name>``) for a tunable intervention, or ``instrument`` for a precomputed one applied but never
  trained (finetuning-scheduler never thaws an instrument, and promotes instruments only to latent roles). Every
  interpretune module declares them through its cooperative ``tunable_roles()``, which its venngram view reads, so a
  schedule selects one intervention with ``role:intervention:<name>`` and all of them with ``role:intervention``;
- :class:`InterventionCfg`, an intervention declared in a module's configuration (``tuning_interventions``), which
  every interpretune module attaches once its model is built (:func:`attach_configured_interventions`), so an example
  needs no module code to steer.

Interventions without parameters (an ablation, a patch, a dose of a concept direction) need no role and no seam: they
are applied by the analysis ops (``model_fwd_intervention``, ``feature_intervention_forward``) during a pass.

Sites are module paths under the root, as latent models' are, so the same intervention attaches to a core module's
``model.model.layers.N``, an nnsight module's ``hf_model.model.layers.N`` or a TransformerLens bridge's
``model.blocks.N.hook_out``. Venngram site addresses (``point:resid_post & layers:13``) replace the per-backend
spellings once interpretune resolves them (#864, #876).

A trained intervention is held fixed in a later stage with finetuning-scheduler's per-phase ``refreeze`` key, rather
than by attaching it frozen.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from interpretune.tuning.latents import _first, _require_reachable, _with_first

# the `nn.ModuleDict` on the root that holds attached interventions, by name
INTERVENTIONS_ATTR = "interventions"

#: The ops a trainable intervention applies. Other registered ops are refused by name until a test covers them.
TRAINABLE_OPS = frozenset({"add", "scale"})

_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")
_HOOKS_ATTR = "_intervention_hooks"
_SITES_ATTR = "_intervention_sites"


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


def attach_intervention(
    root: nn.Module, name: str, intervention: nn.Module, site: str, *, frozen: bool = False
) -> nn.Module:
    """Register ``intervention`` on ``root`` as ``interventions.<name>`` and splice it in at ``site``.

    The intervention rewrites the output of the module at ``site`` (the first element of a tuple output, as HF decoder
    layers return): the output becomes ``intervention(output)``. Its parameters become ``root``'s
    (``interventions.<name>.<...>``), so they are saved in its state dict and reachable by its tunable view.
    :func:`intervention_roles` declares them ``intervention:<name>``, or ``instrument`` when ``frozen``. The
    intervention moves to the site's device (a model placed by ``device_map`` is not moved again by the core loop), and
    keeps its dtype.

    Args:
        root: The module the intervention is registered on (an interpretune module is its view's root).
        name: The intervention's name: letters, digits, ``_`` and ``-``.
        intervention: An ``nn.Module`` mapping the site's activations to their rewritten value.
        site: Path, under ``root``, of the module whose output is rewritten.
        frozen: Declare its parameters ``instrument`` rather than ``intervention:<name>``: for a precomputed
            intervention applied but never trained.

    Returns:
        ``intervention``.

    Raises:
        ValueError: ``name`` is not a valid name or is already attached, or ``root`` is a circuit-tracer composition
            whose forward never runs through ``site``.
        AttributeError: the site's module does not exist under ``root``.
    """
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ValueError(f"Intervention name {name!r} must be letters, digits, '_' and '-' only")
    if not isinstance(intervention, nn.Module):
        raise ValueError(f"{type(intervention).__name__} is not an nn.Module")
    held = _interventions(root, create=True)
    assert held is not None
    if name in held:
        raise ValueError(f"An intervention named {name!r} is already attached; detach it first")
    target = root.get_submodule(site)
    _require_reachable(root, "interventions", target)
    if (device := _site_device(root, target)) is not None:
        intervention.to(device=device)

    def rewrite(module: nn.Module, args: tuple[Any, ...], output: Any) -> Any:
        return _with_first(output, intervention(_first(output)))

    handle = target.register_forward_hook(rewrite)
    held[name] = intervention
    hooks: dict[str, Any] = root.__dict__.setdefault(_HOOKS_ATTR, {})
    sites: dict[str, tuple[str, bool]] = root.__dict__.setdefault(_SITES_ATTR, {})
    hooks[name], sites[name] = handle, (site, frozen)
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
    intervention = held[name]
    del held[name]
    return intervention


def intervention_sites(root: nn.Module) -> Mapping[str, str]:
    """The site of each intervention attached to ``root``, by name."""
    return {name: site for name, (site, _) in root.__dict__.get(_SITES_ATTR, {}).items()}


def intervention_roles(root: nn.Module) -> dict[str, str]:
    """The role of every attached intervention's parameters and buffers, by their names under ``root``."""
    held = _interventions(root)
    if held is None:
        return {}
    sites = root.__dict__.get(_SITES_ATTR, {})
    roles: dict[str, str] = {}
    for name, intervention in held.items():
        _, frozen = sites[name]
        role = "instrument" if frozen else f"intervention:{name}"
        tensors = [*intervention.named_parameters(), *intervention.named_buffers()]
        roles.update({f"{INTERVENTIONS_ATTR}.{name}.{n}": role for n, _ in tensors})
    return roles


@dataclass(frozen=True, kw_only=True)
class InterventionCfg:
    """A trainable intervention declared in a module's configuration, attached once the module's model is built.

    Args:
        name: The intervention's name (``interventions.<name>``).
        site: Path, under the module, of the module whose output it rewrites.
        op: The op it applies (:data:`TRAINABLE_OPS`): ``add`` (a steering vector, the default) or ``scale``.
        shape: The operand's shape (an ``int``, or a list or tuple of them). Defaults to the model's hidden size
            (``config.hidden_size`` of an HF or nnsight model, ``cfg.d_model`` of a TransformerLens one): one vector
            over a residual-stream site.
        scale_factor: A fixed multiplier on the operand.

    A configured intervention's operand starts where its op is the identity and is tunable (``intervention:<name>``).
    A frozen one would be a no-op, so there is no ``frozen`` setting: a later stage holds a trained operand fixed with
    ``refreeze``, and a precomputed one is attached in code (``attach_intervention(..., frozen=True)``).
    """

    name: str
    site: str
    op: str = "add"
    shape: int | tuple[int, ...] | None = None
    scale_factor: float = 1.0

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
    for cfg in map(as_intervention_cfg, cfgs):
        shape = cfg.shape if cfg.shape is not None else _model_hidden_size(root)
        if shape is None:
            raise ValueError(f"Intervention {cfg.name!r} needs a `shape`: the model's hidden size is not readable")
        intervention = OpIntervention(cfg.op, shape, scale_factor=cfg.scale_factor)
        attached.append(attach_intervention(root, cfg.name, intervention, cfg.site))
    return attached
