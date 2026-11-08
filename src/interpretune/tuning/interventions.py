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
"""Learned interventions spliced into a module's forward pass, as tunable ``intervention:<name>`` tensors (#812).

An *intervention* rewrites one site's activations with parameters of its own: a steering vector adds a learned
direction to a residual stream, and other kinds (a learned ablation, a low-rank edit) rewrite a site in other ways. This
module is the seam they share, the counterpart of :mod:`interpretune.tuning.latents` for modules that rewrite a site
rather than model it through latents:

- :class:`SteeringVector`, the first kind: ``x + scale * vector``, starting at zero so the steered model starts out
  identical to the unsteered one;
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

from interpretune.tuning.latents import _first, _refuse_circuit_tracer, _with_first

# the `nn.ModuleDict` on the root that holds attached interventions, by name
INTERVENTIONS_ATTR = "interventions"

_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")
_HOOKS_ATTR = "_intervention_hooks"
_SITES_ATTR = "_intervention_sites"


class SteeringVector(nn.Module):
    """A learned direction added to every position of a site's activations: ``x + scale * vector``.

    The vector starts at zero, so attaching it changes nothing until it trains.

    Args:
        width: The site's activation width (a residual stream's ``d_model``).
        scale: A fixed multiplier on the vector.
    """

    kind = "steer"

    def __init__(self, width: int, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale
        self.vector = nn.Parameter(torch.zeros(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` steered by the vector."""
        return x + self.scale * self.vector.to(x.dtype)


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
        ValueError: ``name`` is not a valid name or is already attached, or ``root`` is a circuit-tracer composition.
        AttributeError: the site's module does not exist under ``root``.
    """
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ValueError(f"Intervention name {name!r} must be letters, digits, '_' and '-' only")
    _refuse_circuit_tracer(root, "interventions")
    if not isinstance(intervention, nn.Module):
        raise ValueError(f"{type(intervention).__name__} is not an nn.Module")
    held = _interventions(root, create=True)
    assert held is not None
    if name in held:
        raise ValueError(f"An intervention named {name!r} is already attached; detach it first")
    target = root.get_submodule(site)
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
    """An intervention declared in a module's configuration, attached once the module's model is built.

    Args:
        name: The intervention's name (``interventions.<name>``).
        site: Path, under the module, of the module whose output it rewrites.
        kind: Which intervention to build: ``steer``, a :class:`SteeringVector`.
        width: The site's activation width. Defaults to the model's (``config.hidden_size`` of an HF or nnsight model,
            ``cfg.d_model`` of a TransformerLens one).
        scale: The steering vector's fixed multiplier.

    A configured steering vector starts at zero and is tunable (``intervention:<name>``). A frozen one would be a no-op,
    so there is no ``frozen`` setting: a later stage holds a trained vector fixed with ``refreeze``, and a precomputed
    vector is attached in code (``attach_intervention(..., frozen=True)``).
    """

    name: str
    site: str
    kind: str = "steer"
    width: int | None = None
    scale: float = 1.0


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


def _model_width(root: nn.Module) -> int | None:
    model = getattr(root, "model", None)
    for cfg_attr, width_attr in (("config", "hidden_size"), ("cfg", "d_model")):
        width = getattr(getattr(model, cfg_attr, None), width_attr, None)
        if isinstance(width, int):
            return width
    return None


def attach_configured_interventions(
    root: nn.Module, cfgs: Sequence[InterventionCfg | Mapping[str, Any]]
) -> list[nn.Module]:
    """Build and attach each configured intervention (an :class:`InterventionCfg` or its mapping) in order.

    Returns:
        The attached interventions.

    Raises:
        ValueError: a kind is not one this module builds, or no width is given and the model's cannot be read.
        TypeError: an entry is neither an :class:`InterventionCfg` nor a mapping.
    """
    attached = []
    for cfg in map(as_intervention_cfg, cfgs):
        if cfg.kind != SteeringVector.kind:
            raise ValueError(
                f"Intervention {cfg.name!r} has kind {cfg.kind!r}; a configuration builds only {SteeringVector.kind!r}"
                " (attach other kinds with `attach_intervention`)"
            )
        width = cfg.width if cfg.width is not None else _model_width(root)
        if width is None:
            raise ValueError(f"Intervention {cfg.name!r} needs a `width`: the model's activation width is not readable")
        steer = SteeringVector(width, scale=cfg.scale)
        attached.append(attach_intervention(root, cfg.name, steer, cfg.site))
    return attached
