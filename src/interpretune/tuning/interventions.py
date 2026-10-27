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
"""Learned interventions spliced into a module's forward pass, as tunable ``intervention:<kind>`` tensors (#812).

An *intervention* rewrites one site's activations with parameters of its own: a steering vector adds a learned
direction to a residual stream, and other kinds (a learned ablation, a low-rank edit) rewrite a site in other ways. This
module is the seam they share, the counterpart of :mod:`interpretune.tuning.latents` for modules that rewrite a site
rather than model it through latents:

- :class:`SteeringVector`, the first kind: ``x + scale * vector``, starting at zero so the steered model starts out
  identical to the unsteered one;
- :func:`attach_intervention` and :func:`detach_intervention`, which register an intervention under the root's
  ``interventions`` and splice it in with a forward hook on the module whose output it rewrites;
- :func:`intervention_roles`, the roles to declare: ``intervention:<kind>`` for a tunable intervention, or
  ``instrument`` for one applied but never trained (a precomputed steering vector, say: finetuning-scheduler never
  thaws an instrument, and promotes instruments only to latent roles);
- :class:`InterventionsMixin`, which declares those roles through ``tunable_roles()``.

Sites are module paths under the root, as latent models' are, so the same intervention attaches to a core module's
``model.model.layers.N``, an nnsight module's ``hf_model.model.layers.N`` or a TransformerLens bridge's
``model.blocks.N.hook_out``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from interpretune.tuning.latents import _first, _with_first

# the `nn.ModuleDict` on the root that holds attached interventions, by name
INTERVENTIONS_ATTR = "interventions"

_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")
_HOOKS_ATTR = "_intervention_hooks"
_SITES_ATTR = "_intervention_sites"


class SteeringVector(nn.Module):
    """A learned direction added to every position of a site's activations: ``x + scale * vector``.

    The vector starts at zero, so attaching it changes nothing until it trains. Its kind, and so its role, is
    ``steer`` (``intervention:steer``).

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


def attach_intervention(
    root: nn.Module, name: str, intervention: nn.Module, site: str, *, kind: str | None = None, frozen: bool = False
) -> nn.Module:
    """Register ``intervention`` on ``root`` as ``interventions.<name>`` and splice it in at ``site``.

    The intervention rewrites the output of the module at ``site`` (the first element of a tuple output, as HF decoder
    layers return): the output becomes ``intervention(output)``. Its parameters become ``root``'s
    (``interventions.<name>.<...>``), so they are saved in its state dict and reachable by its tunable view.
    :func:`intervention_roles` declares them ``intervention:<kind>``, or ``instrument`` when ``frozen``.

    Args:
        root: The module the intervention is registered on (an interpretune module is its view's root).
        name: The intervention's name: letters, digits, ``_`` and ``-``.
        intervention: An ``nn.Module`` mapping the site's activations to their rewritten value.
        site: Path, under ``root``, of the module whose output is rewritten.
        kind: The intervention's kind, its role's suffix (``intervention:<kind>``). Defaults to the intervention's
            ``kind`` attribute (``steer`` for a :class:`SteeringVector`).
        frozen: Declare its parameters ``instrument`` rather than ``intervention:<kind>``.

    Returns:
        ``intervention``.

    Raises:
        ValueError: ``name`` or ``kind`` is not a valid name, or ``name`` is already attached.
        AttributeError: the site's module does not exist under ``root``.
    """
    resolved = kind if kind is not None else getattr(intervention, "kind", None)
    for label, value in (("name", name), ("kind", resolved)):
        if not isinstance(value, str) or not _NAME_RE.fullmatch(value):
            raise ValueError(f"Intervention {label} {value!r} must be letters, digits, '_' and '-' only")
    assert isinstance(resolved, str)
    if not isinstance(intervention, nn.Module):
        raise ValueError(f"{type(intervention).__name__} is not an nn.Module")
    held = _interventions(root, create=True)
    assert held is not None
    if name in held:
        raise ValueError(f"An intervention named {name!r} is already attached; detach it first")
    target = root.get_submodule(site)

    def rewrite(module: nn.Module, args: tuple[Any, ...], output: Any) -> Any:
        return _with_first(output, intervention(_first(output)))

    handle = target.register_forward_hook(rewrite)
    held[name] = intervention
    hooks: dict[str, Any] = root.__dict__.setdefault(_HOOKS_ATTR, {})
    sites: dict[str, tuple[str, str, bool]] = root.__dict__.setdefault(_SITES_ATTR, {})
    hooks[name], sites[name] = handle, (site, resolved, frozen)
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
    return {name: site for name, (site, _, _) in root.__dict__.get(_SITES_ATTR, {}).items()}


def intervention_roles(root: nn.Module) -> dict[str, str]:
    """The role of every attached intervention's parameters and buffers, by their names under ``root``."""
    held = _interventions(root)
    if held is None:
        return {}
    sites = root.__dict__.get(_SITES_ATTR, {})
    roles: dict[str, str] = {}
    for name, intervention in held.items():
        _, kind, frozen = sites[name]
        role = "instrument" if frozen else f"intervention:{kind}"
        tensors = [*intervention.named_parameters(), *intervention.named_buffers()]
        roles.update({f"{INTERVENTIONS_ATTR}.{name}.{n}": role for n, _ in tensors})
    return roles


class InterventionsMixin:
    """Declares the roles of the interventions attached to a module through its ``tunable_roles()``.

    Compose it into an interpretune module (any backend) and attach interventions with :meth:`attach_intervention` in a
    ``register_tunable_modules()`` override, which runs once the model exists and before the module's view is built.
    """

    def attach_intervention(
        self, name: str, intervention: nn.Module, site: str, *, kind: str | None = None, frozen: bool = False
    ) -> nn.Module:
        """:func:`attach_intervention` on this module."""
        return attach_intervention(self, name, intervention, site, kind=kind, frozen=frozen)  # type: ignore[arg-type]

    def detach_intervention(self, name: str) -> nn.Module:
        """:func:`detach_intervention` on this module."""
        return detach_intervention(self, name)  # type: ignore[arg-type]

    def tunable_roles(self) -> dict[str, str]:
        """The cooperative roles, plus ``intervention:<kind>`` (or ``instrument``) for each attached
        intervention."""
        roles = super().tunable_roles()  # type: ignore[misc]
        roles.update(intervention_roles(self))  # type: ignore[arg-type]
        return roles
