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
"""Latent models spliced into a module's forward pass, as tunable ``latent:<name>`` tensors (#812).

A *latent model* models a site's activations through latents: a sparse autoencoder reads and rewrites one site, a
transcoder reads one site (an MLP's input) and rewrites another (that MLP's output), and a block-sparse featurizer or
any other dictionary does either. This module fixes only the seam such models share, so that any of them can be
attached at a site and scheduled by role, whichever backend the module wraps:

- :class:`LatentModel`, the protocol: ``encode`` activations to latents and ``decode`` latents back. sae_lens SAEs and
  circuit-tracer transcoders already satisfy it;
- :class:`LatentSite`, where the model reads (a module's input or output) and which module's output it rewrites, and
  how (:data:`LatentMode`);
- :func:`attach_latent_model` and :func:`detach_latent_model`, which register the model under the root's
  ``latent_models`` and splice it in with forward hooks;
- :func:`latent_model_roles`, the roles to declare: ``latent:<name>`` for a tunable latent model, or ``instrument`` for
  one held frozen until a schedule promotes it (finetuning-scheduler's ``promote`` phase key, core loop only);
- :class:`LatentModelsMixin`, which declares those roles through ``tunable_roles()``.

Sites are module paths under the root, so the same latent model attaches wherever a module computes the activations: a
core module's ``model.model.layers.N.mlp``, an nnsight module's ``hf_model.model.layers.N.mlp`` or a TransformerLens
bridge's hook points. Which latent models tune well, and how, is open research, so this module provides the seam and no
recipe.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import torch
from torch import nn

# the `nn.ModuleDict` on the root that holds attached latent models, by name
LATENT_MODELS_ATTR = "latent_models"

# How a latent model rewrites its site's output:
# - "replace": the output becomes the reconstruction, so the model runs on what the latents capture;
# - "replace_with_error": the output keeps its value (the reconstruction plus its detached error), while gradients reach
#   the latent model through the reconstruction, as sae_lens's error term does;
# - "observe": the output is unchanged and the latents are recorded (`last_latents`), for analysis.
LatentMode = Literal["replace", "replace_with_error", "observe"]

_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")
_HOOKS_ATTR = "_latent_model_hooks"
_SITES_ATTR = "_latent_model_sites"


@runtime_checkable
class LatentModel(Protocol):
    """Anything that encodes activations to latents and decodes latents back."""

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Latents for activations ``x``."""
        ...

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Activations reconstructed from latents ``z``."""
        ...


@dataclass(frozen=True)
class LatentSite:
    """Where a latent model reads and which module's output it rewrites.

    Args:
        read: Path, under the root, of the module whose activations the latent model encodes.
        write: Path of the module whose output it rewrites. Defaults to ``read`` (an SAE); a transcoder reads an MLP's
            input and writes that MLP's output.
        read_from: Whether the ``read`` module's ``"output"`` (the default) or its first positional ``"input"`` is
            encoded.
        mode: How the output is rewritten (:data:`LatentMode`).
    """

    read: str
    write: str | None = None
    read_from: Literal["input", "output"] = "output"
    mode: LatentMode = "replace_with_error"

    @property
    def write_path(self) -> str:
        """The module whose output is rewritten."""
        return self.write if self.write is not None else self.read


def _first(value: Any) -> torch.Tensor:
    return value[0] if isinstance(value, tuple) else value


def _with_first(value: Any, tensor: torch.Tensor) -> Any:
    return (tensor, *value[1:]) if isinstance(value, tuple) else tensor


class _Splice:
    """The forward hooks splicing one latent model in at its site."""

    def __init__(self, latent: nn.Module, site: LatentSite) -> None:
        self.latent, self.site = latent, site
        self.read_value: torch.Tensor | None = None

    def capture(self, module: nn.Module, args: tuple[Any, ...], output: Any = None) -> None:
        self.read_value = _first(args) if self.site.read_from == "input" else _first(output)

    def rewrite(self, module: nn.Module, args: tuple[Any, ...], output: Any) -> Any:
        x, self.read_value = self.read_value, None
        if x is None:  # the read module did not run in this forward pass
            return output
        latents = self.latent.encode(x)  # type: ignore[operator]
        if self.site.mode == "observe":
            self.latent.last_latents = latents.detach()
            return output
        target = _first(output)
        recon = self.latent.decode(latents).to(target.dtype)  # type: ignore[operator]
        if self.site.mode == "replace_with_error":
            recon = recon + (target - recon).detach()
        return _with_first(output, recon)


def _latent_models(root: nn.Module, create: bool = False) -> nn.ModuleDict | None:
    held = root._modules.get(LATENT_MODELS_ATTR)
    if isinstance(held, nn.ModuleDict):
        return held
    if held is not None:
        raise AttributeError(f"{type(root).__name__}.{LATENT_MODELS_ATTR} is not an nn.ModuleDict")
    if not create:
        return None
    root.add_module(LATENT_MODELS_ATTR, held := nn.ModuleDict())
    return held


def attach_latent_model(
    root: nn.Module, name: str, latent: nn.Module, site: LatentSite, *, frozen: bool = False
) -> nn.Module:
    """Register ``latent`` on ``root`` as ``latent_models.<name>`` and splice it in at ``site``.

    The latent model's parameters become ``root``'s (``latent_models.<name>.<...>``), so they are saved in its state
    dict and reachable by its tunable view. :func:`latent_model_roles` declares them ``latent:<name>``, or
    ``instrument`` when ``frozen``: an instrument is never thawed or checkpointed until a schedule promotes it to
    ``latent:<name>``.

    Args:
        root: The module the latent model is registered on (an interpretune module is its view's root).
        name: The latent model's name, also its role's (``latent:<name>``): letters, digits, ``_`` and ``-``.
        latent: The latent model, an ``nn.Module`` with ``encode`` and ``decode`` (:class:`LatentModel`).
        site: Where it reads and writes.
        frozen: Declare its parameters ``instrument`` rather than ``latent:<name>``.

    Returns:
        ``latent``.

    Raises:
        ValueError: ``name`` is not a valid name or is already attached, or ``latent`` is not a latent model.
        AttributeError: a site's module does not exist under ``root``.
    """
    if not _NAME_RE.fullmatch(name):
        raise ValueError(f"Latent model name {name!r} must be letters, digits, '_' and '-' only")
    if not isinstance(latent, nn.Module) or not isinstance(latent, LatentModel):
        raise ValueError(f"{type(latent).__name__} is not an nn.Module with encode() and decode()")
    held = _latent_models(root, create=True)
    assert held is not None
    if name in held:
        raise ValueError(f"A latent model named {name!r} is already attached; detach it first")
    read, write = root.get_submodule(site.read), root.get_submodule(site.write_path)
    splice = _Splice(latent, site)
    handles = [
        read.register_forward_pre_hook(splice.capture)
        if site.read_from == "input"
        else read.register_forward_hook(splice.capture),
        write.register_forward_hook(splice.rewrite),
    ]
    held[name] = latent
    hooks: dict[str, list[Any]] = root.__dict__.setdefault(_HOOKS_ATTR, {})
    sites: dict[str, tuple[LatentSite, bool]] = root.__dict__.setdefault(_SITES_ATTR, {})
    hooks[name], sites[name] = handles, (site, frozen)
    return latent


def detach_latent_model(root: nn.Module, name: str) -> nn.Module:
    """Remove the latent model ``name`` from ``root`` and its site's forward pass.

    Returns:
        The detached latent model.

    Raises:
        KeyError: no latent model of that name is attached.
    """
    held = _latent_models(root)
    if held is None or name not in held:
        raise KeyError(f"No latent model named {name!r} is attached")
    for handle in root.__dict__.get(_HOOKS_ATTR, {}).pop(name, ()):
        handle.remove()
    root.__dict__.get(_SITES_ATTR, {}).pop(name, None)
    latent = held[name]
    del held[name]
    return latent


def latent_model_sites(root: nn.Module) -> Mapping[str, LatentSite]:
    """The site of each latent model attached to ``root``, by name."""
    return {name: site for name, (site, _) in root.__dict__.get(_SITES_ATTR, {}).items()}


def latent_model_roles(root: nn.Module) -> dict[str, str]:
    """The role of every attached latent model's parameters and buffers, by their names under ``root``."""
    held = _latent_models(root)
    if held is None:
        return {}
    sites = root.__dict__.get(_SITES_ATTR, {})
    roles: dict[str, str] = {}
    for name, latent in held.items():
        role = "instrument" if sites.get(name, (None, False))[1] else f"latent:{name}"
        tensors = [*latent.named_parameters(), *latent.named_buffers()]
        roles.update({f"{LATENT_MODELS_ATTR}.{name}.{n}": role for n, _ in tensors})
    return roles


class LatentModelsMixin:
    """Declares the roles of the latent models attached to a module through its ``tunable_roles()``.

    Compose it into an interpretune module (any backend) and attach latent models with :meth:`attach_latent_model` in a
    ``register_tunable_modules()`` override, which runs once the model exists and before the module's view is built.
    """

    def attach_latent_model(self, name: str, latent: nn.Module, site: LatentSite, *, frozen: bool = False) -> nn.Module:
        """:func:`attach_latent_model` on this module."""
        return attach_latent_model(self, name, latent, site, frozen=frozen)  # type: ignore[arg-type]

    def detach_latent_model(self, name: str) -> nn.Module:
        """:func:`detach_latent_model` on this module."""
        return detach_latent_model(self, name)  # type: ignore[arg-type]

    def tunable_roles(self) -> dict[str, str]:
        """The cooperative roles, plus ``latent:<name>`` (or ``instrument``) for each attached latent model."""
        roles = super().tunable_roles()  # type: ignore[misc]
        roles.update(latent_model_roles(self))  # type: ignore[arg-type]
        return roles
