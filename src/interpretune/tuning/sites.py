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
r"""
Site Addresses for the Tuning Seam
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A trainable intervention (:func:`~interpretune.tuning.interventions.attach_intervention`) and a latent model
(:class:`~interpretune.tuning.latents.LatentSite`) are spliced in at a site. A site is a module path under the
interpretune module, which differs per backend for the same activation (``model.model.layers.13`` on a core module,
``hf_model.model.layers.13`` on an nnsight module, ``model.blocks.13.hook_out`` on a TransformerLens bridge), or a
venngram site address, which does not: ``point:hook_resid_post & layers:13`` names the same activation on each.

:func:`resolve_site` turns an address into the module the seam hooks on this composition, and whether it reads or
rewrites that module's input or its output:

- the address is evaluated by :func:`~interpretune.analysis.venngram.sites.resolve_sites`, so its spellings and
  refusals are the analysis surface's, and it must select exactly one site, of the activation's value (``of: grad``
  is refused: a gradient is not a place to splice a module);
- on a TransformerLens composition the point resolves to one of the model's hook points, through the same spellings
  its capture and intervention paths accept, and the seam rewrites the hook point's output;
- on a core/HF or nnsight composition the point resolves through the architecture's component map
  (:class:`~interpretune.analysis.backends.hook_mapping.HookNameResolver`) to an HF module and the side of it that
  holds the activation, under the HF model's path in the interpretune module.

A point a composition cannot place is refused, naming the point and the composition; it is never mapped onto a nearby
module. An address may reference a named set of an address book (``@steer_site``), so a shared book drives
interventions and splices as it drives schedules: the seam takes the book's sets (:func:`book_sets`) beside the
address. A string without a ``:`` that does not start with ``@`` is a module path, as before.

.. warning:: This is an experimental API and may change in a future release.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Literal

from torch import nn

__all__ = ["ResolvedSite", "book_sets", "is_site_address", "resolve_site"]


@dataclass(frozen=True)
class ResolvedSite:
    """The module a site address resolves to on one composition.

    Args:
        path: The module's path under the interpretune module.
        io: Whether the activation is the module's first positional ``"input"`` or its ``"output"``.
        hook: The site in the vocabulary's spelling (``blocks.<layer>.<point>``).
    """

    path: str
    io: Literal["input", "output"]
    hook: str


def is_site_address(site: str) -> bool:
    """Whether ``site`` is a venngram site address rather than a module path: module paths never hold ``:`` and
    never start with ``@`` (a reference to an address book's named set)."""
    return ":" in site or site.startswith("@")


def book_sets(book: Any) -> dict[str, Any]:
    """The named sets of an address book, for ``@<set>`` references in a site address.

    Args:
        book: A venngram ``AddressBook``, a book document (a mapping), a local path to one, or a venngram store
            reference to a published one.

    Returns:
        The book's sets by name (``AddressBook.sets()``).
    """
    from pathlib import Path

    from venngram.book import AddressBook
    from venngram.store import is_reference, load

    if isinstance(book, AddressBook):
        return book.sets()
    if isinstance(book, Mapping):
        return AddressBook.from_mapping(book).sets()
    if isinstance(book, str) and is_reference(book):
        loaded = load(book)
        if not isinstance(loaded, AddressBook):
            raise ValueError(f"{book!r} is a {type(loaded).__name__}, not an address book")
        return loaded.sets()
    return AddressBook.from_file(Path(book)).sets()


def _path_of(root: nn.Module, target: Any) -> str | None:
    return next((n for n, m in root.named_modules() if m is target), None)


def _tl_model(root: nn.Module) -> Any | None:
    model = getattr(root, "model", None)
    return model if isinstance(model, nn.Module) and hasattr(model, "hook_dict") else None


def _hf_model(root: nn.Module) -> tuple[str, nn.Module] | None:
    """The HF model under ``root`` and its path: ``hf_model`` on an nnsight composition, else ``model``."""
    for attr in ("hf_model", "model"):
        model = getattr(root, attr, None)
        if isinstance(model, nn.Module) and getattr(getattr(model, "config", None), "model_type", None) is not None:
            return attr, model
    return None


def _n_layers(root: nn.Module) -> int:
    if (tl := _tl_model(root)) is not None:
        return int(tl.cfg.n_layers)
    if (hf := _hf_model(root)) is not None:
        config = hf[1].config
        for attr in ("num_hidden_layers", "n_layer"):
            if isinstance(n := getattr(config, attr, None), int):
                return n
    raise ValueError(
        f"A site address needs the model's layer count, and {type(root).__name__} has no TransformerLens model"
        " (`model.cfg.n_layers`) or HF model (`config.num_hidden_layers`) to read it from; name the site by its"
        " module path"
    )


def _tl_site(root: nn.Module, model: Any, hook: str) -> ResolvedSite:
    from interpretune.adapters.transformer_lens.backends import _build_available_hook_map
    from interpretune.analysis.backends.interventions import expand_intervention_patterns

    try:
        (actual, *_) = expand_intervention_patterns([hook], _build_available_hook_map(model))[hook]
    except ValueError:
        raise ValueError(
            f"Site {hook!r} names no hook point {type(model).__name__} exposes, in any spelling the vocabulary knows"
        ) from None
    path = _path_of(root, model.hook_dict[actual])
    if path is None:
        raise ValueError(f"Hook point {actual!r} is not a module under {type(root).__name__}")
    return ResolvedSite(path, "output", hook)


def _hf_site(root: nn.Module, prefix: str, model: nn.Module, hook: str) -> ResolvedSite:
    from interpretune.analysis.backends.hook_mapping import HookNameResolver

    architectures = getattr(model.config, "architectures", None)
    resolver = HookNameResolver(architectures[0] if architectures else type(model).__name__)
    path, io = resolver.resolve(hook)
    if io not in ("input", "output"):
        raise ValueError(f"Site {hook!r} resolves to {io!r} on {resolver.architecture}, not a module input or output")
    return ResolvedSite(f"{prefix}.{path}" if path else prefix, io, hook)


def resolve_site(root: nn.Module, address: str, sets: Mapping[str, Any] | None = None) -> ResolvedSite:
    """The module ``address`` names on ``root``'s composition, and the side of it that holds the activation.

    Args:
        root: The interpretune module.
        address: A venngram site address selecting exactly one site, such as ``point:hook_resid_post & layers:13``,
            or a reference to one of ``sets`` (``@steer_site``).
        sets: An address book's named sets (:func:`book_sets`), for ``@<set>`` references.

    Returns:
        The resolved site.

    Raises:
        ValueError: the address is malformed, selects no site or more than one, names a gradient, names a point
            this composition does not expose, or ``root`` holds neither a TransformerLens model nor an HF model.
    """
    from interpretune.analysis.venngram.sites import resolve_sites

    try:
        selection = resolve_sites(address, n_layers=_n_layers(root), sets=sets)
    except Exception as err:  # venngram's and the vocabulary's refusals arrive as several exception types
        raise ValueError(f"Site address {address!r}: {err}") from err
    if selection.flow != "value":
        raise ValueError(f"Site address {address!r} names a gradient (`of: grad`); the tuning seam splices at values")
    if len(selection.sites) != 1:
        raise ValueError(
            f"Site address {address!r} selects {len(selection.sites)} sites ({', '.join(selection.sites[:4])}"
            f"{', ...' if len(selection.sites) > 4 else ''}); a splice takes one site, so attach one per layer"
        )
    (hook,) = selection.sites
    if (tl := _tl_model(root)) is not None:
        return _tl_site(root, tl, hook)
    if (hf := _hf_model(root)) is not None:
        return _hf_site(root, *hf, hook)
    raise ValueError(
        f"{type(root).__name__} holds neither a TransformerLens model nor an HF model, so site address {address!r}"
        " has nothing to resolve on; name the site by its module path"
    )
