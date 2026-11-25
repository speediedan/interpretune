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
Pinning an Address Book to the Models It Was Written Against
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A venngram address book that names latent models (``module:latent_models.sae``, ``role: latent:sae``,
``features: sae``) selects the same tensors wherever it loads, but it means the same thing only on the base model and
the dictionaries it was written against. Its ``requires`` pins them (venngram's book schema 8):

- ``base_model``: the model's id and the revision it resolved to;
- ``latent_models``: by name, each one's ``release`` (an SAE Lens release and id, or a Hub repository and revision) and
  the version of each frame its units are read in.

:func:`pinned_book` writes the pins of the models a module has loaded into a book's document, and
:func:`check_book_pins` refuses a book whose pins differ from them, naming each entry that does
(``venngram.PinMismatchError``). Interpretune checks every book it reads for a module: the book of a configured
intervention and :func:`~interpretune.tuning.sites.book_sets`, and a treatment
(:func:`~interpretune.analysis.venngram.fetch.load_recipe`). A pinned book read with no module to check it against is
refused. A schedule's ``addresses`` are finetuning-scheduler's to read, and it does not check pins, so it refuses a
schedule that uses a pinned book's sets.

A module reports what its latent models were loaded from with ``latent_model_releases()``: the ``release`` each was
attached with (:func:`~interpretune.tuning.latents.attach_latent_model`), the configured release and id of a spliced
SAE Lens SAE (by its ``attached_saes`` key), and the repository and revision of circuit-tracer's transcoders
(``transcoder``) where ``transcoder_set`` states a revision. A model that states no source cannot be pinned, and never
satisfies a pin.

.. warning:: This is an experimental API and may change in a future release.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

from interpretune.utils import MisconfigurationException

__all__ = ["book_pins", "check_book_pins", "loaded_pins", "pinned_book"]


def _catalog_frames(catalog: Any) -> dict[str, dict[str, str]]:
    """The version of each frame a catalog declares, by latent model name and frame name.

    Raises:
        MisconfigurationException: A latent model's frame is declared in two versions, so neither is the one it has.
    """
    frames: dict[str, dict[str, str]] = {}
    for frame in catalog.frames if catalog is not None else ():
        by_name = frames.setdefault(frame.target.name, {})
        if by_name.setdefault(frame.name, frame.version) != frame.version:
            raise MisconfigurationException(
                f"latent model {frame.target.name!r} declares frame {frame.name!r} in versions"
                f" {sorted((by_name[frame.name], frame.version))}, so a book cannot pin one of them"
            )
    return frames


def loaded_pins(
    module: Any, *, catalog: Any = None, releases: Mapping[str, Mapping[str, str]] | None = None
) -> dict[str, Any]:
    """What ``module`` has loaded, in the form ``venngram.AddressBook.check_pins`` compares with a book's pins.

    Args:
        module: The composition whose models are read.
        catalog: The ``LatentCatalog`` whose frames' versions are reported, when the book's units are read in one.
        releases: The releases of latent models the module does not hold, by name: those a catalog binds, for instance.

    Returns:
        ``{"base_model": {"id", "revision"}, "latent_models": {<name>: {"release", "frames"}}}``, with ``None`` for an
        id or revision the module does not state and an empty ``release`` for a latent model with no stated source.

    Raises:
        MisconfigurationException: ``releases`` contradicts what the module reports for one of its latent models.
    """
    from interpretune.hub.subject import model_identity
    from interpretune.tuning.latents import latent_model_releases

    declared = getattr(module, "latent_model_releases", None)
    found: dict[str, dict[str, str]] = latent_model_releases(module) if declared is None else declared()
    for name, release in (releases or {}).items():
        if found.get(name) and dict(found[name]) != dict(release):
            raise MisconfigurationException(
                f"latent model {name!r} was loaded from {dict(found[name])}, not from the release given for it,"
                f" {dict(release)}"
            )
        found[name] = dict(release)
    frames = _catalog_frames(catalog)
    model_id, revision = model_identity(module)
    latents = {
        name: {"release": dict(found.get(name, {})), "frames": frames.get(name, {})} for name in {**found, **frames}
    }
    return {"base_model": {"id": model_id, "revision": revision}, "latent_models": latents}


def book_pins(
    module: Any,
    latent_models: Collection[str] | None = None,
    *,
    catalog: Any = None,
    releases: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """The ``requires`` entries pinning a book to the models ``module`` has loaded.

    Args:
        module: The composition the book was written against.
        latent_models: The latent models the book names. Defaults to every one loaded.
        catalog: See :func:`loaded_pins`.
        releases: See :func:`loaded_pins`.

    Returns:
        ``{"base_model": ..., "latent_models": ...}``, without ``latent_models`` when there are none.

    Raises:
        MisconfigurationException: The base model's id or revision, or a latent model's release, is not known, or a
            named latent model is not loaded. A pin that leaves out what it was written against checks nothing.
    """
    loaded = loaded_pins(module, catalog=catalog, releases=releases)
    base, latents = loaded["base_model"], loaded["latent_models"]
    if missing := [key for key in ("id", "revision") if base[key] is None]:
        raise MisconfigurationException(
            f"{type(module).__name__} does not state its base model's {' or '.join(missing)}, so a book cannot be"
            " pinned to it: load the model from a Hub repository, where its revision resolves to a commit"
        )
    names = sorted(latents) if latent_models is None else list(latent_models)
    if absent := [name for name in names if name not in latents]:
        raise MisconfigurationException(f"latent models {absent} are not loaded (loaded: {sorted(latents)})")
    if unstated := [name for name in names if not latents[name]["release"]]:
        raise MisconfigurationException(
            f"latent models {unstated} state no release, so a book cannot be pinned to them: pass `release` to"
            " `attach_latent_model`, give circuit-tracer's `transcoder_set` a revision (`<repository>@<revision>`), or"
            " pass `releases`"
        )
    pins: dict[str, Any] = {"base_model": base}
    if names:
        pins["latent_models"] = {name: {k: v for k, v in latents[name].items() if v} for name in names}
    return pins


def pinned_book(
    document: Mapping[str, Any],
    module: Any,
    latent_models: Collection[str] | None = None,
    *,
    catalog: Any = None,
    releases: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """``document``, a book's, with the pins of the models ``module`` has loaded (:func:`book_pins`) in its
    ``requires``, at the schema version that carries them, parsed by venngram before it is returned.

    Raises:
        MisconfigurationException: As :func:`book_pins`, or ``document`` already pins its models.
    """
    from venngram.book import PIN_KEYS, PINS_SCHEMA, AddressBook

    requires = dict(document.get("requires") or {})
    if already := [key for key in PIN_KEYS if key in requires]:
        raise MisconfigurationException(f"the book already pins its models (`requires`: {already})")
    requires.update(book_pins(module, latent_models, catalog=catalog, releases=releases))
    version = max(PINS_SCHEMA, document.get("schema_version", PINS_SCHEMA))
    pinned = {**document, "schema_version": version, "requires": requires}
    AddressBook.from_mapping(pinned, source="pinned book")
    return pinned


def check_book_pins(
    book: Any, module: Any, *, catalog: Any = None, releases: Mapping[str, Mapping[str, str]] | None = None
) -> None:
    """Refuse a book whose pins, or an imported book's, differ from the models ``module`` has loaded.

    Args:
        book: A venngram ``AddressBook``. One that pins nothing is accepted.
        module: The composition it will be used on, or ``None`` where there is none to check it against.
        catalog: See :func:`loaded_pins`.
        releases: See :func:`loaded_pins`.

    Raises:
        PinMismatchError: (``venngram``) The models differ, naming every entry that does.
        ValueError: ``module`` is ``None`` and the book pins its models.
    """
    pins = book.pins()
    if not pins:
        return
    if module is None:
        raise ValueError(
            f"{book.source}: the book pins the models it was written against ({', '.join(sorted(pins))}), and there is"
            " no module to check them against; pass the module it will be used on"
        )
    loaded = loaded_pins(module, catalog=catalog, releases=releases)
    book.check_pins(base_model=loaded["base_model"], latent_models=loaded["latent_models"])
