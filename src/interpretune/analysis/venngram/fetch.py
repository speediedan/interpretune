r"""
Fetching Recipes by Reference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wherever interpretune names a recipe (and, in time, an address book), it accepts a venngram document reference
(``<store>:<ref>[@<revision>]:<path>[#<sha256>]``, see ``venngram.store``) as well as a local path or an inline
mapping. A fetched recipe comes back with the exact reference it resolved to, :func:`venngram.store.pin` applied,
so a caller can record that in its provenance and reproduce the run: provenance keeps only hashes and references,
never fetched content.

.. warning:: This is an experimental API and may change in a future release.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from interpretune.utils import MisconfigurationException


def _require(module: str):
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise MisconfigurationException(
            "The venngram integration needs the optional `venngram` extra: install `interpretune[venngram]`."
        ) from e


@dataclass(frozen=True)
class FetchedRecipe:
    """A recipe and where it came from: the exact (pinned) reference for a store fetch, else the local source."""

    recipe: Any  # venngram.recipe.Recipe, typed loosely so importing this module needs no venngram install
    reference: str | None  #: the pinned document reference, when fetched from a store
    source: str  #: where the document was read from, for provenance (a path, or ``"<inline>"``)


def load_recipe(source: str | Path | Mapping[str, Any] | Any) -> FetchedRecipe:
    """Read a recipe from a document reference, a local path, a mapping, or a recipe already in hand.

    A string that parses as a venngram document reference (``venngram.store.is_reference``) is fetched
    manifest-first and hash-checked by venngram, and the returned reference is pinned. Any other string (or a
    ``Path``) is a local file. A mapping is an inline document. A ``venngram.recipe.Recipe`` passes through.

    Raises:
        MisconfigurationException: If venngram is not installed, the fetched document is not a recipe, or the
            source is none of these. A fetch or parse failure from venngram itself propagates unchanged.
    """
    recipe_mod, store = _require("venngram.recipe"), _require("venngram.store")
    if isinstance(source, recipe_mod.Recipe):
        return FetchedRecipe(source, None, "<inline>")
    if isinstance(source, Mapping):
        return FetchedRecipe(recipe_mod.Recipe.from_mapping(source), None, "<inline>")
    if isinstance(source, Path) or (isinstance(source, str) and not store.is_reference(source)):
        return FetchedRecipe(recipe_mod.Recipe.from_file(source), None, str(source))
    if isinstance(source, str):
        loaded = store.load(source)
        if not isinstance(loaded, recipe_mod.Recipe):
            raise MisconfigurationException(
                f"{source}: the fetched document is a {type(loaded).__name__}, not a recipe."
            )
        return FetchedRecipe(loaded, str(store.pin(source)), source)
    raise MisconfigurationException(
        f"A recipe source must be a document reference, a path, a mapping or a Recipe, not {type(source).__name__}."
    )
