r"""
venngram Integration
^^^^^^^^^^^^^^^^^^^^

Interpretune's side of venngram's consumer-owned integration (interpretune#887): what this package honours of a
venngram recipe, declared as a ``venngram.ops.Support`` record derived from a backend's own support records, and
the refuse-by-name check a recipe runs against it before anything is applied. Applying a recipe translates its
treatments into intervention specs, and exporting writes intervention specs back as a book carrying a treatment;
nothing here executes a treatment.

venngram is imported on first use, so importing interpretune does not import it.

.. warning:: This is an experimental API and may change in a future release.
"""

from interpretune.analysis.venngram.apply import recipe_interventions
from interpretune.analysis.venngram.export import export_book
from interpretune.analysis.venngram.fetch import FetchedRecipe, load_recipe
from interpretune.analysis.venngram.support import check_recipe, require_supported, support_mapping, venngram_support

__all__ = [
    "FetchedRecipe",
    "check_recipe",
    "export_book",
    "load_recipe",
    "recipe_interventions",
    "require_supported",
    "support_mapping",
    "venngram_support",
]
