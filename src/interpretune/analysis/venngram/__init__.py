r"""
venngram Integration
^^^^^^^^^^^^^^^^^^^^

Interpretune's side of venngram's consumer-owned integration (interpretune#887): what this package honours of a
venngram recipe, declared as a ``venngram.ops.Support`` record derived from a backend's own support records, and
the refuse-by-name check a recipe runs against it before anything is applied. Applying a recipe translates its
treatments into intervention specs (for activation values, or for gradients under ``of: grad``), or a feature treatment
into circuit-tracer's feature tuples; a site address resolves to the hook names it selects; and exporting writes
intervention specs back as a book carrying a treatment; nothing here executes a treatment.

venngram is imported on first use, so importing interpretune does not import it.

.. warning:: This is an experimental API and may change in a future release.
"""

from interpretune.analysis.venngram.apply import recipe_feature_interventions, recipe_interventions, unit_interventions
from interpretune.analysis.venngram.export import export_book
from interpretune.analysis.venngram.fetch import FetchedRecipe, load_recipe
from interpretune.analysis.venngram.sites import SiteSelection, UnitSiteSelection, resolve_sites, resolve_unit_sites
from interpretune.analysis.venngram.support import (
    check_recipe,
    require_supported,
    support_mapping,
    venngram_feature_subsets,
    venngram_feature_support,
    venngram_support,
)
from interpretune.analysis.venngram.values import normalized_unit_doses

__all__ = [
    "FetchedRecipe",
    "SiteSelection",
    "UnitSiteSelection",
    "check_recipe",
    "export_book",
    "load_recipe",
    "normalized_unit_doses",
    "recipe_feature_interventions",
    "recipe_interventions",
    "require_supported",
    "resolve_sites",
    "resolve_unit_sites",
    "support_mapping",
    "venngram_feature_subsets",
    "venngram_feature_support",
    "venngram_support",
    "unit_interventions",
]
