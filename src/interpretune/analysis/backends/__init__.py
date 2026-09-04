"""Model and analysis backends for analysis operations.

This package is one of the sanctioned import surfaces for op implementations (bundled, local, or
hub) — see ``docs/custom_ops_composition_guide.md``. It provides:

- :mod:`~interpretune.analysis.backends.protocols`: the ``ModelBackend`` / ``AnalysisBackend``
  protocols a framework or analysis package implements
- :mod:`~interpretune.analysis.backends.capabilities`: capability enums, module capability
  aggregation, and the named-backend registry
- :mod:`~interpretune.analysis.backends.interventions`: declarative intervention specs and their
  expansion/application
- :mod:`~interpretune.analysis.backends.feature_selection`: feature-selection specs and filters

The four modules above are the seam. The concrete backends live one level down in
each adapter's own package (``interpretune.adapters.<name>.backends``),
which is **not** part of the op-author surface: an op consumes backends through the protocols and
capability helpers, never by importing one, and the publishability lint enforces that by excluding
``interpretune.adapters`` from op imports. ``hook_mapping`` stays here rather than with an adapter
because the seam itself uses it (``interventions``) and it exports ``HOOK_ALIAS_GROUPS``
through this façade, so moving it would invert the layering.

This module is a re-export façade only: ``__all__`` below is the public surface. Note that
``resolve_analysis_backend`` treats ``backends.<name>`` as a by-name lookup namespace, so only
modules that call ``register_analysis_backend`` at import time participate in that resolution; the
four seam modules deliberately do not.
"""

from __future__ import annotations

from interpretune.analysis.backends.capabilities import (
    ANALYSIS_BACKEND_REGISTRY,
    AnalysisBackendCapability,
    BackendCapability,
    Capability,
    InterventionMode,
    InterventionSupport,
    LatentModelSupport,
    ModuleCapabilities,
    PositionScope,
    get_analysis_backend,
    get_model_backend,
    get_module_capabilities,
    normalize_backend_capability,
    register_analysis_backend,
    require_analysis_backend,
    resolve_analysis_backend,
)
from interpretune.analysis.backends.feature_selection import (
    FeatureSelectionSpec,
    apply_feature_score_sign_filter,
    apply_feature_selection_filter,
    apply_optional_feature_sign_filter,
    augment_feature_rows_for_selection,
    select_top_feature_indices,
)
from interpretune.analysis.backends.interventions import (
    HOOK_ALIAS_GROUPS,
    InterventionDict,
    InterventionSpec,
    InterventionValue,
    apply_intervention,
    iter_intervention_axes,
    normalize_intervention_mode,
    normalize_position_scope,
    require_intervention_mode,
    require_intervention_support,
    require_position_scope,
    build_intervention_dict,
    expand_intervention_patterns,
    get_intervention_target_shape,
    resolve_interventions,
)
from interpretune.analysis.backends.protocols import (
    AnalysisBackend,
    ModelBackend,
    ModelBackendCore,
    SupportsGradients,
    SupportsIntervention,
    SupportsLatentModels,
)

__all__ = [
    "ANALYSIS_BACKEND_REGISTRY",
    "AnalysisBackend",
    "AnalysisBackendCapability",
    "BackendCapability",
    "Capability",
    "FeatureSelectionSpec",
    "HOOK_ALIAS_GROUPS",
    "InterventionDict",
    "InterventionSpec",
    "PositionScope",
    "InterventionValue",
    "ModelBackend",
    "ModelBackendCore",
    "SupportsGradients",
    "SupportsIntervention",
    "SupportsLatentModels",
    "ModuleCapabilities",
    "apply_feature_score_sign_filter",
    "apply_feature_selection_filter",
    "apply_intervention",
    "normalize_position_scope",
    "require_position_scope",
    "require_intervention_support",
    "require_intervention_mode",
    "normalize_intervention_mode",
    "iter_intervention_axes",
    "LatentModelSupport",
    "InterventionSupport",
    "InterventionMode",
    "apply_optional_feature_sign_filter",
    "augment_feature_rows_for_selection",
    "build_intervention_dict",
    "expand_intervention_patterns",
    "get_analysis_backend",
    "get_intervention_target_shape",
    "get_model_backend",
    "get_module_capabilities",
    "normalize_backend_capability",
    "register_analysis_backend",
    "require_analysis_backend",
    "resolve_analysis_backend",
    "resolve_interventions",
    "select_top_feature_indices",
]
