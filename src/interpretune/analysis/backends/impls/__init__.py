"""Concrete backend implementations, one module per third-party library.

These implement the protocols defined in :mod:`interpretune.analysis.backends.protocols` for a specific package:
``transformer_lens`` and ``nnsight`` provide ``ModelBackend`` implementations, ``circuit_tracer`` provides an
``AnalysisBackend``. They are selected through the named-backend registry
(:func:`interpretune.analysis.backends.get_analysis_backend` / ``resolve_analysis_backend``), so callers normally reach
them by name rather than by import.

**This subpackage is not part of the op-author seam.** Op implementations (bundled, local, or hub) consume backends
through the sanctioned surfaces at the package root -- the protocols, the capability helpers, intervention and
feature-selection utilities -- never by importing a concrete backend, which is what would entangle an op with one
package. ``tests/core/test_bundled_op_publishability.py`` enforces that: the package root is a sanctioned prefix and
this subpackage is explicitly excluded from it.

That is narrower than "private". Adapters legitimately import these modules to wire a framework up, and a third-party
backend author is meant to read them as worked examples of implementing the protocols, which is why they live under a
public name rather than an underscored one.
"""

from __future__ import annotations

__all__: list[str] = []
