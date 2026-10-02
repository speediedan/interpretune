"""The adapter-selection guide must list every registered module combination (#256).

A hand-written guide drifts the moment a new composition registers: the page still reads
authoritatively while missing the new combination. This pins the guide's combination list against
the live ``CompositionRegistry`` -- registering a combination without documenting it fails here,
naming the missing entry.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
GUIDE = REPO_ROOT / "docs" / "source" / "usage" / "adapter_selection_guide.md"


def _registered_module_combinations() -> set[str]:
    import interpretune as it

    combos = set()
    for key in it.ADAPTER_REGISTRY.registry.keys():
        if not key or key[0] != "module":
            continue
        names = sorted(a.value if hasattr(a, "value") else str(a) for a in key[1:])
        combos.add("(" + ", ".join(names) + ("," if len(names) == 1 else "") + ")")
    return combos


class TestAdapterGuideListsEveryCombination:
    def test_every_registered_combination_is_documented(self):
        """Each registry combination appears verbatim in the guide."""
        text = GUIDE.read_text(encoding="utf-8")
        missing = [c for c in sorted(_registered_module_combinations()) if c not in text]
        assert not missing, (
            "registered module combinations missing from docs/source/usage/adapter_selection_guide.md:\n  "
            + "\n  ".join(missing)
        )

    def test_guide_names_the_backend_matrix_rather_than_duplicating_it(self):
        """The compatibility matrix lives in one place; the guide links, not copies."""
        text = GUIDE.read_text(encoding="utf-8")
        assert "circuit_tracer_backend_support" in text
