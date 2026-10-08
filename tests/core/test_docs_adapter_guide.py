"""The adapter-selection guide must list every registered module combination (#256).

A hand-written guide drifts the moment a new composition registers: the page still reads
authoritatively while missing the new combination. This pins the guide's combination list against
the live ``CompositionRegistry`` -- registering a combination without documenting it fails here,
naming the missing entry.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
DOCS_SOURCE = REPO_ROOT / "docs" / "source"
GUIDE = DOCS_SOURCE / "usage" / "adapter_selection_guide.md"


def _registered_module_combinations() -> set[str]:
    import interpretune as it

    combos = set()
    for key in it.ADAPTER_REGISTRY.registry.keys():
        if not key or key[0] != "module":
            continue
        names = sorted(a.value if hasattr(a, "value") else str(a) for a in key[1:])
        combos.add("(" + ", ".join(names) + ("," if len(names) == 1 else "") + ")")
    return combos


def _documented_module_combinations(text: str) -> set[str]:
    """The combination bullets under the guide's ``Registered module combinations`` heading."""
    section = text.split("## Registered module combinations", 1)[1].split("\n## ", 1)[0]
    return set(re.findall(r"^- `(\([^`]*\))`$", section, flags=re.MULTILINE))


class TestAdapterGuideListsEveryCombination:
    def test_documented_combinations_match_the_registry(self):
        """The guide lists exactly the registered combinations: none missing, none stale."""
        documented = _documented_module_combinations(GUIDE.read_text(encoding="utf-8"))
        registered = _registered_module_combinations()
        assert documented, "no combination bullets found under 'Registered module combinations'"
        assert not registered - documented, (
            "registered module combinations missing from docs/source/usage/adapter_selection_guide.md:\n  "
            + "\n  ".join(sorted(registered - documented))
        )
        assert not documented - registered, (
            "combinations documented in docs/source/usage/adapter_selection_guide.md but not registered:\n  "
            + "\n  ".join(sorted(documented - registered))
        )

    def test_doc_references_resolve(self):
        """Every ``{doc}`` target names an existing page.

        Sphinx resolves a relative target against the page's own directory and only warns on a miss, so a
        target written from the docs root (``usage/...`` inside ``usage/``) renders as unlinked text in a build
        that still passes.
        """
        text = GUIDE.read_text(encoding="utf-8")
        targets = re.findall(r"\{doc\}`[^`<]*<([^>]+)>`", text)
        assert any("circuit_tracer_backend_support" in t for t in targets), "the backend matrix is not linked"
        unresolved = []
        for target in targets:
            base = DOCS_SOURCE if target.startswith("/") else GUIDE.parent
            if not (base / f"{target.lstrip('/')}.md").exists():
                unresolved.append(target)
        assert not unresolved, f"unresolved {{doc}} targets in the adapter-selection guide: {unresolved}"
