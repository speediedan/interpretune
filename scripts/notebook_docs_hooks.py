"""Interpretune's hooks for the docs-artifact renderer: the analysis-op surface an artifact was rendered against.

The renderer (``src/interpretune/utils/notebook_docs.py``) is repository-neutral; ``[tool.notebook-publishing]``
names this file as ``docs_hooks``. What lives here is the part only interpretune has: every artifact is stamped
with the bundled op names it was rendered against, so ``--check-stale`` can tell when its captured OUTPUTS mention
an op that no longer exists even though its sources have not changed.

Standard library plus pyyaml, and pyyaml is optional: docs-build runs ``--check-stale`` once BEFORE installing
requirements, so it fails fast, and there the op-surface half degrades to unavailable (reported as a note) while
the engine's source-drift half still runs. The same workflow runs the check again after the install.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_OPS_DIR = REPO_ROOT / "src" / "interpretune" / "analysis" / "ops" / "bundled"
RENDER_COMMAND = "python scripts/render_notebook_docs_artifacts.py"

# Where the rendered op surface is stamped, so `--check-stale` can tell whether an artifact's captured
# OUTPUTS have gone stale even when its sources have not.
OP_SURFACE_KEY = "interpretune_op_surface"


def bundled_op_names() -> set[str]:
    """Live bundled op names, read from YAML rather than the dispatcher.

    Deliberately NOT `DISPATCHER._op_definitions`: that reflects whatever the current session has loaded,
    including any hub collection pulled at the time, so a gate built on it would give different answers to
    different people. The bundled YAML is the same for everyone with the same checkout, which is the
    property a CI gate needs. Empty when pyyaml is absent, which means the surface could not be READ.
    """
    try:
        import yaml
    except ImportError:
        return set()

    names: set[str] = set()
    for path in sorted(BUNDLED_OPS_DIR.rglob("*.yaml")):
        content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for key, value in content.items():
            if key == "composite_operations" and isinstance(value, dict):
                names |= set(value)
            elif key != "collection" and isinstance(value, dict):
                names.add(key)
    return names


def stamp_op_surface(notebook: dict[str, Any], op_names: set[str]) -> None:
    """Record the op surface an artifact was rendered against, in notebook metadata."""
    notebook.setdefault("metadata", {})[OP_SURFACE_KEY] = sorted(op_names)


def stale_output_references(artifact: dict[str, Any], current_ops: set[str], output_text: str) -> list[str]:
    """Op names an artifact's OUTPUTS mention that the current op surface no longer has.

    This is the gap the engine's source comparison structurally cannot close: its signature is built from
    ``(cell_type, source)``, so output drift is not merely unchecked, it is unrepresentable. A renamed op
    appearing only in captured output therefore passes every CPU job and leaves the docs site rendering a
    symbol that no longer exists.

    Comparing outputs directly would need a rendering environment, and a re-render rewrites thousands of
    lines of timings and object ids for a one-symbol rename. This compares the recorded op SURFACE instead:
    names the artifact was rendered against, minus names that exist now, intersected with what its outputs
    actually mention. No execution, no GPU, and it only fires when a specific dead name is really present.

    Returns an empty list for an artifact with no recorded surface (absence of a stamp is not evidence of
    freshness; the check reports those separately), and for an unreadable surface: with ``current_ops``
    empty, "recorded minus current" would be the whole recorded surface, and every stamped artifact would
    read as stale at once over a missing import.
    """
    if not current_ops:
        return []
    recorded = artifact.get("metadata", {}).get(OP_SURFACE_KEY)
    if not recorded:
        return []
    return sorted(name for name in set(recorded) - current_ops if name in output_text)


def unstamped_output_references(artifact: dict[str, Any], current_ops: set[str], output_text: str) -> list[str]:
    """Current op names an artifact's OUTPUTS mention that its recorded surface does NOT.

    The inverse of `stale_output_references`: an artifact re-rendered while the stamping machinery changed
    under it carries CURRENT outputs with a STALE stamp. `stale_output_references` stays silent (the outputs
    are genuinely current), but every op the stamp fails to record is invisible to the next rename's drift
    check, so the gate is silently weakened for exactly the files most recently touched.

    Matching is deliberately asymmetric with `stale_output_references`. The dead-name direction wants
    maximal sensitivity, so it uses plain substring: `sae_correct_acts` surviving inside
    `sae_correct_acts_impl` is still a dead reference on the page. This direction wants precision (it
    accuses a STAMP, not the page), so it matches on word boundaries: an output mentioning only
    `logit_diffs_latent` must not read as mentioning `logit_diffs`.

    The remedy also differs: a dead name in outputs needs a RE-EXECUTE; a current name missing from the
    stamp needs only a RE-STAMP (`--restamp`), because the outputs already describe the current world.
    """
    if not current_ops:
        return []
    recorded = set(artifact.get("metadata", {}).get(OP_SURFACE_KEY) or [])
    if not recorded:
        return []  # no stamp at all is the check's "no recorded op surface" note, not a stamp gap
    return sorted(name for name in current_ops - recorded if re.search(rf"\b{re.escape(name)}\b", output_text))


def _stamp(notebook: dict[str, Any]) -> None:
    stamp_op_surface(notebook, bundled_op_names())


def _check(artifact: dict[str, Any], *, rel: str, output_text: str) -> list[tuple[str, str]]:
    current = bundled_op_names()
    if not current:
        return []  # reported once by `_notes`, rather than as drift it cannot measure
    if dead := stale_output_references(artifact, current, output_text):
        return [("drift", f"STALE artifact OUTPUT (re-execute it): {rel}: outputs reference {', '.join(dead)}")]
    if not artifact.get("metadata", {}).get(OP_SURFACE_KEY):
        # Reported, not failed: an artifact predating the stamp cannot be checked, and failing on that would make
        # adopting the stamp a wall rather than a ramp. Silence would read as "checked and clean".
        return [("note", f"note: no recorded op surface, output drift not checkable: {rel}")]
    if gaps := unstamped_output_references(artifact, current, output_text):
        return [
            (
                "drift",
                f"STALE artifact STAMP (re-stamp it, no execution needed: {RENDER_COMMAND} --restamp --notebook "
                f"<name>): {rel}: outputs mention {', '.join(gaps)} absent from the recorded surface",
            )
        ]
    return []


def _notes() -> list[str]:
    if not bundled_op_names():
        return ["note: pyyaml unavailable, output-drift check skipped (source drift still checked)"]
    return []


def _restamp(artifact: dict[str, Any], *, output_text: str) -> tuple[str, str]:
    current = bundled_op_names()
    if not current:
        # Refuse rather than stamp an empty surface: an empty stamp would turn every later drift check into a
        # no-op for that file.
        raise RuntimeError("op surface unreadable (pyyaml unavailable)")
    if sorted(current) == (artifact.get("metadata", {}).get(OP_SURFACE_KEY) or []):
        return "unchanged", ""
    if dead := stale_output_references(artifact, current, output_text):
        # MECHANICAL refusal, not an advisory one: restamping sets recorded == current, after which
        # `recorded - current` is empty forever and the dead-name check can never fire again for this artifact.
        return (
            "refused",
            f"outputs still reference {', '.join(dead)} -- re-execute this artifact instead of restamping it",
        )
    stamp_op_surface(artifact, current)
    return "restamped", ""


HOOKS = {"stamp": _stamp, "check": _check, "notes": _notes, "restamp": _restamp}
