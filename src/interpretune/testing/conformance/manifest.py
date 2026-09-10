"""The manifest of conformance cases that bind every out-of-tree backend.

A ``@conformance_case()`` with no capability gate is a requirement on EVERY model backend, in this repository and
out of it: a backend cannot decline it by declaring less. Adding one is therefore a breaking change to every
adapter repository that runs the suite, and it produces no signal here, because the only implementations this
repository can see are the ones written against the new case. The first time it happened, an out-of-tree
backend's default branch went red at the merge and stayed red, unobserved, until an unrelated push ran its suite
fifteen hours later.

This manifest makes the act visible at the point of adding. Every ungated case is listed here with what it binds,
the pull request that introduced it, and how its adoption downstream was tracked; a unit test in this repository
refuses by name an ungated case that is not listed, and a row whose case no longer exists. Growing the ungated
set then costs an edit to this file, which a reviewer sees in the diff and reads as what it is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from interpretune.testing.conformance.gates import _CASE_GATES, Gate

#: The class whose ungated cases bind every model backend. Cases are keyed by their qualified name.
MODEL_BACKEND_CASE_CLASS = "ModelBackendConformance"

#: An issue or pull request reference, in this repository (``#452``) or another (``owner/repo#31``).
_TRACKING_REF = re.compile(r"^(?:[\w.-]+/[\w.-]+)?#\d+$")


@dataclass(frozen=True)
class UngatedCase:
    """One case every out-of-tree backend must satisfy, and the record of how that requirement was introduced."""

    name: str
    """The case's qualified name, ``ModelBackendConformance.test_...``."""
    binds: str
    """Which targets the case binds, as the gate selects them: every backend, or every backend of one family."""
    introduced: str
    """The pull request that added the case, as an issue-shaped reference."""
    adoption: str
    """How the out-of-tree backends' adoption was tracked, or why none was needed."""


_EVERY = "every model backend"
_HF_NATIVE = "every model backend whose target declares the hf_native forward family"

UNGATED_CASES: tuple[UngatedCase, ...] = (
    UngatedCase(
        f"{MODEL_BACKEND_CASE_CLASS}.test_session_composes_and_declarations_are_coherent",
        _EVERY,
        "#452",
        "introduced with the suite; the bundled backends and the first hub adapter adopted the suite as a whole",
    ),
    UngatedCase(
        f"{MODEL_BACKEND_CASE_CLASS}.test_undeclared_capabilities_are_refused_by_name",
        _EVERY,
        "#452",
        "introduced with the suite; the refusal is the shared gate's, so a backend satisfies it by declaring "
        "only what it implements",
    ),
    UngatedCase(
        f"{MODEL_BACKEND_CASE_CLASS}.test_runner_produces_the_store_schema",
        _EVERY,
        "#452",
        "introduced with the suite",
    ),
    UngatedCase(
        f"{MODEL_BACKEND_CASE_CLASS}.test_cache_op_stores_logits_and_every_requested_point",
        _EVERY,
        "#452",
        "introduced with the suite",
    ),
    UngatedCase(
        f"{MODEL_BACKEND_CASE_CLASS}.test_block_output_is_the_next_block_input",
        _EVERY,
        "#452",
        "introduced with the suite",
    ),
    UngatedCase(
        f"{MODEL_BACKEND_CASE_CLASS}.test_capture_converges_on_the_forward",
        _HF_NATIVE,
        "#452",
        "introduced with the suite; a target opts out only by not being hf_native",
    ),
    UngatedCase(
        f"{MODEL_BACKEND_CASE_CLASS}.test_answer_logits_converge_on_the_forward",
        _HF_NATIVE,
        "#452",
        "introduced with the suite; a target opts out only by not being hf_native",
    ),
    UngatedCase(
        f"{MODEL_BACKEND_CASE_CLASS}.test_the_scope_discriminator_tells_the_scopes_apart",
        _HF_NATIVE,
        "#452",
        "introduced with the suite; runs on the HF model alone, so the backend has nothing to adopt",
    ),
    UngatedCase(
        f"{MODEL_BACKEND_CASE_CLASS}.test_a_batch_above_the_declared_limit_is_refused_by_name",
        "every target that declares it takes one prompt at a time",
        "#455",
        "added from the first hub adoption's own findings, so that adapter satisfied it at introduction",
    ),
    UngatedCase(
        f"{MODEL_BACKEND_CASE_CLASS}.test_the_backend_declares_what_it_can_capture",
        _EVERY,
        "#534",
        "adoption was not tracked; the first hub adapter's default branch went red at the merge and adopted the "
        "capture record in its own tree afterwards (#553 records the class)",
    ),
    UngatedCase(
        f"{MODEL_BACKEND_CASE_CLASS}.test_every_declared_point_is_captured",
        _EVERY,
        "#534",
        "adoption was not tracked; see test_the_backend_declares_what_it_can_capture (#553)",
    ),
    UngatedCase(
        f"{MODEL_BACKEND_CASE_CLASS}.test_a_point_outside_the_declaration_is_refused_by_name",
        _EVERY,
        "#534",
        "adoption was not tracked; see test_the_backend_declares_what_it_can_capture (#553)",
    ),
    UngatedCase(
        f"{MODEL_BACKEND_CASE_CLASS}.test_supplied_settings_survive_composition",
        "every target that supplies module_cfg_extras (the case skips, by name, when none were supplied)",
        "#551",
        "self-selecting: a backend is bound only once its target opts in by supplying extras, and the bundled "
        "targets that did were adjusted in the same pull request",
    ),
)


def binds_every_backend(gate: Gate) -> bool:
    """Whether a gate lets no backend decline the case by what it declares.

    A capability gate (positive or negative) selects on a declaration the backend controls. A gate on the forward family
    or the prompt shape selects on a property of the TARGET, which every backend of that kind has, so the case still
    binds all of them; those gates are ungated for the manifest's purpose.
    """
    return gate.capability is None


def ungated_cases() -> dict[str, Gate]:
    """The registered cases that bind every model backend, keyed by qualified name.

    Importing the case module registers the cases, and that module imports pytest, so this is resolved on call.
    """
    import interpretune.testing.conformance.cases  # noqa: F401  (registers the cases)

    return {
        name: gate
        for name, gate in _CASE_GATES.items()
        if name.startswith(f"{MODEL_BACKEND_CASE_CLASS}.") and binds_every_backend(gate)
    }


def manifest_drift(registered: dict[str, Gate] | None = None) -> tuple[list[str], list[str]]:
    """``(unlisted, stale)``: ungated cases the manifest does not list, and manifest rows with no such case."""
    registered = ungated_cases() if registered is None else registered
    listed = {row.name for row in UNGATED_CASES}
    unlisted = sorted(set(registered) - listed)
    stale = sorted(listed - set(registered))
    return unlisted, stale


def check_manifest(registered: dict[str, Gate] | None = None) -> None:
    """Refuse, by name, an ungated case the manifest does not list or a row whose case is gone or gated.

    Raises ``AssertionError`` with the names and what to do: list the case with its adoption tracking, gate it on
    a capability, or remove the stale row.
    """
    for row in UNGATED_CASES:
        if not _TRACKING_REF.match(row.introduced):
            raise AssertionError(
                f"manifest row {row.name} has introduced={row.introduced!r}; expected an issue-shaped reference "
                "such as '#452' or 'owner/repo#31'"
            )
        if not row.adoption.strip():
            raise AssertionError(
                f"manifest row {row.name} records no adoption; say how downstream adoption was tracked"
            )
    unlisted, stale = manifest_drift(registered)
    problems = []
    if unlisted:
        problems.append(
            "ungated conformance case(s) not in the manifest: "
            + ", ".join(unlisted)
            + ". An ungated case binds every out-of-tree backend the moment it merges and produces no signal here. "
            "Either gate it on a capability a backend can decline, or add it to UNGATED_CASES in "
            "interpretune/testing/conformance/manifest.py with the pull request that adds it and the issue "
            "tracking its adoption downstream."
        )
    if stale:
        problems.append(
            "manifest row(s) with no ungated case of that name: "
            + ", ".join(stale)
            + ". The case was removed, renamed, or is now capability-gated; remove or rename its row."
        )
    if problems:
        raise AssertionError("\n".join(problems))
