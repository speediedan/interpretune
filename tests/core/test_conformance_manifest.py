"""The manifest of ungated conformance cases is the forcing function on growing the set.

An ungated case binds every out-of-tree backend and produces no signal in this repository, so the only thing that can
make adding one visible is a check here that fails until the manifest is edited.
"""

from __future__ import annotations

import pytest

from interpretune.analysis.backends.capabilities import ModelBackendCapability
from interpretune.testing.conformance.gates import Gate
from interpretune.testing.conformance.manifest import (
    MODEL_BACKEND_CASE_CLASS,
    UNGATED_CASES,
    UngatedCase,
    binds_every_backend,
    check_manifest,
    manifest_drift,
    ungated_cases,
)


def test_the_manifest_matches_the_registered_ungated_cases():
    check_manifest()
    unlisted, stale = manifest_drift()
    assert (unlisted, stale) == ([], [])


def test_the_ungated_set_is_measured_not_recalled():
    """The predicate keys on the capability axis only: family and prompt-shape gates still bind every backend."""
    registered = ungated_cases()
    assert registered, "no ungated case is registered; the suite's always-on cases are missing"
    assert all(name.startswith(f"{MODEL_BACKEND_CASE_CLASS}.") for name in registered)
    assert all(gate.capability is None for gate in registered.values())
    assert binds_every_backend(Gate())
    assert binds_every_backend(Gate(family="hf_native"))
    assert binds_every_backend(Gate(single_prompt=True))
    assert not binds_every_backend(Gate(capability=ModelBackendCapability.GRADIENTS))
    assert not binds_every_backend(Gate(capability=ModelBackendCapability.ACTIVATION_INTERVENTION, negative=True))


def test_a_planted_ungated_case_is_refused_by_name():
    """Positive control: the check can fail. A new ungated case trips it, naming the case and what to do."""
    planted = dict(ungated_cases())
    planted[f"{MODEL_BACKEND_CASE_CLASS}.test_a_case_nobody_listed"] = Gate()
    with pytest.raises(AssertionError) as info:
        check_manifest(planted)
    message = str(info.value)
    assert "test_a_case_nobody_listed" in message
    assert "binds every out-of-tree backend" in message
    assert "UNGATED_CASES" in message


def test_a_planted_gated_case_does_not_trip_the_check():
    """The manifest is about the ungated set; a capability-gated case needs no row."""
    registered = dict(ungated_cases())
    # ungated_cases() has already filtered by the predicate; a gated addition never reaches the manifest
    gated = {f"{MODEL_BACKEND_CASE_CLASS}.test_gated": Gate(capability=ModelBackendCapability.GRADIENTS)}
    assert not any(binds_every_backend(g) for g in gated.values())
    check_manifest(registered)


def test_a_stale_row_is_refused_by_name(monkeypatch):
    """The other direction: a row whose case is gone (removed, renamed or gated) is refused, not ignored."""
    ghost = UngatedCase(f"{MODEL_BACKEND_CASE_CLASS}.test_removed_last_week", "every model backend", "#1", "n/a")
    monkeypatch.setattr("interpretune.testing.conformance.manifest.UNGATED_CASES", UNGATED_CASES + (ghost,))
    with pytest.raises(AssertionError) as info:
        check_manifest()
    assert "test_removed_last_week" in info.value.args[0]
    assert "remove or rename its row" in info.value.args[0]


def test_a_row_without_tracking_is_refused(monkeypatch):
    """A row must carry an issue-shaped provenance, so 'n/a' cannot stand in for the adoption record."""
    bad = UngatedCase(f"{MODEL_BACKEND_CASE_CLASS}.test_block_output_is_the_next_block_input", "x", "later", "")
    monkeypatch.setattr("interpretune.testing.conformance.manifest.UNGATED_CASES", (bad,))
    with pytest.raises(AssertionError, match="issue-shaped reference"):
        check_manifest()
