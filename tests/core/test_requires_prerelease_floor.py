"""An interpretune floor that excludes only development builds says so, rather than reading as a broken
environment.

`>=0.1.0` is what an author writes when targeting the current version, and under PEP 440 it excludes `0.1.0.dev349`,
which is what everyone working on an unreleased version has installed. The check stays strict; the message names the
cause and the spelling that admits development builds.
"""

from __future__ import annotations

import pytest

from interpretune.utils import requirements
from interpretune.utils.requirements import requirement_status


def _message(monkeypatch, spec: str, installed: str) -> list[str]:
    monkeypatch.setattr(requirements, "installed_version", lambda dist: installed)
    return [u.message for u in requirement_status({"interpretune": spec}, "t")]


def test_a_floor_excluding_only_development_builds_names_the_cause(monkeypatch):
    (message,) = _message(monkeypatch, ">=0.1.0", "0.1.0.dev349+ge6df61ba3")
    assert "development build of 0.1.0" in message
    assert "'>=0.1.0.dev0'" in message
    assert "egg-info" not in message  # the stale-metadata hint names a different cause, so it must not appear here


@pytest.mark.parametrize(
    ("spec", "installed"),
    [(">=0.2.0", "0.1.0.dev349+ge6df61ba3"), (">=0.2.0", "0.1.0")],
    ids=["dev-build-below-a-later-floor", "release-below-the-floor"],
)
def test_a_genuine_conflict_keeps_the_ordinary_message(monkeypatch, spec, installed):
    """The explanation is reserved for the pre-release case: a build whose own release is ALSO too old is a real
    conflict, and telling its user the environment is fine would be the misdiagnosis in the other direction."""
    (message,) = _message(monkeypatch, spec, installed)
    assert "development build" not in message
    assert f"but {installed!r} is installed" in message


def test_a_dev_floor_admits_the_development_build(monkeypatch):
    """Positive control: the spelling the message recommends does satisfy the check."""
    assert _message(monkeypatch, ">=0.1.0.dev0", "0.1.0.dev349+ge6df61ba3") == []
