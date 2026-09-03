"""A hub adapter must not take a bundled adapter's name without being asked to.

The failure this guards is quiet by construction. Both definitions resolve to the SAME ``Adapter`` enum
member -- `register_dynamic_adapter` returns the existing member for a name already present -- so after the
fact there is no object to interrogate: the member cannot say which code it reaches. A session composed with
the wrong adapter looks exactly like one composed with the right one until behavior diverges.

So the tests assert the CONJUNCTION rather than either half. "Refused by default" and "allowed under the
opt-in" are each satisfiable by a broken implementation (one that always refuses; one that always allows),
and neither alone distinguishes a working precedence from a constant.
"""

from __future__ import annotations

import pytest

from interpretune.hub.precedence import (
    IT_ADAPTER_PRECEDENCE_ENV_VAR,
    AdapterCandidate,
    AdapterShadowError,
    adapter_info,
    adapter_precedence,
    bundled_adapter_names,
    enforce_adapter_precedence,
    prefer_adapters,
    record_hub_adapter,
)

HUB_REPO = "someone/it-circuit-tracer-adapter"
UNIQUE_NAME = "a_name_no_bundled_adapter_uses"


@pytest.fixture(autouse=True)
def _isolate_precedence_state(monkeypatch):
    """Module-level state is session-wide by design, so each test must start from a known empty one.

    Without this a test that opts in leaks the opt-in into every test after it, and the default-refuse assertions would
    pass or fail depending on execution ORDER -- the kind of coupling that turns a real regression into an
    unreproducible flake.
    """
    import interpretune.hub.precedence as prec

    monkeypatch.delenv(IT_ADAPTER_PRECEDENCE_ENV_VAR, raising=False)
    monkeypatch.setattr(prec, "_preferred_adapter_namespaces", [])
    monkeypatch.setattr(prec, "_hub_adapter_sources", {})
    yield


#: A stand-in bundled adapter for the LOGIC tests. Fixed rather than sampled from live discovery so the
#: precedence assertions are deterministic and, more importantly, CANNOT SILENTLY SKIP. The first version of
#: this module took the name from `bundled_adapter_names()` and skipped when discovery was empty; in an
#: environment whose installed metadata predated the entry-point group that skipped 10 of 15 tests while the
#: summary read "5 passed". A precedence rule is not the place to learn that lesson twice: an environment
#: fact should decide whether the DISCOVERY test runs, never whether the logic is checked at all.
STAND_IN_BUNDLED = "circuit_tracer"


@pytest.fixture
def bundled_name(monkeypatch):
    """Pin the bundled set for the logic under test, independent of what this env has installed."""
    import interpretune.hub.precedence as prec

    monkeypatch.setattr(prec, "bundled_adapter_names", lambda: {STAND_IN_BUNDLED})
    return STAND_IN_BUNDLED


class TestDiscoveryBackstheStandIn:
    """The one test that MAY skip on environment, isolated so its skipping costs only itself.

    It is what keeps `STAND_IN_BUNDLED` honest: the logic tests above are about precedence, but they would
    all still pass if the name they use had stopped being a bundled adapter, quietly testing a rule against
    an adapter nobody ships.
    """

    def test_the_stand_in_is_really_a_bundled_adapter(self):
        names = bundled_adapter_names()
        if not names:
            pytest.skip(
                "installed interpretune metadata predates the `interpretune.adapters` entry-point group; "
                "reinstall (`uv pip install -e .`) to exercise real discovery here"
            )
        assert STAND_IN_BUNDLED in names, (
            f"{STAND_IN_BUNDLED!r} is no longer a bundled adapter, so the precedence tests in this module "
            "are exercising their rule against a name interpretune does not ship. Pick a current one."
        )


class TestDefaultRefusal:
    def test_a_colliding_component_is_refused(self, bundled_name):
        with pytest.raises(AdapterShadowError) as excinfo:
            enforce_adapter_precedence(HUB_REPO, [bundled_name], source=f"{HUB_REPO}@abc123")
        assert bundled_name in str(excinfo.value)

    def test_the_refusal_names_the_opt_in(self, bundled_name):
        """A refusal a user cannot act on is half a diagnosis.

        The message must carry BOTH routes, because the two audiences are different: an interactive user can call the
        verb, a scripted or CI run can only set the variable.
        """
        with pytest.raises(AdapterShadowError) as excinfo:
            enforce_adapter_precedence(HUB_REPO, [bundled_name], source=f"{HUB_REPO}@abc123")
        message = str(excinfo.value)
        assert "prefer_adapters" in message
        assert IT_ADAPTER_PRECEDENCE_ENV_VAR in message
        assert HUB_REPO in message, "the message must name the component the user has to opt into"

    def test_a_non_colliding_component_is_not_refused(self):
        """POSITIVE CONTROL: the refusal is caused by the COLLISION, not by being a hub component.

        Without this, an implementation that refused every hub adapter would pass every other test in this
        class while making the whole component kind unusable.
        """
        enforce_adapter_precedence(HUB_REPO, [UNIQUE_NAME], source=f"{HUB_REPO}@abc123")

    def test_a_mixed_component_is_refused_and_names_only_the_colliding_half(self, bundled_name):
        enforce_adapter_precedence  # - referenced for readability of the assertion below
        with pytest.raises(AdapterShadowError) as excinfo:
            enforce_adapter_precedence(HUB_REPO, [UNIQUE_NAME, bundled_name], source=f"{HUB_REPO}@abc123")
        message = str(excinfo.value)
        assert bundled_name in message
        assert UNIQUE_NAME not in message, (
            "naming a non-colliding adapter in the refusal would send the user looking for a second problem "
            "that does not exist"
        )


class TestOptIn:
    def test_the_opt_in_permits_the_previously_refused_component(self, bundled_name):
        prefer_adapters(HUB_REPO)
        enforce_adapter_precedence(HUB_REPO, [bundled_name], source=f"{HUB_REPO}@abc123")

    def test_the_opt_in_is_per_component(self, bundled_name):
        """NEGATIVE CONTROL on the opt-in's scope: opting into one component must not open the gate for all.

        An implementation that set a single global "allow shadowing" flag would pass the test above and silently grant
        every other component the same permission.
        """
        prefer_adapters(HUB_REPO)
        with pytest.raises(AdapterShadowError):
            enforce_adapter_precedence("other/unrelated-adapter", [bundled_name], source="other@def456")

    def test_the_env_var_is_honored_and_read_late(self, bundled_name, monkeypatch):
        """Env parity for scripted runs, read on every call rather than cached at import.

        This module is imported early; a value exported afterwards would never be seen by a cached read. Setting it
        here, AFTER import, is what makes that property observable rather than assumed.
        """
        monkeypatch.setenv(IT_ADAPTER_PRECEDENCE_ENV_VAR, HUB_REPO)
        assert HUB_REPO in adapter_precedence()
        enforce_adapter_precedence(HUB_REPO, [bundled_name], source=f"{HUB_REPO}@abc123")

    def test_clearing_restores_the_default(self, bundled_name):
        prefer_adapters(HUB_REPO)
        prefer_adapters()  # no arguments clears
        with pytest.raises(AdapterShadowError):
            enforce_adapter_precedence(HUB_REPO, [bundled_name], source=f"{HUB_REPO}@abc123")


class TestAdapterInfoReportsTheConjunction:
    """The two halves must be asserted TOGETHER, per test, against the same state.

    Splitting them across tests would let a constant pass: an `is_shadowing_bundled` hardwired to ``False``
    satisfies the default case, and one hardwired to ``True`` satisfies the opt-in case. Only asserting both
    the flag AND the precedence that produced it, in one state, distinguishes a real resolution.
    """

    def test_without_the_opt_in_bundled_wins_and_nothing_reports_as_shadowing(self, bundled_name):
        record_hub_adapter(bundled_name, component=HUB_REPO, revision="abc123def456")
        resolution = adapter_info(bundled_name)
        assert resolution.active.source == "bundled"
        assert resolution.is_shadowing_bundled is False
        assert resolution.precedence == (), "no opt-in is in force, so no precedence should be reported"
        assert any(c.component == HUB_REPO for c in resolution.alternatives), (
            "the hub definition must still be REPORTED as an alternative; a resolution that hides the "
            "loser makes the shadowing invisible at the moment someone is asking about it"
        )

    def test_with_the_opt_in_the_hub_adapter_wins_and_says_so(self, bundled_name):
        record_hub_adapter(bundled_name, component=HUB_REPO, revision="abc123def456")
        prefer_adapters(HUB_REPO)
        resolution = adapter_info(bundled_name)
        assert resolution.active.source == "hub"
        assert resolution.active.component == HUB_REPO
        assert resolution.is_shadowing_bundled is True
        assert HUB_REPO in resolution.precedence, (
            "the flag alone does not explain itself; the precedence that produced it is what a user needs "
            "in order to turn it off"
        )
        assert any(c.source == "bundled" for c in resolution.alternatives)

    def test_a_hub_adapter_with_a_unique_name_is_not_shadowing(self):
        """NEGATIVE CONTROL on the flag: 'came from the hub' is not the same as 'is shadowing'.

        An implementation that reported `is_shadowing_bundled` for every hub adapter would pass the opt-in
        test above while crying wolf on the ordinary case this whole component kind exists to serve.
        """
        record_hub_adapter(UNIQUE_NAME, component=HUB_REPO, revision="abc123def456")
        resolution = adapter_info(UNIQUE_NAME)
        assert resolution.active.source == "hub"
        assert resolution.is_shadowing_bundled is False

    def test_an_unknown_name_raises_rather_than_reporting_an_empty_resolution(self):
        """'What does this name resolve to' has no honest empty answer when the name resolves to nothing."""
        with pytest.raises(KeyError, match="no adapter named"):
            adapter_info("definitely_not_an_adapter_name")

    def test_the_rendered_resolution_states_the_shadowing_in_words(self, bundled_name):
        """Printing it is the intended use, so the human-readable form must carry the finding.

        A flag a user has to know to check is weaker than a line they cannot miss when they print the object.
        """
        record_hub_adapter(bundled_name, component=HUB_REPO, revision="abc123def456")
        prefer_adapters(HUB_REPO)
        assert "shadowing" in str(adapter_info(bundled_name)).lower()


class TestCandidateSerialization:
    def test_absent_fields_are_omitted_rather_than_null(self):
        """A bundled adapter HAS no revision; recording ``null`` invites reading it as a failed lookup."""
        record = AdapterCandidate(name="core", source="bundled").to_dict()
        assert record == {"name": "core", "source": "bundled"}

    def test_hub_provenance_round_trips(self):
        record = AdapterCandidate(
            name="interp_engine", source="hub", component=HUB_REPO, revision="abc123def456"
        ).to_dict()
        assert record["component"] == HUB_REPO and record["revision"] == "abc123def456"
