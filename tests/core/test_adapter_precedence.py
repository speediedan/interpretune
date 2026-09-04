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


# --------------------------------------------------------------------------------------------------
# The executable parity test: WHY precedence has to exist at all.
# --------------------------------------------------------------------------------------------------

PARITY_ADAPTER = "parity_fixture_adapter"

PARITY_ENTRYPOINT = '''
    class ParityAdapterModule:
        """Stand-in for a real adapter mixin; the rails do not care what it composes."""

    class ParityAdapter:
        @classmethod
        def register_adapter_ctx(cls, adapter_ctx_registry) -> None:
            adapter_ctx_registry.register(
                lead_adapter="parity_fixture_adapter",
                component_key="module",
                adapter_combination=("core", "parity_fixture_adapter"),
                composition_classes=(ParityAdapterModule,),
                description="hub-delivered composition, for the parity comparison",
            )
'''


class BundledStyleModule:
    """Stand-in for a bundled adapter's composition class."""


@pytest.fixture()
def restore_adapter_enum():
    """Undo dynamic ``Adapter`` members a test adds; the enum is process-global state."""
    from interpretune.adapters import registration
    from interpretune.protocol import Adapter

    names = set(Adapter._member_map_)
    yield
    for name in set(Adapter._member_map_) - names:
        member = Adapter._member_map_.pop(name)
        Adapter._member_names_.remove(name)
        Adapter._value2member_map_.pop(member.value, None)
        type.__delattr__(Adapter, name)
        registration._DYNAMIC_ADAPTERS.pop(name, None)


@pytest.fixture()
def parity_registry(tmp_path, monkeypatch, restore_adapter_enum):
    """One registry holding two compositions of DIFFERENT provenance: one bundled-style, one hub-delivered."""
    import textwrap

    import yaml

    from interpretune.adapters.registration import CompositionRegistry
    from interpretune.hub.adapters import load_hub_adapter
    from interpretune.hub.components import local_publish
    from interpretune.hub.trust import IT_TRUST_REMOTE_CODE_ENV_VAR

    component = tmp_path / "component"
    component.mkdir(parents=True, exist_ok=True)
    (component / "it_component.yaml").write_text(
        yaml.safe_dump(
            {
                "it_schema_version": 1,
                "kinds": ["adapters"],
                "adapters": {
                    "entrypoint": "parity_entry.py",
                    "declares": [PARITY_ADAPTER],
                    "compositions": [{"component": "module", "adapters": ["core", PARITY_ADAPTER]}],
                },
            }
        ),
        encoding="utf-8",
    )
    (component / "parity_entry.py").write_text(textwrap.dedent(PARITY_ENTRYPOINT), encoding="utf-8")
    cache = tmp_path / "components"
    local_publish(component, "org/parity-fixture", cache_dir=cache)
    monkeypatch.setenv(IT_TRUST_REMOTE_CODE_ENV_VAR, "1")

    registry = CompositionRegistry()
    # The bundled side, registered exactly the way a bundled adapter's `register_adapter_ctx` does.
    registry.register(
        lead_adapter="core",
        component_key="module",
        adapter_combination=("core",),
        composition_classes=(BundledStyleModule,),
        description="bundled-style composition, for the parity comparison",
    )
    members = load_hub_adapter("org/parity-fixture", cache_dir=cache, registry=registry)
    return registry, members[0]


class TestHubAndBundledCompositionsAreIndistinguishable:
    """A hub-delivered composition must be a first-class citizen -- and that is EXACTLY why precedence exists.

    This is the negative control for the whole precedence rule, and it belongs beside it rather than in a
    parity module of its own. The rule above only earns its strictness if the two provenances are genuinely
    indistinguishable to a consumer: were a hub composition visibly different in the registry, shadowing
    would announce itself and a default-refuse would be over-engineering.

    So these tests assert the two halves of one claim. **Parity (R-I):** same retrieval call, same key
    shape, same entry shape, same enumeration -- a hub adapter is not second-class. **And its consequence:**
    the registry CONTRACT carries no provenance, so a silent substitution leaves nothing a consumer could
    branch on or notice, and `it.hub.adapter_info` is the instrument that recovers it.

    **One channel does survive, and what depends on it is a THREE-way split, not two.** Hub composition
    classes carry a revision-scoped ``__module__`` naming the component, because the loader imports each
    cached revision under its own synthetic module name:

    - **Registry contract** -- exposes no provenance (key shape, entry type, arity, enumeration). The claim
      the tests below make.
    - **Diagnostic** -- ``__module__`` as a debugging channel when a composition is misbehaving. Real, and
      not something code may branch on.
    - **Functionally load-bearing, tracked as #432** -- anything round-tripping a class by DOTTED PATH
      inherits the revision: ``class_path:`` YAML, ``instantiate_class(..., import_only=True)``, pickling,
      any config recording a class by qualified name. #432 is this same fact from the other side --
      registration parity holds while CONFIGURATION parity breaks, because a hub component has no stable
      dotted path to put in a ``class_path:`` at all.

    Saying only "diagnostic, not contract" would invite a reader to conclude nothing depends on it, and
    #432 is a standing counterexample.

    An earlier draft of this class asserted the stronger "nothing anywhere reveals provenance" and failed on
    first run -- correctly. **That is the good failure mode**: an assertion stronger than the truth fails
    immediately and cheaply, where one weaker than the truth passes forever and carries a false premise into
    the PR body, stated confidently. Prefer the assertion that can fail today over the one that would be
    comfortable.
    """

    @staticmethod
    def _keys(registry, hub_member):
        from interpretune.protocol import Adapter

        bundled = ("module", *registry.canonicalize_composition((Adapter.core,)))
        hub = ("module", *registry.canonicalize_composition((Adapter.core, hub_member)))
        return bundled, hub

    def test_both_are_retrieved_by_the_same_call(self, parity_registry):
        registry, hub_member = parity_registry
        bundled_key, hub_key = self._keys(registry, hub_member)
        assert registry.get(bundled_key) is not None, "the bundled-style composition must be retrievable"
        assert registry.get(hub_key) is not None, (
            "the hub-delivered composition must be retrievable through the SAME accessor; a separate lookup "
            "path would make hub adapters second-class and defeat the point of the component kind"
        )

    def test_the_key_shapes_are_the_same(self, parity_registry):
        """Positional string component key, then ``Adapter`` MEMBERS -- for both provenances.

        Worth asserting explicitly because the members are the subtle half: a hub adapter's member is
        created dynamically, and a key holding a plain string instead would still look right when printed.
        """
        from interpretune.protocol import Adapter

        registry, hub_member = parity_registry
        bundled_key, hub_key = self._keys(registry, hub_member)
        for key in (bundled_key, hub_key):
            assert isinstance(key[0], str)
            assert all(isinstance(part, Adapter) for part in key[1:]), (
                f"{key!r} holds a non-member adapter; compositions must key on Adapter members in both cases"
            )

    def test_both_appear_in_the_same_enumeration(self, parity_registry):
        registry, hub_member = parity_registry
        bundled_key, hub_key = self._keys(registry, hub_member)
        available = registry.available_compositions()
        for key in (bundled_key, hub_key):
            assert any(set(key[1:]) <= set(candidate) for candidate in available), (
                f"{key!r} is missing from `available_compositions()`; a composition a consumer cannot "
                "discover is not at parity however it was delivered"
            )

    def test_the_entries_have_the_same_shape(self, parity_registry):
        registry, hub_member = parity_registry
        bundled_key, hub_key = self._keys(registry, hub_member)
        bundled, hub = registry.get(bundled_key), registry.get(hub_key)
        assert type(bundled) is type(hub), (
            f"registry entries differ by provenance ({type(bundled)} vs {type(hub)}), so a consumer could "
            "branch on delivery mechanism -- which is precisely what compositional parity forbids"
        )

    def test_the_registry_contract_does_not_expose_provenance(self, parity_registry):
        """THE POINT, stated at the granularity that is actually true.

        The claim is about the registry's CONTRACT -- what a consumer retrieves and can branch on: the key,
        the entry, the enumeration. None of those differ by provenance, which is the parity guarantee and
        also why a silent substitution would leave nothing for a consumer to notice.

        **An earlier version of this test asserted more than that** -- that nothing anywhere in the entry's
        rendered form named its origin -- and it failed immediately, correctly. See the test below: a hub
        composition's classes carry a synthetic ``__module__`` that names the component. That is a real
        channel, and pretending otherwise would have put a false premise under the precedence rationale.
        """
        registry, hub_member = parity_registry
        bundled_key, hub_key = self._keys(registry, hub_member)
        bundled, hub = registry.get(bundled_key), registry.get(hub_key)
        assert type(bundled) is type(hub)
        assert len(bundled) == len(hub), "entry arity differs by provenance, so a consumer could branch on it"
        # The composition tuple holds classes in both cases -- not a class for one and a descriptor or
        # wrapper for the other, which is the shape a second-class delivery path would take.
        assert all(isinstance(c, type) for c in bundled) and all(isinstance(c, type) for c in hub)

    def test_provenance_survives_only_on_the_class_module_and_that_is_diagnostic_not_contract(self, parity_registry):
        """Pins the one channel that DOES carry origin, so neither its removal nor its use is silent.

        `_import_adapter_entrypoint` imports each component under a revision-scoped synthetic module name so
        two cached revisions cannot collide in ``sys.modules``. A side effect is that hub-delivered classes
        say where they came from, which is a genuinely useful thing when debugging a composition that is not
        behaving.

        It is NOT registry contract -- code must not branch on it to decide how to treat a composition,
        since the name is a loader implementation detail and depending on it would recreate the second-class
        delivery path parity exists to prevent.

        But it is not merely cosmetic either, and #432 is the reason to say so here: because the path
        carries a revision that changes on every publish, nothing a hub component defines has a stable
        dotted path, so ``class_path:`` YAML cannot name a hub component's classes at all. Registration
        parity holds; configuration parity does not. Any stable-alias scheme answering #432 therefore has to
        resolve the PRECEDENCE-WINNING component, which is where that issue meets this module.

        If a refactor ever flattens these module names, this test fails -- and both consequences (a lost
        debugging channel, and a changed premise under #432) become a decision rather than a side effect.
        """
        registry, hub_member = parity_registry
        _, hub_key = self._keys(registry, hub_member)
        modules = {c.__module__ for c in registry.get(hub_key)}
        assert any(m.startswith("it_hub_adapters.") for m in modules), (
            f"hub composition classes no longer carry a revision-scoped module name ({modules!r}). If that "
            "was deliberate, note that the only remaining way to identify a hub-delivered composition when "
            "debugging is `it.hub.adapter_info`."
        )

    def test_the_fixture_really_delivered_a_HUB_composition(self, parity_registry):
        """Guard against the whole class above passing vacuously.

        Every test here compares a hub-delivered composition against a bundled-style one. If the fixture
        ever stopped actually loading through the hub path -- a changed cache layout, a silently swallowed
        failure -- the comparisons would still run and still pass, against two bundled-style entries. This
        pins that the hub half is genuinely hub-delivered.
        """
        from interpretune.adapters.registration import dynamic_adapters

        _, hub_member = parity_registry
        assert hub_member.name == PARITY_ADAPTER
        assert hub_member.name in dynamic_adapters(), (
            "the fixture's adapter is not a DYNAMIC member, so it did not come through the hub load path "
            "and every parity comparison in this class is comparing bundled against bundled"
        )

    def test_adapter_info_is_the_instrument_that_can_tell_them_apart(self, parity_registry):
        """Positive control on the consequence: what the registry cannot say, `adapter_info` can.

        Without this the test above is only half an argument -- it establishes that provenance is invisible
        without establishing that anything recovers it, which would make the situation hopeless rather than
        merely requiring an instrument.
        """
        registry, hub_member = parity_registry
        record_hub_adapter(hub_member.name, component="org/parity-fixture", revision="abc123def456")
        resolution = adapter_info(hub_member.name)
        assert resolution.active.component == "org/parity-fixture"
        assert resolution.active.source == "hub"
