"""The ONE core test that exercises a real published adapter component.

Interpretune's suite owns the rails (``test_hub_adapters.py``, against a fixture, always run) and this
single contract test (against the published component, opt-in). The adapter's own behaviour is tested
in the adapter's repository, so an interp-engine release that moves a signature turns that repo red
rather than this one.

**Opt-in, not an ``IT_RUN_*`` selector.** ``tests/conftest.py``'s ``pytest_collection_modifyitems``
REPLACES the collected list for those flags rather than adding to it, so introducing a fourth one
would silently deselect the rest of the suite. This uses its own environment variable and skips.

**Pinned by revision, deliberately.** Tracking a moving ref would let an unrelated component update
turn a core build red, which trains people to ignore red. Bumping the pin is a decision someone makes.
"""

from __future__ import annotations

import os

import pytest

#: Set to the published component's repo id to enable this test.
COMPONENT_ENV_VAR = "IT_HUB_ADAPTER_CONTRACT_REPO"
#: Set to the revision the contract was verified against. Required: an unpinned run is not a contract.
REVISION_ENV_VAR = "IT_HUB_ADAPTER_CONTRACT_REVISION"

pytestmark = pytest.mark.skipif(
    not os.environ.get(COMPONENT_ENV_VAR) or not os.environ.get(REVISION_ENV_VAR),
    reason=(
        f"set {COMPONENT_ENV_VAR} and {REVISION_ENV_VAR} to exercise a published adapter component. "
        "Skipped by default: it needs the network, executes remote code, and depends on an artifact "
        "this repository does not control."
    ),
)


@pytest.fixture(scope="module")
def component() -> tuple[str, str]:
    return os.environ[COMPONENT_ENV_VAR], os.environ[REVISION_ENV_VAR]


@pytest.fixture()
def trusted(monkeypatch):
    """Opt in explicitly.

    The gate defaults to refusing, and that default is deliberate.
    """
    from interpretune.hub.trust import IT_TRUST_REMOTE_CODE_ENV_VAR

    monkeypatch.setenv(IT_TRUST_REMOTE_CODE_ENV_VAR, "1")


class TestPublishedComponentSatisfiesTheRails:
    """Four assertions: it loads, it registers, it composes, and its declared surface is reachable.

    Deliberately NOT asserted here: anything about what the adapter computes. Duplicating the
    component's own suite means two places to update when its dependency moves a signature, and the
    duplicate is the one that rots.
    """

    def test_pull_then_load_registers_the_declared_adapters(self, component, trusted):
        import interpretune as it
        from interpretune.adapters.registration import CompositionRegistry
        from interpretune.hub.adapters import load_hub_adapter
        from interpretune.protocol import Adapter

        repo_id, revision = component
        it.hub.pull(repo_id, revision=revision)

        registry = CompositionRegistry()
        members = load_hub_adapter(repo_id, registry=registry)

        assert members, "a component declaring adapters must contribute at least one"
        for member in members:
            assert Adapter[member.name] is member, "the declared adapter must be a usable Adapter member"
        assert registry.available_compositions(), "loading must register at least one composition"

    def test_declared_compositions_match_the_manifest(self, component, trusted):
        """The manifest advertises a surface; loading must deliver exactly that surface."""
        from interpretune.adapters.registration import CompositionRegistry
        from interpretune.hub.adapters import load_hub_adapter
        from interpretune.hub.components import resolve_component_manifest

        repo_id, _ = component
        manifest, _, _ = resolve_component_manifest(repo_id)
        declared = set((manifest.get("adapters") or {}).get("declares") or [])

        registry = CompositionRegistry()
        members = load_hub_adapter(repo_id, registry=registry)
        assert {m.name for m in members} == declared

    def test_refusing_the_trust_gate_refuses_the_load(self, component, monkeypatch):
        """The gate is the whole safety story for a kind that composes into the MRO."""
        from interpretune.adapters.registration import CompositionRegistry
        from interpretune.hub.adapters import load_hub_adapter
        from interpretune.hub.trust import IT_TRUST_REMOTE_CODE_ENV_VAR, RemoteCodeNotTrustedError

        repo_id, _ = component
        monkeypatch.delenv(IT_TRUST_REMOTE_CODE_ENV_VAR, raising=False)
        with pytest.raises(RemoteCodeNotTrustedError):
            load_hub_adapter(repo_id, registry=CompositionRegistry())

    def test_the_components_own_surface_is_reachable(self, component, trusted):
        """A component may expose more than it registers; the accessor is how a caller reaches it."""
        from interpretune.adapters.registration import CompositionRegistry
        from interpretune.hub.adapters import load_hub_adapter, loaded_adapter_module

        repo_id, _ = component
        load_hub_adapter(repo_id, registry=CompositionRegistry())
        module = loaded_adapter_module(repo_id)
        assert module.__name__.startswith("it_hub_adapters.")
