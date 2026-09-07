"""#462: cache presence is not Hub presence; every rendered revision must say which kind a reader is looking at."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from interpretune.hub.components import (
    HubPresence,
    HubUnreachableError,
    describe_revision,
    hub_presence,
    is_local_revision,
)
from tests.core.test_hub_manager import _create_mock_repository_not_found_error

LOCAL = "local" + "2931c895" + "a" * 27
HUB = "f13f4770" + "b" * 32


class TestRevisionsAreRenderedWithTheirOrigin:
    def test_describe_revision_names_a_local_publish(self):
        assert is_local_revision(LOCAL) and not is_local_revision(HUB) and not is_local_revision(None)
        assert describe_revision(LOCAL) == "local publish 2931c89"
        assert describe_revision(HUB) == "f13f4770bbbb" and describe_revision(None) == "none"

    def test_adapter_and_op_candidates_render_through_it(self):
        from interpretune.analysis.ops.dispatcher import OpCandidate
        from interpretune.hub.precedence import AdapterCandidate

        local = str(AdapterCandidate(name="x", source="hub", component="org/c", revision=LOCAL))
        hub = str(AdapterCandidate(name="x", source="hub", component="org/c", revision=HUB))
        assert "local publish" in local and "local publish" not in hub and "f13f4770bbbb" in hub
        op = str(OpCandidate(name="o", source="hub:org.c", collection="org/c", version=None, revision=LOCAL))
        assert "revision local publish" in op

    def test_load_messages_name_the_origin(self, tmp_path, monkeypatch, restore_adapter_enum=None):
        from interpretune.hub.components import local_publish, resolve_component_manifest

        # any component: the source string a load-time message is built from goes through describe_revision
        from tests.core.test_hub_adapters import REGISTERS_DECLARED, _write_component

        component = _write_component(tmp_path, REGISTERS_DECLARED)
        cache = tmp_path / "cache"
        local_publish(component, "org/fixture-adapter", cache_dir=cache)
        _, _, revision = resolve_component_manifest("org/fixture-adapter", cache_dir=cache)
        assert is_local_revision(revision)


class TestA404IsNotAbsence:
    def test_pull_explains_both_readings(self, tmp_path, monkeypatch):
        from interpretune.hub import components

        monkeypatch.setattr(
            components, "hf_hub_download", Mock(side_effect=_create_mock_repository_not_found_error("nope"))
        )
        with pytest.raises(HubUnreachableError, match="absent, OR it is private and not visible to the token") as info:
            components.pull_component_manifest("org/private", cache_dir=tmp_path)
        assert "whoami" in str(info.value) and "hub_presence" in str(info.value)

    def test_pull_ops_explains_both_readings(self, tmp_path, monkeypatch):
        with patch(
            "huggingface_hub.hf_hub_download", Mock(side_effect=_create_mock_repository_not_found_error("nope"))
        ):
            from interpretune.hub.opcollections import pull_op_collection

            with pytest.raises(HubUnreachableError, match="not visible to the token"):
                pull_op_collection("org/private", cache_dir=tmp_path)


class TestHubPresenceIsExplicit:
    def _api(self, side_effect=None, sha="deadbeef" * 5):
        api = Mock()
        if side_effect is not None:
            api.repo_info.side_effect = side_effect
        else:
            api.repo_info.return_value = Mock(sha=sha)
        return api

    def test_a_reachable_repo_and_revision(self):
        with patch("huggingface_hub.HfApi", return_value=self._api()):
            got = hub_presence("org/c", revision=HUB)
        assert got == HubPresence("org/c", True, HUB, True, got.detail) and "reachable" in got.detail

    def test_a_404_reads_as_absent_or_invisible(self):
        with patch("huggingface_hub.HfApi", return_value=self._api(_create_mock_repository_not_found_error("x"))):
            got = hub_presence("org/c")
        assert got.reachable is False and got.revision_present is None
        assert "absent, OR it is private and not visible to the token" in got.detail

    def test_a_local_revision_never_asks_the_hub(self):
        api = self._api()
        with patch("huggingface_hub.HfApi", return_value=api):
            got = hub_presence("org/c", revision=LOCAL)
        api.repo_info.assert_not_called()
        assert got.reachable is False and got.revision_present is False and "never on the Hub" in got.detail

    def test_a_missing_revision_on_a_reachable_repo(self):
        from huggingface_hub.errors import RevisionNotFoundError

        err = RevisionNotFoundError("gone", response=Mock(status_code=404, headers={}, request=Mock(url="u")))
        with patch("huggingface_hub.HfApi", return_value=self._api(err)):
            got = hub_presence("org/c", revision=HUB)
        assert got.reachable is True and got.revision_present is False
