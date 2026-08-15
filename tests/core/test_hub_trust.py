"""#255: the trust gate for hub-resident code — default-deny, call-time, non-interactive.

The suite as a whole opts in (tests/conftest.py), so every test here removes that consent explicitly. That asymmetry is
deliberate: the gate is only meaningful if something asserts the refusal, and a blanket-trusted suite would never notice
the gate disappearing.
"""

from __future__ import annotations

import pytest

from interpretune.hub.trust import (
    IT_TRUST_REMOTE_CODE_ENV_VAR,
    RemoteCodeNotTrustedError,
    ensure_remote_code_trusted,
    remote_code_trust,
    trust_opt_in_message,
)


class TestTrustResolution:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, None),
            ("1", True),
            ("true", True),
            ("TRUE", True),
            ("yes", True),
            (" yes ", True),
            ("0", False),
            ("false", False),
            ("no", False),
            ("", False),
            ("maybe", False),
        ],
        ids=[
            "unset",
            "one",
            "true",
            "TRUE-cased",
            "yes",
            "whitespace-padded",
            "zero",
            "false",
            "no",
            "empty-but-set",
            "unrecognized",
        ],
    )
    def test_three_states_from_env(self, monkeypatch, value, expected):
        """Unset must stay distinguishable from opt-out: they warrant different messages."""
        if value is None:
            monkeypatch.delenv(IT_TRUST_REMOTE_CODE_ENV_VAR, raising=False)
        else:
            monkeypatch.setenv(IT_TRUST_REMOTE_CODE_ENV_VAR, value)
        assert remote_code_trust() is expected

    def test_trust_is_read_at_call_time_not_import_time(self, monkeypatch):
        """The bug this replaced: opting in from a running notebook cell silently did nothing."""
        monkeypatch.delenv(IT_TRUST_REMOTE_CODE_ENV_VAR, raising=False)
        assert remote_code_trust() is None
        monkeypatch.setenv(IT_TRUST_REMOTE_CODE_ENV_VAR, "1")
        assert remote_code_trust() is True  # no re-import, no reload

    def test_refusal_names_the_repo_the_gesture_and_the_docs(self, monkeypatch):
        monkeypatch.delenv(IT_TRUST_REMOTE_CODE_ENV_VAR, raising=False)
        with pytest.raises(RemoteCodeNotTrustedError) as exc:
            ensure_remote_code_trusted("someorg/somerepo", what="the prompt-config entrypoint 'x.py'")
        message = str(exc.value)
        assert "someorg/somerepo" in message and "x.py" in message
        assert IT_TRUST_REMOTE_CODE_ENV_VAR in message  # the opt-in gesture is spelled out
        assert "interpretune.hub.pull" in message  # inspect-before-trusting escape hatch
        assert "revision" in message  # pinning escape hatch
        assert "interpretune.org" in message

    def test_explicit_opt_out_refuses_too(self, monkeypatch):
        monkeypatch.setenv(IT_TRUST_REMOTE_CODE_ENV_VAR, "0")
        with pytest.raises(RemoteCodeNotTrustedError):
            ensure_remote_code_trusted("someorg/somerepo", what="the entrypoint")

    def test_opt_in_permits(self, monkeypatch):
        monkeypatch.setenv(IT_TRUST_REMOTE_CODE_ENV_VAR, "yes")
        ensure_remote_code_trusted("someorg/somerepo", what="the entrypoint")  # does not raise

    def test_gate_is_not_interactive(self, monkeypatch):
        """Regression pin: the unset path must never read stdin (notebooks, CI and Windows all lose)."""
        monkeypatch.delenv(IT_TRUST_REMOTE_CODE_ENV_VAR, raising=False)

        def _explode(*args, **kwargs):
            raise AssertionError("the trust gate prompted for input")

        monkeypatch.setattr("builtins.input", _explode)
        with pytest.raises(RemoteCodeNotTrustedError):
            ensure_remote_code_trusted("someorg/somerepo", what="the entrypoint")
        assert "prompt" not in trust_opt_in_message("someorg/somerepo", "the entrypoint").lower()


@pytest.fixture()
def unexecuted_entrypoint(monkeypatch):
    """Withdraw consent AND any prior execution of the seed entrypoint, i.e. a fresh process.

    Necessary because the suite opts in globally: by the time these tests run, another test has
    usually executed the entrypoint already, and the gate deliberately does not re-challenge a
    module that has already run (see test_already_executed_module_is_not_re_challenged).
    """
    import sys

    monkeypatch.delenv(IT_TRUST_REMOTE_CODE_ENV_VAR, raising=False)
    for name in [n for n in sys.modules if n.startswith("it_hub_components.speediedan__prompt_configs")]:
        monkeypatch.delitem(sys.modules, name)


class TestPromptConfigEntrypointIsGated:
    """The exec_module path (#255's real hole): importing a component entrypoint runs its code."""

    def test_entrypoint_import_refused_without_consent(self, unexecuted_entrypoint):
        from interpretune.hub.promptconfigs import import_cached_entrypoint

        with pytest.raises(RemoteCodeNotTrustedError, match="prompt-config entrypoint"):
            import_cached_entrypoint("speediedan/prompt-configs")

    def test_compose_ref_resolution_refused_without_consent(self, unexecuted_entrypoint):
        from interpretune.hub.promptconfigs import instantiate_prompt_cfg_node

        with pytest.raises(RemoteCodeNotTrustedError):
            instantiate_prompt_cfg_node({"compose_ref": "speediedan/prompt-configs#GemmaPromptConfig"})

    def test_already_executed_module_is_not_re_challenged(self, monkeypatch):
        """The gate protects EXECUTION.

        Once code has run, withholding the module protects nothing.
        """
        from interpretune.hub.promptconfigs import import_cached_entrypoint

        monkeypatch.setenv(IT_TRUST_REMOTE_CODE_ENV_VAR, "1")
        first = import_cached_entrypoint("speediedan/prompt-configs")
        monkeypatch.delenv(IT_TRUST_REMOTE_CODE_ENV_VAR, raising=False)
        assert import_cached_entrypoint("speediedan/prompt-configs") is first

    def test_manifest_resolution_is_data_and_stays_ungated(self, monkeypatch):
        """Reading what a component DECLARES must not require consent to RUN it."""
        from interpretune.hub.components import resolve_component_manifest

        monkeypatch.delenv(IT_TRUST_REMOTE_CODE_ENV_VAR, raising=False)
        manifest, _, _ = resolve_component_manifest("speediedan/prompt-configs")
        assert "promptconfigs" in manifest.get("kinds", [])
        # and the manifest is exactly how a user learns which file they would be executing
        assert manifest["promptconfigs"]["entrypoint"]


class TestOpDiscoveryDegradesRatherThanRaising:
    """Ops discovery is best-effort: an undeclared preference must not fail the first op access."""

    def test_unset_trust_warns_with_advice_and_yields_no_hub_ops(self, monkeypatch, tmp_path):
        from unittest.mock import MagicMock, patch

        from interpretune.analysis.ops.compiler.cache_manager import OpDefinitionsCacheManager

        monkeypatch.delenv(IT_TRUST_REMOTE_CODE_ENV_VAR, raising=False)
        hub_cache = tmp_path / "hub"
        hub_cache.mkdir()
        manager = OpDefinitionsCacheManager(cache_dir=tmp_path / "cache")
        repo = MagicMock(repo_id="someorg/ops-collection", repo_type="model")
        with (
            patch("interpretune.analysis.IT_ANALYSIS_HUB_CACHE", hub_cache),
            patch(
                "interpretune.analysis.ops.compiler.cache_manager.scan_cache_dir",
                return_value=MagicMock(repos=[repo]),
            ),
            pytest.warns(match="Refusing to execute analysis ops"),
        ):
            assert manager.discover_hub_yaml_files() == []

    def test_explicit_opt_out_skips_quietly_without_the_advice(self, monkeypatch, tmp_path):
        from interpretune.analysis.ops.compiler.cache_manager import OpDefinitionsCacheManager

        monkeypatch.setenv(IT_TRUST_REMOTE_CODE_ENV_VAR, "0")
        manager = OpDefinitionsCacheManager(cache_dir=tmp_path / "cache")
        with pytest.warns(match="deliberate opt-out"):
            assert manager.add_hub_yaml_files() == []
