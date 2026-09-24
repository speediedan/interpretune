"""Unit tests for the opencode explanation-CLI route (fake executable, no network).

A shell script stands in for ``opencode``: it records its argv, prints canned stdout JSON event
lines (the error shape below is the measured ``opencode run --format json`` envelope), and
honors a mode env var for failure, delete-failure, and sleep (timeout) paths.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import tempfile
import warnings

import pytest

from interpretune.utils.neuronpedia_explanations import (
    OPENCODE_EXPLANATION_CLI_SPEC,
    ExplanationCliSpec,
    NeuronpediaExplanationError,
    build_explanation_cli_env,
    clean_explanation_text,
    extract_response_text_from_cli_events,
    extract_session_id_from_cli_events,
    invoke_explanation_cli,
    resolve_explanation_cli_spec,
)

FAKE_SCRIPT = """#!/bin/sh
echo "$@" >> "$FAKE_MARKER_DIR/argv.log"
prev=""
for a in "$@"; do
  if [ "$prev" = "-f" ]; then cp "$a" "$FAKE_MARKER_DIR/prompt_copy.md" 2>/dev/null; fi
  prev="$a"
done
if [ "$1" = "session" ]; then
  if [ "$FAKE_MODE" = "faildelete" ]; then exit 1; fi
  echo "$3" >> "$FAKE_MARKER_DIR/deleted.log"
  exit 0
fi
if [ "$FAKE_MODE" = "sleep" ]; then echo $$ > "$FAKE_MARKER_DIR/sleeper.pid"; sleep 30; fi
cat "$FAKE_STDOUT_FILE"
exit "${FAKE_EXIT_CODE:-0}"
"""

SUCCESS_EVENT = {
    "type": "message",
    "sessionID": "ses_ok1",
    "part": {"text": "Method: 1\nReason: test.\nExplanation: test-widget"},
}
# The measured `opencode run --format json` error envelope.
FAIL_EVENT = {"type": "error", "sessionID": "ses_fail1", "error": {"message": "boom"}}


@pytest.fixture()
def fake_cli(tmp_path, monkeypatch):
    """An executable fake `opencode` on PATH plus a marker dir; returns (spec, marker_dir, set_output)."""
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    exe = tmp_path / "fakeopencode"
    exe.write_text(FAKE_SCRIPT)
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("FAKE_MARKER_DIR", str(marker_dir))
    monkeypatch.setenv("FAKE_MODE", "success")
    monkeypatch.delenv("IT_EXPLANATION_CLI", raising=False)

    def set_output(*events, exit_code=0):
        out_file = tmp_path / f"canned_{len(list(tmp_path.glob('canned_*')))}.jsonl"
        out_file.write_text("\n".join(json.dumps(e) for e in events) + "\n")
        monkeypatch.setenv("FAKE_STDOUT_FILE", str(out_file))
        monkeypatch.setenv("FAKE_EXIT_CODE", str(exit_code))

    set_output(SUCCESS_EVENT)
    spec = ExplanationCliSpec(
        executable="fakeopencode",
        prompt_args=(),
        model_env_var=None,
        provider_type_env_var=None,
        provider_base_url_env_var=None,
        provider_api_key_env_var=None,
        model_args=("run", "--pure", "-m", "opencode/{model}", "--format", "json"),
        prompt_via_file=True,
        cleanup_session=True,
        kill_process_group_on_timeout=True,
        json_event_output=True,
    )
    return spec, marker_dir, set_output


def _run_argv(marker_dir):
    """The run call's argv (the log's first line; the session delete follows it)."""
    return (marker_dir / "argv.log").read_text().strip().splitlines()[0]


class TestArgvAssembly:
    def test_model_templated_prompt_in_file_not_argv(self, fake_cli):
        spec, marker_dir, _ = fake_cli
        prompt = "explain feature X with a very long prompt " * 50
        result = invoke_explanation_cli(prompt, explanation_model="my-model", cli_spec=spec)
        assert result.response_text is not None
        argv = _run_argv(marker_dir)
        assert "-m opencode/my-model" in argv
        assert "--format json" in argv
        assert "explain feature X" not in argv
        file_arg = argv.split("-f ", 1)[1].split()[0]
        assert os.path.isabs(file_arg)
        # The prompt file is removed after the call; the fake kept a copy to prove delivery.
        assert (marker_dir / "prompt_copy.md").read_text() == prompt
        assert clean_explanation_text(result.response_text) == "test-widget"

    def test_prompt_file_removed_afterwards(self, fake_cli):
        spec, _, _ = fake_cli
        before = set(os.listdir(tempfile.gettempdir()))
        invoke_explanation_cli("hi", cli_spec=spec)
        after = set(os.listdir(tempfile.gettempdir()))
        assert {d for d in after - before if d.startswith("it_explanation_prompt_")} == set()


class TestSessionCleanup:
    def test_success_deletes_session(self, fake_cli):
        spec, marker_dir, _ = fake_cli
        invoke_explanation_cli("hi", cli_spec=spec)
        assert (marker_dir / "deleted.log").read_text().strip() == "ses_ok1"

    def test_failure_still_cleans_up_then_raises(self, fake_cli):
        spec, marker_dir, set_output = fake_cli
        set_output(FAIL_EVENT, exit_code=3)
        with pytest.raises(NeuronpediaExplanationError, match="boom"):
            invoke_explanation_cli("hi", cli_spec=spec)
        assert (marker_dir / "deleted.log").read_text().strip() == "ses_fail1"

    def test_failed_delete_warns_but_returns(self, fake_cli, monkeypatch):
        spec, _, _ = fake_cli
        monkeypatch.setenv("FAKE_MODE", "faildelete")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = invoke_explanation_cli("hi", cli_spec=spec)
        assert result.response_text is not None
        assert any("session delete" in str(w.message) for w in caught)

    def test_timeout_kills_process_group(self, fake_cli, monkeypatch):
        spec, marker_dir, _ = fake_cli
        monkeypatch.setenv("FAKE_MODE", "sleep")
        with pytest.raises(subprocess.TimeoutExpired):
            invoke_explanation_cli("hi", timeout_seconds=1, cli_spec=spec)
        sleeper_pid = int((marker_dir / "sleeper.pid").read_text().strip())
        with pytest.raises(ProcessLookupError):
            os.kill(sleeper_pid, 0)

    def test_missing_executable_refused_by_name(self):
        spec = ExplanationCliSpec(executable="definitely-not-on-path-xyz")
        with pytest.raises(NeuronpediaExplanationError, match="Could not find"):
            invoke_explanation_cli("hi", cli_spec=spec)


class TestEventParsing:
    def test_session_id_from_error_shape(self):
        stdout = '{"type":"error","sessionID":"ses_x1","error":{"message":"Upstream request failed"}}'
        assert extract_session_id_from_cli_events(stdout) == "ses_x1"

    def test_session_id_absent_is_none(self):
        assert extract_session_id_from_cli_events("plain text\nnot json\n") is None

    def test_error_events_excluded_from_response(self):
        stdout = '{"type":"error","sessionID":"s","error":{"message":"should not leak"}}'
        with pytest.raises(NeuronpediaExplanationError, match="no response text"):
            extract_response_text_from_cli_events(stdout)

    def test_synthetic_nested_events_mechanism(self):
        """The collection mechanism (not a schema claim): nested strings gathered, joined."""
        stdout = json.dumps({"type": "message", "part": {"text": "Explanation: gizmo"}})
        assert "gizmo" in extract_response_text_from_cli_events(stdout)

    def test_empty_events_refused(self):
        with pytest.raises(NeuronpediaExplanationError, match="no response text"):
            extract_response_text_from_cli_events('{"type":"ping"}\nnot json\n')


class TestSpecResolution:
    def test_opencode_override_selects_full_route(self, monkeypatch):
        monkeypatch.setenv("IT_EXPLANATION_CLI", "opencode")
        assert resolve_explanation_cli_spec() is OPENCODE_EXPLANATION_CLI_SPEC

    def test_other_override_swaps_executable_only(self, monkeypatch):
        monkeypatch.setenv("IT_EXPLANATION_CLI", "mycli")
        resolved = resolve_explanation_cli_spec()
        assert resolved.executable == "mycli"
        assert resolved.prompt_args == ("-p",)

    def test_opencode_spec_takes_no_env_model(self):
        env = build_explanation_cli_env(OPENCODE_EXPLANATION_CLI_SPEC, base_env={})
        assert "COPILOT_MODEL" not in env
        assert env == {}
