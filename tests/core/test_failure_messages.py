"""First-run failure-mode UX: each mode must fail with an actionable message (#253).

The acceptance contract is message CONTENT, not exception type: a first-time user sees prose, so
every case asserts the words that tell them what to do next. Cases live here as they land; the
extras mode is pinned in tests/core/test_bundled_composition_requires.py (both miss directions),
the trust mode by the hub trust-posture suite.
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

from interpretune.utils.exceptions import MisconfigurationException
from interpretune.utils.hf_auth_errors import (
    interpret_hf_model_load_error,
    model_id_from_pretrained_kwargs,
)


def _as_transformers_raises(hub_cls, status):
    """A typed Hub error chained under a generic OSError, the shape transformers raises on a failed load.

    Measured against the live Hub: a nonexistent repository arrives as ``OSError`` caused by
    ``RepositoryNotFoundError`` (status 401), and a gated one without a token as ``OSError`` caused by
    ``GatedRepoError`` (status 401). The outer message names neither, so classification must read the cause.
    """
    request = httpx.Request("GET", "https://huggingface.co/org/gated-model/resolve/main/config.json")
    hub_error = hub_cls(f"{status} Client Error", response=httpx.Response(status, request=request))
    try:
        try:
            raise hub_error
        except HfHubHTTPError as cause:
            raise OSError("org/gated-model is not a local folder and is not a valid model identifier") from cause
    except OSError as outer:
        return outer


def _interpret(error, **kwargs):
    kwargs.setdefault("model_id", "org/gated-model")
    kwargs.setdefault("access_token", None)
    kwargs.setdefault("auth_env_key", "HF_TOKEN")
    return interpret_hf_model_load_error(error, **kwargs)


class TestGatedModelAuthMessages:
    """Each Hub refusal must decode to its own instruction."""

    def test_401_without_a_token_names_the_credential(self):
        """No token offered: say where a token goes, not just 'unauthorized'."""
        err = _interpret(_as_transformers_raises(GatedRepoError, 401))
        assert isinstance(err, MisconfigurationException)
        assert "HF_TOKEN" in str(err) and "huggingface-cli login" in str(err)

    def test_401_with_a_token_blames_the_token(self):
        """A refused token must not read as a missing one (different fix)."""
        err = _interpret(_as_transformers_raises(GatedRepoError, 401), access_token="hf_stale")
        assert isinstance(err, MisconfigurationException)
        assert "refused the configured token" in str(err)

    def test_403_names_license_acceptance(self):
        """The repository gate is a license click, not a credential problem."""
        err = _interpret(_as_transformers_raises(GatedRepoError, 403), access_token="hf_fine")
        assert isinstance(err, MisconfigurationException)
        assert "huggingface.co/org/gated-model" in str(err)

    @pytest.mark.parametrize("status", [401, 404])
    def test_repository_not_found_names_both_readings(self, status):
        """Nonexistent OR private-to-someone-else; the Hub answers 401 to an anonymous request for either."""
        err = _interpret(_as_transformers_raises(RepositoryNotFoundError, status))
        assert isinstance(err, MisconfigurationException)
        assert "does not exist, or it is private" in str(err) and "HF_TOKEN" in str(err)

    def test_bare_http_401_is_a_token_problem(self):
        """A typed 401 from a non-repository endpoint reads like the gated case."""
        err = _interpret(_as_transformers_raises(HfHubHTTPError, 401), access_token="hf_stale")
        assert "refused the configured token" in str(err)

    @pytest.mark.parametrize(
        "error",
        [
            RuntimeError("size mismatch for lm_head.weight: copying a param with shape torch.Size([50401, 768])"),
            OSError("[Errno 2] No such file or directory: '/cache/models--org--m/snapshots/abc/model.safetensors'"),
            ValueError("the requested attention implementation is not found"),
            OSError("401 Client Error: Unauthorized for url"),
            RuntimeError("CUDA out of memory"),
        ],
        ids=["shape_with_401_digits", "missing_cache_file", "not_found_text", "auth_text_untyped", "oom"],
    )
    def test_unrecognized_errors_propagate_untouched(self, error):
        """Only a typed Hub error is interpreted; message text never is, however Hub-like it reads."""
        assert _interpret(error, access_token="hf_x") is None


class TestModelIdRecovery:
    def test_known_kwarg_keys_resolve(self):
        assert model_id_from_pretrained_kwargs({"pretrained_model_name_or_path": "org/m"}) == "org/m"

    def test_unknown_shape_falls_back(self):
        assert model_id_from_pretrained_kwargs({}) == "the configured model"
        assert model_id_from_pretrained_kwargs(None) == "the configured model"


class TestTroubleshootingPageStaysInSync:
    """The troubleshooting page quotes messages the code actually emits, and lists exactly the covered modes.

    Each quoted phrase is checked against a message generated by the code that raises it, so a reworded message fails
    here instead of leaving the page quoting text no user will ever see.
    """

    DOCS = Path(__file__).parent.parent.parent / "docs" / "source"
    PAGE = DOCS / "usage" / "when_things_go_wrong.md"

    @staticmethod
    def _flat(text: str) -> str:
        return " ".join(text.split())

    def _assert_quoted(self, message: str, *phrases: str) -> None:
        page, message = self._flat(self.PAGE.read_text(encoding="utf-8")), self._flat(message)
        for phrase in phrases:
            assert phrase in message, f"the code no longer emits {phrase!r}"
            assert phrase in page, f"the troubleshooting page no longer quotes {phrase!r}"

    def test_covered_modes_listed_and_uncovered_named(self):
        text = self.PAGE.read_text(encoding="utf-8")
        for mode in (
            "Unknown adapter combination",
            "Gated-model download failures",
            "Remote-code trust gate",
            "Unreachable hub component",
        ):
            assert f"## {mode}" in text, f"troubleshooting page dropped mode: {mode}"
        assert "CUDA-absent" in text and "cache-root" in text

    def test_composition_miss_messages_are_quoted(self):
        from interpretune.adapters.registration import CompositionRegistry
        from interpretune.utils.requirements import requirement_status

        with pytest.raises(KeyError) as miss:
            CompositionRegistry().get(("core", "no-such-adapter-xyz"))
        self._assert_quoted(str(miss.value), "was not found in the registry", "Available valid compositions")
        unmet = requirement_status({"pip": ["a-package-nobody-has"]}, source="x")[0].message
        self._assert_quoted(unmet, "requires pip package", "which is not installed", "Install it with")

    def test_gated_model_message_is_quoted(self):
        err = _interpret(_as_transformers_raises(RepositoryNotFoundError, 401))
        self._assert_quoted(str(err), "no repository with that id is visible to this request")

    def test_trust_and_unreachable_messages_are_quoted(self):
        from interpretune.hub.components import _explain_404
        from interpretune.hub.trust import trust_opt_in_message

        self._assert_quoted(
            trust_opt_in_message("org/repo", "analysis ops"),
            "Refusing to execute analysis ops from",
            "Interpretune does not execute hub-resident code unless you opt in",
        )
        self._assert_quoted(
            str(_explain_404("org/repo")),
            "The Hub returned 404 for",
            "the repo is absent, OR it is private and not visible to the token in use",
        )

    def test_doc_references_resolve(self):
        """Sphinx resolves a relative {doc} target against the page's directory and only warns on a miss."""
        targets = re.findall(r"\{doc\}`[^`<]*<([^>]+)>`", self.PAGE.read_text(encoding="utf-8"))
        assert targets, "the page links no other docs"
        unresolved = [
            t
            for t in targets
            if not ((self.DOCS if t.startswith("/") else self.PAGE.parent) / f"{t.lstrip('/')}.md").exists()
        ]
        assert not unresolved, f"unresolved {{doc}} targets: {unresolved}"
