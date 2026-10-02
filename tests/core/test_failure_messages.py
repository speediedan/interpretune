"""First-run failure-mode UX: each mode must fail with an actionable message (#253).

The acceptance contract is message CONTENT, not exception type: a first-time user sees prose, so
every case asserts the words that tell them what to do next. Cases live here as they land; the
extras mode is pinned in tests/core/test_bundled_composition_requires.py (both miss directions),
the trust mode by the hub trust-posture suite.
"""

from __future__ import annotations


from interpretune.utils.exceptions import MisconfigurationException
from interpretune.utils.hf_auth_errors import (
    interpret_hf_model_load_error,
    model_id_from_pretrained_kwargs,
)


def _interpret(text, **kwargs):
    kwargs.setdefault("model_id", "org/gated-model")
    kwargs.setdefault("access_token", None)
    kwargs.setdefault("auth_env_key", "HF_TOKEN")
    return interpret_hf_model_load_error(OSError(text), **kwargs)


class TestGatedModelAuthMessages:
    """The three gated-weight failures must decode to three different instructions."""

    def test_401_without_a_token_names_the_credential(self):
        """No token offered: say where a token goes, not just 'unauthorized'."""
        err = _interpret("401 Client Error: Unauthorized for url")
        assert isinstance(err, MisconfigurationException)
        assert "HF_TOKEN" in str(err) and "huggingface-cli login" in str(err)

    def test_401_with_a_token_blames_the_token(self):
        """A refused token must not read as a missing one (different fix)."""
        err = _interpret("401 Client Error: Unauthorized for url", access_token="hf_stale")
        assert isinstance(err, MisconfigurationException)
        assert "refused the configured token" in str(err)

    def test_403_names_license_acceptance(self):
        """The repository gate is a license click, not a credential problem."""
        err = _interpret(
            "403 Client Error: Forbidden - You must accept the license to access this repository",
            access_token="hf_fine",
        )
        assert isinstance(err, MisconfigurationException)
        assert "huggingface.co/org/gated-model" in str(err)

    def test_404_names_both_readings(self):
        """404 is nonexistent OR private-to-someone-else; the message must carry both."""
        err = _interpret("404 Client Error: Not Found for url")
        assert isinstance(err, MisconfigurationException)
        assert "private to another account" in str(err)

    def test_unrecognized_errors_propagate_untouched(self):
        """A classifier that explains everything explains nothing: unknown stays unknown."""
        assert _interpret("CUDA out of memory") is None
        assert _interpret("Connection reset by peer") is None


class TestModelIdRecovery:
    def test_known_kwarg_keys_resolve(self):
        assert model_id_from_pretrained_kwargs({"pretrained_model_name_or_path": "org/m"}) == "org/m"

    def test_unknown_shape_falls_back(self):
        assert model_id_from_pretrained_kwargs({}) == "the configured model"
        assert model_id_from_pretrained_kwargs(None) == "the configured model"
