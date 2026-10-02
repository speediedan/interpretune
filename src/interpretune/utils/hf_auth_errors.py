"""Actionable messages for Hugging Face model-load auth failures (first-run UX).

Downloading gated weights fails in three ways a first-time user cannot tell apart from the raw
Hub error: no token configured, a token the Hub rejects, and a gated repository whose license the
account has not accepted. ``huggingface_hub`` collapses all three into ``OSError`` /
``HfHubHTTPError`` chains whose status codes a user must know to decode, so model init interprets
them here, at the load site, where the model id and the configured credential are still in hand.

Pure by construction: the interpreter takes the exception plus the two facts it needs and returns
an exception (or ``None`` for anything it does not recognize, letting the original propagate), so
every case is assertable on CPU with synthetic errors and no network.
"""

from __future__ import annotations

from typing import Any

from interpretune.utils.exceptions import MisconfigurationException

_MODEL_ID_KEYS = ("pretrained_model_name_or_path", "model_name", "model_id", "repo_id")


def model_id_from_pretrained_kwargs(pretrained_kwargs: dict[str, Any] | None) -> str:
    """The model id a load was attempted for, or a generic reference when it is unknowable."""
    for key in _MODEL_ID_KEYS:
        value = (pretrained_kwargs or {}).get(key)
        if value:
            return str(value)
    return "the configured model"


def interpret_hf_model_load_error(
    error: BaseException, *, model_id: str, access_token: str | None, auth_env_key: str | None
) -> MisconfigurationException | None:
    """Map a model-download failure to an actionable error, or ``None`` when unrecognized.

    The three gated-model cases decode by status: 401 with no token means nothing was offered;
    401 with a token means the token itself was refused; 403 means the repository gate (license
    acceptance) stopped an otherwise authenticated request; 404 means the id resolves to nothing
    the request can see -- nonexistent, or private to another account. Anything else propagates
    untouched: an unknown failure keeps today's behavior rather than gaining a wrong explanation.
    """
    text = f"{type(error).__name__}: {error}"
    lowered = text.lower()
    has_401 = "401" in lowered or "unauthorized" in lowered or "unauthenticated" in lowered
    has_403 = "403" in lowered or "forbidden" in lowered
    gated_hint = "gated" in lowered or "license" in lowered or "access to" in lowered
    credential = f" ${auth_env_key}" if auth_env_key else " a token"

    if has_401 and access_token is None:
        return MisconfigurationException(
            f"Could not download {model_id}: the Hub refused the request as unauthenticated and no "
            f"token was configured. Set{credential}, or run `huggingface-cli login` so the cached "
            f"credential applies. Original error: {text}"
        )
    if has_401:
        return MisconfigurationException(
            f"Could not download {model_id}: the Hub refused the configured token. Check that it is "
            f"still valid and carries access to this repository (regenerate at "
            f"huggingface.co/settings/tokens if in doubt). Original error: {text}"
        )
    if has_403 or gated_hint:
        return MisconfigurationException(
            f"Could not download {model_id}: the repository gate stopped the request. Accept the "
            f"model license at huggingface.co/{model_id} with the account owning the token, then "
            f"retry. Original error: {text}"
        )
    if "404" in lowered or "not found" in lowered or "no such" in lowered:
        return MisconfigurationException(
            f"Could not download {model_id}: the Hub reports it does not exist. If the id is "
            f"correct, it is private to another account (the Hub answers 404 rather than 403 for "
            f"repositories a token cannot see) -- request access or switch to a token that has it. "
            f"Original error: {text}"
        )
    return None
