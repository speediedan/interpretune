"""Actionable messages for Hugging Face model-load auth failures (first-run UX).

Downloading gated weights fails in ways a first-time user cannot tell apart from the raw error: no token
configured, a token the Hub rejects, a gated repository whose license the account has not accepted, and a
repository the request cannot see at all. transformers wraps each in a generic ``OSError`` whose text names
none of them reliably, but it chains the typed ``huggingface_hub`` error (``GatedRepoError``,
``RepositoryNotFoundError``, ``HfHubHTTPError``) as the cause, with the HTTP response attached. Model init
interprets that typed error here, at the load site, where the model id and the configured credential are
still in hand.

Classification reads exception TYPES and status codes, never message text: a load can fail for many reasons
whose messages happen to contain "401", "not found" or "no such" (a state-dict shape, a missing cache file),
and an unrecognized failure must keep today's behavior rather than gain a wrong explanation. Pure by
construction: every case is assertable on CPU with synthetic typed errors and no network.
"""

from __future__ import annotations

from typing import Any, Iterator

from interpretune.utils.exceptions import MisconfigurationException

_MODEL_ID_KEYS = ("pretrained_model_name_or_path", "model_name", "model_id", "repo_id")


def model_id_from_pretrained_kwargs(pretrained_kwargs: dict[str, Any] | None) -> str:
    """The model id a load was attempted for, or a generic reference when it is unknowable."""
    for key in _MODEL_ID_KEYS:
        value = (pretrained_kwargs or {}).get(key)
        if value:
            return str(value)
    return "the configured model"


def _exception_chain(error: BaseException) -> Iterator[BaseException]:
    """The error and everything it was raised from or during, outermost first, each once."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def interpret_hf_model_load_error(
    error: BaseException, *, model_id: str, access_token: str | None, auth_env_key: str | None
) -> MisconfigurationException | None:
    """Map a model-download failure to an actionable error, or ``None`` when unrecognized.

    Only a typed ``huggingface_hub`` HTTP error in the exception chain is interpreted. A gated repository
    answers 401 to an unauthenticated or refused request and 403 to an authenticated account that has not
    accepted its license; a repository the request cannot see at all (nonexistent, or private to another
    account) raises ``RepositoryNotFoundError`` whatever the status. Anything else propagates untouched.
    """
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    hub_error = next((e for e in _exception_chain(error) if isinstance(e, HfHubHTTPError)), None)
    if hub_error is None:
        return None
    status = getattr(getattr(hub_error, "response", None), "status_code", None)
    credential = f" ${auth_env_key}" if auth_env_key else " a token"
    original = f"Original error: {type(error).__name__}: {error}"

    # GatedRepoError subclasses RepositoryNotFoundError, so test it first; a bare 401/403 HTTP error from a
    # non-repository endpoint reads the same way as the gated case.
    refused_by_auth = isinstance(hub_error, GatedRepoError) or (
        status in (401, 403) and not isinstance(hub_error, RepositoryNotFoundError)
    )
    if refused_by_auth:
        if status == 401 and access_token is None:
            return MisconfigurationException(
                f"Could not download {model_id}: the Hub refused the request as unauthenticated and no token "
                f"was configured. Set{credential}, or run `huggingface-cli login` so the cached credential "
                f"applies. {original}"
            )
        if status == 401:
            return MisconfigurationException(
                f"Could not download {model_id}: the Hub refused the configured token. Check that it is still "
                f"valid and carries access to this repository (regenerate at huggingface.co/settings/tokens if "
                f"in doubt). {original}"
            )
        return MisconfigurationException(
            f"Could not download {model_id}: the repository gate stopped the request. Accept the model "
            f"license at huggingface.co/{model_id} with the account owning the token, then retry. {original}"
        )
    if isinstance(hub_error, RepositoryNotFoundError):
        token_hint = (
            f" No token was configured, so a private repository cannot be seen: set{credential} or run "
            "`huggingface-cli login`."
            if access_token is None
            else " If the id is correct, request access or switch to a token that has it."
        )
        return MisconfigurationException(
            f"Could not download {model_id}: no repository with that id is visible to this request. It does "
            f"not exist, or it is private to an account the request is not authenticated as.{token_hint} "
            f"{original}"
        )
    return None
