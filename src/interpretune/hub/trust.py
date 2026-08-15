"""The one trust gate for code that arrives from the Hub (interpretune#255).

Interpretune's value proposition is running other people's analysis components, so "what am I
consenting to execute?" needs one answer in one place. Two paths execute hub-resident Python:
op-collection modules (analysis ops) and prompt-config entrypoints (``compose_ref``). Both
consult this module; everything else on the hub path is data (YAML manifests, configurations,
parquet) and is not gated here.

Two properties this module deliberately has, both of which the prior implementation lacked:

- **Default-deny.** An unset ``IT_TRUST_REMOTE_CODE`` denies rather than trusts. Publication of
  the seed components makes third-party ``it.hub`` loads possible for the first time, and a
  framework whose point is loading other people's code cannot ship "trust everything" as the
  value a user gets by not deciding.
- **Read at CALL time, never at import time.** The value it replaced was captured into a module
  constant when ``interpretune.analysis`` was imported, so the obvious notebook gesture
  (``import interpretune`` … ``os.environ["IT_TRUST_REMOTE_CODE"] = "1"``) silently did nothing.
  The opt-in has to work where the user actually is.

Deliberately NOT interactive. The previous unset path delegated to transformers'
``resolve_trust_remote_code``, which prompts on stdin under a 15-second ``SIGALRM``; in a
notebook, a CI job, or a Windows process that is variously a hang, a silent 15-second stall, or
an immediate error, and a single declined prompt aborted discovery for every remaining repo. A
deterministic refusal carrying the exact opt-in gesture is more useful than a prompt nobody is
present to answer.
"""

from __future__ import annotations

import os

#: Session-scoped opt-in. Accepted affirmative spellings mirror the values the warning text has
#: always advertised; any other set value is a deliberate opt-OUT, and unset is "not decided".
IT_TRUST_REMOTE_CODE_ENV_VAR = "IT_TRUST_REMOTE_CODE"
_AFFIRMATIVE = frozenset({"1", "true", "yes"})

TRUST_DOCS_URL = "https://interpretune.org/en/latest/usage/hub_trust_posture.html"


class RemoteCodeNotTrustedError(RuntimeError):
    """Executing hub-resident code was refused because remote code is not trusted this session."""


def remote_code_trust() -> bool | None:
    """Resolve the current trust decision: ``True`` (opt-in), ``False`` (opt-out), ``None`` (unset).

    The three states stay distinct because they warrant different messages: an explicit opt-out is
    a decision already taken and needs no advice, while "unset" is a user who has not been told
    what the choice is.
    """
    raw = os.environ.get(IT_TRUST_REMOTE_CODE_ENV_VAR)
    if raw is None:
        return None
    return raw.strip().lower() in _AFFIRMATIVE


def trust_opt_in_message(repo_id: str, what: str) -> str:
    """The actionable half of a refusal: what was refused, what it would have meant, how to allow it."""
    return (
        f"Refusing to execute {what} from {repo_id!r}: loading it runs code published by that repo "
        f"inside this process, with your filesystem, network and Hugging Face token. Interpretune "
        f"does not execute hub-resident code unless you opt in.\n"
        f"  Inspect it first: interpretune.hub.pull({repo_id!r}) caches the repo without executing "
        f"anything, so you can read the entrypoint the manifest names.\n"
        f"  Then opt in for this session: os.environ[{IT_TRUST_REMOTE_CODE_ENV_VAR!r}] = '1' "
        f"(or export {IT_TRUST_REMOTE_CODE_ENV_VAR}=1 before starting the process).\n"
        f"  Pin a revision when you pull so trusted code cannot change under you.\n"
        f"  Details: {TRUST_DOCS_URL}"
    )


def ensure_remote_code_trusted(repo_id: str, *, what: str) -> None:
    """Raise unless the user has opted in to executing hub-resident code.

    Used where refusing to execute means the caller cannot proceed at all (importing a prompt-config
    entrypoint). Discovery paths that can degrade to "load fewer things" use
    :func:`remote_code_trusted` instead.
    """
    if remote_code_trust() is not True:
        raise RemoteCodeNotTrustedError(trust_opt_in_message(repo_id, what))


def remote_code_trusted(repo_id: str, *, what: str) -> bool:
    """Whether hub-resident code may execute, warning ONCE per unset session rather than raising.

    Discovery is best-effort by nature (a user with cached op collections still gets a working session without them), so
    an unset trust decision skips with advice instead of failing the import. An explicit opt-out skips silently: that
    decision has already been made.
    """
    trust = remote_code_trust()
    if trust is True:
        return True
    if trust is None:
        from interpretune.utils.logging import rank_zero_warn

        rank_zero_warn(trust_opt_in_message(repo_id, what))
    return False
