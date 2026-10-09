# When Things Go Wrong

First-run failures in interpretune refuse by name: each mode below fails with a message that says
what happened and what to do next, and each is pinned by a test asserting message content, not
just the exception type. If you meet a bare traceback instead, that is a bug: file it with the
message and what you ran.

Covers: adapter composition misses, gated-model auth, remote-code trust, unreachable hub
components. CUDA-absent and cache-root modes are not covered yet.

## Unknown adapter combination

Requesting an adapter context with no registered composition raises `KeyError` naming the key
and every available composition. When the miss is really an absent backend, the error says so
and names the extra:

```text
The composition key `(<Adapter.core: 'core'>, <Adapter.circuit_tracer: 'circuit_tracer'>)` was not
found in the registry. Available valid compositions: ...
Unavailable in this environment:
  - circuit_tracer: interpretune.adapters.circuit_tracer: requires pip package 'circuit-tracer',
    which is not installed. Install it with: uv pip install 'circuit-tracer'
```

A miss with no availability note is a genuinely nonexistent combination: check spelling against
the {doc}`adapter-selection guide <adapter_selection_guide>`.

## Gated-model download failures

A failed model download reaches you as a generic `OSError` from transformers. Model init reads the
typed Hugging Face Hub error chained underneath it, and its HTTP status, and turns each case into
its own instruction:

- **No token offered (401, nothing configured):** set the credential the config names
  (`$HF_TOKEN` or your `os_env_model_auth_key`), or run `huggingface-cli login`.
- **Token refused (401, token configured):** the token itself is bad; regenerate it and check
  repository access.
- **Repository gate (403):** accept the model license at `huggingface.co/<model-id>` with the
  token's account, then retry.
- **Repository not visible:** "no repository with that id is visible to this request". It does not
  exist, or it is private to an account the request is not authenticated as; with no token
  configured, the message also says how to set one.

A failure without a typed Hub error underneath (a state-dict shape mismatch, a missing cache file)
propagates as the original error, however Hub-like its text reads: an unknown failure keeps
today's behavior rather than gaining a wrong explanation.

## Remote-code trust gate

Loading hub code (op collections and similar) refuses unless the session opts in, naming what
is being asked, what declining does, and how to pre-approve:

```text
Refusing to execute analysis ops from '<repo>': loading it runs code published by that repo
inside this process ... Interpretune does not execute hub-resident code unless you opt in.
```

Inspect first (`interpretune.hub.pull` caches without executing), then opt in for the session
with `IT_TRUST_REMOTE_CODE=1`, pinning a revision. See
{doc}`hub trust posture <hub_trust_posture>`.

## Unreachable hub component

Pulling a hub component whose repository answers 404 raises `HubUnreachableError`, whose message
carries both readings, because the Hub answers 404 rather than 403 for a private repository a
token cannot see:

```text
The Hub returned 404 for '<repo>': the repo is absent, OR it is private and not visible to the
token in use. ...
```

Check the token's repository scope before treating the component as absent; a cached snapshot is
not evidence that the repository still exists on the Hub.
