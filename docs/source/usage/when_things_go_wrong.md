# When Things Go Wrong

First-run failures in interpretune refuse by name: each mode below fails with a message that says
what happened and what to do next, and each is pinned by a test asserting message content, not
just the exception type. If you meet a bare traceback instead, that is a bug — file it with the
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

A miss with no availability note is a genuinely nonexistent combination — check spelling against
the {doc}`adapter-selection guide <usage/adapter_selection_guide>`.

## Gated-model download failures

Downloading gated weights fails in three ways the raw Hub error collapses into one. Model init
decodes them:

- **No token offered (401, nothing configured):** set the credential the config names
  (`$HF_TOKEN` or your `os_env_model_auth_key`), or run `huggingface-cli login`.
- **Token refused (401, token configured):** the token itself is bad — regenerate it and check
  repository access.
- **Repository gate (403, or license mentioned):** accept the model license at
  `huggingface.co/<model-id>` with the token's account, then retry.
- **Not found (404):** the id resolves to nothing visible — nonexistent, or private to another
  account (the Hub answers 404 rather than 403 for those). Pinned by
  `HubUnreachableError`, which carries both readings.

Anything else propagates as the original error: an unknown failure keeps today's behavior
rather than gaining a wrong explanation.

## Remote-code trust gate

Loading hub code (op collections and similar) refuses unless the session opts in, naming what
is being asked, what declining does, and how to pre-approve:

```text
Refusing to execute analysis ops from '<repo>': loading it runs code published by that repo
inside this process ... Interpretune does not execute hub-resident code unless you opt in.
```

Inspect first (`interpretune.hub.pull` caches without executing), then opt in for the session
with `IT_TRUST_REMOTE_CODE=1`, pinning a revision. See
{doc}`hub trust posture <usage/hub_trust_posture>`.
