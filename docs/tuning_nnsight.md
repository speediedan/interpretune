# Tuning: nnsight

[Tuning](tuning.md) describes how every composition declares and attaches its Fine-Tuning Scheduler view.

A local, dispatched nnsight composition registers the nnsight model's HF module on the Interpretune module as `hf_model`
(in `register_tunable_modules()`), so the optimizer, checkpoints and the default view, venngram's composed view mounted
at `hf_model`, reach its parameters. nnsight builds the model through its own constructor, so a `lora_cfg` or
`bitsandbytesconfig` in `hf_from_pretrained_cfg` is not applied
([interpretune#942](https://github.com/speediedan/interpretune/issues/942)). A composition
executing remotely (NDIF) has no local weights to tune, and one built with `dispatch=False` keeps its weights on the
meta device; both are refused, each with its reason.

The paths to fuller local nnsight tuning, such as tuning applied through nnsight's own tracing, are tracked in
[interpretune#814](https://github.com/speediedan/interpretune/issues/814).

## Remote (NDIF) tuning in bounded sessions

A served model's own weights are never tuned remotely. What a remote session can train is what it creates itself:
adapter or intervention tensors, applied to the frozen base inside the session, with their optimizer and loop built
there too. This follows nnsight's
[remote training pattern](https://github.com/ndif-team/nnsight/blob/main/docs/patterns/remote-training.md): a tensor
or optimizer built on the client never reaches the server's copy, and only plain tensors come back.

A session is bounded, so `interpretune.tuning.remote` runs training as a sequence of chunks:

- `RemoteChunkState` is what a chunk starts from and returns: each role's tensor, the optimizer's state, the step, the
  data cursor and the RNG state, as plain tensors (`to_portable()` / `from_portable()`).
- `RemoteIdentity` is what every chunk of a run agrees on: the served model's repository and revision, each role's
  shape and dtype, and the schedule version. Only `adapter:<name>` and `intervention:<name>` roles are accepted.
- `run_remote_chunk(model, state, batches, apply_roles, loss_fn, ...)` runs one chunk inside `model.session(...)`:
  one trace per batch applies the roles and computes the loss, and the optimizer steps after it. `train_roles` limits a
  chunk to one phase's roles; the rest keep their values.
- `RemoteCheckpointLedger` commits a returned chunk only on top of the last committed one. A second return of the same
  chunk is a no-op, and a chunk produced for another revision or layout, a gap, or a step that went backwards is
  refused. A chunk that fails or never returns leaves the committed state as it was, and the next one starts from it.

```python
from interpretune.tuning.remote import RemoteCheckpointLedger, RemoteChunkState, RemoteIdentity, run_remote_chunk

params = {"intervention:steer": torch.zeros(hidden_size)}
start = RemoteChunkState(RemoteIdentity.for_params(repo_id, revision, params), chunk_index=0, step=0, params=params)
ledger = RemoteCheckpointLedger(start)


def apply_roles(model, roles):
    site = model.model.layers[12]
    site.output = site.output + roles["intervention:steer"]  # check the module's output type first


def loss_fn(model, batch):
    return model.lm_head.output[:, -1, target_id].mean()


for chunk in chunks:  # each chunk is a few batches: one bounded remote job
    returned, losses = run_remote_chunk(model, ledger.committed, chunk, apply_roles, loss_fn,
                                        optimizer_kwargs={"lr": 1e-2}, remote=True)
    ledger.commit(returned)
```

A decoder block may return its hidden states as a tensor rather than a tuple: indexing `output[0]` then selects the
first batch row alone, so check what the site's module returns before editing it.

A tensor returned on the meta device is refused: an NDIF key without the trusted tag runs sessions against a meta
model, so such a chunk trained nothing. The mode is tested in process, through nnsight's local simulation, and in a
separate process that holds its own copy of the model and returns only the saved values; it has not yet been run
against NDIF itself.

Concrete example usage for this adapter is to come; it is tracked in [interpretune#812](https://github.com/speediedan/interpretune/issues/812).
