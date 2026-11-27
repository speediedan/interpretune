"""Exact cross-batch op state when the analysis runner shares batches out across ranks (placement ``ddp``).

An op that declares ``op_state`` gives batch ``k`` a result that depends on every batch before it (``concept_direction``
writes the running direction over batches ``0..k``). Under data shares a rank never sees the other ranks' earlier
batches, so the analysis runner keeps the one-process result with two mechanisms, chosen from where the stateful ops
sit in the composition (:func:`plan_share_execution`):

- **Replay**, for a stateful op that is the composition's last: the ranks run the composition on their shares, each
  recording the op's per-batch inputs, and after the shares are merged rank 0 runs the op again over the recorded
  inputs in one-process order, with fresh state, and rewrites the op's output columns (:func:`replay_rows`).
- **A serial prefix pass**, for a stateful op whose outputs later ops consume: every rank first runs the composition up
  to and including that op over the whole batch sequence, in one-process order, keeping the outputs of the batches it
  takes; it then runs only the remainder on its own batches, each starting from its prefix outputs. The prefix is
  computed alike on every rank, so nothing is sent between ranks.

The op classes read the active plan through :func:`active_share`; the runner sets it for a run with
:func:`share_execution`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import torch

_ACTIVE: ContextVar[ShareExecution | None] = ContextVar("interpretune_share_execution", default=None)

BatchKey = tuple[int, int]  # (epoch, batch index in the whole sequence)


def _members(op: Any) -> list[Any]:
    return list(getattr(op, "composition", None) or [op])


def _detached(value: Any) -> Any:
    """``value`` with every tensor in it detached and moved to the CPU, so it can be kept and sent between
    ranks."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    from transformers import BatchEncoding

    if isinstance(value, BatchEncoding):  # kept a BatchEncoding, so an impl reading `batch.input_ids` still can
        return BatchEncoding({k: _detached(v) for k, v in value.items()})
    if isinstance(value, Mapping):
        return {k: _detached(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_detached(v) for v in value)
    return value


@dataclass
class ShareExecution:
    """How one analysis run's stateful ops are kept exact under data shares, and what the run has recorded.

    Attributes:
        replay_op: The name of the stateful op replayed after the merge (the composition's last), or None.
        prefix_end: The index in the composition of the last stateful op a later op consumes, or None. The ops up to
            and including it run in the serial prefix pass.
        in_prefix_pass: Whether the prefix pass is running, in which case a composition stops after ``prefix_end``.
        prefix_outputs: The prefix pass's analysis batch for each batch this rank takes, by :data:`BatchKey`.
        recorded: The replay op's inputs for each batch this rank ran, by :data:`BatchKey`.
    """

    replay_op: str | None = None
    prefix_end: int | None = None
    in_prefix_pass: bool = False
    prefix_outputs: dict[BatchKey, Any] = field(default_factory=dict)
    recorded: dict[BatchKey, dict[str, Any]] = field(default_factory=dict)

    @staticmethod
    def key(module: Any, batch_idx: int | None) -> BatchKey:
        """The :data:`BatchKey` of the batch ``module`` is analyzing: its current epoch and ``batch_idx``."""
        return int(getattr(module, "current_epoch", 0) or 0), int(batch_idx or 0)

    def start(self, module: Any, analysis_batch: Any, batch_idx: int | None) -> tuple[Any, int]:
        """The analysis batch a composition starts from, and the index of its first op to run.

        Outside the prefix pass, a composition with a prefix starts after it, from the batch's prefix outputs.
        """
        if self.prefix_end is None or self.in_prefix_pass:
            return analysis_batch, 0
        prefix = self.prefix_outputs.pop(self.key(module, batch_idx))
        for name, value in prefix.items():
            analysis_batch[name] = value
        return analysis_batch, self.prefix_end + 1

    def stops_before(self, index: int) -> bool:
        """Whether the prefix pass ends before the op at ``index``."""
        return self.in_prefix_pass and self.prefix_end is not None and index > self.prefix_end

    def record(self, module: Any, op: Any, analysis_batch: Any, batch: Any, batch_idx: int | None) -> None:
        """Keep what the replay op needs to run again on this batch: its input fields and the raw batch."""
        if self.in_prefix_pass or getattr(op, "name", None) != self.replay_op:
            return
        names = list((getattr(op, "input_schema", None) or {}).keys())
        inputs = {name: _detached(analysis_batch[name]) for name in names if name in analysis_batch}
        self.recorded[self.key(module, batch_idx)] = {"inputs": inputs, "batch": _detached(batch)}


def plan_share_execution(op: Any) -> ShareExecution | None:
    """The :class:`ShareExecution` that keeps ``op``'s cross-batch state exact under data shares, or None when no
    member of ``op`` declares ``op_state``."""
    members = _members(op)
    stateful = [i for i, member in enumerate(members) if getattr(member, "op_state", None) is not None]
    if not stateful:
        return None
    last = len(members) - 1
    consumed = [i for i in stateful if i < last]
    replay = getattr(members[last], "name", None) if stateful[-1] == last else None
    return ShareExecution(replay_op=replay, prefix_end=consumed[-1] if consumed else None)


def active_share() -> ShareExecution | None:
    """The run's :class:`ShareExecution`, while the analysis runner shares batches out; None otherwise."""
    return _ACTIVE.get()


@contextmanager
def share_execution(plan: ShareExecution | None) -> Iterator[ShareExecution | None]:
    """Make ``plan`` the active :class:`ShareExecution` for the enclosed block."""
    token = _ACTIVE.set(plan)
    try:
        yield plan
    finally:
        _ACTIVE.reset(token)


@contextmanager
def prefix_pass(plan: ShareExecution) -> Iterator[None]:
    """Run the enclosed block as the serial prefix pass of ``plan``."""
    plan.in_prefix_pass = True
    try:
        yield
    finally:
        plan.in_prefix_pass = False


def replay_rows(
    module: Any,
    analysis_cfg: Any,
    plan: ShareExecution,
    recorded: Mapping[BatchKey, Mapping[str, Any]],
    order: Sequence[BatchKey],
) -> dict[BatchKey, dict[str, Any]]:
    """Run ``plan``'s replay op again over ``recorded`` inputs in one-process ``order``, with fresh state, and
    return its stored output fields for each batch.

    State is reset first, and at each epoch boundary as the op's ``op_state`` asks, as the runner does for one process.
    The op is called with the analysis inputs generation gave it, and each output goes through the analysis config's
    ``save_batch``, so it is encoded as generation encoded it.
    """
    from interpretune.analysis.execution import build_analysis_inputs
    from interpretune.analysis.ops.base import AnalysisBatch
    from interpretune.analysis.optools import resolve_tokenizer

    op = next(member for member in _members(analysis_cfg.op) if getattr(member, "name", None) == plan.replay_op)
    outputs = list((getattr(op, "output_schema", None) or {}).keys())
    tokenizer = resolve_tokenizer(module)
    analysis_inputs = build_analysis_inputs(module, analysis_cfg=analysis_cfg)
    analysis_cfg.reset_op_state()
    rows: dict[BatchKey, dict[str, Any]] = {}
    epoch = None
    for key in order:
        if epoch is not None and key[0] != epoch:
            analysis_cfg.reset_op_state(epoch_boundary=True)
        epoch = key[0]
        entry = recorded[key]
        module.current_epoch = key[0]
        replayed = op._call_with_resolved_params(
            module, AnalysisBatch(entry["inputs"]), entry["batch"], key[1], analysis_inputs=analysis_inputs
        )
        stored = next(iter(analysis_cfg.save_batch(replayed, entry["batch"], tokenizer=tokenizer)))
        rows[key] = {name: stored[name] for name in outputs if name in stored}
    return rows
