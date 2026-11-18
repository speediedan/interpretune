# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bounded remote-session tuning: the client's checkpoint contract.

A remote (NDIF) nnsight session trains only what it creates itself: the trainable tensors, their optimizer and the
loop are built inside ``model.session(remote=True)``, and only plain tensors come back. A session is bounded, so a run
is a sequence of chunks, each starting from the last state the client committed. This module is that client side: the
state a chunk carries out and back (:class:`RemoteChunkState`), the identity it must match (:class:`RemoteIdentity`),
and the ledger that commits a returned chunk only on top of the previous one (:class:`RemoteCheckpointLedger`).

Nothing here assumes remote state survives a session: a chunk that fails, or never returns, leaves the committed state
as it was, and the next chunk starts again from it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

# The portable chunk payload version `RemoteChunkState.from_portable` reads.
REMOTE_CHUNK_FORMAT = 1

# The role kinds a remote session may create and train; the served base is never trainable remotely.
TRAINABLE_ROLE_KINDS = ("adapter", "intervention")


class RemoteTuningRefusal(ValueError):
    """A remote tuning request, chunk or checkpoint this contract cannot vouch for."""


def check_role(role: str) -> str:
    """Return ``role`` if a remote session may train it, else refuse naming it.

    Only roles the session creates itself (``adapter:<name>``, ``intervention:<name>``) are trainable: the served
    model's own parameters, packed or quantized storage included, stay frozen and are never read back.
    """
    kind, sep, name = role.partition(":")
    if not sep or not name or kind not in TRAINABLE_ROLE_KINDS:
        kinds = ", ".join(f"{k}:<name>" for k in TRAINABLE_ROLE_KINDS)
        raise RemoteTuningRefusal(
            f"Remote tuning trains only tensors the session creates ({kinds}); {role!r} is not one, and the served "
            "base model is never tuned remotely."
        )
    return role


@dataclass(frozen=True)
class RemoteIdentity:
    """What every chunk of one run must agree on.

    Args:
        repo_id: The served model's repository id.
        revision: The served model's revision. A chunk produced against another revision is refused.
        layout: Each trainable role's ``(role, shape, dtype)``, in a fixed order.
        schedule_version: The version of the phase schedule the run follows.
    """

    repo_id: str
    revision: str
    layout: tuple[tuple[str, tuple[int, ...], str], ...]
    schedule_version: str = "1"

    def __post_init__(self) -> None:
        if not self.layout:
            raise RemoteTuningRefusal("A remote tuning run needs at least one trainable role.")
        roles = [check_role(role) for role, _, _ in self.layout]
        if len(set(roles)) != len(roles):
            raise RemoteTuningRefusal(f"Duplicate roles in the remote tuning layout: {roles}.")

    @classmethod
    def for_params(
        cls, repo_id: str, revision: str, params: Mapping[str, torch.Tensor], **kwargs: Any
    ) -> RemoteIdentity:
        """The identity whose layout is ``params``' roles, shapes and dtypes."""
        layout = tuple((role, tuple(t.shape), str(t.dtype)) for role, t in params.items())
        return cls(repo_id=repo_id, revision=revision, layout=layout, **kwargs)

    def check_params(self, params: Mapping[str, torch.Tensor]) -> None:
        """Refuse ``params`` unless they are exactly this layout's roles, shapes and dtypes."""
        expected = {role: (shape, dtype) for role, shape, dtype in self.layout}
        actual = {role: (tuple(t.shape), str(t.dtype)) for role, t in params.items()}
        if expected != actual:
            missing = sorted(set(expected) - set(actual))
            unknown = sorted(set(actual) - set(expected))
            changed = sorted(r for r in set(expected) & set(actual) if expected[r] != actual[r])
            raise RemoteTuningRefusal(
                f"Chunk tensors do not match the run's layout: missing {missing}, unknown {unknown}, "
                f"changed shape or dtype {changed}."
            )


@dataclass
class RemoteChunkState:
    """The state one chunk starts from and the next is built from.

    Args:
        identity: The run's :class:`RemoteIdentity`.
        chunk_index: ``0`` for the initial state, then one per returned chunk.
        step: Optimizer steps taken so far.
        params: Each trainable role's tensor.
        optimizer_state: The optimizer's ``state_dict()``, tensors included.
        data_cursor: The position in the data the next chunk reads from.
        rng_state: The generator state the next chunk resumes, if the run is seeded.
    """

    identity: RemoteIdentity
    chunk_index: int
    step: int
    params: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any] = field(default_factory=dict)
    data_cursor: int = 0
    rng_state: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.identity.check_params(self.params)
        if self.chunk_index < 0 or self.step < 0 or self.data_cursor < 0:
            raise RemoteTuningRefusal(
                f"Negative chunk index, step or data cursor ({self.chunk_index}, {self.step}, {self.data_cursor})."
            )
        for role, tensor in self.params.items():
            if tensor.device.type == "meta":
                raise RemoteTuningRefusal(
                    f"Role {role!r} came back on the meta device: the session that produced it never held real "
                    "weights (an untrusted NDIF key runs a meta model), so it trained nothing."
                )

    def digest(self) -> str:
        """A content hash over every field, so two returns of the same chunk can be told apart from two chunks."""
        h = hashlib.sha256()
        _feed(h, (self.identity, self.chunk_index, self.step, self.data_cursor))
        _feed(h, self.params)
        _feed(h, self.optimizer_state)
        _feed(h, self.rng_state)
        return h.hexdigest()

    def to_portable(self) -> dict[str, Any]:
        """Plain tensors and primitives only: no module, Envoy or optimizer object."""
        ident = self.identity
        return {
            "format": REMOTE_CHUNK_FORMAT,
            "identity": {
                "repo_id": ident.repo_id,
                "revision": ident.revision,
                "layout": [[role, list(shape), dtype] for role, shape, dtype in ident.layout],
                "schedule_version": ident.schedule_version,
            },
            "chunk_index": self.chunk_index,
            "step": self.step,
            "params": {role: t.detach().cpu().clone() for role, t in self.params.items()},
            "optimizer_state": _to_cpu(self.optimizer_state),
            "data_cursor": self.data_cursor,
            "rng_state": None if self.rng_state is None else self.rng_state.detach().cpu().clone(),
        }

    @classmethod
    def from_portable(cls, payload: Mapping[str, Any]) -> RemoteChunkState:
        """Read a :meth:`to_portable` payload, refusing a format this version does not know."""
        fmt = payload.get("format")
        if fmt != REMOTE_CHUNK_FORMAT:
            raise RemoteTuningRefusal(
                f"Remote chunk payload format {fmt!r} is not one this version reads ({REMOTE_CHUNK_FORMAT})."
            )
        ident = payload["identity"]
        identity = RemoteIdentity(
            repo_id=ident["repo_id"],
            revision=ident["revision"],
            layout=tuple((role, tuple(shape), dtype) for role, shape, dtype in ident["layout"]),
            schedule_version=ident["schedule_version"],
        )
        return cls(
            identity=identity,
            chunk_index=payload["chunk_index"],
            step=payload["step"],
            params=dict(payload["params"]),
            optimizer_state=dict(payload["optimizer_state"]),
            data_cursor=payload["data_cursor"],
            rng_state=payload["rng_state"],
        )


class RemoteCheckpointLedger:
    """Commits returned chunks in order, so no chunk advances the run twice or out of turn.

    Args:
        initial: The state before the first chunk (``chunk_index == 0``).
    """

    def __init__(self, initial: RemoteChunkState) -> None:
        if initial.chunk_index != 0:
            raise RemoteTuningRefusal(f"A ledger starts from chunk 0, not chunk {initial.chunk_index}.")
        self._history = [initial]
        self._digests = [initial.digest()]

    @property
    def committed(self) -> RemoteChunkState:
        """The last committed state: where the next chunk starts."""
        return self._history[-1]

    @property
    def history(self) -> tuple[RemoteChunkState, ...]:
        """Every committed state, initial first."""
        return tuple(self._history)

    def commit(self, returned: RemoteChunkState) -> bool:
        """Commit ``returned`` if it is the next chunk, and say whether it advanced the run.

        A second return of the chunk already committed (same content) is a no-op returning ``False``. A different
        chunk at a committed index, a gap, another identity, or a step or data cursor that went backwards is refused.
        """
        last = self.committed
        if returned.identity != last.identity:
            raise RemoteTuningRefusal(
                f"Chunk {returned.chunk_index} was produced for {returned.identity}, not this run's {last.identity}."
            )
        digest = returned.digest()
        if returned.chunk_index <= last.chunk_index:
            if self._digests[returned.chunk_index] == digest:
                return False
            raise RemoteTuningRefusal(
                f"Chunk {returned.chunk_index} is already committed with different contents; a chunk is committed once."
            )
        if returned.chunk_index != last.chunk_index + 1:
            raise RemoteTuningRefusal(
                f"Chunk {returned.chunk_index} does not follow committed chunk {last.chunk_index}."
            )
        if returned.step < last.step or returned.data_cursor < last.data_cursor:
            raise RemoteTuningRefusal(
                f"Chunk {returned.chunk_index} moved backwards: step {last.step} -> {returned.step}, data cursor "
                f"{last.data_cursor} -> {returned.data_cursor}."
            )
        self._history.append(returned)
        self._digests.append(digest)
        return True


def _to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, Mapping):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def _feed(h: Any, obj: Any) -> None:
    if isinstance(obj, torch.Tensor):
        t = obj.detach().cpu().contiguous()
        h.update(f"T{t.dtype}{tuple(t.shape)}".encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    elif isinstance(obj, Mapping):
        h.update(b"{")
        for key in sorted(obj, key=repr):
            _feed(h, key)
            _feed(h, obj[key])
        h.update(b"}")
    elif isinstance(obj, (list, tuple)):
        h.update(b"[")
        for item in obj:
            _feed(h, item)
        h.update(b"]")
    else:
        h.update(repr(obj).encode())
