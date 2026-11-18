"""The client checkpoint contract for bounded remote-session tuning (interpretune#775)."""

from __future__ import annotations

import pytest
import torch

from interpretune.tuning.remote import (
    REMOTE_CHUNK_FORMAT,
    RemoteCheckpointLedger,
    RemoteChunkState,
    RemoteIdentity,
    RemoteTuningRefusal,
    check_role,
)


def _params(seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {"adapter:lora_a": torch.randn(4, 2, generator=g), "intervention:steer": torch.randn(4, generator=g)}


def _optimizer_state(params: dict[str, torch.Tensor], steps: int) -> dict:
    tensors = [p.clone().requires_grad_(True) for p in params.values()]
    opt = torch.optim.AdamW(tensors, lr=0.1)
    for _ in range(steps):
        sum(t.square().sum() for t in tensors).backward()
        opt.step()
        opt.zero_grad()
    return opt.state_dict()


def _identity(**kwargs) -> RemoteIdentity:
    return RemoteIdentity.for_params("openai-community/gpt2", "607a30d7", _params(), **kwargs)


def _chunk(index: int, seed: int | None = None, step: int | None = None, cursor: int | None = None, **kwargs):
    params = _params(index if seed is None else seed)
    steps = 2 * index if step is None else step
    return RemoteChunkState(
        identity=kwargs.pop("identity", _identity()),
        chunk_index=index,
        step=steps,
        params=params,
        optimizer_state=_optimizer_state(params, steps),
        data_cursor=8 * index if cursor is None else cursor,
        rng_state=torch.Generator().manual_seed(index).get_state(),
    )


class TestRoles:
    @pytest.mark.parametrize("role", ["adapter:lora", "intervention:steer"])
    def test_session_created_roles_are_trainable(self, role):
        assert check_role(role) == role

    @pytest.mark.parametrize("role", ["base", "model.layers.0.mlp.weight", "fitted:x", "instrument:t", "adapter:"])
    def test_any_other_role_is_refused_by_name(self, role):
        with pytest.raises(RemoteTuningRefusal, match="never tuned remotely"):
            check_role(role)

    def test_layout_refuses_a_base_role_and_duplicates(self):
        with pytest.raises(RemoteTuningRefusal, match="'base'"):
            RemoteIdentity("m", "r", (("base", (2,), "torch.float32"),))
        with pytest.raises(RemoteTuningRefusal, match="Duplicate"):
            RemoteIdentity("m", "r", (("adapter:a", (2,), "torch.float32"),) * 2)
        with pytest.raises(RemoteTuningRefusal, match="at least one"):
            RemoteIdentity("m", "r", ())


class TestChunkState:
    def test_portable_round_trip_preserves_every_field(self):
        state = _chunk(2)
        payload = state.to_portable()
        assert payload["format"] == REMOTE_CHUNK_FORMAT
        restored = RemoteChunkState.from_portable(payload)
        assert restored.identity == state.identity
        assert restored.digest() == state.digest()
        restored.params["adapter:lora_a"].add_(1.0)  # the payload is a copy, not a view of the live state
        assert state.digest() == _chunk(2).digest()

    def test_an_unknown_format_is_refused(self):
        payload = _chunk(1).to_portable()
        payload["format"] = REMOTE_CHUNK_FORMAT + 1
        with pytest.raises(RemoteTuningRefusal, match="format"):
            RemoteChunkState.from_portable(payload)

    def test_layout_mismatch_is_refused_naming_the_roles(self):
        params = _params()
        params["adapter:lora_a"] = torch.zeros(4, 3)
        params["adapter:extra"] = torch.zeros(1)
        del params["intervention:steer"]
        with pytest.raises(RemoteTuningRefusal, match=r"missing \['intervention:steer'\].*unknown \['adapter:extra'\]"):
            RemoteChunkState(identity=_identity(), chunk_index=1, step=1, params=params)

    def test_meta_tensors_are_refused(self):
        """An untrusted NDIF key runs a meta model: what it returns trained nothing and must not be committed."""
        params = {role: t.to("meta") for role, t in _params().items()}
        with pytest.raises(RemoteTuningRefusal, match="meta device"):
            RemoteChunkState(identity=_identity(), chunk_index=1, step=1, params=params)

    def test_digest_changes_with_optimizer_state(self):
        a, b = _chunk(1), _chunk(1)
        assert a.digest() == b.digest()
        b.optimizer_state["state"][0]["exp_avg"].add_(1e-3)
        assert a.digest() != b.digest()


class TestLedger:
    def test_chunks_commit_in_order(self):
        ledger = RemoteCheckpointLedger(_chunk(0))
        assert ledger.commit(_chunk(1)) and ledger.commit(_chunk(2))
        assert [s.chunk_index for s in ledger.history] == [0, 1, 2]
        assert ledger.committed.step == 4

    def test_a_duplicate_return_does_not_advance_twice(self):
        ledger = RemoteCheckpointLedger(_chunk(0))
        assert ledger.commit(_chunk(1)) is True
        assert ledger.commit(_chunk(1)) is False
        assert len(ledger.history) == 2

    def test_a_conflicting_duplicate_is_refused(self):
        ledger = RemoteCheckpointLedger(_chunk(0))
        ledger.commit(_chunk(1))
        with pytest.raises(RemoteTuningRefusal, match="already committed"):
            ledger.commit(_chunk(1, seed=99))

    def test_a_failed_chunk_leaves_the_committed_state(self):
        ledger = RemoteCheckpointLedger(_chunk(0))
        ledger.commit(_chunk(1))
        before = ledger.committed.digest()
        with pytest.raises(RemoteTuningRefusal, match="does not follow"):
            ledger.commit(_chunk(3))  # chunk 2 never came back
        assert ledger.committed.digest() == before and ledger.committed.chunk_index == 1
        assert ledger.commit(_chunk(2))  # the retry from the committed state goes through

    def test_another_revision_is_refused(self):
        ledger = RemoteCheckpointLedger(_chunk(0))
        stale = RemoteIdentity.for_params("openai-community/gpt2", "0000000", _params())
        with pytest.raises(RemoteTuningRefusal, match="not this run's"):
            ledger.commit(_chunk(1, identity=stale))

    def test_another_schedule_version_is_refused(self):
        ledger = RemoteCheckpointLedger(_chunk(0))
        with pytest.raises(RemoteTuningRefusal, match="not this run's"):
            ledger.commit(_chunk(1, identity=_identity(schedule_version="2")))

    def test_steps_or_cursor_going_backwards_are_refused(self):
        ledger = RemoteCheckpointLedger(_chunk(0, step=0, cursor=0))
        ledger.commit(_chunk(1, step=4, cursor=16))
        with pytest.raises(RemoteTuningRefusal, match="moved backwards"):
            ledger.commit(_chunk(2, step=3, cursor=24))
        with pytest.raises(RemoteTuningRefusal, match="moved backwards"):
            ledger.commit(_chunk(2, step=6, cursor=8))

    def test_a_ledger_starts_at_chunk_zero(self):
        with pytest.raises(RemoteTuningRefusal, match="chunk 0"):
            RemoteCheckpointLedger(_chunk(1))
