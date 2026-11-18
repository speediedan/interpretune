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
    run_remote_chunk,
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


D_MODEL = 16


@pytest.fixture(scope="module")
def tiny_lm(tmp_path_factory):
    """A seeded random tiny GPT-2 served through nnsight, standing in for the remote frozen base."""
    from nnsight import TransformersModel
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    torch.manual_seed(0)
    config = GPT2Config(n_layer=2, n_head=2, n_embd=D_MODEL, vocab_size=len(tokenizer), n_positions=16)
    path = tmp_path_factory.mktemp("tiny_gpt2_remote")
    GPT2LMHeadModel(config).eval().save_pretrained(path)
    tokenizer.save_pretrained(path)
    return TransformersModel(str(path), task="text-generation", device="cpu", dispatch=True)


def _batches(n: int, start: int = 0) -> list[torch.Tensor]:
    return [torch.randint(0, 100, (2, 6), generator=torch.Generator().manual_seed(i)) for i in range(start, start + n)]


def _apply(model, params):
    # GPT-2's block returns its hidden states as a tensor here, so ``output[0]`` would be batch row 0 alone.
    site = model.transformer.h[0]
    site.output = site.output + params["intervention:steer"]


def _loss(model, batch):
    return model.lm_head.output[:, -1, 7].mean()


def _initial() -> RemoteChunkState:
    params = {"intervention:steer": torch.zeros(D_MODEL)}
    identity = RemoteIdentity.for_params("tiny-gpt2", "local", params)
    return RemoteChunkState(identity=identity, chunk_index=0, step=0, params=params)


def _oracle(lm, batches):
    """The same training in plain torch on the HF model: a forward hook adds the steering vector."""
    hf = lm._module
    steer = torch.nn.Parameter(torch.zeros(D_MODEL))
    opt = torch.optim.AdamW([steer], lr=0.1)
    hook = hf.transformer.h[0].register_forward_hook(lambda m, i, o: o + steer)
    losses = []
    try:
        for ids in batches:
            loss = hf(ids).logits[:, -1, 7].mean()
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())
    finally:
        hook.remove()
        hf.zero_grad(set_to_none=True)
    return steer.detach(), opt.state_dict(), losses


class TestSessionChunk:
    KW = dict(optimizer="AdamW", optimizer_kwargs={"lr": 0.1}, remote=False)

    def test_matches_a_plain_torch_oracle_and_leaves_the_base_unchanged(self, tiny_lm):
        base = {k: v.clone() for k, v in tiny_lm._module.state_dict().items()}
        state, losses = run_remote_chunk(tiny_lm, _initial(), _batches(4), _apply, _loss, **self.KW)
        tiny_lm._module.zero_grad(set_to_none=True)
        steer, opt_state, oracle_losses = _oracle(tiny_lm, _batches(4))
        assert losses == pytest.approx(oracle_losses, abs=1e-6)
        torch.testing.assert_close(state.params["intervention:steer"], steer)
        torch.testing.assert_close(state.optimizer_state["state"][0]["exp_avg"], opt_state["state"][0]["exp_avg"])
        assert state.params["intervention:steer"].abs().sum() > 0  # it trained
        assert (state.chunk_index, state.step, state.data_cursor) == (1, 4, 4)
        after = tiny_lm._module.state_dict()
        assert all(torch.equal(base[k], after[k]) for k in base)

    def test_two_chunks_resume_exactly_one_uninterrupted_chunk(self, tiny_lm):
        whole, _ = run_remote_chunk(tiny_lm, _initial(), _batches(4), _apply, _loss, **self.KW)
        ledger = RemoteCheckpointLedger(_initial())
        for start in (0, 2):
            returned, _ = run_remote_chunk(tiny_lm, ledger.committed, _batches(2, start), _apply, _loss, **self.KW)
            assert ledger.commit(returned)
        resumed = ledger.committed
        torch.testing.assert_close(resumed.params["intervention:steer"], whole.params["intervention:steer"])
        torch.testing.assert_close(
            resumed.optimizer_state["state"][0]["exp_avg_sq"], whole.optimizer_state["state"][0]["exp_avg_sq"]
        )
        assert (resumed.step, resumed.data_cursor) == (whole.step, whole.data_cursor)

    def test_dropping_the_optimizer_state_breaks_resume_parity(self, tiny_lm):
        """The planted defect the parity check exists for: a resume that restarts the optimizer still trains, and
        still returns a plausible loss, but does not reach the uninterrupted result."""
        whole, _ = run_remote_chunk(tiny_lm, _initial(), _batches(4), _apply, _loss, **self.KW)
        first, _ = run_remote_chunk(tiny_lm, _initial(), _batches(2), _apply, _loss, **self.KW)
        first.optimizer_state = {}
        second, losses = run_remote_chunk(tiny_lm, first, _batches(2, 2), _apply, _loss, **self.KW)
        assert all(abs(x) < 10 for x in losses)
        assert not torch.allclose(second.params["intervention:steer"], whole.params["intervention:steer"])

    def test_a_role_outside_the_phase_keeps_its_value(self, tiny_lm):
        params = {"intervention:steer": torch.zeros(D_MODEL), "intervention:idle": torch.ones(3)}
        start = RemoteChunkState(RemoteIdentity.for_params("tiny-gpt2", "local", params), 0, 0, params)
        state, _ = run_remote_chunk(
            tiny_lm, start, _batches(2), _apply, _loss, train_roles=["intervention:steer"], **self.KW
        )
        torch.testing.assert_close(state.params["intervention:idle"], torch.ones(3))
        with pytest.raises(RemoteTuningRefusal, match="not in this run's layout"):
            run_remote_chunk(tiny_lm, start, _batches(1), _apply, _loss, train_roles=["adapter:nope"], **self.KW)

    def test_an_unknown_optimizer_is_refused(self, tiny_lm):
        with pytest.raises(RemoteTuningRefusal, match="not a torch.optim optimizer"):
            run_remote_chunk(tiny_lm, _initial(), _batches(1), _apply, _loss, optimizer="Nope", remote=False)
