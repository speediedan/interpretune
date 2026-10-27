"""The core training loop's lr scheduler stepping, step budget, per-epoch metrics, callbacks, validation with
gradients and checkpoint state (#761)."""

from __future__ import annotations

import io
import random
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from interpretune.config import SessionRunnerCfg
from interpretune.protocol import LRSchedulerConfig
from interpretune.runners.core import (
    CoreLoopContext,
    checkpoint_state,
    core_train_loop,
    load_checkpoint_state,
    step_lr_schedulers,
)
from interpretune.utils import MisconfigurationException


class TinyModule(nn.Module):
    """The surface the core loop uses, over a one-layer model."""

    def __init__(self, lr: float = 1.0, epoch_hooks: bool = True, batch_end_hook: bool = True) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.model = nn.Linear(2, 1)
        self.optimizers = [torch.optim.SGD(self.model.parameters(), lr=lr)]
        self.lr_scheduler_configs: list[LRSchedulerConfig] = []
        self.it_cfg = SimpleNamespace(profiler_cfg=SimpleNamespace(which=None))
        self.current_epoch, self.global_step = 0, 0
        self.dtype, self.device = torch.float32, torch.device("cpu")
        self._logged_metrics: dict[str, list[float]] = {}
        self.calls: list[str] = []
        self.val_grad_modes: list[tuple[bool, bool]] = []
        if epoch_hooks:
            self.on_train_epoch_start = lambda: self.calls.append("epoch_start")
            self.on_train_epoch_end = lambda: self.calls.append("epoch_end")
        if batch_end_hook:
            self.on_train_batch_end = lambda outputs, batch, batch_idx: None

    def log(self, name, value, *args, **kwargs):
        value = value.detach().item() if isinstance(value, torch.Tensor) else value
        self._logged_metrics.setdefault(name, []).append(value)

    def batch_to_device(self, batch):
        return batch

    def training_step(self, batch, batch_idx):
        loss = self.model(batch).pow(2).mean()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.val_grad_modes.append((torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
        self.log("val_loss", float(self.current_epoch + batch_idx))


def _datamodule(n_train: int = 2, n_val: int = 2):
    batches = [torch.ones(1, 2) * (i + 1) for i in range(max(n_train, n_val))]
    return SimpleNamespace(train_dataloader=lambda: batches[:n_train], val_dataloader=lambda: batches[:n_val])


def _train(module, max_epochs=1, **kwargs):
    return core_train_loop(module, _datamodule(**kwargs.pop("dm", {})), -1, -1, max_epochs, **kwargs)


class Recorder:
    """A callback recording every hook it receives."""

    def __init__(self) -> None:
        self.events: list[tuple] = []

    def on_train_start(self, ctx):
        self.events.append(("train_start",))

    def on_train_epoch_start(self, ctx):
        self.events.append(("epoch_start", ctx.epoch))

    def on_train_step_end(self, ctx, outputs):
        self.events.append(("step_end", ctx.optimizer_step))

    def on_validation_end(self, ctx, metrics):
        self.events.append(("validation_end", ctx.epoch, dict(metrics)))

    def on_train_epoch_end(self, ctx, metrics):
        self.events.append(("epoch_end", ctx.epoch))

    def on_train_end(self, ctx):
        self.events.append(("train_end",))


class TestLRSchedulerStepping:
    def test_step_interval_steps_after_each_optimizer_step(self):
        module = TinyModule()
        sched = torch.optim.lr_scheduler.StepLR(module.optimizers[0], step_size=1, gamma=0.5)
        module.lr_scheduler_configs = [LRSchedulerConfig(sched, interval="step")]
        _train(module, max_epochs=2, dm={"n_train": 3})
        assert sched.last_epoch == 6
        assert module.optimizers[0].param_groups[0]["lr"] == pytest.approx(0.5**6)

    def test_epoch_interval_honors_frequency(self):
        module = TinyModule()
        sched = torch.optim.lr_scheduler.StepLR(module.optimizers[0], step_size=1, gamma=0.5)
        module.lr_scheduler_configs = [LRSchedulerConfig(sched, interval="epoch", frequency=2)]
        _train(module, max_epochs=4)
        assert sched.last_epoch == 2

    def test_reduce_on_plateau_steps_with_its_monitor(self):
        module = TinyModule()
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(module.optimizers[0], patience=0, factor=0.5)
        module.lr_scheduler_configs = [LRSchedulerConfig(sched, interval="epoch", monitor="val_loss")]
        _train(module, max_epochs=3)  # val_loss rises every epoch, so each epoch after the first reduces
        assert module.optimizers[0].param_groups[0]["lr"] == pytest.approx(0.25)

    def test_reduce_on_plateau_without_monitor_refuses(self):
        module = TinyModule()
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(module.optimizers[0])
        module.lr_scheduler_configs = [LRSchedulerConfig(sched, interval="epoch")]
        with pytest.raises(MisconfigurationException, match="needs a `monitor`"):
            _train(module)

    def test_reduce_on_plateau_missing_metric_refuses_naming_it(self):
        module = TinyModule()
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(module.optimizers[0])
        module.lr_scheduler_configs = [LRSchedulerConfig(sched, interval="epoch", monitor="nope")]
        with pytest.raises(MisconfigurationException, match="monitors 'nope'"):
            _train(module)

    def test_reduce_on_plateau_missing_metric_non_strict_skips(self):
        module = TinyModule()
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(module.optimizers[0])
        module.lr_scheduler_configs = [LRSchedulerConfig(sched, interval="epoch", monitor="nope", strict=False)]
        with pytest.warns(UserWarning, match="Skipping a `ReduceLROnPlateau` step"):
            _train(module)

    def test_no_schedulers_is_a_no_op(self):
        step_lr_schedulers(SimpleNamespace(lr_scheduler_configs=None), "epoch", 1)


class TestStepBudget:
    def test_max_steps_stops_mid_epoch_and_still_validates(self):
        module = TinyModule()
        recorder = Recorder()
        ctx = _train(module, max_epochs=5, max_steps=3, callbacks=[recorder])
        assert ctx.optimizer_step == 3
        assert [e for e in recorder.events if e[0] == "epoch_start"] == [("epoch_start", 0), ("epoch_start", 1)]
        # the epoch in which the budget is spent still validates
        assert [e[1] for e in recorder.events if e[0] == "validation_end"] == [0, 1]

    def test_global_step_still_counts_every_dispatched_step(self):
        module = TinyModule()
        ctx = _train(module, max_epochs=1)
        assert (ctx.optimizer_step, module.global_step) == (2, 4)  # two training and two validation steps


class TestCallbacksAndHooks:
    def test_callback_order(self):
        module = TinyModule()
        recorder = Recorder()
        _train(module, max_epochs=1, dm={"n_train": 2, "n_val": 1}, callbacks=[recorder])
        assert [e[0] for e in recorder.events] == [
            "train_start",
            "epoch_start",
            "step_end",
            "step_end",
            "validation_end",
            "epoch_end",
            "train_end",
        ]
        assert module.calls == ["epoch_start", "epoch_end"]

    def test_partial_callbacks_are_fine(self):
        class OnlyEnd:
            ended = False

            def on_train_end(self, ctx):
                OnlyEnd.ended = True

        _train(TinyModule(), callbacks=[OnlyEnd()])
        assert OnlyEnd.ended

    def test_should_stop_ends_after_the_current_step(self):
        class StopAtTwo:
            def on_train_step_end(self, ctx, outputs):
                ctx.should_stop = ctx.optimizer_step >= 2

        ctx = _train(TinyModule(), max_epochs=3, dm={"n_train": 5}, callbacks=[StopAtTwo()])
        assert (ctx.optimizer_step, ctx.epoch) == (2, 0)

    def test_epoch_hooks_are_optional(self):
        _train(TinyModule(epoch_hooks=False), max_epochs=2)

    def test_batch_end_hook_is_optional(self):
        """A module without ``on_train_batch_end`` trains, as one without the other hooks does (#817)."""
        module = TinyModule(batch_end_hook=False)
        assert not hasattr(module, "on_train_batch_end")
        before = module.model.weight.detach().clone()
        ctx = _train(module, max_epochs=1)
        assert ctx.optimizer_step == 2 and not torch.equal(module.model.weight, before)

    def test_optimizer_is_read_every_step(self):
        module = TinyModule()
        replacement = torch.optim.SGD(module.model.parameters(), lr=0.0)

        class Swap:
            def on_train_step_end(self, ctx, outputs):
                ctx.module.optimizers = [replacement]

        stepped = []
        original_step = replacement.step
        replacement.step = lambda *a, **k: (stepped.append(1), original_step(*a, **k))[1]
        _train(module, dm={"n_train": 3}, callbacks=[Swap()])
        assert len(stepped) == 2  # every step after the first uses the replacement


class TestEpochMetrics:
    def test_metrics_are_averaged_and_reset_per_epoch(self, capsys):
        module = TinyModule()
        recorder = Recorder()
        _train(module, max_epochs=2, callbacks=[recorder])
        val_metrics = [e[2] for e in recorder.events if e[0] == "validation_end"]
        # val_loss is current_epoch + batch_idx: epoch 0 logs 0 and 1, epoch 1 logs 1 and 2
        assert [m["val_loss"] for m in val_metrics] == [0.5, 1.5]
        assert all("train_loss" in m for m in val_metrics)
        assert module._logged_metrics == {}
        out = capsys.readouterr().out
        assert "Epoch 0 end" in out and "Epoch 1 end" in out


class TestValidationGradMode:
    def test_default_validates_under_inference_mode(self):
        module = TinyModule()
        _train(module)
        assert set(module.val_grad_modes) == {(False, True)}

    def test_validate_with_grad(self):
        module = TinyModule()
        _train(module, validate_with_grad=True)
        assert set(module.val_grad_modes) == {(True, False)}


class TestCheckpointState:
    def _module_with_scheduler(self):
        module = TinyModule()
        sched = torch.optim.lr_scheduler.StepLR(module.optimizers[0], step_size=1, gamma=0.5)
        module.lr_scheduler_configs = [LRSchedulerConfig(sched, interval="step")]
        module.optimizers[0].param_groups[0]["momentum"] = 0.9  # optimizer state worth restoring
        return module

    def test_round_trip_restores_module_optimizer_scheduler_rng_and_counters(self):
        class Counter:
            def __init__(self):
                self.n = 0

            def on_train_step_end(self, ctx, outputs):
                self.n += 1

            def state_dict(self):
                return {"n": self.n}

            def load_state_dict(self, state):
                self.n = state["n"]

        module, counter = self._module_with_scheduler(), Counter()
        ctx = _train(module, max_epochs=2, callbacks=[counter])
        buffer = io.BytesIO()  # serialized, as a real checkpoint is: the state holds live references
        torch.save(checkpoint_state(module, ctx), buffer)
        weights = {k: v.clone() for k, v in module.state_dict().items()}
        lr = module.optimizers[0].param_groups[0]["lr"]
        expected_draws = (random.random(), torch.rand(1))  # the draws a resumed run must reproduce
        _train(module, max_epochs=1, callbacks=[counter])  # move everything on
        counter.n = 99

        buffer.seek(0)
        load_checkpoint_state(module, torch.load(buffer, weights_only=False), ctx)
        assert all(torch.equal(module.state_dict()[k], v) for k, v in weights.items())
        assert module.optimizers[0].param_groups[0]["lr"] == lr
        assert module.lr_scheduler_configs[0].scheduler.last_epoch == 4
        assert (ctx.optimizer_step, module.current_epoch, counter.n) == (4, 1, 4)
        assert (random.random(), torch.rand(1)) == (expected_draws[0], pytest.approx(expected_draws[1]))

    def test_module_hooks_can_extend_the_checkpoint(self):
        module = TinyModule()
        module.on_save_checkpoint = lambda checkpoint: checkpoint.update(extra=1)
        seen = []
        module.on_load_checkpoint = lambda checkpoint: seen.append(checkpoint["extra"])
        state = checkpoint_state(module)
        load_checkpoint_state(module, state)
        assert seen == [1]

    def test_optimizer_count_mismatch_refuses(self):
        module = TinyModule()
        state = checkpoint_state(module)
        module.optimizers.append(torch.optim.SGD(module.model.parameters(), lr=0.1))
        with pytest.raises(MisconfigurationException, match="holds 1 optimizers but the module has 2"):
            load_checkpoint_state(module, state)

    @pytest.mark.parametrize("as_path", [False, True], ids=["mapping", "path"])
    def test_resume_from_continues_as_the_uninterrupted_run(self, tmp_path, as_path):
        """A checkpoint taken in `on_train_epoch_end` resumes at the next epoch with the epoch-interval lr
        scheduler step it was taken before (#809)."""

        def module_with_epoch_scheduler():
            module = TinyModule(lr=0.1)
            sched = torch.optim.lr_scheduler.StepLR(module.optimizers[0], step_size=1, gamma=0.5)
            module.lr_scheduler_configs = [LRSchedulerConfig(sched, interval="epoch")]
            return module

        class Save:
            def __init__(self):
                self.checkpoint = None

            def on_train_epoch_end(self, ctx, metrics):
                if ctx.epoch == 1:
                    buffer = io.BytesIO()
                    torch.save(ctx.checkpoint_state(), buffer)
                    self.checkpoint = buffer.getvalue()

        ref, save = module_with_epoch_scheduler(), Save()
        ref_ctx = _train(ref, max_epochs=4, callbacks=[save])
        checkpoint = torch.load(io.BytesIO(save.checkpoint), weights_only=False)
        assert checkpoint["loop"]["epoch_lr_pending"] and checkpoint["loop"]["epoch"] == 1
        if as_path:
            torch.save(checkpoint, tmp_path / "ckpt.pt")
        resume_from = tmp_path / "ckpt.pt" if as_path else checkpoint

        resumed, recorder = module_with_epoch_scheduler(), Recorder()
        ctx = _train(resumed, max_epochs=4, callbacks=[recorder], resume_from=resume_from)
        assert [e[1] for e in recorder.events if e[0] == "epoch_start"] == [2, 3]
        assert all(torch.equal(p, q) for p, q in zip(ref.parameters(), resumed.parameters()))
        assert resumed.lr_scheduler_configs[0].scheduler.last_epoch == 4
        assert resumed.optimizers[0].param_groups[0]["lr"] == ref.optimizers[0].param_groups[0]["lr"]
        assert (ctx.optimizer_step, resumed.global_step) == (ref_ctx.optimizer_step, ref.global_step)

    def test_callbacks_load_before_the_module_and_optimizers(self):
        """A callback's `load_state_dict` runs first, so it may rebuild the optimizers the checkpoint's state loads
        into."""
        module = TinyModule()
        order = []

        class Rebuild:
            def state_dict(self):
                return {}

            def load_state_dict(self, state):
                order.append("callback")
                module.optimizers = [torch.optim.SGD(module.model.parameters(), lr=0.5)]

        ctx = CoreLoopContext(module=module, callbacks=(Rebuild(),))
        state = ctx.checkpoint_state()
        module.load_state_dict = lambda sd, _load=module.load_state_dict: (order.append("module"), _load(sd))[1]
        ctx.load_checkpoint_state(state)
        assert order == ["callback", "module"] and module.optimizers[0].param_groups[0]["lr"] == 1.0

    def test_context_shortcuts(self):
        module = TinyModule()
        ctx = CoreLoopContext(module=module, optimizer_step=7)
        state = ctx.checkpoint_state()
        ctx.optimizer_step = 0
        ctx.load_checkpoint_state(state)
        assert ctx.optimizer_step == 7


def test_session_runner_cfg_fields_reach_the_loop():
    """`SessionRunner` passes its config's fields to the loop as keyword arguments."""
    module, recorder = TinyModule(), Recorder()
    cfg = SessionRunnerCfg(
        module=module,
        datamodule=_datamodule(),
        max_epochs=3,
        max_steps=1,
        callbacks=[recorder],
        validate_with_grad=True,
    )
    ctx = core_train_loop(**cfg.__dict__)
    assert ctx.optimizer_step == 1 and recorder.events[0] == ("train_start",)
    assert set(module.val_grad_modes) == {(True, False)}
