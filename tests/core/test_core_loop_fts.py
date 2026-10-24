"""Finetuning-scheduler's ``FTSController`` driven from the core training loop through :class:`FTSCoreCallback`
(#762)."""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("finetuning_scheduler.core.controller", reason="finetuning-scheduler predates FTSController")

from tests.configuration import get_deepcopied_session  # noqa: E402
from tests.core import test_core_loop_training as core_loop  # noqa: E402
from interpretune.config import SessionRunnerCfg  # noqa: E402
from interpretune.runners.core import SessionRunner  # noqa: E402
from interpretune.tuning.core_loop import FTSCoreCallback  # noqa: E402
from interpretune.utils import MisconfigurationException  # noqa: E402

TWO_PHASES = {0: {"params": ["model.bias"], "max_transition_epoch": 1}, 1: {"params": ["model.weight"]}}


class Snapshots:
    """Records the module's parameters after every epoch, and the optimizer the loop would use next."""

    def __init__(self) -> None:
        self.params: list[dict[str, torch.Tensor]] = []
        self.optimizers: list[object] = []

    def on_train_epoch_end(self, ctx, metrics):
        self.params.append({n: p.detach().clone() for n, p in ctx.module.named_parameters()})
        self.optimizers.append(ctx.optimizer)


def _sgd(lr: float = 0.1):
    return partial(torch.optim.SGD, lr=lr)


def test_phases_thaw_in_order_and_the_loop_steps_the_controllers_optimizer():
    module = core_loop.TinyModule()
    initial = {n: p.detach().clone() for n, p in module.named_parameters()}
    fts = FTSCoreCallback(TWO_PHASES, controller_kwargs={"optimizer": _sgd()})
    snaps = Snapshots()
    core_loop._train(module, max_epochs=2, callbacks=[fts, snaps])

    first, second = snaps.params
    # phase 0 (epoch 0) trains only the bias
    assert torch.equal(first["model.weight"], initial["model.weight"])
    assert not torch.equal(first["model.bias"], initial["model.bias"])
    # phase 1 (epoch 1) thaws the weight
    assert not torch.equal(second["model.weight"], first["model.weight"])
    assert fts.controller.curr_depth == 1
    # the loop stepped the controller's optimizer, not the module's original one
    assert all(opt is fts.controller.optimizer for opt in snaps.optimizers)
    assert module.optimizers == [fts.controller.optimizer] and module.lr_scheduler_configs == []
    assert fts.controller.global_step == 4  # two batches per epoch, counted by the controller's own step hook


def test_the_optimizer_defaults_to_the_modules_optimizer_init():
    module = core_loop.TinyModule()
    module.it_cfg.optimizer_init = {"class_path": "torch.optim.SGD", "init_args": {"lr": 0.05}}
    fts = FTSCoreCallback(TWO_PHASES)
    core_loop._train(module, max_epochs=1, callbacks=[fts])
    optimizer = fts.controller.optimizer
    assert isinstance(optimizer, torch.optim.SGD) and optimizer.param_groups[0]["lr"] == 0.05


def test_an_optimizer_is_required():
    module = core_loop.TinyModule()
    with pytest.raises(MisconfigurationException, match="needs an optimizer"):
        core_loop._train(module, max_epochs=1, callbacks=[FTSCoreCallback(TWO_PHASES)])


def test_a_schedule_longer_than_the_loop_budget_is_reported(recwarn):
    module = core_loop.TinyModule()
    schedule = {
        0: {"params": ["model.bias"], "max_transition_epoch": 1},
        1: {"params": ["model.weight"], "max_transition_epoch": 5},
    }
    core_loop._train(
        module, max_epochs=2, callbacks=[FTSCoreCallback(schedule, controller_kwargs={"optimizer": _sgd()})]
    )
    assert any("beyond the core loop's `max_epochs` 2" in str(w.message) for w in recwarn.list)


def test_a_controller_stop_ends_the_loop():
    ctx = SimpleNamespace(should_stop=False)
    fts = FTSCoreCallback(TWO_PHASES)
    fts.controller = SimpleNamespace(should_stop=True)
    fts._sync_stop(ctx)
    assert ctx.should_stop


def test_a_factory_excludes_the_schedule_and_kwargs():
    with pytest.raises(MisconfigurationException, match="not both"):
        FTSCoreCallback(TWO_PHASES, controller_factory=lambda module: None)


def test_a_core_session_trains_phase_by_phase(get_it_session__core_cust__initonly):
    """Through a real session and `SessionRunner`: the controller is rooted at the module, names come from its
    view, and its optimizer replaces the one the session configured."""
    session = get_deepcopied_session(get_it_session__core_cust__initonly.it_session)
    module = session.module
    *_, earlier, last = [n for n, p in module.named_parameters() if p.requires_grad]
    schedule = {0: {"params": [last], "max_transition_epoch": 1}, 1: {"params": [earlier]}}
    fts = FTSCoreCallback(schedule)  # the optimizer comes from the module's `optimizer_init`
    snaps = Snapshots()
    initial = {n: p.detach().clone() for n, p in module.named_parameters()}
    run_cfg = SessionRunnerCfg(
        it_session=session, max_epochs=2, limit_train_batches=1, limit_val_batches=1, callbacks=[fts, snaps]
    )
    SessionRunner(run_cfg=run_cfg).train()
    assert fts.controller.view.root is module
    assert torch.equal(snaps.params[0][earlier], initial[earlier])
    assert not torch.equal(snaps.params[0][last], initial[last])
    assert not torch.equal(snaps.params[1][earlier], snaps.params[0][earlier])
    assert fts.controller.curr_depth == 1 and module.optimizers == [fts.controller.optimizer]
