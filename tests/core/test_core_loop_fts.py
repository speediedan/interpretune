"""Finetuning-scheduler's ``TuningScheduler`` driven from the core training loop through :class:`FTSCoreCallback`
(#762)."""

from __future__ import annotations

from functools import partial

import pytest
import torch

pytest.importorskip(
    "finetuning_scheduler.core.tuning_scheduler", reason="finetuning-scheduler predates TuningScheduler"
)

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


def test_phases_thaw_in_order_and_the_loop_steps_the_tuning_schedulers_optimizer():
    module = core_loop.TinyModule()
    initial = {n: p.detach().clone() for n, p in module.named_parameters()}
    fts = FTSCoreCallback(TWO_PHASES, tuning_scheduler_kwargs={"optimizer": _sgd()})
    snaps = Snapshots()
    core_loop._train(module, max_epochs=2, callbacks=[fts, snaps])

    first, second = snaps.params
    # phase 0 (epoch 0) trains only the bias
    assert torch.equal(first["model.weight"], initial["model.weight"])
    assert not torch.equal(first["model.bias"], initial["model.bias"])
    # phase 1 (epoch 1) thaws the weight
    assert not torch.equal(second["model.weight"], first["model.weight"])
    assert fts.tuning_scheduler.curr_depth == 1
    # the loop stepped the tuning scheduler's optimizer, not the module's original one
    assert all(opt is fts.tuning_scheduler.optimizer for opt in snaps.optimizers)
    assert module.optimizers == [fts.tuning_scheduler.optimizer] and module.lr_scheduler_configs == []
    assert (
        fts.tuning_scheduler.global_step == 4
    )  # two batches per epoch, counted by the tuning scheduler's own step hook


def test_the_optimizer_defaults_to_the_modules_optimizer_init():
    module = core_loop.TinyModule()
    module.it_cfg.optimizer_init = {"class_path": "torch.optim.SGD", "init_args": {"lr": 0.05}}
    fts = FTSCoreCallback(TWO_PHASES)
    core_loop._train(module, max_epochs=1, callbacks=[fts])
    optimizer = fts.tuning_scheduler.optimizer
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
        module, max_epochs=2, callbacks=[FTSCoreCallback(schedule, tuning_scheduler_kwargs={"optimizer": _sgd()})]
    )
    assert any("beyond the core loop's `max_epochs` 2" in str(w.message) for w in recwarn.list)


class Ends:
    """Records the epochs started and whether training ended."""

    def __init__(self) -> None:
        self.epochs: list[int] = []
        self.ended = False

    def on_train_epoch_start(self, ctx):
        self.epochs.append(ctx.epoch)

    def on_train_end(self, ctx):
        self.ended = True


def test_a_tuning_scheduler_stop_ends_the_loop():
    """A transition policy stopping after epoch 0 ends a real loop: one epoch, then `train_end`."""
    from finetuning_scheduler.core.transitions import CallableTransition, Decision

    stop_after_first = CallableTransition(lambda c: (Decision.STOP, "test stop") if c.epoch >= 1 else Decision.CONTINUE)
    schedule = {0: {"params": ["model.bias"]}, 1: {"params": ["model.weight"]}}
    fts = FTSCoreCallback(schedule, tuning_scheduler_kwargs={"optimizer": _sgd(), "transition": stop_after_first})
    ends = Ends()
    core_loop._train(core_loop.TinyModule(), max_epochs=3, callbacks=[fts, ends])
    assert ends.epochs == [0] and ends.ended
    assert fts.tuning_scheduler.should_stop and fts.tuning_scheduler.stop_reason == "test stop"


def test_a_tuning_scheduler_rooted_elsewhere_is_refused():
    from finetuning_scheduler.core.tuning_scheduler import TuningScheduler

    other = core_loop.TinyModule()
    fts = FTSCoreCallback(tuning_scheduler_factory=lambda module: TuningScheduler(other, TWO_PHASES, optimizer=_sgd()))
    with pytest.raises(MisconfigurationException, match="not at the session's module"):
        core_loop._train(core_loop.TinyModule(), max_epochs=1, callbacks=[fts])


def test_a_core_loop_checkpoint_is_refused_on_resume():
    """`checkpoint_state` holds none of the tuning scheduler's state, so loading it into the callback refuses."""
    from interpretune.runners.core import checkpoint_state, load_checkpoint_state

    saved = {}

    class Save:
        def on_train_end(self, ctx):
            saved["ckpt"], saved["ctx"] = checkpoint_state(ctx.module, ctx), ctx

    module = core_loop.TinyModule()
    fts = FTSCoreCallback(TWO_PHASES, tuning_scheduler_kwargs={"optimizer": _sgd()})
    core_loop._train(module, max_epochs=1, callbacks=[fts, Save()])
    assert saved["ckpt"]["callbacks"]["0:FTSCoreCallback"] == {"resume_through_tuning_scheduler": True}
    with pytest.raises(MisconfigurationException, match="resume through the tuning scheduler"):
        load_checkpoint_state(module, saved["ckpt"], saved["ctx"])


def test_an_epoch_interval_lr_scheduler_steps_once_per_epoch():
    """The module's own lr scheduler configs are dropped, so the tuning scheduler's scheduler steps exactly once an
    epoch."""
    module = core_loop.TinyModule()
    module.it_cfg.optimizer_init = {"class_path": "torch.optim.SGD", "init_args": {"lr": 0.1}}
    module.it_cfg.lr_scheduler_init = {"class_path": "torch.optim.lr_scheduler.StepLR", "init_args": {"step_size": 1}}
    module.it_cfg.pl_lrs_cfg = {"interval": "epoch", "frequency": 1}
    schedule = {0: {"params": ["model.bias", "model.weight"]}}
    fts = FTSCoreCallback(schedule)
    core_loop._train(module, max_epochs=3, callbacks=[fts])
    assert fts.tuning_scheduler.lr_scheduler.last_epoch == 3
    assert fts.tuning_scheduler.optimizer.param_groups[0]["lr"] == pytest.approx(0.1 * 0.1**3)


def test_a_factory_excludes_the_schedule_and_kwargs():
    with pytest.raises(MisconfigurationException, match="not both"):
        FTSCoreCallback(TWO_PHASES, tuning_scheduler_factory=lambda module: None)


def test_a_core_session_trains_phase_by_phase(get_it_session__core_cust__initonly):
    """Through a real session and `SessionRunner`: the tuning scheduler is rooted at the module, names come from
    its view, and its optimizer replaces the one the session configured."""
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
    assert fts.tuning_scheduler.view.root is module
    assert torch.equal(snaps.params[0][earlier], initial[earlier])
    assert not torch.equal(snaps.params[0][last], initial[last])
    assert not torch.equal(snaps.params[1][earlier], snaps.params[0][earlier])
    assert fts.tuning_scheduler.curr_depth == 1 and module.optimizers == [fts.tuning_scheduler.optimizer]


_FTS_CALLBACK_PATH = "interpretune.tuning.core_loop.FTSCoreCallback"


def _run_cfg_parser():
    """The run configuration's parse surface, as `ITCLI` builds it."""
    from jsonargparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_class_arguments(
        SessionRunnerCfg, "run_cfg", instantiate=False, sub_configs=True, skip={"it_session", "module", "datamodule"}
    )
    return parser


def test_a_callback_given_as_configuration_is_instantiated():
    """`SessionRunnerCfg` builds a callback given as `class_path`/`init_args`, and leaves objects as they are."""
    snaps = Snapshots()
    entry = {"class_path": _FTS_CALLBACK_PATH, "init_args": {"schedule": TWO_PHASES}}
    run_cfg = SessionRunnerCfg(module=core_loop.TinyModule(), datamodule=object(), callbacks=[entry, snaps])
    fts, kept = run_cfg.callbacks
    assert isinstance(fts, FTSCoreCallback) and fts.schedule == TWO_PHASES and kept is snaps


def test_fts_core_callback_round_trips_through_a_parsed_configuration(tmp_path, get_it_session__core_cust__initonly):
    """A core-loop FTS run built from its parsed configuration alone: the configuration dumps and parses back
    unchanged, and two runs built from it train the same phases under the same tuning scheduler configuration."""
    import yaml

    module = get_it_session__core_cust__initonly.it_session.module
    *_, earlier, last = [n for n, p in module.named_parameters() if p.requires_grad]
    schedule_path = tmp_path / "schedule.yaml"
    schedule_path.write_text(
        yaml.safe_dump({0: {"params": [last], "max_transition_epoch": 1}, 1: {"params": [earlier]}})
    )
    config = {
        "run_cfg": {
            "max_epochs": 2,
            "limit_train_batches": 1,
            "limit_val_batches": 1,
            "callbacks": [
                {
                    "class_path": _FTS_CALLBACK_PATH,
                    "init_args": {"schedule": str(schedule_path), "tuning_scheduler_kwargs": {"max_depth": 1}},
                }
            ],
        }
    }
    parser = _run_cfg_parser()
    parsed = parser.parse_string(yaml.safe_dump(config))
    reparsed = parser.parse_string(parser.dump(parsed))
    assert reparsed == parsed

    results = []
    for ns in (parsed, reparsed):
        session = get_deepcopied_session(get_it_session__core_cust__initonly.it_session)
        run_cfg = SessionRunnerCfg(it_session=session, **ns.run_cfg.as_dict())
        (fts,) = run_cfg.callbacks
        assert isinstance(fts, FTSCoreCallback)
        SessionRunner(run_cfg=run_cfg).train()
        results.append((fts.tuning_scheduler.curr_depth, fts.tuning_scheduler.config_hash))
    assert results[0] == results[1] and results[0][0] == 1
