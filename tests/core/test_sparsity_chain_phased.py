"""The sparsity chain (#698 to #700) as one phased run on the core loop, CPU-sized (#899): roles, the boundary's
order (restore best, end hooks, gates, start hooks), a gate's veto, run-fitted state in checkpoints and on resume,
a promotion, and the training streams untouched by hooks that draw random numbers."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("venngram.tuning", reason="venngram, which holds the tunable view protocol, is not installed")

from finetuning_scheduler.core.checkpoint import load_payload  # noqa: E402
from venngram.tuning.views import CanonicalView  # noqa: E402

from interpretune.runners.core import core_train_loop  # noqa: E402
from interpretune.tuning.core_loop import FTSCoreCallback  # noqa: E402
from tests.core import sparsity_chain as sc  # noqa: E402

MAX_EPOCHS = 5


class BoundaryProbe:
    """A gate (run after the chain's own) recording, at each boundary, whether the module holds the best
    checkpoint's weights and the graphs the end hook built; it always accepts."""

    seen: list = []

    def __call__(self, event):
        module = event.tuning_scheduler.model
        best = load_payload(event.best_path)["model"]
        state = module.state_dict()
        restored = all(torch.equal(state[k], v) for k, v in best.items() if not k.startswith("model.enforce_"))
        BoundaryProbe.seen.append(
            (
                event.depth,
                restored,
                {k: (m.clone(), keep.clone()) for k, (m, keep) in getattr(module, "graphs", {}).items()},
            )
        )
        return True


def _kwargs(tmp_path, veto_depths=(), probe=False, hooks=True, **extra):
    kwargs = {
        "restore_best": True,
        "max_depth": 1,
        "checkpoint": {
            "class_path": "finetuning_scheduler.core.checkpoint.BestCheckpoint",
            "init_args": {"dirpath": str(tmp_path), "monitor": "val_ce", "save_initial": True},
        },
        **extra,
    }
    if hooks:
        kwargs.update(sc.hook_config(veto_depths))
        if probe:
            kwargs["gates"] = kwargs["gates"] + [{"class_path": f"{__name__}.BoundaryProbe"}]
    return kwargs


def _module(**module_kwargs) -> sc.ChainModule:
    module = sc.ChainModule(**module_kwargs)
    object.__setattr__(module, "tunable_view", CanonicalView(module, roles=sc.roles(module)))
    return module


def _run(tmp_path, max_epochs=MAX_EPOCHS, promote=False, module_kwargs=None, extra_hooks=(), **kwargs):
    module = _module(**(module_kwargs or {}))
    kwargs = _kwargs(tmp_path, **kwargs)
    for when, entry in extra_hooks:
        kwargs["phase_hooks"][when] = kwargs["phase_hooks"][when] + [entry]
    fts = FTSCoreCallback(sc.chain_schedule(promote), tuning_scheduler_kwargs=kwargs)
    core_train_loop(module, module.datamodule(), -1, -1, max_epochs, callbacks=[fts])
    return module, fts.tuning_scheduler


def test_the_boundary_restores_best_then_builds_graphs_gates_and_fills_the_fitted_buffers(tmp_path):
    BoundaryProbe.seen = []
    module, ts = _run(tmp_path, probe=True)
    assert module.events[:6] == [
        ("calibrate", 0),
        ("payload", 0),
        ("graphs", 0),
        ("gate", 0, 0),
        ("fill", 1),
        ("payload", 1),
    ]
    assert module.events[6][:2] == ("gate", 1) and len(module.events) == 7  # the final gate
    assert ts.curr_depth == 1 and not ts.should_stop
    # at the boundary the module held the best checkpoint's weights when the end hook built its graphs
    depth, restored, graphs = BoundaryProbe.seen[0]
    assert depth == 0 and restored and set(graphs) == set(sc.REG_LAYERS)
    # and phase 1's enforcement is exactly those graphs
    for layer, (mean, keep) in graphs.items():
        assert torch.equal(getattr(module.model, f"enforce_mean_{layer}"), mean)
        assert torch.equal(getattr(module.model, f"enforce_keep_{layer}"), keep) and int(keep.sum()) == 4
    assert module.model.enforce and module.reg_weight == pytest.approx(0.3)
    assert [r["accept"] for r in ts.gate_records if r["gate"].endswith("FeatureShareGate")] == [True, True]
    assert all(0.0 <= r["record"]["share"] <= 1.0 for r in ts.gate_records if r["record"])


def test_roles_keep_instruments_frozen_and_out_of_checkpoints_while_fitted_state_is_saved(tmp_path):
    module, ts = _run(tmp_path)
    tcs = {n: p for n, p in module.named_parameters() if n.startswith("model.tcs.")}
    assert tcs and not any(p.requires_grad for p in tcs.values())
    payload = load_payload(ts._checkpoint.best_path)
    assert not any(k.startswith("model.tcs.") for k in payload["model"])
    assert {f"model.enforce_{kind}_{layer}" for kind in ("mean", "keep") for layer in sc.REG_LAYERS} <= set(
        payload["model"]
    )
    assert payload["statefuls"]["normalizer"] == {"value": module.normalizer.value} and module.normalizer.value > 0


def test_a_vetoed_recovery_phase_restores_the_sparsity_phase_best(tmp_path):
    BoundaryProbe.seen = []
    module, ts = _run(tmp_path, veto_depths=(1,), probe=True)
    assert ts.should_stop and "veto: phase 1" in ts.stop_reason
    boundary = load_payload(ts._checkpoint.boundary_path(1))["model"]
    state = module.state_dict()
    assert all(torch.equal(state[k], v) for k, v in boundary.items())


def test_a_promoted_transcoder_trains_in_the_recovery_phase(tmp_path):
    module, ts = _run(tmp_path, promote=True)
    w_enc = dict(module.named_parameters())[f"model.tcs.{sc.REG_LAYERS[-1]}.W_enc"]
    assert ts.curr_depth == 1 and w_enc.requires_grad
    assert any(p is w_enc for g in ts.optimizer.param_groups for p in g["params"])
    others = [p for n, p in module.named_parameters() if n.startswith("model.tcs.") and p is not w_enc]
    assert not any(p.requires_grad for p in others)


def test_the_normalizer_survives_resume(tmp_path):
    module, ts = _run(tmp_path, max_epochs=3)
    saved, fitted = ts.resume_state(rng=False), module.normalizer.value
    resumed = _module(seed=1)  # a different untuned model calibrates a different normalizer
    fts = FTSCoreCallback(sc.chain_schedule(), tuning_scheduler_kwargs=_kwargs(tmp_path / "resumed"))
    fts.tuning_scheduler = fts.build_tuning_scheduler(resumed)
    fts.tuning_scheduler.setup()  # runs phase 0's start hooks: the calibration
    assert resumed.normalizer.value != fitted
    fts.tuning_scheduler.load_resume_state(saved)
    assert resumed.normalizer.value == fitted and fts.tuning_scheduler.curr_depth == ts.curr_depth


@pytest.mark.parametrize("isolate", [True, False], ids=["isolated", "control"])
def test_hooks_drawing_random_numbers_leave_training_unchanged(tmp_path, isolate):
    """With dropout drawing from the training stream, a hook that draws random numbers changes nothing when the
    hooks' streams are isolated (the default), and changes the run when they are not (the positive control)."""
    noise = {"hook": {"class_path": f"{sc.__name__}.DrawNoise"}, "isolate_rng": isolate}
    torch.manual_seed(0)
    base, _ = _run(tmp_path / "base", module_kwargs={"dropout": 0.2})
    torch.manual_seed(0)
    noisy, _ = _run(tmp_path / "noisy", module_kwargs={"dropout": 0.2}, extra_hooks=[("end", noise), ("start", noise)])
    same = all(torch.equal(p, q) for p, q in zip(base.parameters(), noisy.parameters()))
    assert same is isolate
