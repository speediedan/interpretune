"""The RTE phased fine-tuning example (#765) runs the same schedule on the core loop and under Lightning, with the
same outcome.

``it_examples.tuning.rte_phased_ft`` drives ``rte_ft.gpt2.core`` on CPU on both hosts: ``FTSCoreCallback`` with a
``PlateauTransition`` and a ``BestCheckpoint``, and the ``FinetuningScheduler`` callback with ``FTSEarlyStopping`` and
``FTSCheckpoint``. Both must thaw the same parameters each epoch, transition in the same epochs, and keep the same best
checkpoint. The run is long enough for one transition by ``max_transition_epoch`` and one by a validation-loss plateau.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "finetuning_scheduler.core.tuning_scheduler", reason="finetuning-scheduler predates TuningScheduler"
)
pytest.importorskip("lightning")

from it_examples.tuning import rte_phased_ft  # noqa: E402


def test_the_core_loop_and_lightning_agree_phase_by_phase(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    # a large `min_delta` makes the second phase plateau within a CPU-sized run
    kwargs = dict(max_epochs=6, limit_train_batches=2, limit_val_batches=2, min_delta=2.0, device="cpu")
    core = rte_phased_ft.run(
        "core", "gpt2", dataset_dir=tmp_path / "core_ds", ckpt_dir=str(tmp_path / "core"), **kwargs
    )
    lightning = rte_phased_ft.run(
        "lightning", "gpt2", dataset_dir=tmp_path / "l_ds", ckpt_dir=str(tmp_path / "lightning"), **kwargs
    )
    # phase 0 ends at its `max_transition_epoch` (2); phase 1 on a validation-loss plateau
    assert core["transition_epochs"] == lightning["transition_epochs"] == [2, 5]
    assert core["final_depth"] == lightning["final_depth"] == 2
    assert core["best_depth"] == lightning["best_depth"] == 2
    assert [e["thawed"] for e in core["epochs"]] == [e["thawed"] for e in lightning["epochs"]]
    assert [len(e["thawed"]) for e in core["epochs"]] == [26, 26, 74, 74, 74, 148]
    assert [e["val_loss"] for e in core["epochs"]] == pytest.approx(
        [e["val_loss"] for e in lightning["epochs"]], rel=1e-5
    )
    assert core["best_score"] == pytest.approx(lightning["best_score"], rel=1e-5)
