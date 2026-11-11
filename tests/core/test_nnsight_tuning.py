"""Tuning the local nnsight cases nothing else exercises (#942): the ``(lightning, nnsight)`` composition under the
``FinetuningScheduler`` callback, and a ``lora_cfg`` on an nnsight composition, which is refused by name rather
than silently ignored."""

from __future__ import annotations

import pytest
import torch

from tests.runif import RunIf

PHASE_0 = r"hf_model\.transformer\.ln_f\..*"
PHASE_1 = r"hf_model\.transformer\.h\.11\..*"


def _names(module, pattern: str) -> set[str]:
    import re

    return {n for n, _ in module.named_parameters() if re.fullmatch(pattern, n)}


@RunIf(lightning=True)
class TestLightningNNsightPhasedRun:
    def test_each_phase_changes_only_what_it_thaws(self, get_it_session__l_ns_gpt2__setup, tmp_path):
        from lightning.pytorch import Callback, Trainer
        from finetuning_scheduler import FinetuningScheduler

        session = get_it_session__l_ns_gpt2__setup.it_session
        module = session.module
        snapshots: list[dict[str, torch.Tensor]] = []

        class Snapshot(Callback):
            def on_train_epoch_end(self, trainer, pl_module):
                snapshots.append({n: p.detach().clone() for n, p in pl_module.named_parameters()})

        before = {n: p.detach().clone() for n, p in module.named_parameters()}
        schedule = {0: {"params": [PHASE_0], "max_transition_epoch": 1}, 1: {"params": [PHASE_1]}}
        trainer = Trainer(
            default_root_dir=tmp_path,
            accelerator="cpu",
            devices=1,
            max_epochs=2,
            limit_train_batches=2,
            limit_val_batches=1,
            num_sanity_val_steps=0,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[FinetuningScheduler(ft_schedule=schedule, max_depth=1), Snapshot()],
        )
        trainer.fit(module, datamodule=session.datamodule)
        assert len(snapshots) == 2
        phase_0, phase_1 = _names(module, PHASE_0), _names(module, PHASE_1)
        assert phase_0 and phase_1

        def changed(a, b):
            return {n for n in a if not torch.equal(a[n], b[n])}

        first, second = changed(before, snapshots[0]), changed(snapshots[0], snapshots[1])
        assert first and first <= phase_0, sorted(first - phase_0)[:5]
        assert second & phase_1 and second <= phase_0 | phase_1, sorted(second - phase_0 - phase_1)[:5]


@pytest.mark.parametrize(
    "hf_settings, named",
    [
        pytest.param({"lora_cfg": {"r": 8, "target_modules": ["c_attn"]}}, "`lora_cfg`", id="lora"),
        pytest.param({"bitsandbytesconfig": {"load_in_4bit": True}}, "`bitsandbytesconfig`", id="bitsandbytes"),
    ],
)
def test_an_hf_setting_nnsight_never_applies_is_refused_by_name(hf_settings, named):
    """The configuration would otherwise say adapters (or quantization) while the run trains the base unchanged."""
    from interpretune.adapters.nnsight.config import ITNNsightConfig, NNsightConfig
    from interpretune.config.mixins import HFFromPretrainedConfig
    from interpretune.utils import MisconfigurationException

    with pytest.raises(MisconfigurationException, match=f"sets {named}, which an nnsight composition does not apply"):
        ITNNsightConfig(
            model_name_or_path="openai-community/gpt2",
            nnsight_cfg=NNsightConfig(model_name="openai-community/gpt2", device_map="cpu", dispatch=True),
            hf_from_pretrained_cfg=HFFromPretrainedConfig(**hf_settings),
        )


def test_an_hf_config_without_those_settings_still_builds():
    """``pretrained_kwargs`` and a model head are what the bundled nnsight configs carry, and they stay
    accepted."""
    from interpretune.adapters.nnsight.config import ITNNsightConfig, NNsightConfig
    from interpretune.config.mixins import HFFromPretrainedConfig

    cfg = ITNNsightConfig(
        model_name_or_path="openai-community/gpt2",
        nnsight_cfg=NNsightConfig(model_name="openai-community/gpt2", device_map="cpu", dispatch=True),
        hf_from_pretrained_cfg=HFFromPretrainedConfig(
            pretrained_kwargs={"dtype": "float32"}, model_head="transformers.GPT2LMHeadModel"
        ),
    )
    assert not cfg.hf_from_pretrained_cfg.lora_cfg
