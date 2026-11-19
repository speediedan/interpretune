"""The task-neutral ``speediedan/prompt-analysis`` component, published locally from its in-tree source.

Its model-only configurations must carry no prompts and no attribution targets, so circuit-tracer's salient-logit
selection is what a caller gets unless they name targets; its datamodule serves only the prompts it is given.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import interpretune as it
from interpretune.config import ITDataModuleConfig
from interpretune.hub import components

COMPONENT_DIR = Path(__file__).parent.parent.parent / "src" / "it_examples" / "examples" / "prompt_analysis"
REPO_ID = "speediedan/prompt-analysis"
CONFIG_KEYS = sorted(yaml.safe_load((COMPONENT_DIR / "it_component.yaml").read_text())["module"]["configs"])


@pytest.fixture
def prompt_analysis_cache(tmp_path) -> Path:
    cache = tmp_path / "components"
    components.local_publish(
        COMPONENT_DIR, REPO_ID, entrypoint_src=COMPONENT_DIR / "prompt_analysis.py", cache_dir=cache
    )
    return cache


@pytest.mark.parametrize("key", [k for k in CONFIG_KEYS if k.startswith("prompt_analysis.")])
def test_a_model_only_configuration_sets_no_prompts_and_no_targets(prompt_analysis_cache, key):
    dm_cfg, m_cfg, dm_cls, m_cls = it.hub.load(REPO_ID, key, cache_dir=prompt_analysis_cache)
    assert dm_cls.__name__ == "PromptDataModule"
    assert m_cls.__name__ == "CircuitTracerModule"
    assert dm_cfg.prompt_cfg.prompts == []
    assert m_cfg.circuit_tracer_cfg.analysis_target_tokens is None
    assert m_cfg.circuit_tracer_cfg.target_token_ids is None


@pytest.mark.parametrize("key", [k for k in CONFIG_KEYS if k.startswith("capital_states.")])
def test_the_capital_states_variant_names_its_prompt_and_targets(prompt_analysis_cache, key):
    dm_cfg, m_cfg, _, _ = it.hub.load(REPO_ID, key, cache_dir=prompt_analysis_cache)
    (prompt,) = dm_cfg.prompt_cfg.prompts
    assert prompt.lower().endswith("the capital of the state containing dallas is")
    assert m_cfg.circuit_tracer_cfg.analysis_target_tokens == ["\u2581Austin", "\u2581Dallas"]


def test_every_configuration_has_a_model_only_counterpart():
    """An experiment variant adds prompts and targets to a model configuration the component also offers bare."""
    model_only = {k.split(".", 1)[1] for k in CONFIG_KEYS if k.startswith("prompt_analysis.")}
    assert {k.split(".", 1)[1] for k in CONFIG_KEYS} == model_only


def _datamodule(cache: Path, prompts: list[str]):
    """A ``PromptDataModule`` over ``prompts`` with the small gpt2 tokenizer, its classes taken from the load."""
    dm_cfg, _, dm_cls, _ = it.hub.load(REPO_ID, "prompt_analysis.gemma2.circuit_tracer+nnsight", cache_dir=cache)
    itdm_cfg = ITDataModuleConfig(
        model_name_or_path="gpt2",
        task_name="prompt_analysis",
        tokenizer_id_overrides={"pad_token_id": 50256},
        tokenizer_kwargs={"model_input_names": ["input"], "padding_side": "left"},
        prompt_cfg=type(dm_cfg.prompt_cfg)(prompts=prompts),
        eval_batch_size=2,
    )
    return dm_cls(itdm_cfg)


def test_the_datamodule_serves_the_configured_prompts_in_order(prompt_analysis_cache):
    prompts = ["Fact: the capital of the state containing Dallas is", "The Eiffel Tower is in"]
    dm = _datamodule(prompt_analysis_cache, prompts)
    dm.setup()
    batches = list(dm.test_dataloader())
    assert [len(b["input"]) for b in batches] == [2]
    decoded = dm.tokenizer.batch_decode(batches[0]["input"], skip_special_tokens=True)
    assert decoded == prompts


def test_changed_prompts_are_served_on_the_next_setup(prompt_analysis_cache):
    dm = _datamodule(prompt_analysis_cache, ["first prompt"])
    dm.setup()
    dm.itdm_cfg.prompt_cfg.prompts = ["second prompt"]
    dm.setup()
    (batch,) = list(dm.predict_dataloader())
    assert dm.tokenizer.batch_decode(batch["input"], skip_special_tokens=True) == ["second prompt"]


@pytest.mark.parametrize("loader", ["test_dataloader", "predict_dataloader"])
def test_a_model_only_setup_succeeds_and_serving_no_prompts_is_refused_by_name(prompt_analysis_cache, loader):
    dm = _datamodule(prompt_analysis_cache, [])
    dm.setup()
    with pytest.raises(ValueError, match="datamodule_cfg.prompt_cfg.prompts"):
        getattr(dm, loader)()


@pytest.mark.parametrize("loader", ["train_dataloader", "val_dataloader"])
def test_label_splits_are_refused_by_name(prompt_analysis_cache, loader):
    dm = _datamodule(prompt_analysis_cache, ["a prompt"])
    dm.setup()
    with pytest.raises(NotImplementedError, match="carry no labels"):
        getattr(dm, loader)()
