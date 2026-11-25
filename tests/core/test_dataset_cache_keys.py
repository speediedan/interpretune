"""One session's prepared dataset is not another's: the default directory is keyed, and a save replaces whole
files.

Keyed by task alone, the directory was shared by every model, tokenizer and prompt configuration of the task. On a fresh
datasets cache a gpt2 session once reused a ``pytest_rte_hf`` subset that a gemma or llama session had tokenized first;
the ids beyond gpt2's vocabulary faulted the device (#794). And a session that prepared its data while another trained
rewrote the Arrow files the second had memory-mapped, whose labels turned to garbage (#1220).
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import datasets
import pytest

from interpretune.base.datamodules import ITDataModule
from interpretune.config.datamodule import ChatTemplatePromptConfig, ITDataModuleConfig, PromptConfig
from tests.modules import BaseTestDataModule


class _Tokenizer:
    def __init__(self, name_or_path: str = "gpt2", vocab: int = 50257) -> None:
        self.name_or_path = name_or_path
        self._vocab = vocab

    def __len__(self) -> int:
        return self._vocab


GEMMA = ("google/gemma-2-2b-it", 256000)


@pytest.fixture
def cache_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "cache"
    monkeypatch.setenv("HF_DATASETS_CACHE", str(root))
    return root.resolve()


def _keyed(tokenizer=None, *, extra=None, **cfg) -> Path:
    itdm_cfg = ITDataModuleConfig(**{"task_name": "rte", "model_name_or_path": "gpt2"} | cfg)
    itdm_cfg.key_default_dataset_path(tokenizer or _Tokenizer(), **(extra or {}))
    return Path(itdm_cfg.dataset_path)


class TestDefaultDatasetPathKey:
    def test_the_key_stays_under_the_task_and_is_stable(self, cache_root):
        path = _keyed()
        assert path.parent == cache_root and path.name.startswith("rte__")
        assert _keyed() == path

    @pytest.mark.parametrize(
        "changed",
        [
            pytest.param({"model_name_or_path": "gpt2-medium"}, id="model"),
            pytest.param({"prompt_cfg": ChatTemplatePromptConfig()}, id="prompt_class"),
            pytest.param({"prompt_cfg": PromptConfig(cust_task_prompt={"context": "x"})}, id="prompt_field"),
            pytest.param({"max_seq_length": 128}, id="max_seq_length"),
            pytest.param({"text_fields": ("premise",)}, id="text_fields"),
            pytest.param({"prepare_data_map_cfg": {"batched": True}}, id="map_cfg"),
        ],
    )
    def test_what_changes_the_rows_changes_the_path(self, cache_root, changed):
        assert _keyed(**changed) != _keyed()

    @pytest.mark.parametrize(
        "unchanged",
        [
            pytest.param({"train_batch_size": 1, "eval_batch_size": 1}, id="batch_size"),
            pytest.param({"dataloader_kwargs": {"num_workers": 2}}, id="dataloader"),
            pytest.param({"signature_columns": ["input_ids"]}, id="signature_columns"),
        ],
    )
    def test_what_does_not_change_the_rows_keeps_the_path(self, cache_root, unchanged):
        assert _keyed(**unchanged) == _keyed()

    def test_the_tokenizer_and_a_datamodules_own_inputs_are_keyed(self, cache_root):
        assert len({_keyed(), _keyed(_Tokenizer(*GEMMA)), _keyed(extra={"entrypoint": "abc"})}) == 3

    def test_two_datamodules_of_one_task_get_separate_directories(self, cache_root, monkeypatch):
        class Other(ITDataModule):
            def prepare_data(self, target_model=None) -> None:
                """Writes other rows for the same task."""

        monkeypatch.setattr(ITDataModule, "configure_tokenizer", lambda self: _Tokenizer())
        monkeypatch.setattr("interpretune.base.datamodules._import_class", lambda _: lambda *a, **k: None)
        paths = []
        for cls in (ITDataModule, Other):
            itdm_cfg = ITDataModuleConfig(task_name="rte", model_name_or_path="gpt2")
            cls(itdm_cfg)
            paths.append(itdm_cfg.dataset_path)
        assert paths[0] != paths[1]

    def test_a_subclass_field_is_keyed(self, cache_root):
        @dataclass(kw_only=True)
        class SampledCfg(ITDataModuleConfig):
            seed: int = 0

        paths = set()
        for seed in (0, 1):
            itdm_cfg = SampledCfg(task_name="rte", model_name_or_path="gpt2", seed=seed)
            itdm_cfg.key_default_dataset_path(_Tokenizer())
            paths.add(itdm_cfg.dataset_path)
        assert len(paths) == 2

    def test_a_reused_config_is_rekeyed_for_the_session_tokenizer(self, cache_root):
        itdm_cfg = ITDataModuleConfig(task_name="rte", model_name_or_path="gpt2")
        paths = []
        for tokenizer in (_Tokenizer(), _Tokenizer(*GEMMA), _Tokenizer()):
            itdm_cfg.key_default_dataset_path(tokenizer)
            paths.append(itdm_cfg.dataset_path)
        assert paths[0] == paths[2] != paths[1]

    def test_an_explicit_path_is_left_alone(self, cache_root, tmp_path):
        explicit = (tmp_path / "explicit").resolve()
        assert _keyed(dataset_path=explicit) == explicit
        # set on the configuration after it was built, as a caller that owns a run directory does
        itdm_cfg = ITDataModuleConfig(task_name="rte", model_name_or_path="gpt2")
        itdm_cfg.dataset_path = explicit
        itdm_cfg.key_default_dataset_path(_Tokenizer())
        assert itdm_cfg.dataset_path == explicit
        # the task's own directory, named explicitly, is explicit too
        assert _keyed(dataset_path=cache_root / "rte") == cache_root / "rte"

    def test_ranks_of_one_run_agree(self, cache_root):
        """A path derived from an object's ``repr`` or from hash order would differ between processes."""
        script = (
            "from interpretune.config.datamodule import ChatTemplatePromptConfig, ITDataModuleConfig\n"
            "class T:\n    name_or_path = 'gpt2'\n    def __len__(self): return 50257\n"
            "cfg = ITDataModuleConfig(task_name='rte', model_name_or_path='gpt2', text_fields=('a', 'b'),\n"
            "    prompt_cfg=ChatTemplatePromptConfig(),\n"
            "    tokenizer_kwargs={'s': {'x', 'y', 'z'}, 'o': object(), 'f': len})\n"
            "cfg.key_default_dataset_path(T())\n"
            "print(cfg.dataset_path)\n"
        )
        runs = [
            subprocess.run([sys.executable, "-c", script], capture_output=True, encoding="utf-8", check=True)
            for _ in range(2)
        ]
        first, second = (run.stdout.strip().splitlines()[-1] for run in runs)
        assert first == second and Path(first).parent == cache_root

    def test_the_test_subset_is_keyed_by_its_entrypoint(self):
        """The suite's datamodules reuse a prepared subset, so the entrypoint that encodes it is part of the
        key."""
        assert set(BaseTestDataModule.prepared_dataset_key_extra(SimpleNamespace())) == {"entrypoint"}  # type: ignore[arg-type]


def _splits(label: int, rows: int = 2000) -> datasets.DatasetDict:
    split = datasets.Dataset.from_dict({"labels": [label] * rows, "input_ids": [[label] * 16] * rows})
    return datasets.DatasetDict({"train": split, "validation": split})


class _DataModule(ITDataModule):
    def __init__(self, dataset_path: Path) -> None:
        self.itdm_cfg = SimpleNamespace(dataset_path=dataset_path)  # type: ignore[assignment]
        self._module = None


class TestSavePreparedDataset:
    def test_round_trip(self, tmp_path):
        dm = _DataModule(tmp_path / "a" / "dataset")
        dm.save_prepared_dataset(_splits(3))
        dm.setup()
        assert {name: set(split["labels"]) for name, split in dm.dataset.items()} == {"train": {3}, "validation": {3}}
        assert [p.name for p in (tmp_path / "a").iterdir()] == ["dataset"]  # nothing staged is left behind

    @pytest.mark.skipif(sys.platform == "win32", reason="Windows refuses to replace a mapped file")
    def test_a_reader_keeps_the_rows_it_opened(self, tmp_path):
        dm = _DataModule(tmp_path / "dataset")
        dm.save_prepared_dataset(_splits(1))
        dm.setup()
        dm.save_prepared_dataset(_splits(0, rows=10))
        assert set(dm.dataset["train"]["labels"]) == {1} and len(dm.dataset["train"]) == 2000
        fresh = _DataModule(tmp_path / "dataset")
        fresh.setup()
        assert set(fresh.dataset["train"]["labels"]) == {0}

    def test_shards_of_an_earlier_save_are_removed(self, tmp_path):
        target = tmp_path / "dataset"
        dm = _DataModule(target)
        _splits(1).save_to_disk(target, num_shards={"train": 4, "validation": 1})
        dm.save_prepared_dataset(_splits(2))
        assert len(list((target / "train").glob("*.arrow"))) == 1
        dm.setup()
        assert set(dm.dataset["train"]["labels"]) == {2}

    def test_a_blocked_replace_keeps_the_existing_dataset(self, tmp_path, monkeypatch):
        dm = _DataModule(tmp_path / "dataset")
        dm.save_prepared_dataset(_splits(1))

        def blocked(*_):
            raise PermissionError("mapped")

        monkeypatch.setattr("interpretune.base.datamodules.os.replace", blocked)
        with pytest.warns(UserWarning, match="memory-mapped"):
            dm.save_prepared_dataset(_splits(2))
        monkeypatch.undo()
        dm.setup()
        assert set(dm.dataset["train"]["labels"]) == {1}
