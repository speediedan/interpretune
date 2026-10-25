"""The cached test subsets are keyed by tokenizer as well as task (#794).

On a fresh datasets cache, a gpt2 session once reused a ``pytest_rte_hf`` subset that a gemma or llama session had
tokenized first. The ids beyond gpt2's vocabulary faulted the device in the first embedding lookup, a sticky fault
that took 48 later GPU tests with it on both vendors.
"""

from __future__ import annotations

from types import SimpleNamespace

from tests.modules import BaseTestDataModule


class _Tokenizer:
    def __init__(self, name_or_path: str, vocab: int) -> None:
        self.name_or_path = name_or_path
        self._vocab = vocab

    def __len__(self) -> int:
        return self._vocab


def _datamodule(task_name: str, dataset_path, tokenizer):
    dm = SimpleNamespace(itdm_cfg=SimpleNamespace(task_name=task_name, dataset_path=dataset_path), tokenizer=tokenizer)
    BaseTestDataModule._key_default_dataset_path_by_tokenizer(dm)  # type: ignore[arg-type]
    return dm.itdm_cfg.dataset_path


def test_tokenizers_sharing_a_task_get_separate_subsets(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path))
    default = tmp_path / "pytest_rte_hf"
    gpt2 = _datamodule("pytest_rte_hf", default, _Tokenizer("gpt2", 50257))
    gemma = _datamodule("pytest_rte_hf", default, _Tokenizer("google/gemma-2-2b-it", 256000))
    assert gpt2 != gemma
    assert {gpt2.parent, gemma.parent} == {tmp_path.resolve()}
    assert gpt2.name.startswith("pytest_rte_hf__") and gemma.name.startswith("pytest_rte_hf__")
    # the same tokenizer always finds its own subset again
    assert _datamodule("pytest_rte_hf", default, _Tokenizer("gpt2", 50257)) == gpt2


def test_an_already_keyed_path_is_rekeyed_for_the_session_tokenizer(tmp_path, monkeypatch):
    # a config object reused by a session with another tokenizer must not keep the first session's key
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path))
    gpt2 = _datamodule("pytest_rte_hf", tmp_path / "pytest_rte_hf", _Tokenizer("gpt2", 50257))
    assert _datamodule("pytest_rte_hf", gpt2, _Tokenizer("meta-llama/Llama-3.2-3B-Instruct", 128256)) != gpt2


def test_an_explicit_dataset_path_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "cache"))
    explicit = tmp_path / "force_prepare_tests_ds"
    assert _datamodule("pytest_rte_hf", explicit, _Tokenizer("gpt2", 50257)) == explicit
