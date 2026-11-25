"""The measures the example tutorials share."""

from __future__ import annotations

import huggingface_hub
import pytest
import torch

from interpretune.harness.measures import answer_margin, cached_revision


def test_answer_margin_reads_the_last_position():
    logits = torch.zeros(2, 3, 5)
    logits[:, -1, 1], logits[:, -1, 4] = torch.tensor([3.0, -1.0]), torch.tensor([1.0, 2.0])
    logits[:, 0, 1] = 100.0  # an earlier position is not read
    torch.testing.assert_close(answer_margin(logits, (1, 4)), torch.tensor([2.0, -3.0]))


def test_cached_revision_reads_the_snapshot_commit(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshots" / "c1899de2" / "config.json"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda repo, name: str(snapshot))
    assert cached_revision("org/model") == "c1899de2"


def test_cached_revision_refuses_a_model_the_cache_does_not_hold(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda repo, name: None)
    with pytest.raises(ValueError, match="org/model is not in the local Hugging Face cache"):
        cached_revision("org/model")
