"""Streamed AnalysisStores: decode parity with a loaded store, refusals, and what a streamed pull reads."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from interpretune.hub.artifacts import IT_ARTIFACT_ENVELOPE, ArtifactEnvelopeError

ROWS = 7


def _features():
    from datasets import Array2D, Features, Sequence, Value

    return Features(
        {
            "input_string": Value("string"),
            "input_tokens": Sequence(Value("int64")),
            "active_features": Array2D(shape=(None, 3), dtype="int64"),
            "logit_diff": Value("float32"),
            "graph_cfg_json": Value("string"),
        }
    )


def _format_kwargs() -> dict:
    from interpretune.analysis.ops.base import ColCfg

    return {
        "col_cfg": {
            "active_features": ColCfg(datasets_dtype="int64", array_shape=(None, 3), sequence_type=False).to_dict(),
            "input_tokens": ColCfg(datasets_dtype="int64").to_dict(),
        }
    }


@pytest.fixture()
def parquet_store(tmp_path) -> Path:
    """A store's Parquet in several row groups, with ragged, 2D, scalar and string columns."""
    from datasets import Dataset

    dataset = Dataset.from_dict(
        {
            "input_string": [f"premise {i}" for i in range(ROWS)],
            "input_tokens": [list(range(i + 1)) for i in range(ROWS)],
            "active_features": [[[i, i + 1, i + 2]] * (i % 3 + 1) for i in range(ROWS)],
            "logit_diff": [i / 2 for i in range(ROWS)],
            "graph_cfg_json": [json.dumps({"k": i}) for i in range(ROWS)],
        },
        features=_features(),
    )
    path = tmp_path / "validation-00000-of-00001.parquet"
    dataset.to_parquet(str(path), batch_size=2)
    return path


def _stores(path: Path):
    from datasets import load_dataset

    from interpretune.analysis import AnalysisStore

    files = {"validation": [str(path)]}
    loaded = AnalysisStore(
        dataset=load_dataset("parquet", data_files=files, split="validation"), it_format_kwargs=_format_kwargs()
    )
    streamed = AnalysisStore(
        dataset=load_dataset("parquet", data_files=files, split="validation", streaming=True),
        it_format_kwargs=_format_kwargs(),
    )
    return loaded, streamed


def _same(a, b) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if not isinstance(a, str) and hasattr(a, "__iter__"):
        a, b = list(a), list(b)
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


class TestStreamedStoreDecodesAsALoadedStore:
    """The stream's rows go through the formatter a Dataset's rows go through, so the two agree value for value."""

    def test_a_stream_is_held_as_stream_with_no_dataset(self, parquet_store):
        _, streamed = _stores(parquet_store)
        assert streamed.stream is not None and streamed.dataset is None and streamed.streaming

    def test_iterated_rows_match(self, parquet_store):
        loaded, streamed = _stores(parquet_store)
        loaded_rows, streamed_rows = list(loaded), list(streamed)
        assert len(streamed_rows) == ROWS
        assert isinstance(streamed_rows[0]["active_features"], torch.Tensor)  # col_cfg reached the stream
        assert _same(loaded_rows, streamed_rows)

    @pytest.mark.parametrize("column", list(_features()))
    def test_each_column_matches(self, parquet_store, column):
        loaded, streamed = _stores(parquet_store)
        assert _same(loaded[column], streamed[column])

    def test_stacked_scalar_columns_match(self, parquet_store):
        loaded, streamed = _stores(parquet_store)
        loaded.stack_batches = streamed.stack_batches = True
        assert _same(loaded["logit_diff"], streamed["logit_diff"])

    def test_several_columns_read_in_one_pass_match(self, parquet_store):
        loaded, streamed = _stores(parquet_store)
        columns = ["input_tokens", "logit_diff", "input_string"]
        assert _same(loaded[columns], streamed[columns])

    def test_select_columns_narrows_the_stream(self, parquet_store):
        _, streamed = _stores(parquet_store)
        streamed.select_columns(["input_string", "logit_diff"])
        assert sorted(next(iter(streamed))) == ["input_string", "logit_diff"]

    def test_stream_attributes_are_reachable_through_the_store(self, parquet_store):
        _, streamed = _stores(parquet_store)
        assert streamed.column_names == list(_features())


class TestStreamedStoreRefusals:
    @pytest.mark.parametrize("key", [0, slice(0, 2)])
    def test_positional_access_is_refused_by_name(self, parquet_store, key):
        _, streamed = _stores(parquet_store)
        with pytest.raises(TypeError, match="no positional access"):
            streamed[key]

    def test_a_stream_without_features_is_refused_by_name(self):
        from datasets import IterableDataset

        from interpretune.analysis import AnalysisStore

        stream = IterableDataset.from_generator(lambda: iter([{"x": 1}]))
        assert stream.features is None
        with pytest.raises(ValueError, match="declares none"):
            AnalysisStore(dataset=stream)

    def test_an_envelope_for_a_streamed_store_is_refused(self, parquet_store):
        from interpretune.hub.artifacts import build_analysis_store_envelope

        _, streamed = _stores(parquet_store)
        with pytest.raises(ArtifactEnvelopeError, match="streamed AnalysisStore"):
            build_analysis_store_envelope(streamed)


class TestStreamedPull:
    """``pull_analysis_store(streaming=True)``: the files the envelope names, at its commit, projected."""

    REPO = "someorg/store"
    COMMIT = "0123456789abcdef0123456789abcdef01234567"

    @pytest.fixture()
    def envelope_snapshot(self, parquet_store, tmp_path, monkeypatch):
        """The envelope in an HF-cache layout, served for any download, so the pull's commit is ``COMMIT``."""
        from datasets import load_dataset

        from interpretune.analysis import AnalysisStore
        from interpretune.hub.artifacts import build_analysis_store_envelope
        from tests.core.test_hub_artifacts import SMALL_STORE_SUBJECT

        store = AnalysisStore(
            dataset=load_dataset("parquet", data_files={"validation": [str(parquet_store)]}, split="validation"),
            it_format_kwargs=_format_kwargs(),
        )
        store.subject = SMALL_STORE_SUBJECT
        envelope = build_analysis_store_envelope(store, interchange={"files": ["data/" + parquet_store.name]})
        snapshot = tmp_path / "cache" / "snapshots" / self.COMMIT
        snapshot.mkdir(parents=True)
        (snapshot / IT_ARTIFACT_ENVELOPE).write_text(json.dumps(envelope), encoding="utf-8")
        monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: str(snapshot / IT_ARTIFACT_ENVELOPE))
        monkeypatch.setattr(
            "huggingface_hub.snapshot_download", lambda *a, **k: pytest.fail("a streamed pull downloads no snapshot")
        )
        return envelope

    def test_streams_the_listed_files_at_the_envelope_commit(
        self, envelope_snapshot, parquet_store, tmp_path, monkeypatch
    ):
        import interpretune.hub.artifacts as artifacts
        from tests.core.test_hub_artifacts import SMALL_STORE_SUBJECT

        calls = []
        real = artifacts._stream_parquet

        def _local(urls, split, **kwargs):
            calls.append((urls, split, kwargs))
            return real([str(parquet_store)], split, **kwargs)

        monkeypatch.setattr(artifacts, "_stream_parquet", _local)
        store = artifacts.pull_analysis_store(self.REPO, streaming=True, columns=["logit_diff"], cache_dir=tmp_path)
        ((urls, split, kwargs),) = calls
        assert urls == [f"hf://datasets/{self.REPO}@{self.COMMIT}/data/{parquet_store.name}"]
        assert split == "validation" and kwargs["columns"] == ["logit_diff"]
        assert kwargs["storage_options"] == {"hf": {"block_size": artifacts.STREAMING_BLOCK_SIZE, "token": None}}
        assert store.subject == SMALL_STORE_SUBJECT
        assert [row["logit_diff"].item() for row in store] == [i / 2 for i in range(ROWS)]

    def test_an_envelope_listing_no_files_is_refused_by_name(self):
        from interpretune.hub.artifacts import _published_parquet_urls

        with pytest.raises(ArtifactEnvelopeError, match="lists no Parquet files"):
            _published_parquet_urls(self.REPO, self.COMMIT, {"artifacts": {"interchange": {"format": "parquet"}}})

    def test_columns_without_streaming_is_refused_before_any_download(self, monkeypatch):
        from interpretune.hub.artifacts import pull_analysis_store

        monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: pytest.fail("no download expected"))
        with pytest.raises(ValueError, match="projects a streamed read"):
            pull_analysis_store(self.REPO, columns=["logit_diff"])


class TestStreamedReadCost:
    """A streamed read stops where it stops: measured on a filesystem that counts range requests the way
    ``HfFileSystemFile`` makes them (an fsspec buffered file whose ``_fetch_range`` is one request)."""

    @pytest.fixture()
    def counting_fs(self):
        import fsspec
        from fsspec.spec import AbstractBufferedFile, AbstractFileSystem

        stats = {"requests": 0, "bytes": 0}

        class _CountingFile(AbstractBufferedFile):
            def _fetch_range(self, start, end):
                with open(self.path, "rb") as handle:
                    handle.seek(start)
                    data = handle.read(end - start)
                stats["requests"] += 1
                stats["bytes"] += len(data)
                return data

        class _CountingFS(AbstractFileSystem):
            protocol = "itcountfs"

            def __init__(self, *args, block_size=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.block_size = block_size

            @classmethod
            def _strip_protocol(cls, path):
                path = str(path)
                path = path[len("itcountfs://") :] if path.startswith("itcountfs://") else path
                return "/" + path.lstrip("/")

            def info(self, path, **kwargs):
                path = self._strip_protocol(path)
                return {"name": path, "size": os.path.getsize(path), "type": "file"}

            def ls(self, path, detail=True, **kwargs):
                entry = self.info(path)
                return [entry] if detail else [entry["name"]]

            def _open(self, path, mode="rb", block_size=None, **kwargs):
                size = block_size or self.block_size or "default"
                return _CountingFile(self, self._strip_protocol(path), mode=mode, block_size=size, **kwargs)

        fsspec.register_implementation("itcountfs", _CountingFS, clobber=True)
        return stats

    @pytest.fixture()
    def grouped_parquet(self, tmp_path) -> Path:
        """About 4 MB of float rows in 16 row groups, beside a small scalar column."""
        import numpy as np
        from datasets import Dataset

        rows = 512
        dataset = Dataset.from_dict(
            {
                "label": list(range(rows)),
                "values": np.random.default_rng(0).standard_normal((rows, 2048)).astype(np.float32),
            }
        )
        path = tmp_path / "grouped.parquet"
        dataset.to_parquet(str(path), batch_size=rows // 16)
        return path

    def _first_row(self, path, stats, **load_kwargs) -> int:
        from datasets import load_dataset

        stats.update(requests=0, bytes=0)
        stream = load_dataset(
            "parquet",
            data_files={"validation": [f"itcountfs://{path}"]},
            split="validation",
            streaming=True,
            **load_kwargs,
        )
        next(iter(stream))
        return stats["bytes"]

    def test_reading_the_first_row_fetches_its_row_group_not_the_file(self, counting_fs, grouped_parquet):
        from interpretune.hub.artifacts import STREAMING_BLOCK_SIZE, _stream_parquet

        size = grouped_parquet.stat().st_size
        counting_fs.update(requests=0, bytes=0)
        stream = _stream_parquet(
            [f"itcountfs://{grouped_parquet}"],
            "validation",
            columns=None,
            storage_options={"itcountfs": {"block_size": STREAMING_BLOCK_SIZE}},
            token=None,
        )
        next(iter(stream))
        assert counting_fs["bytes"] < size / 4, (counting_fs, size)

    def test_control_the_default_scan_reads_the_whole_file_for_one_row(self, counting_fs, grouped_parquet):
        """Without ``pre_buffer=False`` the same read fetches every row group: the bound above is the option's."""
        size = grouped_parquet.stat().st_size
        read = self._first_row(grouped_parquet, counting_fs, storage_options={"itcountfs": {"block_size": 64 * 1024}})
        assert read >= size * 0.9, (read, size)

    def test_a_projected_read_fetches_little_of_the_other_columns(self, counting_fs, grouped_parquet):
        from interpretune.hub.artifacts import STREAMING_BLOCK_SIZE, _stream_parquet

        counting_fs.update(requests=0, bytes=0)
        stream = _stream_parquet(
            [f"itcountfs://{grouped_parquet}"],
            "validation",
            columns=["label"],
            storage_options={"itcountfs": {"block_size": STREAMING_BLOCK_SIZE}},
            token=None,
        )
        assert [row["label"] for row in stream] == list(range(512))
        # each row group costs its small label chunk plus one read-ahead block, about 1 MB of this 6 MB file in all
        assert counting_fs["bytes"] < grouped_parquet.stat().st_size / 4, counting_fs
