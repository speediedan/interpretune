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


class TestBatchValue:
    """``AnalysisStore.batch_value`` gives one row's value of one column, decoding nothing else."""

    @pytest.mark.parametrize("stack_batches", [False, True])
    @pytest.mark.parametrize("column", list(_features()))
    def test_it_is_the_rows_value(self, parquet_store, column, stack_batches):
        loaded, streamed = _stores(parquet_store)
        loaded.stack_batches = streamed.stack_batches = stack_batches
        for i in (0, 1, 1, 4, ROWS - 1, 2):  # repeated, skipping ahead, and an earlier row
            row = loaded.dataset[i][column]
            assert _same(row, loaded.batch_value(column, i)) and _same(row, streamed.batch_value(column, i))

    def test_a_column_or_row_the_store_lacks(self, parquet_store):
        for store in _stores(parquet_store):
            assert store.batch_value("absent", 0) is None
            with pytest.raises(IndexError):
                store.batch_value("logit_diff", ROWS)
            assert store.batch_value("logit_diff", ROWS - 1) is not None  # and the store still reads

    def test_a_dataset_decodes_no_other_column(self, parquet_store, monkeypatch):
        from interpretune.analysis.formatters import ITAnalysisFormatter

        loaded, _ = _stores(parquet_store)
        decoded, format_row = [], ITAnalysisFormatter.format_row

        def recorded(self, pa_table):
            decoded.append(pa_table.column_names)
            return format_row(self, pa_table)

        monkeypatch.setattr(ITAnalysisFormatter, "format_row", recorded)
        loaded.batch_value("logit_diff", 3)
        loaded.dataset[3]
        assert decoded == [["logit_diff"], list(_features())]  # the second is the control: a row decodes them all

    def test_batches_read_in_order_cost_one_pass_over_a_stream(self, parquet_store, monkeypatch):
        from datasets import IterableDataset

        _, streamed = _stores(parquet_store)
        passes, iterate = [], IterableDataset.__iter__

        def recorded(self):
            passes.append(self.column_names)
            return iterate(self)

        monkeypatch.setattr(IterableDataset, "__iter__", recorded)
        for i in range(ROWS):
            for _ in range(3):
                streamed.batch_value("input_tokens", i)
        assert passes == [["input_tokens"]]
        streamed.batch_value("input_tokens", 0)
        assert len(passes) == 2

    def test_each_read_is_its_own_value(self, parquet_store):
        for store in _stores(parquet_store):
            store.batch_value("input_tokens", 2).fill_(-1)
            assert store.batch_value("input_tokens", 2).tolist() == [0, 1, 2]

    def test_replaced_or_reformatted_rows_are_read_anew(self, parquet_store):
        loaded, streamed = _stores(parquet_store)
        for store in (loaded, streamed):
            store.batch_value("input_string", 0)
        loaded.dataset = loaded.dataset.select([5])
        streamed._attach_stream(streamed.stream.skip(5))
        for store in (loaded, streamed):
            assert store.batch_value("input_string", 0) == "premise 5"
        assert isinstance(loaded.batch_value("input_tokens", 0), torch.Tensor)
        loaded.dataset.set_format(type="python")  # in place
        assert loaded.batch_value("input_tokens", 0) == list(range(6))

    @pytest.mark.parametrize("duplicate", ["copy", "deepcopy", "pickle"])
    def test_a_copy_starts_without_the_readers(self, parquet_store, duplicate):
        import copy
        import pickle

        duplicates = {
            "copy": copy.copy,
            "deepcopy": copy.deepcopy,
            "pickle": lambda store: pickle.loads(pickle.dumps(store)),
        }
        for store in _stores(parquet_store):
            store.batch_value("input_string", 3)
            other = duplicates[duplicate](store)
            assert not other._column_views and not other._stream_cursors
            assert other.batch_value("input_string", 1) == "premise 1"
            assert store._column_views or store._stream_cursors

    def test_a_bound_batch_reads_its_fields_through_it(self, parquet_store, monkeypatch):
        from types import SimpleNamespace

        from interpretune.analysis import AnalysisBatch, AnalysisStore

        loaded, streamed = _stores(parquet_store)
        for store in (loaded, streamed):
            reads: list = []
            batch_value = AnalysisStore.batch_value
            with monkeypatch.context() as patched:
                patched.setattr(
                    AnalysisStore,
                    "batch_value",
                    lambda self, column, i: reads.append((column, i)) or batch_value(self, column, i),
                )
                patched.setattr(AnalysisStore, "__getitem__", lambda self, key: pytest.fail(f"read {key!r} whole"))
                module = SimpleNamespace(analysis_cfg=SimpleNamespace(input_store=store))
                batch = AnalysisBatch().bind_resolution_context(module, batch_idx=4)
                assert hasattr(batch, "input_tokens") and batch.input_tokens.tolist() == [0, 1, 2, 3, 4]
                assert batch.get("input_string") == "premise 4"
            assert set(reads) == {("input_tokens", 4), ("input_string", 4)}


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

        checked = []
        monkeypatch.setattr(artifacts, "_stream_parquet", _local)
        monkeypatch.setattr(artifacts, "_warn_on_streamed_read_cost", lambda *a: checked.append(a))
        store = artifacts.pull_analysis_store(self.REPO, streaming=True, columns=["logit_diff"], cache_dir=tmp_path)
        ((urls, split, kwargs),) = calls
        # the cost check reads the same files, with the same projection, before the stream is built
        ((source, files, checked_urls, columns, options),) = checked
        assert source == f"{self.REPO}@{self.COMMIT}" and files == [f"data/{parquet_store.name}"]
        assert checked_urls == urls and columns == ["logit_diff"] and options == kwargs["storage_options"]
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
                # a POSIX path can arrive with its leading slash stripped; a Windows one starts with its drive
                return path if os.name == "nt" else "/" + path.lstrip("/")

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
            data_files={"validation": [f"itcountfs://{path.as_posix()}"]},
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
            [f"itcountfs://{grouped_parquet.as_posix()}"],
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
            [f"itcountfs://{grouped_parquet.as_posix()}"],
            "validation",
            columns=["label"],
            storage_options={"itcountfs": {"block_size": STREAMING_BLOCK_SIZE}},
            token=None,
        )
        assert [row["label"] for row in stream] == list(range(512))
        # each row group costs its small label chunk plus one read-ahead block, about 1 MB of this 6 MB file in all
        assert counting_fs["bytes"] < grouped_parquet.stat().st_size / 4, counting_fs

    def test_the_footer_estimate_costs_one_request_of_one_block(self, counting_fs, grouped_parquet):
        from interpretune.hub.artifacts import STREAMING_BLOCK_SIZE, _streamed_first_row_bytes

        counting_fs.update(requests=0, bytes=0)
        _, groups = _streamed_first_row_bytes(f"itcountfs://{grouped_parquet.as_posix()}", None, {"itcountfs": {}})
        assert groups == 16
        assert counting_fs["requests"] == 1 and counting_fs["bytes"] <= STREAMING_BLOCK_SIZE

    @pytest.mark.parametrize("columns", [None, ["values"], ["label"]])
    def test_the_estimate_tracks_what_the_first_row_fetches(self, counting_fs, grouped_parquet, columns):
        """Within a factor of two of the measured fetch, for the full read and both projections."""
        from interpretune.hub.artifacts import STREAMING_BLOCK_SIZE, _stream_parquet, _streamed_first_row_bytes

        url = f"itcountfs://{grouped_parquet.as_posix()}"
        options = {"itcountfs": {"block_size": STREAMING_BLOCK_SIZE}}
        estimate, _ = _streamed_first_row_bytes(url, columns, options)
        counting_fs.update(requests=0, bytes=0)
        next(iter(_stream_parquet([url], "validation", columns=columns, storage_options=options, token=None)))
        measured = counting_fs["bytes"]
        # small projections are dominated by the footer and block rounding, so they get an absolute allowance too
        assert measured / 2 - 4 * STREAMING_BLOCK_SIZE <= estimate <= 2 * measured + 4 * STREAMING_BLOCK_SIZE

    def test_one_row_group_costs_the_whole_file(self, counting_fs, tmp_path):
        """A store pushed before the row-group target: one group, so its first row is its whole file."""
        import numpy as np
        import pyarrow.parquet as pq
        from datasets import Dataset

        from interpretune.hub.artifacts import _streamed_first_row_bytes

        path = tmp_path / "one_group.parquet"
        values = np.random.default_rng(0).standard_normal((256, 1024)).astype(np.float32)
        Dataset.from_dict({"values": values}).to_parquet(str(path), batch_size=256)
        estimate, groups = _streamed_first_row_bytes(f"itcountfs://{path.as_posix()}", None, {"itcountfs": {}})
        assert groups == 1
        group = pq.ParquetFile(path).metadata.row_group(0)
        assert estimate == sum(group.column(c).total_compressed_size for c in range(group.num_columns))
        assert estimate > 0.9 * path.stat().st_size  # the one group is nearly all of the file

    def test_a_costly_first_row_warns_with_the_remedies(self, counting_fs, grouped_parquet, monkeypatch):
        import interpretune.hub.artifacts as artifacts

        url = f"itcountfs://{grouped_parquet.as_posix()}"
        values, _ = artifacts._streamed_first_row_bytes(url, ["values"], {"itcountfs": {}})
        label, _ = artifacts._streamed_first_row_bytes(url, ["label"], {"itcountfs": {}})
        assert label < values
        monkeypatch.setattr(artifacts, "STREAMED_FIRST_ROW_WARN_BYTES", (label + values) // 2)
        with pytest.warns(UserWarning, match=r"some/store@abc.*columns=.*re-publish"):
            artifacts._warn_on_streamed_read_cost("some/store@abc", ["g.parquet"], [url], ["values"], {"itcountfs": {}})
        # the control: the projection under the threshold does not warn
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            artifacts._warn_on_streamed_read_cost("some/store@abc", ["g.parquet"], [url], ["label"], {"itcountfs": {}})

    def test_a_listed_file_missing_at_the_commit_is_refused(self, counting_fs, tmp_path):
        from interpretune.hub.artifacts import _warn_on_streamed_read_cost

        missing = tmp_path / "absent.parquet"
        with pytest.raises(ArtifactEnvelopeError, match=r"some/store@abc: the envelope lists 'data/absent.parquet'"):
            _warn_on_streamed_read_cost(
                "some/store@abc",
                ["data/absent.parquet"],
                [f"itcountfs://{missing.as_posix()}"],
                None,
                {"itcountfs": {}},
            )
