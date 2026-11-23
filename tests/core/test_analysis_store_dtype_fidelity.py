"""AnalysisStore dtype and shape fidelity: write, Parquet interchange, then pull eager and streamed.

Every case writes through ``analysis_store_from_batches`` (the op schema's own writer), publishes the store's Parquet
the way ``push_analysis_store`` does, and reads it back three ways: the Arrow store in memory, ``pull_analysis_store``
eager and ``pull_analysis_store(streaming=True)``. The table these tests pin is in
``docs/analysis_store_serialization.md``; a change to any row fails here.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from interpretune.analysis.store_dtypes import STORE_TORCH_DTYPES, StoreDtypeError

COMMIT = "c" * 40
REPO = "someorg/dtype-store"

#: Column shapes a schema can declare: a scalar, a sequence, a 2-D array with a variable first dimension and a ragged
#: 3-D array (two variable dimensions, which a schema stores as nested sequences).
SHAPES = {
    "scalar": dict(sequence_type=False),
    "sequence": dict(),
    "array2d": dict(array_shape=(None, 3), sequence_type=False),
    "ragged3d": dict(array_shape=(None, None, 2), sequence_type=False),
}


def _row_shape(kind: str, row: int) -> tuple[int, ...]:
    return {"scalar": (), "sequence": (4 + row,), "array2d": (2 + row, 3), "ragged3d": (2, 1 + row, 2)}[kind]


def _values(dtype: torch.dtype, shape: tuple[int, ...], row: int) -> torch.Tensor:
    """Values that use the dtype's full precision and range, so any narrowing on the way changes some of them."""
    g = torch.Generator().manual_seed(row)
    if dtype == torch.bool:
        return torch.rand(shape, generator=g) > 0.5
    if dtype.is_floating_point:
        return (torch.randn(shape, generator=g, dtype=torch.float64) * 3 + 0.123456789012345).to(dtype)
    info = torch.iinfo(dtype)
    vals = torch.randint(max(info.min, -(2**40)), min(info.max, 2**40), shape, generator=g, dtype=torch.int64)
    if vals.numel():
        vals.view(-1)[0] = info.max  # the extreme shows any overflow
    return vals.to(dtype)


def _module():
    """The dimensions a schema reads from a module, and nothing else."""
    tokenizer = SimpleNamespace(vocab_size=32, model_max_length=32)
    return SimpleNamespace(
        datamodule=SimpleNamespace(itdm_cfg=SimpleNamespace(eval_batch_size=1)),
        it_cfg=SimpleNamespace(),
        model=SimpleNamespace(tokenizer=tokenizer),
    )


def _op(declared: str, kind: str, **extra):
    from interpretune.analysis.ops.base import AnalysisOp, ColCfg, OpSchema

    return AnalysisOp(
        name="dtype_probe",
        description="one tensor column",
        output_schema=OpSchema({"x": ColCfg(datasets_dtype=declared, **{**SHAPES[kind], **extra})}),
    )


def _write(op, tensors):
    from interpretune.analysis import AnalysisBatch, analysis_store_from_batches

    batches = []
    for t in tensors:
        batch = AnalysisBatch()
        batch.x = t
        batches.append(batch)
    return analysis_store_from_batches(_module(), batches, op=op)


def _publish(store, tmp_path, monkeypatch):
    """Write the store's Parquet as ``push_analysis_store`` does and serve it, with its envelope, to the pull."""
    from interpretune.hub import artifacts
    from interpretune.hub.subject import StoreSubject

    store.subject = StoreSubject(model_id="someorg/tiny")
    data = tmp_path / "snapshots" / COMMIT / "data"
    data.mkdir(parents=True)
    parquet = data / "validation-00000-of-00001.parquet"
    with artifacts._row_group_target(artifacts.ANALYSIS_STORE_ROW_GROUP_SIZE):
        store.dataset.to_parquet(str(parquet))
    envelope = artifacts.build_analysis_store_envelope(store, interchange={"files": ["data/" + parquet.name]})
    env_path = data.parent / artifacts.IT_ARTIFACT_ENVELOPE
    env_path.write_text(json.dumps(envelope), encoding="utf-8")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: str(env_path))
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: str(data))
    real = artifacts._stream_parquet
    monkeypatch.setattr(artifacts, "_stream_parquet", lambda urls, split, **kw: real([str(parquet)], split, **kw))
    # The streamed-read cost guard reads each listed file's Parquet footer from the Hub, which this fake does not
    # serve; the guard has its own tests in test_hub_artifact_streaming.py, and these test what the read returns.
    monkeypatch.setattr(artifacts, "_warn_on_streamed_read_cost", lambda *a: None)


def _read_all(store, tmp_path, monkeypatch) -> dict[str, list[torch.Tensor]]:
    from interpretune.hub.artifacts import pull_analysis_store

    _publish(store, tmp_path, monkeypatch)
    eager = pull_analysis_store(REPO, cache_dir=tmp_path)
    streamed = pull_analysis_store(REPO, cache_dir=tmp_path, streaming=True)
    return {
        "arrow": [store.dataset[i]["x"] for i in range(len(store.dataset))],
        "eager": [eager.dataset[i]["x"] for i in range(len(eager.dataset))],
        "streamed": [row["x"] for row in streamed],
    }


def _assert_faithful(written, reads, dtype):
    for reader, got in reads.items():
        assert len(got) == len(written), reader
        for src, back in zip(written, got):
            assert isinstance(back, torch.Tensor), reader
            assert back.dtype == dtype, f"{reader}: read {back.dtype}, declared {dtype}"
            assert back.shape == src.shape, f"{reader}: read shape {tuple(back.shape)}, wrote {tuple(src.shape)}"
            assert torch.equal(back, src.to(dtype)), f"{reader}: values changed"


@pytest.mark.parametrize("kind", list(SHAPES))
@pytest.mark.parametrize("declared", list(STORE_TORCH_DTYPES))
def test_each_declared_dtype_round_trips_unchanged(declared, kind, tmp_path, monkeypatch):
    """A tensor of the declared dtype reads back with that dtype, its shape and its values, from every reader."""
    dtype = STORE_TORCH_DTYPES[declared]
    written = [_values(dtype, _row_shape(kind, r), r) for r in range(2)]
    store = _write(_op(declared, kind), written)
    _assert_faithful(written, _read_all(store, tmp_path, monkeypatch), dtype)


@pytest.mark.parametrize("kind", list(SHAPES))
def test_bfloat16_is_stored_exactly_as_float32(kind, tmp_path, monkeypatch):
    """Arrow has no bfloat16 type; every bfloat16 value is a float32 value, so a float32 column holds it
    exactly."""
    written = [_values(torch.bfloat16, _row_shape(kind, r), r) for r in range(2)]
    store = _write(_op("float32", kind), written)
    _assert_faithful(written, _read_all(store, tmp_path, monkeypatch), torch.float32)


@pytest.mark.parametrize(
    "source, declared",
    [
        (torch.float16, "float32"),
        (torch.int8, "float32"),
        (torch.int16, "int64"),
        (torch.int32, "float64"),
        (torch.uint8, "int16"),
        (torch.bool, "int64"),
    ],
)
def test_a_widening_that_keeps_every_value_is_accepted(source, declared, tmp_path, monkeypatch):
    """A tensor whose dtype the declared one contains is stored; the reader gets the declared dtype."""
    written = [_values(source, _row_shape("sequence", r), r) for r in range(2)]
    store = _write(_op(declared, "sequence"), written)
    _assert_faithful(written, _read_all(store, tmp_path, monkeypatch), STORE_TORCH_DTYPES[declared])


def test_a_narrowing_whose_values_survive_is_accepted(tmp_path, monkeypatch):
    """The check is on values: float64 values that are float32 values, and small integers, are stored exactly."""
    written = [torch.tensor([0.5, -2.25, 1024.0], dtype=torch.float64), torch.tensor([3.0, 1e-3], dtype=torch.float64)]
    store = _write(_op("float32", "sequence"), [w.float().double() for w in written])
    _assert_faithful([w.float().double() for w in written], _read_all(store, tmp_path, monkeypatch), torch.float32)
    ints = [torch.tensor([1, -7, 2**24], dtype=torch.int64)]
    _assert_faithful(ints, {"arrow": [_write(_op("float32", "sequence"), ints).dataset[0]["x"]]}, torch.float32)


@pytest.mark.parametrize("kind", list(SHAPES))
@pytest.mark.parametrize(
    "source, declared, shown",
    [
        (torch.float32, "int64", "float32 tensor whose values int64 cannot hold"),
        (torch.int32, "float32", "int32 tensor whose values float32 cannot hold"),
        (torch.int64, "int32", "int64 tensor whose values int32 cannot hold"),
        (torch.int16, "bool", "int16 tensor whose values bool cannot hold"),
        (torch.float32, "bool", "float32 tensor whose values bool cannot hold"),
    ],
)
def test_a_value_the_column_cannot_hold_is_refused_by_name(source, declared, shown, kind):
    """Truncation, integer precision loss, overflow and booleanizing are refused at write time, naming the column.

    Before this check the same writes stored changed values on some shapes and raised a generic Arrow error on others.
    """
    written = [_values(source, _row_shape(kind, 1), 1)]
    with pytest.raises(StoreDtypeError, match=rf"Column 'x' is declared {declared} and was given a {shown}"):
        _write(_op(declared, kind), written)


@pytest.mark.parametrize("source, declared", [(torch.float64, "float32"), (torch.float32, "float16")])
def test_a_float_rounded_to_the_declared_precision_is_stored_with_one_warning(source, declared, tmp_path, monkeypatch):
    """A schema's float dtype is the precision its op stores: a float64 model's values are rounded to it, with a
    warning.

    Bundled ops declare float32 columns, and a model run in float64 produces float64 values for them, so refusing the
    rounding would refuse every float64 run of those ops. The warning names the column once per dtype pair.
    """
    from interpretune.analysis import store_dtypes

    monkeypatch.setattr(store_dtypes, "_ROUNDING_WARNED", set())
    written = [_values(source, _row_shape("sequence", r), r) for r in range(2)]
    with pytest.warns(UserWarning, match=rf"Column 'x' is declared {declared} .* stored rounded") as record:
        store = _write(_op(declared, "sequence"), written)
    assert sum("stored rounded" in str(w.message) for w in record) == 1
    target = STORE_TORCH_DTYPES[declared]
    _assert_faithful([w.to(target) for w in written], _read_all(store, tmp_path, monkeypatch), target)


def test_a_float_that_overflows_the_declared_precision_is_refused():
    """Rounding keeps a value finite; a finite value that would become infinite is a changed value, refused."""
    with pytest.raises(StoreDtypeError, match="float32 tensor whose values float16 cannot hold .*-> inf"):
        _write(_op("float16", "sequence"), [torch.tensor([1.5, 1e5])])


def test_bfloat16_beyond_float16_range_is_refused():
    """Bfloat16's range exceeds float16's: in range the values survive (float16 has the finer mantissa), past it
    not."""
    in_range = torch.tensor([1.5, -3.25, 6e4], dtype=torch.bfloat16)
    assert torch.equal(_write(_op("float16", "sequence"), [in_range]).dataset[0]["x"], in_range.to(torch.float16))
    with pytest.raises(StoreDtypeError, match="bfloat16 tensor whose values float16 cannot hold .*-> inf"):
        _write(_op("float16", "sequence"), [torch.tensor([1.5, 1e5], dtype=torch.bfloat16)])


def test_a_complex_tensor_is_refused_by_name():
    with pytest.raises(StoreDtypeError, match="Column 'x' .* no complex type"):
        _write(_op("float32", "sequence"), [torch.tensor([1 + 2j])])


def test_refusal_checks_tensors_inside_a_per_latent_column():
    """A per-latent column holds a tensor per latent; each is checked against the declared dtype."""
    op = _op("int64", "sequence", per_latent=True)
    from interpretune.analysis import AnalysisBatch
    from interpretune.analysis.ops.base import AnalysisOp

    batch = AnalysisBatch()
    batch.x = {"hook": {0: torch.tensor([2]), 3: torch.tensor([0.5])}}
    with pytest.raises(StoreDtypeError, match="Column 'x' is declared int64 and was given a float32 tensor"):
        AnalysisOp.process_batch(analysis_batch=batch, batch={}, output_schema=op.output_schema)


@pytest.mark.parametrize("declared", ["bfloat16", "torch.bfloat16", "bf16"])
def test_a_declared_dtype_arrow_has_no_type_for_is_refused_by_name(declared):
    from interpretune.analysis.core import schema_to_features

    with pytest.raises(StoreDtypeError, match=r"Column 'x' declares dtype .*bf.*Declare float32"):
        schema_to_features(_module(), schema=_op(declared, "sequence").output_schema)


@pytest.mark.parametrize(
    "handle_dtype, expected",
    [("bfloat16", "float32"), ("torch.bfloat16", "float32"), ("float16", "float16"), (None, "float32")],
)
def test_a_latent_model_column_takes_a_storable_dtype(handle_dtype, expected):
    """A latent model loaded in bfloat16 writes bfloat16 activations; its columns are float32, which holds them."""
    from interpretune.analysis.store_dtypes import latent_model_column_dtype

    assert latent_model_column_dtype(handle_dtype) == expected


def test_a_store_written_from_batches_carries_its_column_configs(tmp_path, monkeypatch):
    """The column configs travel with the store, so a dynamic-dimension swap is undone on every read.

    ``analysis_store_from_batches`` once dropped them: the store's own reads and a pulled copy then returned the
    swapped layout, and the pushed envelope recorded no ``col_cfg`` at all.
    """
    # as an op writes a (batch, positions) tensor: stored with dims 0 and 1 swapped, so the schema declares (None, 3)
    written = [torch.arange(6, dtype=torch.float32).reshape(3, 2), torch.arange(12, dtype=torch.float32).reshape(3, 4)]
    store = _write(_op("float32", "array2d", dyn_dim=1), written)
    assert store.it_format_kwargs["col_cfg"]["x"]["dyn_dim"] == 1
    _assert_faithful(written, _read_all(store, tmp_path, monkeypatch), torch.float32)
