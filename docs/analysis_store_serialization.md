# AnalysisStore Serialization

Interpretune persists analysis outputs through Hugging Face Datasets rather than a bespoke storage layer. The pipeline is intentionally split into three stages:

1. An analysis operation declares an `OpSchema` made of `ColCfg` objects.
2. `schema_to_features()` converts that schema into Hugging Face `Features`.
3. The dataset is materialized with `with_format("interpretune")`, which routes reads through `ITAnalysisFormatter`.

This document describes the current flow, how native and hub ops contribute schema metadata, and the structured graph representation used for circuit-tracer outputs.

## Current Pipeline

### Schema declaration

Each native analysis op in [src/interpretune/analysis/ops/native_analysis_functions.yaml](../src/interpretune/analysis/ops/native_analysis_functions.yaml) declares `input_schema`, `output_schema`, `aliases`, and optional `required_capabilities` entries. Those YAML entries become `ColCfg` instances in [src/interpretune/analysis/ops/base.py](../src/interpretune/analysis/ops/base.py).

Hub ops follow the same shape after download and load through [src/interpretune/analysis/ops/hub_manager.py](../src/interpretune/analysis/ops/hub_manager.py) and [src/interpretune/analysis/ops/dispatcher.py](../src/interpretune/analysis/ops/dispatcher.py). Once loaded, native and hub ops are normalized into the same `AnalysisOp` representation, so serialization is driven by resolved schema rather than by the source registry.

Aliases and composite ops do not create separate storage rules. Aliases resolve to the same concrete op definition, and composite ops serialize only the union of the concrete stage schemas produced by compilation. The formatter therefore sees final compiled column metadata, not whether a column originated in a native op, a hub op, or a composite expansion.

`ColCfg` controls the Arrow representation and formatter behavior:

- `datasets_dtype` selects the primitive element dtype.
- `array_shape` maps to `Array2D` or `Array3D` when the rank is fixed.
- `sequence_type` maps to `Sequence(Value(...))` for variable-length vectors.
- `non_tensor` marks fields that should remain Python values instead of being tensorized by the formatter.
- `per_latent` and `per_latent_model_hook` describe nested latent-model dictionaries.
- formatter kwargs can optionally provide an analysis backend that hydrates richer objects from the primitive row data after Arrow decoding.

`OpSchema` is just a validated `dict[str, ColCfg]` wrapper, so the schema remains data-driven and serializable.

### Feature generation

[schema_to_features()](../src/interpretune/analysis/core.py#L227) converts an `OpSchema` into Hugging Face `Features`.

The important behaviors are:

- Scalar columns become `Value(dtype=...)`.
- Variable-length vectors become `Sequence(Value(dtype=...))`.
- Rank-2 and rank-3 tensors become `Array2D` and `Array3D`.
- Per-latent and per-hook fields become nested dictionaries of those same primitives.

This stage is where Interpretune commits to an Arrow-native layout. If a field is not representable here, the formatter cannot rescue it later. That applies equally to native ops and downloaded hub ops.

### Dataset generation

[dataset_features_and_format()](../src/interpretune/runners/analysis.py#L74) derives both:

- the `Features` object passed into `Dataset.from_generator(...)`, and
- the serialized `col_cfg` payload passed into `with_format("interpretune", col_cfg=...)`.

[generate_analysis_dataset()](../src/interpretune/runners/analysis.py#L116) then builds the dataset from `analysis_store_generator(...)` and immediately applies the custom format.

The stored dataset is still plain Hugging Face data on disk. The custom format only affects how rows, columns, and batches are decoded when read back.

### Custom formatter

[ITAnalysisFormatter](../src/interpretune/analysis/formatters.py) subclasses `TorchFormatter`.

Its role is deliberately narrow:

- preserve `non_tensor` values as Python objects,
- rebuild per-latent nested dictionaries,
- apply dynamic-dimension permutations when `dyn_dim` is configured,
- optionally hand the decoded row or batch to an analysis backend for post-format hydration,
- otherwise defer tensor creation to the standard Hugging Face torch formatter.

This means the formatter is not a serialization escape hatch. It does not deserialize opaque Python objects from opaque blobs. It only tensorizes data that Arrow already knows how to decode, and any richer object hydration has to start from those primitive decoded fields.

### AnalysisStore access

[AnalysisStore](../src/interpretune/analysis/core.py#L348) wraps a dataset and always reads it back through the `interpretune` formatter.

Important access patterns:

- `store["column"]` returns the formatted column values.
- `store[idx]` returns a formatted row.
- protocol-declared fields are also reachable through attribute access such as `store.logit_diffs`.

`AnalysisStore` therefore depends on two things remaining aligned:

- the on-disk Arrow schema, and
- the protocol annotations that decide which fields should be exposed as attributes.

## Structured Graph Representation

Circuit-tracer graphs should be stored as ordinary dataset columns rather than serialized Python blobs. The graph payload is naturally decomposable into Arrow-native primitives:

- `input_string`: `Value("string")`
- `input_tokens`: `Sequence(Value("int64"))`
- `active_features`: `Array2D(shape=(None, 3), dtype="int64")`
- `adjacency_matrix`: `Array2D(shape=(None, None), dtype="float32")`
- `selected_features`: `Sequence(Value("int64"))`
- `activation_values`: `Sequence(Value("float32"))`
- `logit_target_ids`: `Sequence(Value("int64"))`
- `logit_target_tokens`: `Sequence(Value("string"))`
- `logit_probabilities`: `Sequence(Value("float32"))`
- `graph_cfg_json`: `Value("string")`
- `graph_scan_json`: `Value("string")`
- `graph_vocab_size`: `Value("int64")`

This layout keeps the graph:

- portable across Arrow save/load boundaries,
- inspectable without `torch.load`,
- compatible with the existing formatter,
- hydratable back into a circuit-tracer `Graph`.

## Why the Blob Format Was Replaced

The previous implementation stored the full graph as `graph_pt_bytes: Value("binary")` containing a `torch.save(...)` payload. That approach had three problems:

1. The stored value was opaque to Arrow and to any tool other than Python + PyTorch.
2. It bypassed the schema-driven formatter model instead of using it.
3. Persisted binary payloads proved fragile during dataset round-trips.

The structured representation avoids those issues by keeping the persisted state entirely within normal dataset feature types.

## Hydration Boundary

Interpretune should treat graph reconstruction as a protocol boundary instead of hard-coding blob deserialization in op definitions. A graph-like consumer only needs a primitive payload containing tensors, strings, lists, and scalar config metadata. Circuit-tracer can then rebuild `Graph` from that payload, and other integrations can do the same for their own graph types.

The current implementation pushes that package-specific logic into an analysis backend rather than into `definitions.py`. The formatter can optionally call that backend after row decoding, which keeps the persisted dataset primitive while making hydrated graph access more seamless at read time.
