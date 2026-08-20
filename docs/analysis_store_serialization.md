# AnalysisStore Serialization

Interpretune persists analysis outputs through Hugging Face Datasets rather than a bespoke storage layer. The pipeline is intentionally split into three stages:

1. An analysis operation declares an `OpSchema` made of `ColCfg` objects.
2. `schema_to_features()` converts that schema into Hugging Face `Features`.
3. The dataset is materialized with `with_format("interpretune")`, which routes reads through `ITAnalysisFormatter`.

This document describes the current flow, how bundled and hub ops contribute schema metadata, and the structured graph representation used for circuit-tracer outputs.

## Current Pipeline

### Schema declaration

Each bundled analysis op, declared in its op-family YAML under [src/interpretune/analysis/ops/bundled/](../src/interpretune/analysis/ops/bundled/) (e.g. `concept/concept_ops.yaml`), declares `input_schema`, `output_schema`, `aliases`, and optional `required_capabilities` entries. Those YAML entries become `ColCfg` instances in [src/interpretune/analysis/ops/base.py](../src/interpretune/analysis/ops/base.py).

Hub ops follow the same shape after download and load through [src/interpretune/hub/manager.py](../src/interpretune/hub/manager.py) and [src/interpretune/analysis/ops/dispatcher.py](../src/interpretune/analysis/ops/dispatcher.py). Once loaded, bundled and hub ops are normalized into the same `AnalysisOp` representation, so serialization is driven by resolved schema rather than by the source registry.

Aliases and composite ops do not create separate storage rules. Aliases resolve to the same concrete op definition, and composite ops serialize only the union of the concrete stage schemas produced by compilation. The formatter therefore sees final compiled column metadata, not whether a column originated in a bundled op, a hub op, or a composite expansion.

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

This stage is where Interpretune commits to an Arrow-native layout. If a field is not representable here, the formatter cannot rescue it later. That applies equally to bundled ops and downloaded hub ops.

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

## Publishing to the Hub: the `it_artifact.json` envelope

An AnalysisStore published to the Hub travels as a dataset repo: parquet files (the interchange
format; the local store stays Arrow) plus an `it_artifact.json` envelope at the repo root. The
envelope is **descriptive, not authoritative** — the dataset files are the truth, and the envelope
restates identity and provenance so a reader can decide whether this is the artifact they want, and
so the interpretune formatter can be re-attached from the serialized `col_cfg` without re-running
the pipeline.

Two blocks with opposite rules:

- `identity` is written once at first publish and preserved verbatim on every re-push. A store keeps
  its `store_id` across renames, re-pushes and regenerations.
- `provenance` (including `content_fingerprint`) is refreshed freely, so consumers get change
  detection without identity churn.

### Schema versioning policy

Hub artifacts outlive the code that wrote them, and the publisher is usually not the reader. That
makes the envelope the one interpretune surface where the alpha's otherwise-liberal
breaking-change posture does not apply.

**`schema` is a single integer major version.** There is no minor component, deliberately: readers
ignore unknown keys by contract, so shipping an additive optional field needs no version change at
all, and the only thing a version bump has left to communicate is "a reader that does not
understand this cannot be correct". A `major.minor` scheme would encode that same distinction in a
second number that nothing would consult. (`it_component.yaml` uses the same rule under the key
`it_schema_version`.)

The rules:

| Change | Schema bump |
| --- | --- |
| Add an optional field readers may ignore | none |
| Add a field a correct reader must understand | major |
| Remove, rename, or redefine an existing field | major |
| Change what an existing value means | major |

**Readers accept a window, not a single version.** `ARTIFACT_SCHEMA_VERSION` is what this build
writes; `ARTIFACT_SCHEMA_MIN_READABLE` is the oldest it will read. Both live in
`interpretune.hub.artifacts`, and the three failure cases are distinguished because they call for
three different actions:

| Envelope | Reader behavior |
| --- | --- |
| within `[MIN_READABLE, VERSION]` | read it; unknown keys ignored |
| newer than `VERSION` | refuse, naming the upgrade — an older reader cannot safely guess |
| older than `MIN_READABLE` | refuse, pointing at a re-publish |
| `schema` missing or not an integer | refuse: this is not an interpretune artifact envelope |

**Raising the floor retires published artifacts**, so it is a deliberate, documented act rather
than a side effect of a refactor. The floor is currently `1`, the schema is currently `1`, and
`tests/core/fixtures/it_artifact_schema1.json` pins a literal v1 envelope that the test suite reads
on every run — the piece that actually catches an accidental mandatory-field change, since every
other test builds its envelope with the current writer and so drifts along with it.

**Publishing refuses to emit an envelope this build could not read back.** `build_analysis_store_envelope`
returns its result through the same validator the read paths use, so a schema bumped past the
reader's window, or an identity block preserved from a corrupted existing envelope, fails locally
instead of landing on the Hub.

Neuronpedia dashboard corpora carry their own manifests (`dashboards.json` for provenance,
`source_ids.json` for identity), documented in the dashboard pipeline guide. They follow the same
single-integer convention but are read leniently today; if and when they adopt this envelope they
take a distinct `artifact_kind` under the rules above.

### Op-collection provenance: what the envelope does and does not record

The envelope records the interpretune version, a content fingerprint, the analysis backend, and the
`ColCfg` needed to re-attach the formatter. It does **not** record which op collections produced the
store.

That distinction matters once a store's columns come from a hub op collection rather than from the
bundled set. The bundled case is fully covered by `interpretune_version`, because a bundled op's
contract is pinned by the package that shipped it. A hub op's is not: its collection carries its own
version and resolves at a specific commit, so `interpretune_version` alone does not identify the
contract that produced the columns. Reproducing such a store means knowing the collection and its
revision, and today that has to come from outside the envelope.

`provenance` is caller-supplied and merged over the defaults, so a producer can record it now without
any schema change — additive keys are a minor bump under the policy above, and readers ignore keys
they do not know:

```python
import interpretune as it

active = it.hub.op_info("concept_direction").active
it.hub.push_analysis_store(
    store,
    "me/my-analysis-store",
    provenance={
        "op_collections": [
            # `source` is the provenance field: "bundled", "local", or "hub:<user>.<repo>". Do not derive
            # a namespace by splitting the op name -- a bundled op has no namespace to split off.
            {"source": active.source,
             "collection": active.collection,
             "version": active.version,
             "revision": active.revision}
        ]
    },
)
```

**Interpretune now records this for you**, so the explicit form above is an override rather than the
only route. When an `AnalysisCfg` builds or adopts an output store it stamps that store with what its
op resolved to *at write time*, and the envelope reads the stamp:

- one entry per contributing definition, so a composition that mixes a bundled op with a pulled one
  reports both rather than picking one;
- the op name **as written**, bare or fully qualified, alongside the name it resolved to;
- `source`, plus collection name, version and cached revision where the source has them.

A caller-supplied `provenance` still wins, which is what a store loaded from disk needs — such a store
carries no stamp of its own, because nothing observed it being written.

**Provenance is recorded at write time and never reconstructed at push time**, and the distinction is
not fussiness. Precedence is session-mutable (`prefer_ops` / `IT_OP_PRECEDENCE`, the latter re-read on
every access), so a bare name can resolve to a different collection by the time a store is published.
Reconstruction would be silently right or silently wrong on the same inputs, and the store keeps no
record of which naming form a column came from to tell the two apart.

Where there is nothing to record — a store assembled outside the op path, or an op with no registered
definition — the key is **omitted entirely**. Absence reads as absence rather than defaulting to
`bundled`.

### What a recorded entry does and does not identify

`source` is a **category**, not a locator, so how far an entry gets you depends on which category it is:

| `source` | What the entry identifies | Can a reader resolve it? |
| --- | --- | --- |
| `bundled` | the op shipped in the wheel; its contract is pinned by `interpretune_version` in the same envelope | yes, from the version alone |
| `hub:<user.repo>` | collection name, declared version, and the cached revision it was fetched at | yes, it is fetchable |
| `local` | the collection *name* as declared, and nothing more | **no** |

The `local` row is the honest limit. A local collection has no revision to record and its name is not
globally unique, so two unrelated collections may both record `local` / `my_ops`. Entries do stay
distinguishable *from each other* by collection name, but a `local` entry is a **label, not an address**:
it tells a reader that the columns did not come from the wheel or from a fetchable collection, which is
useful, and does not tell them where to find it, which no amount of recording could.
