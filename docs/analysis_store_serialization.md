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
- `site` declares what the column's values name in the model they were computed on: a layer index, hook-name keys, token ids or positions, latent indices, or values in an activation space. It is published with `col_cfg`, and `AnalysisStore.resolve_on` reads it to decide whether the column carries over to another model (see [Reading a store against another model](finding_analysis_stores.md#reading-a-store-against-another-model)).
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

### Dtype fidelity

A column's dtype is the one its schema declares (`array_dtype`, else `datasets_dtype`). What a reader gets back is
that dtype, with the values and shape that were written, from all three readers: the store in memory (Arrow),
`pull_analysis_store` (Parquet, eager) and `pull_analysis_store(streaming=True)` (Parquet, streamed). The table is
measured over scalar, sequence, 2-D (`Array2D`) and ragged 3-D (nested sequence) columns, and
`tests/core/test_analysis_store_dtype_fidelity.py` fails if any row changes.

| Declared dtype | Parquet holds | A reader gets |
| --- | --- | --- |
| `bool` | `boolean` | `torch.bool` |
| `int8`, `int16`, `int32`, `int64` | the same integer width | `torch.int8`, `torch.int16`, `torch.int32`, `torch.int64` |
| `uint8` | `uint8` | `torch.uint8` |
| `float16` | `halffloat` | `torch.float16` |
| `float32` | `float` | `torch.float32` |
| `float64` | `double` | `torch.float64` |
| `bfloat16` | refused when the schema is built, naming the column: Arrow has no bfloat16 type | |

A **bfloat16 tensor** is written to a `float32` column, which holds every bfloat16 value exactly; it reads back as
`float32`. A latent-model column (`per_latent_model_hook`) takes its dtype from the latent model's config, and a
latent model loaded in bfloat16 gets `float32` columns for the same reason.

**Writing a tensor of another dtype** into a column is accepted when casting it to the column's dtype and back
reproduces every value. A float rounded to a narrower declared float is also accepted, with a warning naming the
column once per dtype pair: a schema's float dtype is the precision its op stores, and a model run in float64
produces float64 values for the bundled ops' float32 columns. Anything else is refused with `StoreDtypeError`, which
names the column, both dtypes and the first value the cast would change. The check runs in `AnalysisOp.process_batch`, which both writers go through (the
runner's generator and `analysis_store_from_batches`). Measured examples, where "before" is what a store held without
the check:

| Written into the column | Before | Now |
| --- | --- | --- |
| bfloat16, float16, int8, int16, uint8 or bool into `float32` | values exact | accepted |
| float64 into `float32`, float32 into `float16` | values rounded, silently | rounded, with a warning naming the column |
| a float past the declared float's range (1e5 into `float16`) | stored as `inf`, silently | refused |
| float into `int64` or `int32` | fractions truncated, silently (a generic Arrow error for a 2-D column from the runner) | refused, unless every value is integral |
| int32 or int64 into `float32` | values past $2^{24}$ rounded, silently, in a scalar column; a generic Arrow error in other shapes | refused, unless every value survives |
| int64 into `int32` | a generic Arrow overflow error | refused, naming the column |
| anything but 0 and 1 into `bool` | every non-zero value stored as `True`, silently | refused |

Two reads narrowed silently before this table was measured, and no longer do: a `float64` column read back as
`float32` (rounding every value, from every reader), and `float16`, `int8`, `int16`, `int32` and `uint8` columns read
back as `float32` or `int64`. A store written by `analysis_store_from_batches` also carried no column configs, so a
`dyn_dim` column read back with its dimensions still swapped and its pushed envelope recorded no `col_cfg`; it now
carries them, as the runner's stores do.

### Text columns

`datasets` encodes whatever a string column is given, so a value of the wrong kind is stored as text that looks like
data. `AnalysisOp.process_batch` refuses those with `StoreDtypeError`, naming the column and what to declare. `None`
is a null at any level.

| Written into the column | Before | Now |
| --- | --- | --- |
| a list, dict or tensor into a string column | its Python repr | refused |
| a number or bool into a string column | its digits or `"True"` | refused |
| a string into a list-of-strings column (`sequence_type: true`) | the list of its characters | refused |
| lists nested deeper than the column declares | the inner lists' reprs | refused |

To store structured values, declare their shape or encode them:

- **Nested lists** of strings or numbers: `datasets_dtype` of the element, `non_tensor: true`, `sequence_type: false`
  and one `array_shape` entry per level (`[null, null]` for a list of lists, whose rows may differ in length).
- **Anything else:** `json.dumps(value)` into a string column named `<column>_json`, which an op reads back with
  `load_json_field`.

A column declared `datasets_dtype: object` compiles to a plain string column, so a stored `object` column holds strings
only. `object` is for `intermediate_only` columns, which are never stored.

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
so the Interpretune formatter can be re-attached from the serialized `col_cfg` without re-running
the pipeline.

Two blocks with opposite rules:

- `identity` is written once at first publish and preserved verbatim on every re-push. A store keeps
  its `store_id` across renames, re-pushes and regenerations.
- `provenance` (including `content_fingerprint`) is refreshed freely, so consumers get change
  detection without identity churn.

`artifacts.interchange` records how the Parquet was written: `files`, the published Parquet paths, and
`row_group_target_bytes`, the row-group size target of the push. A streamed pull reads exactly the listed `files`
at the envelope's commit rather than assuming a path layout, and refuses an envelope that lists none. Both fields are
additive, so they needed no schema bump; see [Streaming a store](finding_analysis_stores.md#streaming-a-store).

### The `subject` block: what the store was computed on

A store's columns mean something only against the model, tokenizer, component map and latent
dictionaries that produced them, and none of those can be recovered from the columns. So since schema
version 2 every envelope carries a `subject`:

| key | what it records |
| --- | --- |
| `model.id`, `model.revision` | the model the module loaded, and the resolved commit of its weights when the loader reports one |
| `architecture` | the architecture class name (`config.architectures[0]`) |
| `tokenizer.id` | the tokenizer's `name_or_path` |
| `component_map` | the map that resolved the store's sites: its `architecture`, `source` and `schema_version` |
| `dictionaries` | each attached latent dictionary by `kind` (`sae_lens`: `release`, `sae_id`; `circuit_tracer`: `transcoder_set`) |

`AnalysisCfg.apply` stamps the subject onto the store it writes, from the module the op runs on, for
the reason it stamps op provenance: only write time knows it. Every key is present, and what
interpretune cannot read is recorded as `null` rather than filled with a plausible value. Only the
model id is required.

A store assembled outside an op (with `analysis_store_from_batches`, say) carries no subject, and
publishing it is refused until one is given:

```python
it.hub.push_analysis_store(store, repo_id, subject=it.hub.subject_from_module(module))
```

`it.hub.pull_analysis_store` puts the subject back on the store it returns, as `store.subject`.

### Checks on push

`Dataset.push_to_hub` writes the parquet, and from `datasets` 4.1.0 (interpretune's floor) every
file it writes carries a page index, so nothing checks the files before they upload. After the
upload, the `range-read` check reads one published parquet file back through `HfFileSystem` at the
pushed revision and checks that it decodes. A failure raises `PublishedArtifactCheckError`: the
artifact is already live, so delete it or republish. Pass `skip_checks={"range-read"}` to skip it;
the skip is recorded in the envelope's `provenance` as `publish_checks_skipped`. The same check
interface guards dashboard corpora (see the dashboard pipeline docs).

### What the `interpretune` block re-attaches

Alongside those two, the `interpretune` block carries what a reader needs to rebuild the store's
*reading* surface rather than its identity:

| key | purpose | omitted when |
| --- | --- | --- |
| `col_cfg` | the serialized per-column config the formatter is re-attached from | never |
| `analysis_backend` | a registered backend **name**, not an instance | no backend attached |
| `protocol_cls` | the batch protocol's import path, when an op declared a non-default one | it is the default |
| `protocol_fields` | the field names that protocol declares, its bases' included | it is the default |

The first three record a **portable reference resolved on the way back**, never a live object;
`protocol_fields` records the protocol's attribute surface as plain data. All four are recorded at
write time rather than reconstructed at read time, for the reason op-collection provenance is: what a
name resolves to later is not necessarily what produced the store.

`protocol_cls` closes a gap where a store pushed with a custom protocol came back with the default —
the columns were intact, but the attribute surface the protocol exists to provide was silently absent.
Pulling a store resolves `protocol_cls` only from `interpretune` or from a hub component module already
loaded through the remote-code gate, since importing a path runs its module. Anything else, or a path
that does not name a batch protocol, is **refused**: serving the default in its place is exactly the
failure recording the protocol is meant to prevent.

#### Reading a store without its producer's code

A protocol decides only which columns get attribute access (`store.preds`); every column reads and
decodes through `col_cfg`, which is data. So a reader who wants a third-party store's columns without
trusting its op collection asks for that explicitly:

```python
store = it.hub.pull_analysis_store("someone/store", protocol="schema")
store["preds"]  # decoded exactly as in a resolved pull
store.preds     # attribute access from the recorded protocol_fields
```

`protocol="schema"` imports nothing. The store gets a stand-in protocol declaring the names in
`protocol_fields`, so attribute access matches a resolved pull, a declared column the store lacks
included (it reads as None). The stand-in carries `recorded_protocol` (the path the envelope names)
and `fields_from`. An envelope written before `protocol_fields` existed gets the store's own columns
as its fields (`fields_from == "columns"`), so there a declared column the store lacks raises
`AttributeError` instead. What a schema read gives up is the protocol's own methods, if it defines any.
The default, `protocol="resolve"`, never falls back to a schema read: a caller always gets the
reduced surface by asking for it.

Re-publishing a schema-read store records the original `protocol_cls` path and fields, not the
stand-in's. For discovery before any pull, `describe_analysis_store` returns the cached envelope
(`col_cfg`, `protocol_cls`, `protocol_fields`, `subject`) without the network.

### Schema versioning policy

Hub artifacts outlive the code that wrote them, and the publisher is usually not the reader. That
makes the envelope the one Interpretune surface where the alpha's otherwise-liberal
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
| `schema` missing or not an integer | refuse: this is not an Interpretune artifact envelope |

**Raising the floor retires published artifacts**, so it is a deliberate, documented act rather
than a side effect of a refactor. The floor is currently `1` and the schema is currently `3`: version
2 made the `subject` block mandatory, and a version-1 envelope is still read, with no subject; version 3 put
`site` in every `col_cfg` entry and `n_layers` in the subject, and a version-2 reader refuses it as newer. Each
version in the window has an envelope the writer of that version produced, frozen under
`tests/core/fixtures/` (`it_artifact_schema1.json`, `it_artifact_schema2.json`, `it_artifact_schema3.json`) and read on every run.
That is the piece that catches an accidental mandatory-field change, since every other test builds its
envelope with the current writer and so drifts along with it. The reader matrix runs against those same
fixtures: a reader one version behind refuses the version-3 envelope as newer, and a floor raised to `2`
refuses the version-1 envelope.

**Publishing refuses to emit an envelope this build could not read back.** `build_analysis_store_envelope`
returns its result through the same validator the read paths use, so a schema bumped past the
reader's window, or an identity block preserved from a corrupted existing envelope, fails locally
instead of landing on the Hub.

Neuronpedia dashboard corpora carry their own manifests (`dashboards.json` for provenance,
`source_ids.json` for identity), documented in the dashboard pipeline guide. They follow the same
single-integer convention but are read leniently today; if and when they adopt this envelope they
take a distinct `artifact_kind` under the rules above.

### Op-collection provenance: what the envelope does and does not record

The envelope records the Interpretune version, a content fingerprint, the analysis backend, and the
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
