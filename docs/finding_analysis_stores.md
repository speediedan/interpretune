# Finding and using AnalysisStores

An AnalysisStore pushed with `interpretune.hub.push_analysis_store` is a dataset repo on the Hugging Face Hub: Parquet
files, an `it_artifact.json` envelope and a generated card. This page covers how a session finds stores that others
have published, which publishers it trusts to supply them, and how to pull or stream one. The envelope itself is described in
[AnalysisStore serialization](analysis_store_serialization.md).

## Trusted publishers

A session discovers analysis from **publishers**: Hub namespaces, a user or an organization, whose stores and
dashboard corpora it is willing to list. One list governs both kinds, so a publisher trusted for stores is trusted for
dashboards too.

The list resolves at call time, in this order:

1. the `publishers=` keyword of the discovery call;
2. the `IT_TRUSTED_PUBLISHERS` environment variable, comma-separated;
3. the project's own namespace, `speediedan` (`interpretune.hub.discovery.DEFAULT_TRUSTED_PUBLISHERS`).

```python
import os
import interpretune as it

it.hub.trusted_publishers()                     # ('speediedan',)
os.environ["IT_TRUSTED_PUBLISHERS"] = "speediedan,my-lab"
it.hub.trusted_publishers()                     # ('speediedan', 'my-lab'), read now, not at import
it.hub.trusted_publishers(["my-lab"])           # the keyword wins: ('my-lab',)
```

Each entry must be a bare namespace. A repo id such as `my-lab/some-store`, an empty entry or an empty list is refused
with `UntrustedPublisherSpecError`, rather than matching nothing.

**What trust means here.** Trusting a publisher lets discovery list its artifacts. It runs none of their code: a
pulled store still resolves its `protocol_cls` only from interpretune or from a hub component already loaded through
the remote-code gate, and that gate (`IT_TRUST_REMOTE_CODE`) is separate and default-deny. A store whose protocol you
have not loaded can still be read with `pull_analysis_store(..., protocol="schema")`, which imports nothing. See
[Trusting hub components](hub_trust_posture.md).

Trust is per publisher for now, which keeps the model simple. A finer, per-collection posture may follow if a use for
it appears.

## Finding stores

`it.hub.find_analysis_stores()` lists the stores the trusted publishers have pushed:

```python
rows = it.hub.find_analysis_stores("google/gemma-2-2b", token=os.environ.get("HF_TOKEN"))
for row in rows:
    print(row.repo_id, row.revision[:8], row.schema, row.subject.model_id, row.generated_by)
```

It lists each publisher's datasets tagged `interpretune-analysis-store` (every pushed store carries the tag) and reads
each one's `it_artifact.json` at the commit the listing reports. That is a few KB per store; no Parquet moves. Each
row is a `StoreListing`:

| field | meaning |
| --- | --- |
| `repo_id`, `publisher` | the dataset repo and the trusted namespace it was found under |
| `revision` | the commit the envelope was read at; pull that revision to get exactly this store |
| `schema` | the envelope's schema version |
| `store_id` | the store's identity, preserved across re-pushes |
| `subject` | what the store was computed on (model, revision, architecture, tokenizer, component map, dictionaries); `None` only for a schema-1 store |
| `generated_by`, `num_rows`, `columns` | what produced it and its shape, from the envelope |
| `private` | whether the repo is private |

The optional `model_id` argument matches the store's `subject`. A schema-1 store has no subject, so it is left out of
a filtered listing and appears only in an unfiltered one.

**What is left out, and what raises.** A tagged dataset whose envelope this interpretune cannot read (written by a
newer interpretune, older than the readable floor, missing or malformed) is left out with a
`StoreSchemaSkippedWarning` that names it and says why. One unreadable store never hides the rest. A failure to reach
the Hub raises instead, because a listing that is short by an unreachable store would read as complete.

**Private stores.** Pass `token=` to list and read private stores that token can see. Without one, only public stores
are listed.

## Pulling a store

Pull a listed store at the revision its row records:

```python
row = rows[0]
store = it.hub.pull_analysis_store(row.repo_id, revision=row.revision, token=os.environ.get("HF_TOKEN"))
store.subject          # the same subject the listing showed
```

The envelope resolves first and the Parquet files come from the same commit, so the pulled store is the one the row
described even if the repo has been pushed to since. The interpretune formatter is re-attached from the envelope, so
graph columns hydrate and steering runs from the stored graph without re-running the pipeline. The
`shared_analysis_roundtrip` notebook shows the full push, pull and replicate loop.

## Streaming a store

A full pull downloads every Parquet file before it returns. To read part of a large store, or to start reading before
the rest arrives, stream it:

```python
store = it.hub.pull_analysis_store(
    row.repo_id, revision=row.revision, streaming=True, columns=["input_string", "logit_diff"]
)
for record in store:                  # decoded rows, as a full pull would decode them
    print(record["input_string"], record["logit_diff"])
store["logit_diff"]                   # one column, streamed for that column only
```

A streamed pull downloads only the envelope up front. The store then streams the Parquet files the envelope lists, at
the envelope's commit, through `datasets` streaming. It differs from a pulled store in four ways:

- **It reads only what you ask for.** `columns=` projects the read, so the other columns are not fetched. Reading a
  column by name streams that column alone, again on each access.
- **It has no positions.** It can be iterated and read by column, but indexing by row (`store[0]`, `store[2:5]`) is
  refused, since a stream has no positions to index. Pull without `streaming=True` to index rows.
- **It holds a stream, not a table.** The rows are in `store.stream` (an `IterableDataset`), and `store.dataset` is
  `None`. A streamed store cannot be pushed; pull it in full to republish it.
- **Each read is a range request.** Requests go through `HfFileSystem` in blocks of `block_size` bytes, 64 KiB by
  default (`interpretune.hub.artifacts.STREAMING_BLOCK_SIZE`). The `HfFileSystem` default of 5 MiB is fetched on
  every request, so on the published demo store a projected read of two small columns transfers about 226 KB of a
  32 MB file, where the default block would transfer about 5.4 MB.

Rows decode exactly as a full pull decodes them: the same formatter, the same `col_cfg`, and graph columns hydrate
through the store's analysis backend. The `shared_analysis_roundtrip` notebook streams the published demo store,
`speediedan/ct-concept-steering-demo`, which is public, so its streamed read needs no token.

**How far a read reaches depends on row groups.** A Parquet file is read in row groups, and the stream reads a group
only when a read reaches it. `push_analysis_store` writes row groups of about 16 MB uncompressed
(`ANALYSIS_STORE_ROW_GROUP_SIZE`; override with `row_group_size=`) and records the target in the envelope as
`artifacts.interchange.row_group_target_bytes`. The default was measured on a 142 MB store: reading its first row cost
34 MB at 16 MB groups, against the whole file at the 100 MB `datasets` default, while a scan of two scalar columns took
11 requests rather than the 38 that 4 MB groups need. A store whose rows are each larger than the target (one
attribution graph per row, for example) gets one row per group.

## Dashboard corpora

Dashboard corpora are discovered from the same publishers. `find_dashboard_corpora` and `resolve_bucket_for` (in
`interpretune.utils.neuronpedia_hub_fetch`) take the same `publishers=` keyword and read the same environment
variable; a corpus is a bucket that carries a `dashboards.json`. See
[Finding a published corpus](neuronpedia_dashboard_pipeline.md#finding-a-published-corpus-trusted-publishers).

## Publishing so others can find you

There is no registration step. A store pushed into your namespace is listed for anyone who trusts that namespace, as
soon as the push completes. To have a session discover stores from a publisher, add their namespace to
`IT_TRUSTED_PUBLISHERS` or pass it as `publishers=`.

Listing stores through the Hub's own search (a `library:interpretune` filter on datasets, with a usage snippet) needs
an upstream mapping from the `interpretune` tag, which is planned but not yet in place.
