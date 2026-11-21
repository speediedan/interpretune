# Finding and using AnalysisStores

An AnalysisStore pushed with `interpretune.hub.push_analysis_store` is a dataset repo on the Hugging Face Hub: Parquet
files, an `it_artifact.json` envelope and a generated card. This page covers how a session finds stores that others
have published, which publishers it trusts to supply them, and how to pull one. The envelope itself is described in
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
the remote-code gate, and that gate (`IT_TRUST_REMOTE_CODE`) is separate and default-deny. See
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
