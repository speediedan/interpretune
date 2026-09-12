# Hosted CI and the Hugging Face Hub: warm once, then test offline

The hosted matrix (Ubuntu, Windows, macOS) runs three jobs at once, and they share one Hub API rate
limit. Measured on the two test files that load `gpt2` most, one job issues about a hundred Hub requests,
more than half of them to the rate-limited `/api/` endpoints, and **a warm cache removes almost none of
them**: the Hub client re-validates every cached file on each load, and transformers lists the repository
tree on every tokenizer load. With three jobs doing this at once the Hub answers `429`, the failure lands
in fixture setup, and the test-level reruns burn their attempts inside the same rate-limit window.

The fix is structural rather than a longer retry. Each job **warms** its cache from a manifest, then runs
the suite with **`HF_HUB_OFFLINE=1`**, so the tests issue zero Hub requests regardless of how many jobs
run concurrently. Measured on the same two files, the offline run passes identically with no requests.

## The pieces

| Piece | Where | What it does |
| --- | --- | --- |
| Manifest | `tests/hf_warm_manifest.yaml` | The models and datasets the suite loads. Also the cache key. |
| Warm step | `scripts/warm_hf_cache.py` | Fetches every manifest entry with backoff, using the same call shape the tests use. |
| Actions cache | `ci_test-full.yml` | `~/.cache/huggingface`, keyed on the manifest hash, saved right after warming. |
| Offline pass | `run-pytest-instrumented` action | Runs the suite with `HF_HUB_OFFLINE=1`; `hf_live` tests skip here. |
| Online pass | same action | Runs `-m hf_live` afterwards with the network available. |
| Legible miss | `tests/conftest.py` | A failure caused by an artifact missing from the cache reports the manifest to edit. |

## When a test needs something new from the Hub

Add the repository to the manifest. For a dataset, add the `config_name` the test loads; the warm step
calls `load_dataset` with the same arguments so the prepared cache the test looks up is the one written.
Editing the manifest changes its hash, which refreshes the actions cache on every runner.

A test that genuinely needs the live Hub (it exercises the real pull path, or reads a private repository
with a token) is marked `@pytest.mark.hf_live`. It skips in the offline pass and runs in the online pass,
which is small enough never to approach the rate limit.

### Which of the two: the criterion is what the test exists to exercise

Both options make a failing test pass, so "it fails offline" does not choose between them. The question is
whether a warm cache would leave the test still testing its subject:

> **If a warm cache makes the test pass without exercising the thing it exists to exercise, mark it
> `hf_live`. Otherwise add the repository to the manifest.**

A test that loads a model in order to test something else wants the manifest — the download is setup, and
warming it is exactly the point. A test whose subject *is* the fetch does not: warming the cache turns it
into a test of something else while it stays green, which is the failure mode with no symptom.

That case is not hypothetical. `#490` was a pinned pull that wrote a snapshot but no `refs/main`, so the
documented pull-then-load could not work from nothing. Every local run was green, because an earlier
unpinned pull had left the ref behind. A warmed cache reproduces that state deliberately.

Mark by **intent** rather than reactively. A test that means to reach the Hub declares it whether or not
any pipeline currently runs offline — otherwise it survives until someone adds an offline pass, and then
fails as though the new pass were the defect. The same criterion, from the manifest's side, is in the
header of `tests/hf_warm_manifest.yaml`.

### Verifying the mark, and the check that looks like it fails

Running the marked file on its own does **not** verify anything: `tests/conftest.py` applies the skip from a
session-level hook, and that conftest is never loaded when only `src/it_examples/tests/` is collected. The
mark then appears broken when it is correct — and it fails in whichever direction your machine happens to
be in, passing against a warm local cache or erroring against a clean one. Neither is the skip you are
checking for, which is why the outcome looks like information and is not.

```bash
# WRONG - conftest never loads, so the mark appears broken
pytest src/it_examples/tests/test_x.py

# RIGHT - CI's shape: both paths collected together, so the hook sees the item
HF_HUB_OFFLINE=1 pytest -rs tests/core/test_warm_hf_cache.py src/it_examples/tests/test_x.py
#   expect: SKIPPED ... "needs the live Hub, and HF_HUB_OFFLINE=1 is set"
```

And confirm the mark is on the item you think it is, which the skip alone does not tell you:

```bash
pytest --collect-only -m hf_live src/it_examples/tests/test_x.py
```

Nothing changes for a developer running the suite locally: without `HF_HUB_OFFLINE` set, every test runs
online exactly as before. To reproduce the CI shape locally:

```bash
python scripts/warm_hf_cache.py
HF_HUB_OFFLINE=1 python -m pytest tests src/it_examples/tests
python -m pytest tests src/it_examples/tests -m hf_live
```

## Why the cache is saved before the tests run

`actions/cache` normally saves in a post step that is skipped when the job fails. A red test would then
throw away the warm cache, and the next run would warm from scratch again. Restoring and saving are split
so the save happens as soon as the warm step succeeds.

## The self-hosted GPU pipeline is unchanged

It runs one job at a time on a persistent cache, so it does not hit the shared rate limit. It stays
online.
