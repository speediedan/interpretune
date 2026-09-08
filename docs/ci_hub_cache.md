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
