# Locally verifying dashboard generation and example(s)

Recommended order:

- (a) **[Dev env setup](#a-environment-setup)**: one script, run before any of the below
- (b) **[Example notebooks](#b-example-notebooks)**: an Interpretune demo notebook that downloads, imports and uses a generated dashboard from hf hub
- (c) **[Pre-generated dashboards](#c-pre-generated-dashboards)**: download and import generated dashboards from hf hub
- (d) **[Dashboard regeneration](#d-dashboard-regeneration)**: regenerate the latest lineage scalable-dashboard benchmark

(b) is the shortest path to something to look at: the notebook fetches the dashboard it needs, so only requires the dev env setup script in (a).

---

<a id="a-environment-setup"></a>
<a id="environment-setup"></a>
## (a) Development environment setup

A non-destructive setup script prepares everything the rest of this guide needs — it locates or
clones the four repos, builds the venv, checks the local Postgres, and offers to build any missing
prompt datasets. It doesn't modify existing checkout and doesn't need root. There are a lot of
flexible options if you're interested, but the defaults here should work.

```bash
mkdir -p /tmp/it_eval_repos && cd /tmp/it_eval_repos
git clone https://github.com/speediedan/interpretune.git && cd interpretune

# See the whole plan first, without running anything:
# python scripts/setup_dashboard_benchmark_env.py --dry-run

# Clone/build everything. With no arguments it clones the sibling repos beside this one and puts the
# preserved-baseline worktrees in a dated temp dir (/tmp/it_baseline_trees_YYYYMMDD).
python scripts/setup_dashboard_benchmark_env.py

# To reuse checkouts you already have, point at them instead of cloning:
# python scripts/setup_dashboard_benchmark_env.py --repos-root /tmp/it_eval_repos \
#   --neuronpedia ~/repos/neuronpedia
```

The completion summary prints the exact `source ...` lines to activate the environment:

```bash
source /tmp/it_baseline_trees_YYYYMMDD/benchmark_env.sh && source <venv>/bin/activate
```

Prerequisites the script checks itself: `git` and `uv` on PATH, `docker` if the local Neuronpedia DB
needs bringing up, bash >= 4.3, and HuggingFace access to the gated `google/gemma-3-1b-it`.

---

<a id="b-example-notebooks"></a>
<a id="example-notebooks"></a>
## (b) Example notebooks

**`ct_concept_steering_demo_local_np`** is the notebook to run: a concept-direction steering demo on
`gemma-3-1b-it` against your local Neuronpedia stack, end to end — local dashboards and explanations
(optionally generating any missing explanation per-feature as it goes) feeding feature selection and
steering. It needs `gemma-3-1b-it` dashboards in your database and **downloads and imports them
itself if they are absent**, so there is nothing to prepare beyond a running stack.

### 1. Verify the local Neuronpedia stack

If you have an existing Neuronpedia local dev stack up and running, just make sure your webapp and Postgres are
reachable and note the ports, then set `LOCAL_WEBAPP_URL` / `LOCAL_DB_URL` in the notebook to match.
Nothing below is required if your stack is already up otherwise.

The commands below are for a fresh setup. `make init-env` overwrites `.env` (it prompts
first and aborts unless you confirm), so do not run it against a Neuronpedia env you have
already configured and don't want to overwrite:

```bash
cd neuronpedia
make init-env   # overwrites existing .env after confirming
make webapp-localhost-build && make webapp-localhost-run
```
### 2. Set up the explanation CLI (OPTIONAL)

<details>
<summary>Expand if you want to see missing explanations auto-generated for features involved in the example notebook</summary>

This whole step is optional. The local notebook ships with
`GENERATE_MISSING_LOCAL_EXPLANATIONS = False`, so it reports local explanation coverage without
backfilling and everything else still works. Set this up only to see explanations generated end to
end.

Explanation generation calls an external endpoint with a conforming local CLI rather than a hardcoded provider with activation sourcing and the DB insert being local.

The local provider must accept a one-shot prompt non-interactively (`<cli> -p "<prompt>"`) and print the response to stdout. The GitHub Copilot CLI is the default. If you do not have it (no GitHub Copilot subscription required):

```bash
npm install -g @github/copilot          # needs Node 18+
copilot --version                       # confirm it is on PATH
```

Then pick one of two routes:

a. BYOK (any OpenAI-compatible provider). Three variables can supply the key — highest
precedence first: `IT_EXPLANATION_PROVIDER_API_KEY`, then `COPILOT_PROVIDER_API_KEY`, then
`OPENROUTER_API_KEY` (the variable Neuronpedia uses, so one key serves both sides).

The endpoint follows the *key*, not the variable holding it: any key with the `sk-or-` prefix
routes to OpenRouter from any of the three. An explicit `IT_EXPLANATION_PROVIDER_BASE_URL` will take precedence though.

A free recommended combination validated end to end:

```bash
export IT_EXPLANATION_PROVIDER_API_KEY=sk-or-v1-...            # an OpenRouter key
export IT_EXPLANATION_CLI_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
```

Always set `IT_EXPLANATION_CLI_MODEL` with an OpenRouter key — the default model id belongs to a
different provider and will not resolve there.

For any other OpenAI-compatible endpoint, point it explicitly:

```bash
export IT_EXPLANATION_PROVIDER_API_KEY=<your key>
export IT_EXPLANATION_PROVIDER_BASE_URL=<https://your-endpoint/v1>
export IT_EXPLANATION_CLI_MODEL=<model served by that endpoint>
```

b. The CLI's own auth. This is the less frequently used pattern. With no resolvable API key, the CLI's native auth is used unchanged (for Copilot, `copilot` handles GitHub sign-in). So you need to pass a model your native
provider actually serves as the BYOK default model will not resolve:

```bash
export IT_EXPLANATION_CLI_MODEL=<model your Copilot plan serves>
```

To use a different conforming CLI entirely, set `IT_EXPLANATION_CLI`.

You can generate one explanation directly to confirm the setup before running the notebook (update with your local postgres and target model/source-set etc.):

```bash
python scripts/generate_neuronpedia_feature_explanation.py \
  --model-id gemma-3-1b-it --layer 0 --source-set gemmascope-2-transcoder-262k --index 17 \
  --base-url http://localhost:3000 --insert-into-local-db \
  --local-db-url 'postgres://postgres:postgres@127.0.0.1:5433/postgres'
```

Full reference:
[Explanation CLI configuration](neuronpedia_dashboard_pipeline.md#explanation-cli-configuration)
and [Local explanation note](neuronpedia_dashboard_pipeline.md#local-explanation-note).

</details>

---

<a id="c-pre-generated-dashboards"></a>
<a id="pre-generated-dashboards"></a>
## (c) Pre-generated dashboards: download and import without generating anything

Two dashboards are published as public Hugging Face Storage Buckets, so a local Neuronpedia DB can be
populated without spending GPU hours.

| Corpus | Bucket | Files | Size | Prompts |
| --- | --- | --- | --- | --- |
| **Monology** (generic web text) | [`…__monology__dashboards`](https://huggingface.co/buckets/speediedan/gemma-3-1b-it__gemmascope-2-transcoder-16k__monology__dashboards) | 1016 | 10.15 GiB | 24,576 × 128 tok |
| **RTE** (example-aligned) | [`…__rte__dashboards`](https://huggingface.co/buckets/speediedan/gemma-3-1b-it__gemmascope-2-transcoder-16k__rte__dashboards) | 1068 | 6.06 GiB | 2,490 × 319 tok |

Both are `gemma-3-1b-it` with the `gemma-scope-2-1b-it-transcoders-all` 16k transcoders, all 26
layers. The monology dataset is a typical dense, generic dataset used for many existing Neuronpedia dashboards. The RTE dashboards demo example-aligned dashboards where each sequence is a dataset example.

Both were regenerated 2026-08-07 with multiple Parquet row groups (`row_group_size=4096`), which is
what makes them range-readable: fetching one feature's dashboard over the S3 gateway costs 1.4 MiB of
a 38.5 MiB file rather than all of it. Each corpus records its own layout in `dashboards.json` under
`artifacts.parquet_row_group_size`.

> These land in the same source sets **(d)** would generate into. `--autosuffix-on-exists` below
> keeps both by importing under a timestamped variant; without a collision flag the command refuses
> and tells you what is there. See
> [source-set collisions](neuronpedia_dashboard_pipeline.md#source-set-collisions) for the other options.

```bash
python scripts/fetch_dashboards_from_hub.py \
  --bucket speediedan/gemma-3-1b-it__gemmascope-2-transcoder-16k__monology__dashboards \
  --local-db-url postgres://postgres:postgres@127.0.0.1:5433/postgres \
  --autosuffix-on-exists
```

Swap the bucket id for `…__rte__dashboards` to fetch the other corpus. That is the whole
procedure: it reads the corpus's own `dashboards.json`, picks the matching committed config,
creates the destination, downloads, imports, and prints a count summary. Add `--dry-run` to see the
plan without moving anything or `--dest <dir>` to choose where the corpus lands (default:
`$IT_NP_CACHE/hub_downloads`, else `$HF_HOME/interpretune/neuronpedia/hub_downloads`).

Needs a running local Neuronpedia Postgres — see the
[pipeline guide](neuronpedia_dashboard_pipeline.md). Expect 425,984 `Neuron` rows per corpus
(26 layers × 16384 feats) plus activations (~16.2M monology, ~10.4M RTE), 40–60 minutes.

---

<a id="d-dashboard-regeneration"></a>
<a id="dashboard-regeneration"></a>
<!-- former heading; PR bodies link here, so the anchor outlives the rename -->
<a id="reproducing-the-benchmark"></a>
## (d) Dashboard regeneration

With the environment from **(a)** activated, the benchmark suite runs in one command:

```bash
python scripts/run_dashboard_benchmark_suite.py --mode threeway   # ~25 min
python scripts/run_dashboard_benchmark_suite.py --mode full       # ~2 h, 17 legs
```
Full usage: `scripts/dashboard_benchmark_suite_usage.md` (in the repository).

For a single end-to-end generation rather than the benchmark wave, see the
[quickstart](neuronpedia_dashboard_pipeline.md#quickstart-gemma-3-1b-it-16k-on-monology-single-gpu) —
one command against a committed config, about an hour on a single 24 GiB card. It generates into the
same source set the monology corpus in **(c)** imports into, so the same
[collision flags](neuronpedia_dashboard_pipeline.md#source-set-collisions) apply.

---

## Related documentation

- [Neuronpedia dashboard pipeline](neuronpedia_dashboard_pipeline.md) — generation, import, and
  the [pretokenization CLI](neuronpedia_dashboard_pipeline.md#pretokenize-dashboard-datasets)
- [Developer multi-repo setup](developer_multi_repo_setup.md) — the environment-variable
  contract and build steps
- [Roadmap](../roadmap.md) — Wave 1 / Wave 2 framing
