# Locally verifying dashboard generation and example(s)

Local command guide for:
- Generating the latest lineage scalable-dashboard benchmark
- Running an example notebook that consumes locally generated dashboards.

References an end-to-end basic dashboard generation ([quickstart in the dashboard pipeline guide](neuronpedia_dashboard_pipeline.md#quickstart-gemma-3-1b-it-16k-on-monology-single-gpu)) that can be generated locally for use with the example notebook.

---

## Reproducing the benchmark

We have a non-destructive setup script prepares everything the benchmark suite needs — it locates or clones
the four repos, builds the venv, checks the local Postgres, and offers to build any missing prompt
datasets etc. It never modifies an existing checkout and doesn't need root. There are a lot of flexible options if you're interested but the defaults here should work

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

The completion summary prints the exact `source ...` lines and both suite commands, ready to paste:

```bash
source /tmp/it_baseline_trees_YYYYMMDD/benchmark_env.sh && source <venv>/bin/activate
python scripts/run_dashboard_benchmark_suite.py --mode threeway   # ~25 min
python scripts/run_dashboard_benchmark_suite.py --mode full       # ~2 h, 17 legs
```

Prerequisites the script checks itself: `git` and `uv` on PATH, `docker` if the local Neuronpedia DB
needs bringing up, bash >= 4.3, and HuggingFace access to the **gated** `google/gemma-3-1b-it`.

To repackage existing artifacts without re-running anything:
`--from-existing <artifact_root> --package-root <dir>`.

Full usage: `scripts/dashboard_benchmark_suite_usage.md` (in the repository).

---

## Example notebooks

The quick concept-direction steering demo comes in two variants, so you can start wherever your setup is:

| Notebook | Substrate | Setup required |
| --- | --- | --- |
| `ct_concept_steering_demo` | public gemma-2-2b + [neuronpedia.org](https://www.neuronpedia.org) | GPU + model weights only |
| `ct_concept_steering_demo_local_np` | gemma-3-1b-it + your local Neuronpedia stack | the sections below |

You can start with the public one if you want since it needs no local services and exercises the same analysis path but the local variant is the end-to-end demonstration that uses local dashboards/explanations (and optionally generates any missing explanations per-feature as needed in the notebook!) feeding feature selection and steering. You'll be guided below to run the quick local dashboard generation for `gemma-3-1b-it` ([quickstart in the dashboard pipeline guide](neuronpedia_dashboard_pipeline.md#quickstart-gemma-3-1b-it-16k-on-monology-single-gpu)) before running that local notebook though since it depends on non-public dashboards (shareable dashboards prioritized IT feature forthcoming).

### 1. Bring up the local Neuronpedia stack

If you already run Neuronpedia locally you can skip this — just make sure your webapp and Postgres are
reachable and note the ports, then set `LOCAL_WEBAPP_URL` / `LOCAL_DB_URL` in the notebook to match.
Nothing below is required if your stack is already up.

The commands below are for a fresh setup. `make init-env` overwrites `.env` (it prompts
first and aborts unless you confirm), so do not run it against a Neuronpedia install you have
already configured:

```bash
cd neuronpedia
make init-env                       # FRESH SETUPS ONLY - overwrites .env; then set POSTGRES_HOST_PORT=5433
make webapp-localhost-build && make webapp-localhost-run
```

The webapp serves on `http://localhost:3000`; Postgres on `127.0.0.1:5433`. Features are addressed as
`http://localhost:3000/<modelId>/<layer>-<sourceSetId>/<featureIndex>`, e.g.
`http://localhost:3000/gemma-3-1b-it/0-gemmascope-2-transcoder-262k/17`.

### 2. Generate and import dashboards

See the
[quickstart](neuronpedia_dashboard_pipeline.md#quickstart-gemma-3-1b-it-16k-on-monology-single-gpu)
in the dashboard pipeline guide for the end-to-end generation walkthrough — one command against a
committed config, about an hour on a single 24 GiB card — and
[importing existing bundles](neuronpedia_dashboard_pipeline.md#import-existing-export-bundles-into-the-local-db)
for the backfill case.

The local notebook's defaults expect `gemma-3-1b-it` dashboards for the
`gemmascope-2-transcoder-16k` source set; dashboard and runtime width must match.

If you have a GPU with more VRAM than the reference 4090 (24 GiB), you can push the generation
configuration further — larger `n_features_per_batch` / `n_prompts_in_forward_pass`, or more prompts.
The pipeline guide documents the shapes we validated and where the memory cliffs sit.

### 3. Set up the explanation CLI (OPTIONAL)

<details>
<summary>Expand if you want to see missing explanations auto-generated for features involved in the example notebook</summary>

This whole step is optional. The local notebook ships with
`GENERATE_MISSING_LOCAL_EXPLANATIONS = False`, so it reports local explanation coverage without
backfilling and everything else still works. Set this up only to see explanations generated end to
end.

Explanation generation currently always calls an external endpoint with a conforming local CLI rather than a hardcoded provider. Activation sourcing and the DB insert are local.

The local provider must accept a one-shot prompt non-interactively (`<cli> -p "<prompt>"`) and print the response to stdout. The GitHub Copilot CLI is the default. If you do not have it (no GitHub Copilot subscription required):

```bash
npm install -g @github/copilot          # needs Node 18+
copilot --version                       # confirm it is on PATH
```

Then pick one of two routes:

a. BYOK (any OpenAI-compatible provider). Three variables can supply the key — highest
precedence first: `IT_EXPLANATION_PROVIDER_API_KEY`, then `COPILOT_PROVIDER_API_KEY`, then
`OPENROUTER_API_KEY` (the variable Neuronpedia itself uses, so one key serves both sides).

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

If your local database already has explanations, set `REGENERATE_LOCAL_EXPLANATIONS = True` in the
notebook. Otherwise every feature is skipped and the run reports full coverage without calling the
CLI. Note nothing is deleted, the new explanation is inserted alongside.

You can also generate one explanation directly to confirm the setup before running the notebook:

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

## Pre-generated dashboards: download and import without generating anything

Two corpora are published as public Hugging Face **Storage Buckets**, so a local Neuronpedia can be
populated without spending GPU hours. **No token or account is required** — every command below was
run with no credential present.

| Corpus | Bucket | Files | Size | Prompts |
| --- | --- | --- | --- | --- |
| **RTE** (example-aligned) | [`…__rte__dashboards`](https://huggingface.co/buckets/speediedan/gemma-3-1b-it__gemmascope-2-transcoder-16k__rte__dashboards) | 1067 | 5.84 GiB | 2,490 × 319 tok |
| **Monology** (generic web text) | [`…__monology__dashboards`](https://huggingface.co/buckets/speediedan/gemma-3-1b-it__gemmascope-2-transcoder-16k__monology__dashboards) | 1014 | 10.07 GiB | 24,576 × 128 tok |

Both are `gemma-3-1b-it` with the `gemma-scope-2-1b-it-transcoders-all` 16k transcoders, all 26
layers. RTE uses the prompts the example notebooks run, so its dashboards line up with what those
notebooks show; monology is the generic-text counterpart. They import into **different source sets**
and can coexist in one database.

### 1. Download

The destination's **leaf directory name is used as the run directory**, so keep the names below and
vary the parent if needed.

| corpus | bucket id | leaf directory |
| --- | --- | --- |
| RTE | `speediedan/gemma-3-1b-it__gemmascope-2-transcoder-16k__rte__dashboards` | `gemma-3-1b-it_gemmascope-2-transcoder-16k-rte` |
| monology | `speediedan/gemma-3-1b-it__gemmascope-2-transcoder-16k__monology__dashboards` | `gemma-3-1b-it_gemmascope-2-transcoder-16k` |

```python
from pathlib import Path
from interpretune.utils import download_dashboard_run

# RTE, 5.84 GiB. For monology, swap both the bucket id and the leaf directory per the table above.
download_dashboard_run(
    "speediedan/gemma-3-1b-it__gemmascope-2-transcoder-16k__rte__dashboards",
    Path("~/np_corpora/gemma-3-1b-it_gemmascope-2-transcoder-16k-rte").expanduser(),
)
```

### 2. Import into a local Neuronpedia DB

Needs a running local Neuronpedia Postgres — see the
[pipeline guide](neuronpedia_dashboard_pipeline.md). Point `--run-root` at the **parent** of the
directory you downloaded into:

```bash
# RTE
python scripts/launch_neuronpedia_dashboard_pipeline.py \
  --config scripts/configs/neuronpedia_dashboard/gemmascope-2-transcoder-16k-rte-production.yaml \
  --import-only-local-db \
  --run-root ~/np_corpora \
  --local-db-url postgres://postgres:postgres@127.0.0.1:5433/postgres
```

```bash
# monology -- same shape, different config
python scripts/launch_neuronpedia_dashboard_pipeline.py \
  --config scripts/configs/neuronpedia_dashboard/gemmascope-2-transcoder-16k-monology-24576.yaml \
  --import-only-local-db \
  --run-root ~/np_corpora \
  --local-db-url postgres://postgres:postgres@127.0.0.1:5433/postgres
```

Add `--print-command --dry-run` to either to see the resolved command without executing it.

Each import writes 26 layers × 16,384 features = **425,984 `Neuron` rows** plus activations
(~16.2M for monology, ~10.4M for RTE), taking roughly 40–60 minutes.

### 3. Confirm it landed

```sql
SELECT s."setName", s."modelId", count(DISTINCT s.id) AS sources
FROM "Source" s
WHERE s."setName" LIKE 'gemmascope-2-transcoder-16k%'
GROUP BY s."setName", s."modelId";
```

Expect 26 sources per (set, model) pair — `gemmascope-2-transcoder-16k` for monology and
`gemmascope-2-transcoder-16k-rte` for RTE.

### Notes worth knowing first

- **Source ids travel with the corpus.** Each bucket carries a `source_ids.json` recording the
  Neuronpedia source id per layer, so the import yields the same ids no matter where you unpacked
  it. Earlier corpora inferred ids from the directory name, which meant a renamed download imported
  *successfully* under different ids — a failure that looked like success. This removes that.
- **Each bucket describes itself in `dashboards.json`.** ~3 KB at the bucket root, so you can check
  what a corpus is before committing to a multi-GiB download:

  ```python
  from huggingface_hub import HfApi
  HfApi().download_bucket_files(
      "speediedan/gemma-3-1b-it__gemmascope-2-transcoder-16k__monology__dashboards",
      [("dashboards.json", "dashboards.json")],
  )
  ```

  It records the model, source set, prompt corpus (`24576 prompts × 128 tokens` for monology,
  `2490 × 319` for RTE), the layers actually generated, whether page indexes are present, and the
  `sae-dashboard` / `pyarrow` versions that wrote the files. Nothing imports from it — it is
  descriptive only, so a stale copy cannot misroute an import the way a stale `source_ids.json`
  could.
- **`activation_copy_rows` are included deliberately** (~45% of the payload). The per-batch
  manifests declare that table and the importer raises if it is missing, so a slimmed-down copy is
  not importable at all. It is also the faster of the two import paths.
- **Parquet page indexes are present**, so the files are range-readable. Via the bucket's
  S3-compatible gateway (`https://s3.hf.co/speediedan`) you can query them in place — e.g. DuckDB
  `read_parquet('s3://…')` — without downloading anything.
- **The `hf` CLI (`hf buckets sync …`) may be simpler where it works.** It is deliberately not
  documented here: in our environment a `click`/`typer` incompatibility breaks the `hf` entry point
  entirely, not just the buckets subcommand, so we have not verified those commands and will not
  publish untested ones.

## Related documentation

- [Neuronpedia dashboard pipeline](neuronpedia_dashboard_pipeline.md) — generation, import, and
  the [pretokenization CLI](neuronpedia_dashboard_pipeline.md#pretokenize-dashboard-datasets)
- [Developer multi-repo setup](developer_multi_repo_setup.md) — the environment-variable
  contract and build steps
- [Roadmap](../roadmap.md) — Wave 1 / Wave 2 framing
