# Verifying dashboard generation

How to reproduce the scalable-dashboard benchmark yourself, and how to run the example notebook that
consumes locally generated dashboards. Everything here is a procedure you can execute — claims about
any particular pull request live with that pull request.

---

## Reproducing the benchmark

One guided, non-destructive setup script prepares everything the suite needs — it locates or clones
the four repos, builds the venv, checks the local Postgres, and offers to build any missing prompt
datasets. It never modifies an existing checkout and never needs root.

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

Full usage: `scripts/dashboard_benchmark_suite_usage.md` (in the repository — that file is outside
the docs build).

---

## Try it: the example notebooks

The concept-direction steering demo comes in two variants, so you can start wherever your setup is:

| Notebook | Substrate | Setup required |
| --- | --- | --- |
| `ct_concept_steering_demo` | public gemma-2-2b + [neuronpedia.org](https://www.neuronpedia.org) | GPU + model weights only |
| `ct_concept_steering_demo_local_np` | gemma-3-1b-it + **your** local Neuronpedia stack | the sections below |

Start with the public one — it needs no local services and exercises the same analysis path. The
local variant is the end-to-end demonstration: dashboards you generated, explanations you generated,
feeding feature selection and steering.

Both notebooks state their substrate as plain defaults; there is no mode switch to set.

### 1. Bring up the local Neuronpedia stack

```bash
cd neuronpedia
make init-env                       # then set POSTGRES_HOST_PORT=5433 in .env
make webapp-localhost-build && make webapp-localhost-run
```

The webapp serves on `http://localhost:3000`; Postgres on `127.0.0.1:5433`. Features are addressed as
`http://localhost:3000/<modelId>/<layer>-<sourceSetId>/<featureIndex>`, e.g.
`http://localhost:3000/gemma-3-1b-it/0-gemmascope-2-transcoder-262k/17`.

### 2. Generate and import dashboards

See the [dashboard pipeline guide](neuronpedia_dashboard_pipeline.md) for the end-to-end generation
walkthrough, and
[importing existing bundles](neuronpedia_dashboard_pipeline.md#import-existing-export-bundles-into-the-local-db)
for the backfill case.

The local notebook's defaults expect `gemma-3-1b-it` dashboards for the
`gemmascope-2-transcoder-16k` source set; dashboard and runtime width must match.

If you have a GPU with more VRAM than the reference 4090 (24 GiB), you can push the generation
configuration further — larger `n_features_per_batch` / `n_prompts_in_forward_pass`, or more prompts.
The pipeline guide documents the shapes we validated and where the memory cliffs sit.

### 3. Set up the explanation CLI (OPTIONAL)

<details>
<summary>Expand only if you want to see explanations generated end to end</summary>

**This whole step is optional.** The local notebook ships with
`GENERATE_MISSING_LOCAL_EXPLANATIONS = False`, so it reports local explanation coverage without
backfilling and everything else still works. Set this up only to see explanations generated end to
end.

Explanation generation always calls an **external LLM endpoint**. Activation sourcing and the DB
insert are local; there is no offline explainer.

Explanations drive a *conforming local CLI* rather than a hardcoded provider: it must accept a
one-shot prompt non-interactively (`<cli> -p "<prompt>"`) and print the response to stdout. The
GitHub Copilot CLI is the default. If you do not have it (no GitHub Copilot subscription required):

```bash
npm install -g @github/copilot          # needs Node 18+
copilot --version                       # confirm it is on PATH
```

Then pick one of two routes:

**a. BYOK (any OpenAI-compatible provider).** Three variables can supply the key — highest
precedence first: `IT_EXPLANATION_PROVIDER_API_KEY`, then `COPILOT_PROVIDER_API_KEY`, then
`OPENROUTER_API_KEY` (the variable Neuronpedia itself uses, so one key serves both sides).

The endpoint follows the **key**, not the variable holding it: any key with the `sk-or-` prefix
routes to OpenRouter from any of the three. An explicit `IT_EXPLANATION_PROVIDER_BASE_URL` wins.

Recommended, and the combination validated end to end:

```bash
export IT_EXPLANATION_PROVIDER_API_KEY=sk-or-v1-...            # an OpenRouter key
export IT_EXPLANATION_CLI_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
```

**Always set `IT_EXPLANATION_CLI_MODEL` with an OpenRouter key** — the default model id belongs to a
different provider and will not resolve there. Free-tier slugs carry a `:free` suffix.

For any other OpenAI-compatible endpoint, point it explicitly:

```bash
export IT_EXPLANATION_PROVIDER_API_KEY=<your key>
export IT_EXPLANATION_PROVIDER_BASE_URL=<https://your-endpoint/v1>
export IT_EXPLANATION_CLI_MODEL=<model served by that endpoint>
```

**b. The CLI's own auth.** With no resolvable API key, the CLI's native auth is used unchanged (for
Copilot, `copilot` handles GitHub sign-in). In that case you **must** pass a model your native
provider actually serves — the BYOK default model will not resolve:

```bash
export IT_EXPLANATION_CLI_MODEL=<model your Copilot plan serves>
```

To use a different conforming CLI entirely, set `IT_EXPLANATION_CLI`.

If your local database already has explanations, set `REGENERATE_LOCAL_EXPLANATIONS = True` in the
notebook. Otherwise every feature is skipped and the run reports full coverage **without calling the
CLI once** — which looks identical whether the pipeline works or not. Nothing is deleted; the new
explanation is inserted alongside.

Generate one explanation directly, to confirm the setup before running the notebook:

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

## Related documentation

- [Neuronpedia dashboard pipeline](neuronpedia_dashboard_pipeline.md) — generation, import, and
  the [pretokenization CLI](neuronpedia_dashboard_pipeline.md#pretokenize-dashboard-datasets)
- [Developer multi-repo setup](developer_multi_repo_setup.md) — the environment-variable
  contract and build steps
- [Roadmap](../roadmap.md) — Wave 1 / Wave 2 framing
