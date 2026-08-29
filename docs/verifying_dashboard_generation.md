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
make init-env        # OVERWRITES .env, prompts first; skip it if yours is already configured
make webapp-install  # required on a fresh checkout: the build fails with `env-cmd: not found` without it
make webapp-build && make webapp-run
```

`make webapp-install` is easy to skip because the failure names a missing binary rather than a
missing step: `make webapp-build` on an uninstalled checkout reports `sh: 1: env-cmd: not found`,
which reads like a broken environment rather than an absent `node_modules`.
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
  --local-db-url 'postgres://postgres:postgres@127.0.0.1:5432/postgres'
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
  --local-db-url postgres://postgres:postgres@127.0.0.1:5432/postgres \
  --autosuffix-on-exists
```

Swap the bucket id for `…__rte__dashboards` to fetch the other corpus. That is the whole
procedure: it reads the corpus's own `dashboards.json`, picks the matching committed config,
creates the destination, downloads, imports, and prints a count summary.

**The target database must already be initialised, not merely reachable.** The import writes a
`SourceRelease` row whose `creatorId` references the `bot` user that `prisma/seed.ts` creates, so a
database carrying the schema but no seed data fails with:

```
psycopg.errors.ForeignKeyViolation: insert or update on table "SourceRelease"
  violates foreign key constraint "SourceRelease_creatorId_fkey"
DETAIL: Key (creatorId)=(...) is not present in table "User".
```

That names the symptom rather than the cause. `make db-init` in the neuronpedia checkout supplies
what is missing: it applies migrations, runs `prisma db seed`, and installs the pgvector tuning. A
stack brought up through **(b)** has already done this; a database created any other way has not.

**Check the buckets from an authenticated session, not anonymously.** The unauthenticated Hub API has
been observed reporting a bucket as empty while it was in fact full, and serving a stale manifest for
several minutes after a push. Both failures look like an answer rather than an error, so a maintainer
confirming a corpus anonymously can reach a confident wrong conclusion about what is published. `hf
auth login`, or pass a token, before concluding anything about bucket contents. Add `--dry-run` to see the
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
[quickstart](neuronpedia_dashboard_pipeline.md#quickstart-gemma-3-1b-it-16k-on-monology-single-gpu):
one command against a committed config, about an hour on a single 24 GiB card. It generates into the
same source set the monology corpus in **(c)** imports into, so the same
[collision flags](neuronpedia_dashboard_pipeline.md#source-set-collisions) apply.

### Which tensor the run captures, and how to confirm it did

Read this before regenerating anything for the Gemma Scope 2 transcoders. It is the one part of the
regeneration that cannot be checked by looking at the result.

These transcoders are trained on the block norm's output, which Google's shipped `config.json`
declares as `model.layers.{N}.pre_feedforward_layernorm.output`. Their SAELens metadata declares
`blocks.{N}.hook_mlp_in`, which TransformerLens fires on the residual stream *before* that norm.
Those are different tensors, and not marginally so: reconstructing the transcoder's declared target
on layer 5 gives FVU 0.11 from the declared input against FVU 20585 from `hook_mlp_in`, with a cosine
similarity of 0.088 between the two. See
[the caveat in the pipeline guide](neuronpedia_dashboard_pipeline.md#gemma-scope-2-capture-hook-caveat)
for the full measurement.

The shipped configs resolve this by naming the capture location outright:

```yaml
capture_hook_name: blocks.{layer}.ln2.hook_out
```

Four configs set it and five inherit it through `EXTENDS`, so all nine are covered and a run against
any of them captures the right tensor without further arguments. `{layer}` is substituted per layer,
and the value is a `TransformerBridge` name, which is why these configs also set
`model_wrapper: bridge`.

**Confirm it from the run log, because nothing else can tell you.** The runner announces the decision
once per run:

```
Capturing at 'blocks.5.ln2.hook_out' (explicitly configured) rather than the SAE's declared
hook name 'blocks.5.hook_mlp_in'. ...
```

Grep the log for `Capturing at` and check the hook it names. This matters more than it looks:
compatibility mode leaves **both** hooks live on the model at once, and a generated corpus carries no
trace of which one was read. **A corpus generated with both hooks live and no log evidence of which
was used is indistinguishable from a defective one by inspection of the artifact**, which is the
property that let this go unnoticed through a year of parity testing. Parity compares two legs that
resolve the hook the same way, so it cannot see a difference that is upstream of both.

For these transcoders, **a run log with no `Capturing at` line at all captured at the declared hook**,
because the runner prints only when the capture location differs from the declared one. Scope that
reading to this family: for an SAE whose declared hook already names the right tensor, silence is
correct and expected.

> **Do not "fix" the `hook_point` label to match.** The config carries both, eleven lines apart and
> deliberately disagreeing:
>
> ```yaml
> hook_point: hook_mlp_in                       # the Neuronpedia source LABEL
> capture_hook_name: blocks.{layer}.ln2.hook_out  # where activations are actually read
> ```
>
> `hook_point` is the Neuronpedia source label, drawn from a fixed vocabulary with no term for this
> tensor. Editing it changes the label and not the capture, which yields a corpus that reads correct
> and captures wrong: strictly worse than a mismatch you can see. A label agreeing with the capture is
> also exactly what the defective corpora show, so agreement is not evidence of anything.

### The acceptance check, which needs no baseline at all

The log line above tells you what a run *intended*. To check what it *produced*, compare the corpus's
implied L0 against the number the SAE declares for itself. SAELens carries that per `sae_id` in its own
directory:

```yaml
# sae_lens/pretrained_saes.yaml, gemma-scope-2-1b-it-transcoders-all
- id: layer_5_width_16k_l0_small_affine
  l0: 15
```

The release name is an independent sanity check on the same quantity: `l0_small` against `l0_big`.

Measured across all 52 layers of the two corrected corpora, against a declared `l0` of 15:

```
                        min            median          max
monology       12.8 (0.85x)     17.9 (1.19x)   31.5 (2.10x)
rte            10.1 (0.67x)     17.1 (1.14x)   40.2 (2.68x)
the corpus this replaced        ~451 (30.1x)
```

**A correct run lands in the same neighbourhood as the declared value; a run captured at the wrong
tensor is an order of magnitude above it.** That is the whole signal, and it is large enough that no
threshold is needed to see it: the corrected spread tops out below 3x while the defective corpus sits
at 30x, a full order of magnitude clear of anything measured here.

Two cautions on reading the spread, both learned from it rather than assumed. It is **not** monotonic
in depth: layer 6 (16.3) sits below layer 5 (18.3), so a dip at one layer is not evidence of anything.
And the range is wider than a small sample suggests, 0.67x to 2.68x rather than the 0.85x to 1.45x
that the first three layers profiled implied, so **a single layer landing near 2.5x is ordinary and a
gate set tightly around early observations would have failed on real data.**

**This is the only check here that is absolute.** Everything else compares against something, and every
comparison we had was blind to this defect for one of two structurally different reasons:

- **Parity is blind because the defect is common-mode.** Hook resolution happens in the runner's
  constructor, above the legacy/columnar split, so both legs of any comparison capture at the same hook
  by construction. A differential check cannot see an error shared by both of its legs.
- **Throughput is blind because it measures a quantity the defect does not touch.** In the two-layer
  pilot, layers whose stored output differed by 27% finished within 7 ms of each other: wall time is
  dominated by forward passes, which the hook location does not change. Generation throughput remains a
  useful performance-regression guard and is not evidence about capture location.

Implied L0 escapes both, because it is absolute *and* it measures the stored artifact. A reader holding
one corpus, with no reference corpus and no prior run, can falsify it in a single query.

**Do not read the row counts as a substitute for it.** A corrected deep layer and a defective shallow
one are close or identical on every gross statistic: corrected layer 25 stores 522,029 rows at 47.1%
nonzero with a median max of 0.00, against defective layer 0 at 483,013 rows, 34.2% nonzero, and a
median max of 0.0000. Deep layers are legitimately sparser. Those figures mean something only with the
layer held fixed and the hook varied.

### Before a direct pipeline run: check the tree yourself

`run_dashboard_benchmark_suite.py` refuses to package from dirty repositories unless you pass
`--allow-dirty`. `launch_neuronpedia_dashboard_pipeline.py` has no such guard, so a direct run, which
is what the quickstart above describes, will start against whatever is in your working tree.

```bash
git -C <interpretune> status -sb
grep -c capture_hook_name src/interpretune/utils/neuronpedia_dashboard_pipeline.py   # expect > 0
```

**If that count is zero the tree lacks the capture-hook support regardless of which branch it reports
being on.** A branch name describes what was checked out, not what is in the files now.

---

## Related documentation

- [Neuronpedia dashboard pipeline](neuronpedia_dashboard_pipeline.md) — generation, import, and
  the [pretokenization CLI](neuronpedia_dashboard_pipeline.md#pretokenize-dashboard-datasets)
- [Developer multi-repo setup](developer_multi_repo_setup.md) — the environment-variable
  contract and build steps
- [Roadmap](../roadmap.md) — Wave 1 / Wave 2 framing
