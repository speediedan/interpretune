# Neuronpedia Dashboard Pipeline

This note documents the supported Interpretune path for generating SAEDashboard outputs, converting them to Neuronpedia export bundles, and importing them into a local Neuronpedia database.

## Purpose

Use `interpretune.utils.neuronpedia_dashboard_pipeline` when you want one file-backed, resumable pipeline that:

1. runs SAEDashboard generation for a layer range
2. converts each completed layer into Neuronpedia export format
3. imports the converted bundle into a local Neuronpedia Postgres database
4. emits enough diagnostics to distinguish stalls, kills, and conversion/import seams

This replaces the earlier ad hoc shell resume flow that depended on `tee` and separate manual conversion/import steps.

## Required environment

The current working setup on this host is:

```bash
PYTHON=/mnt/cache/speediedan/.venvs/it_latest/bin/python
HF_HOME=/mnt/cache_extended/speediedan/.cache/huggingface
HF_DATASETS_CACHE=/mnt/cache_extended/speediedan/.cache/huggingface/datasets
HF_HUB_CACHE=/mnt/cache_extended/speediedan/.cache/huggingface/hub
IT_NP_CACHE=/mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia
```

For local DB imports, the pipeline resolves the DB URL from the Neuronpedia local env files, but the explicit localhost URL currently in use is:

```bash
postgres://postgres:postgres@127.0.0.1:5433/postgres
```

## Standard launch example

This is the current Target B pattern for `gemma-3-1b-it` `gemmascope-2-transcoder-16k`:

```bash
RUN_ROOT="/mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia/dashboard_runs"
RUN_DIR="${RUN_ROOT}/gemma-3-1b-it_gemmascope-2-transcoder-16k"
LAUNCH_LOG="${RUN_DIR}/run.resume-24-25.launch.log"

mkdir -p "${RUN_DIR}"

nohup /mnt/cache/speediedan/.venvs/it_latest/bin/python -m interpretune.utils.neuronpedia_dashboard_pipeline \
  --model-name gemma-3-1b-it \
  --model-layers 26 \
  --sae-set gemma-scope-2-1b-it-transcoders-all \
  --neuronpedia-source-set-id gemmascope-2-transcoder-16k \
  --neuronpedia-source-set-description 'Transcoder - 16k' \
  --creator-name 'Google DeepMind' \
  --release-id gemma-scope-2 \
  --release-title 'Gemma Scope 2' \
  --release-url https://huggingface.co/mwhanna/gemma-scope-2-1b-it \
  --hf-weights-repo-id mwhanna/gemma-scope-2-1b-it \
  --hf-weights-path-template 'transcoder_all/layer_{layer}_width_16k_l0_small_affine' \
  --hook-point hook_mlp_in \
  --prompts-huggingface-dataset-path monology/pile-uncopyrighted \
  --start-layer 24 \
  --end-layer 25 \
  --sae-path-template 'layer_{layer}_width_16k_l0_small_affine' \
  --python-executable /mnt/cache/speediedan/.venvs/it_latest/bin/python \
  --cuda-visible-devices 0 \
  --heartbeat-seconds 60 \
  --stall-timeout-seconds 1800 \
  --use-skip-transcoder \
  > "${LAUNCH_LOG}" 2>&1 &
```

## Model metadata guardrail

The local import path does more than features and explanations. `import_neuronpedia_export_bundle_local_db(...)` also imports
bundle-scoped metadata tables such as `model.jsonl` and `release.jsonl`. Those inserts use `ON CONFLICT DO NOTHING`, which
means a correct existing local `Model` row is preserved, but a missing row in a fresh local DB is seeded directly from the
bundle metadata.

That import seam was the real cause of the earlier broken local Gemma graph rows. Older Neuronpedia utility exports were
still emitting `layers: 0` and omitting `defaultGraphSourceSetName`, so importing one of those stale bundles could recreate
a `Model` row that broke `/graph` even though graph upload itself was fine.

Current supported fix:

1. the converter now requires `--model-layers` and writes both `defaultSourceSetName` and `defaultGraphSourceSetName`
2. the checked-in `gemma-3-1b-it` export examples under `neuronpedia_utils/exports/` have been refreshed to match
   `gemmascope-2-transcoder-16k`

If your local DB was already seeded from a pre-fix bundle, repair that `Model` row once in the live DB. After that, stick
to regenerated or refreshed bundles rather than re-importing older `model.jsonl` and `release.jsonl` artifacts.

## What the pipeline now handles automatically

### Nested dashboard leaf resolution

SAEDashboard layer directories do not always place `batch-*.json` files directly under `layer_<n>`. For the Gemma Target B run, the actual converter input for layer `23` was:

```text
.../layer_23/google/gemma-3-1b-it_gemma-scope-2-1b-it-transcoders-all_blocks.23.hook_mlp_in_16384
```

The converter script expects the directory that directly contains `batch-*.json`, not the layer root. The Interpretune pipeline now resolves that leaf directory automatically before conversion.

### Export-root recovery

After conversion, the pipeline resolves the expected export root from the real batch metadata and falls back across candidate export directories if the direct path is absent.

### Diagnostics

At every heartbeat, the pipeline logs:

1. output directory file count
2. output directory byte count
3. `ps` snapshot for the inner runner
4. `nvidia-smi` snapshot for GPU memory/process state

On stall or nonzero exit, it also logs a kernel snapshot via `dmesg`.

## Monitoring

To follow a live run, inspect the file-backed launch log:

```bash
tail -f /mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia/dashboard_runs/gemma-3-1b-it_gemmascope-2-transcoder-16k/run.resume-24-25.launch.log
```

Healthy heartbeat lines look like:

```text
INFO Heartbeat output_dir=... files=2 bytes=1639 elapsed_without_growth=60.1s pid=... ps=... gpu=...
```

If bytes are static for too long, use the logged `ps`, GPU, and kernel snapshots before assuming the job is stuck in Python logic.

## Local import validation

If you want to validate an already-converted bundle directly, use the Python helper with the explicit localhost Postgres URL:

```bash
cd /home/speediedan/repos/interpretune
source /mnt/cache/speediedan/.venvs/it_latest/bin/activate

python - <<'PY'
from pathlib import Path
from interpretune.utils import import_neuronpedia_export_bundle_local_db

bundle = Path(
    '/home/speediedan/repos/neuronpedia/utils/neuronpedia-utils/neuronpedia_utils/exports/'
    'gemma-3-1b-it/23-gemmascope-2-transcoder-16k'
)
summary = import_neuronpedia_export_bundle_local_db(
    bundle,
    local_db_url='postgres://postgres:postgres@127.0.0.1:5433/postgres',
)
print(summary)
PY
```

If you are validating a bundle outside this repo, inspect its `model.jsonl` and `release.jsonl` first. A pre-fix bundle can
still seed stale `Model` metadata into an otherwise fresh local DB.

## How notebook explanation backfill uses dashboard exports

The dashboard/import pipeline and the localhost concept-direction notebook now share the same export artifacts.

The explanation path now behaves like a layered cache hierarchy:

1. `Explanation` rows in the local Neuronpedia DB
2. feature-specific Interpretune cache under `IT_NP_CACHE/.../feature-activations/feature-<index>.jsonl.gz`
3. local Neuronpedia export batches under `<neuronpedia_repo>/.../exports/<model>/<layer-source-set>/activations/batch-*.jsonl.gz`
4. remote public activation-batch fallback downloaded into `IT_NP_CACHE/.../activations/batch-<n>.jsonl.gz`

That layering matters because the notebook uses different recovery steps depending on what already exists.

### Retrieval-only case: explanation already exists locally

If `check_local_explanation_coverage(...)` finds an `Explanation` row for the target `(modelId, layer, index)` tuple, the pipeline does not generate anything.

The path is:

1. notebook computes candidate feature refs
2. local DB coverage check returns `has_local_explanation=True`
3. generation is skipped for that feature
4. the webapp renders the stored `Explanation.description`

No export scan, feature-cache write, or remote download happens in this case.

### Local-source case: explanation missing, activations already available locally

If the explanation is missing but the activation source data is already available, the notebook stays entirely local.

If the feature cache already exists:

1. `load_cached_feature_activations(...)` reads `feature-<index>.jsonl.gz`
2. prompt construction and explanation generation run immediately
3. the new explanation is inserted into the local DB
4. the feature-specific cache file is deleted after successful insert

If only the local export batches exist:

1. `prepare_local_explanation_backfill(...)` scans `activations/batch-*.jsonl.gz`
2. it filters the rows for the exact feature index
3. it writes a feature-specific cache file under `feature-activations/`
4. explanation generation consumes that reduced cache
5. after successful insert, the feature-specific cache file is removed automatically

This is the preferred localhost flow because it reuses the exact export bundle produced by the dashboard pipeline.

### Remote fallback case: explanation missing and no local activations are present

If neither the local export bundle nor the feature cache has the needed activations, explanation generation falls back automatically to the public dataset endpoint.

Current implementation:

1. `load_cached_feature_activations(...)` checks local feature cache and local batch cache first
2. if neither exists, it derives candidate public batch URLs from `candidate_public_activation_batch_urls(...)`
3. `download_url_to_path(...)` downloads the batch automatically to:
  - `IT_NP_CACHE/v1/<model>/<layer-source-set>/activations/batch-<n>.jsonl.gz`
4. the downloaded batch is filtered to the requested feature rows
5. generation continues without any manual user action

Important detail:

- the current fallback remote base is `https://neuronpedia-datasets.s3.amazonaws.com`
- the downloaded file lands in the durable Interpretune batch cache, not in `/tmp`
- this is automatic inside explanation generation; the notebook user does not need a separate preload step

The relevant pieces are:

1. dashboard generation creates Neuronpedia export bundles under:
  - `/home/speediedan/repos/neuronpedia/utils/neuronpedia-utils/neuronpedia_utils/exports/<model>/<layer-source-set>`
2. each bundle includes `activations/batch-*.jsonl.gz`
3. the localhost concept-direction notebook calls `prepare_local_explanation_backfill(...)` before `ensure_local_feature_explanations(...)`
4. that helper searches the local export `activations/` batches for the exact feature index and writes a feature-specific Interpretune cache artifact under:
  - `IT_NP_CACHE/v1/<model>/<layer-source-set>/feature-activations/feature-<index>.jsonl.gz`
5. the explanation-generation helper then reads those cached rows to build/import explanations into the local DB

This is now the preferred local path because exported activation batch numbering is not guaranteed to match the older public-dataset batch-number heuristic. Searching the local export bundle by exact feature index is more reliable for localhost runs.

### Cleanup behavior after successful insert

The explanation pipeline now treats feature-specific cache files as transient staging artifacts.

After `generate_explanation_artifact(...)` successfully inserts the explanation into the local DB:

1. `cleanup_feature_activation_cache(...)` checks whether the activations came from `feature-activations/feature-<index>.jsonl.gz`
2. if so, it deletes that file and removes the empty parent directory when possible
3. batch caches under `IT_NP_CACHE/.../activations/batch-<n>.jsonl.gz` are kept so later features from the same batch do not trigger another download

That means:

- feature caches are disposable
- batch caches are reusable
- the dashboard export bundle remains the long-lived local source of truth

### Manual cache warm-up checklist

If a localhost notebook reports explanation cache misses:

1. verify the export bundle exists for the relevant layer/source-set
2. verify `activations/batch-*.jsonl.gz` is present in that bundle
3. rerun the notebook with `CHECK_LOCAL_EXPLANATION_COVERAGE=true` and `GENERATE_MISSING_LOCAL_EXPLANATIONS=true`
4. inspect the notebook’s `local_explanation_prefetch` JSON block
5. confirm the derived `feature-activations/feature-<index>.jsonl.gz` file exists under `IT_NP_CACHE`

For manual inspection on this host:

```bash
find /home/speediedan/repos/neuronpedia/utils/neuronpedia-utils/neuronpedia_utils/exports/gemma-3-1b-it/19-gemmascope-2-transcoder-16k/activations -maxdepth 1 -name 'batch-*.jsonl.gz' | head

ls -l /mnt/cache_extended/speediedan/.cache/huggingface/interpretune/neuronpedia/v1/gemma-3-1b-it/19-gemmascope-2-transcoder-16k/feature-activations/
```

If you need to force regeneration of previously inserted Copilot-generated explanations before a rerun, delete those rows from the local DB first and then rerun the localhost config:

```sql
DELETE FROM "Explanation"
WHERE "modelId" = 'gemma-3-1b-it'
  AND layer LIKE '%-gemmascope-2-transcoder-16k'
  AND notes LIKE '%interpretune-github-copilot-cli%';
```

## Current operational caution

The pipeline is currently running from the shared `it_latest` environment because the documented Neuronpedia-specific environment on this host is incomplete. That is acceptable for active recovery, but long-running dashboard jobs should eventually move into a dedicated environment with the Neuronpedia and SAEDashboard dependencies pinned independently from the main Interpretune dev environment.
