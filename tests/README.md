# tests/README.md

## CI Environment Variables

- **IT_CI_LOG_LEVEL**: Defaults to `INFO` at the repository level. You can override this variable at the workflow or step levels in GitHub Actions to enable more detailed logging (e.g., set to `DEBUG` for verbose output).

- **CI_RESOURCE_MONITOR**: Defaults to `0` at the repository level. You can override this variable at the workflow or step levels to enable basic CI resource logging. This is useful for debugging resource exhaustion issues on default GitHub Actions runners.

- **IT_RESOURCE_DEBUG**: Canonical opt-in flag for resource diagnostics. When set to `1`, shell helpers, pytest hooks,
  fixture factories, analysis helpers, and serialization helpers all emit structured resource lines. When CUDA is
  available, those same structured lines also include per-GPU allocated/reserved/peak VRAM fields so the local
  harness can append a GPU summary table to the final coverage log.

Representative output from `/tmp/gen_it_coverage_it_latest_20260317150318.log`:

```text
[shell_resource_debug] coverage:bootstrap:it_latest: context=shell rss_gb=0.01 vms_gb=0.02 cuda_available=true cuda_device_count=2 cuda_gpu0_total_gb=23.52 cuda_gpu0_current_allocated_gb=0.00 cuda_gpu0_current_reserved_gb=0.00 cuda_gpu0_peak_allocated_gb=0.00 cuda_gpu0_peak_reserved_gb=0.00 cuda_gpu1_total_gb=7.60 cuda_gpu1_current_allocated_gb=0.00 cuda_gpu1_current_reserved_gb=0.00 cuda_gpu1_peak_allocated_gb=0.00 cuda_gpu1_peak_reserved_gb=0.00
[test_resource_debug] test:start:tests/core/test_adapters_circuit_tracer.py::TestCircuitTracerConfig::test_default_backend_is_transformerlens: context=test nodeid=tests/core/test_adapters_circuit_tracer.py::TestCircuitTracerConfig::test_default_backend_is_transformerlens lifecycle=start rss_gb=1.06 vms_gb=11.67 cuda_available=false cuda_device_count=0
[fixture_resource_debug] it_session_cfg_fixture:ns_gpt2:setup:start: context=fixture kind=it_session_cfg_fixture key=ns_gpt2 scope=class lifecycle=setup_start rss_gb=1.06 vms_gb=11.67 cuda_available=false cuda_device_count=0 path0=/tmp/pytest-of-speediedan/pytest-1671/ns_gpt2_it_session_cfg_fixture0 used_gb0=377.22 free_gb0=491.54
```

The log prefixes remain context-specific (`shell_resource_debug`, `fixture_resource_debug`, `test_resource_debug`,
`analysis_resource_debug`, `op_serialization_resource_debug`), but those prefixes are now controlled by the same
environment variable rather than by separate per-context flags.

For more details, see the main CI workflow configuration in `.github/workflows/ci_test-full.yml`.

## Local CI Reproduction Knobs

## Local Coverage Harness Split

The local `scripts/gen_it_coverage.sh` harness now mirrors the Azure GPU pipeline's phase split:

1. Base pytest runs with `CUDA_VISIBLE_DEVICES=''` so the normal suite remains CPU-only and avoids GPU OOM churn.
2. A second `IT_RUN_CUDA_TESTS=1` pytest pass re-enables regular CUDA/bf16-marked tests that should contribute to
   coverage but do not require standalone isolation.
3. `tests/special_tests.sh --mark_type=standalone` handles the standalone-only slice.
4. `tests/special_tests.sh --mark_type=profile_ci` handles the CI profiling slice.

This split is deliberate: regular CUDA tests still append to the same coverage file, but they run after the CPU-only
baseline has released its fixture state.

Example local run:

```bash
./scripts/manage_standalone_processes.sh --use-nohup \
  ./scripts/gen_it_coverage.sh \
  --repo-home=${PWD} \
  --target-env-name=it_latest \
  --venv-dir=/mnt/cache/${USER}/.venvs \
  --no-rebuild-base \
  --allow-failures \
  --no-reruns
```

Add `--resource-debug` when you want the harness to export `IT_RESOURCE_DEBUG=1` automatically.

Latest validated run from `/tmp/gen_it_coverage_it_latest_20260317150318.log`:

- CPU-only base pytest phase: `979 passed`, `90 skipped`
- CUDA-marked pytest phase: `42 passed`
- standalone special tests: `1 passed`
- `profile_ci` special tests: `6 passed`
- total coverage: `88%`
- GPU 0 peak usage from the generated summary: `3.21 GB` allocated and `3.32 GB` reserved out of `23.52 GB`
  total VRAM, or about `13.6%` / `14.1%`
- GPU 1 remained unused during this run (`0.00 GB` allocated / reserved out of `7.60 GB` total VRAM)
- Largest setup-time reserved-VRAM deltas came from `core_gpt2_peft` (`0.49 GB`) and `core_gpt2_peft_seq` (`0.16 GB`)
- Heaviest peak-reserved fixture observations were `l_tl_bridge_gpt2`, `l_tl_ht_gpt2_sched`, and
  `l_tl_ht_gpt2`, each peaking at about `3.32 GB` reserved VRAM

The test suite now exposes a small set of environment variables specifically for reproducing
 GitHub Actions memory behavior locally:

- `IT_MOCK_RUNNER_RAM_GB`: Overrides `tests.runif.get_runner_ram_gb()` for the current process.
  This forces the same low-memory fixture-scope and cleanup paths used in CI without requiring a
  physically constrained machine.
- `IT_NNSIGHT_CONFIGS_PER_PASS`: Overrides the default `NNsightModelBackend` multi-invoke batch size.
  This is useful when probing resource intensive ops like `model_ablation` / `logit_diffs_attr_ablation` memory tradeoff.
- `IT_RESOURCE_DEBUG`: Enables all structured shell / test / fixture / analysis / serialization resource logs.

Example low-memory reproduction commands:

```bash
# Reproduce the heavy attr-ablation parity path with CI-like low-memory behavior
CUDA_VISIBLE_DEVICES='' IT_MOCK_RUNNER_RAM_GB=32 \
  python -m pytest tests/core/test_model_backend_parity.py::TestLogitDiffsAttrAblationBackendParity::test_logit_diffs_match -q

# Add serialization resource snapshots around the logit_diffs serialization test
CUDA_VISIBLE_DEVICES='' IT_MOCK_RUNNER_RAM_GB=16 IT_RESOURCE_DEBUG=1 \
  python -m pytest tests/core/test_analysis_ops_definitions.py::TestAnalysisOperationsImplementations::test_op_serialization[logit_diffs] -s -q

# Manually compare different NNsight multi-invoke batch sizes
CUDA_VISIBLE_DEVICES='' IT_MOCK_RUNNER_RAM_GB=32 IT_NNSIGHT_CONFIGS_PER_PASS=2 \
  python -m pytest tests/core/test_model_backend_parity.py::TestLogitDiffsAttrAblationBackendParity::test_logit_diffs_match -q
```

## HuggingFace Token Configuration

Some tests require a HuggingFace authentication token to access Hub features (e.g., uploading/downloading
analysis op collections). These tokens are provided via GitHub repository secrets and mapped to environment
variables in the CI workflow.

### Secrets and Environment Variables

| GitHub Secret | CI Env Variable | Purpose |
|---|---|---|
| `HF_GATED_PUBLIC_REPO_AUTH_KEY` | `HF_GATED_PUBLIC_REPO_AUTH_KEY` | Access to gated HuggingFace models (e.g., Gemma, Llama) |
| `HF_TRIVIAL_OP_REPO_EXAMPLE_AUTH_KEY` | `HF_TRIVIAL_OP_REPO_EXAMPLE_AUTH_KEY` | Read/write access for op collection example notebook tests |

These are configured in `.github/workflows/ci_test-full.yml` under the `env:` section of the test job.

### Affected Tests

- **`test_op_collection_notebooks`** (`tests/examples/test_notebooks.py`): Executes the
  `op_collection_example.ipynb` published notebook via papermill. The notebook uploads/downloads
  analysis op collections to a private HuggingFace repo. Requires `HF_TRIVIAL_OP_REPO_EXAMPLE_AUTH_KEY`
  or `HF_TOKEN` to be set. **Skipped automatically** when neither token is available.

### Token Resolution Order (in the notebook)

The notebook (`op_collection_example.ipynb`, Cell 10) resolves the HuggingFace token in this order:

1. `os.environ["HF_TRIVIAL_OP_REPO_EXAMPLE_AUTH_KEY"]` — CI-specific secret
2. `os.environ["HF_TOKEN"]` — Standard HuggingFace token
3. `notebook_login()` — Interactive login prompt (only works in interactive sessions)

### Local Development

To run the op collection notebook test locally:

```bash
# Option 1: Use the dedicated env variable
export HF_TRIVIAL_OP_REPO_EXAMPLE_AUTH_KEY="hf_your_token_here"

# Option 2: Use the standard HF_TOKEN
export HF_TOKEN="hf_your_token_here"

# Then run the test
python -m pytest tests/examples/test_notebooks.py::test_op_collection_notebooks -v
```

The token requires **write** permissions since the notebook creates and uploads to a private HuggingFace
repository (`{username}/trivial_op_repo`).

### Adding a New Secret

To add or update a HuggingFace token secret:

1. Go to the repository's **Settings → Secrets and variables → Actions**
2. Add a new repository secret with the appropriate name
3. Map it in `.github/workflows/ci_test-full.yml` under the `env:` section
4. In the test file, add a `@pytest.mark.skipif` guard so the test skips gracefully when the
   token is unavailable

## Concept-Direction Notebook Config Notes

The concept-direction notebook configs under `tests/concept_direction_approach_parity/configs/` now expose two
intervention-related capabilities that are useful for local experiment reproduction:

- `ANALYSIS.direct_projection.interventions` mirrors `model_fwd_intervention`'s explicit intervention mapping. The
  notebook wrapper injects the runtime `concept_direction` as the `intervention_tensor` when omitted, so configs can
  target non-default hook points like `blocks.0.hook_in` with `project` or `add` modes.
- `ANALYSIS.constrained_feature_selection` entries may include an `activation_value` override alongside the feature
  ref. Missing constrained features are no longer dropped automatically: `extract_top_features` synthesizes candidate
  rows using same-layer or global activation baselines so downstream `feature_intervention_forward` calls can still
  run.

Example constrained feature-selection entry with an explicit activation override:

```yaml
ANALYSIS:
  constrained_feature_selection:
    - ref: [gemma-3-4b-it, gemmascope-2-transcoder-16k, 22, 2975]
      activation_value: 4.5
```

## Test Reruns for Transient Failures

All pytest invocations in our CI and local coverage scripts include `--reruns 2 --reruns-delay 5` by default. This addresses transient `httpx` read timeouts that can occur with HuggingFace `transformers` v5 during model/tokenizer downloads. Tests that fail due to these transient network issues will automatically retry up to 2 times with a 5-second delay between attempts.

The `pytest-rerunfailures` plugin is included in our test dependencies (`pyproject.toml` test group).

**Configurable Rerun Options:**

Local coverage and test scripts (`gen_it_coverage.sh`, `special_tests.sh`, `analyze_test_coverage.py`) support configurable rerun behavior:

- `--no-reruns`: Disable test reruns entirely
- `--reruns=N`: Set number of reruns (default: 2)
- `--reruns-delay=N`: Set delay between reruns in seconds (default: 5)

Example: Run special tests without reruns for faster debugging:
```bash
./tests/special_tests.sh --mark_type=standalone --no-reruns
```

CI scripts use hard-coded rerun values for consistent behavior.

## Profiling & Analysis

We include a couple of developer-facing documents under `tests/` for profiling and fixture analysis:

- `tests/FIXTURE_ANALYSIS.md` — Dynamic fixture benchmark & analysis report (recent export).
- `tests/PROFILING.md` — How-to for generating pytest flamegraphs, speedscope captures, and using `scripts/speedscope_top_packages.py` to parse and sample top stacks by package.

These are intended for maintainers and contributors performing performance investigations and are particularly useful for managing the complexity of our numerous package integrations.

## Benchmark Tests

End-to-end experiment benchmarks live in `tests/benchmarks/`. These validate that experiment configs
produce expected accuracy and are gated by the `benchmark` RunIf mark (`IT_RUN_BENCHMARK_TESTS=1`).

See [`tests/benchmarks/README.md`](benchmarks/README.md) for full usage, registry format, how to add new experiments,
and debug diagnostics.

- `tests/upstream_parity/UPSTREAM_CT_PARITY_DEBUG.md` — Manual upstream circuit-tracer semantic-intervention
  sanity-check workflow,
  including the one-off extractor script and the current three-way reference table.

## Memory Management for NNsight Tests

## Semantic Intervention Parity Pattern

The semantic concept-direction intervention parity coverage now uses a three-anchor pattern for the
NNsight circuit-tracer backend:

1. Upstream reference: `circuit-tracer/tests/test_tutorial_notebook_backends.py::test_attribution_targets_semantic_intervention`
  runs TransformerLens and NNsight serially under explicit cleanup, verifies that both backends widen
  the Austin-vs-Dallas logit gap after semantic feature intervention, and accepts a backend-to-backend
  post-gap difference below `0.5`.
2. Interpretune native CT path: `tests/core/test_analysis_backend_parity.py` rebuilds the semantic
  `CustomTarget` directly against `CircuitTracerNNsightGemma2`, runs `generate_attribution_graph(...)`,
  extracts top features from node influence, and applies native `feature_intervention(...)` values using
  the same `10.0 * activation` scaling used upstream.
3. Interpretune analysis-op path: the same test creates a second fresh NNsight session, computes
  `concept_direction -> compute_attribution_graph -> graph_node_influence -> extract_top_features ->
  feature_intervention_forward`, and compares those results against the native CT baseline.

When the regular parity test fails in a way that suggests an upstream package drift rather than a local regression,
use the manual extractor documented in [tests/upstream_parity/UPSTREAM_CT_PARITY_DEBUG.md](upstream_parity/UPSTREAM_CT_PARITY_DEBUG.md).
That script replays the actual upstream semantic-intervention logic and records current upstream CT NNsight,
Interpretune native CT, and Interpretune analysis-op values into a single JSON payload for sanity checking.

The current Interpretune parity tolerances are intentionally stricter than the upstream TL-vs-NNsight
comparison because both local paths execute against the same NNsight backend and should agree nearly
exactly:

- Concept direction: cosine similarity must exceed `0.999`.
- Top feature identities: exact tuple equality.
- Top-feature activation values: `torch.testing.assert_close(..., rtol=1e-6, atol=1e-6)`.
- Intervention feature ids and positions: exact equality.
- Intervention values: `torch.testing.assert_close(..., rtol=1e-6, atol=1e-6)`.
- Pre- and post-intervention final logits: `torch.testing.assert_close(..., rtol=1e-6, atol=1e-6)`.
- Semantic intervention effect: both paths must widen the Austin-vs-Dallas gap.

Implementation detail: the Interpretune test no longer shells out to subprocess helpers. It now uses
fresh in-process `CircuitTracerNNsightGemma2` sessions separated by `serial_test_cleanup(...)` from
`tests/analysis_resource_utils.py`, which performs best-effort NNsight tracing-state cleanup plus GC /
CUDA cache release between the native and analysis-op phases.

Some NNsight-related parity tests (e.g., `test_parity_ns`, `TestLogitDiffsAttrAblationBackendParity`)
are memory-intensive. Each parametrized test variant loads and retains NNsight model state, and on
memory-constrained CI runners (e.g., Ubuntu with ~7 GB available RAM) the cumulative footprint can
trigger OOM errors.

### Strategy: `cleanup_memory` at the Test Method Level

Rather than isolating these tests with `@RunIf(standalone=True)` — which removes them from regular
CI runs and reduces cross-platform coverage signal — we apply `@pytest.mark.usefixtures("cleanup_memory")`
at the **individual test method** level.

The `cleanup_memory` fixture (defined in `tests/conftest.py`) yields and then calls `gc.collect()` to
free dangling references after each test, limiting peak RSS growth across parametrized variants.

**Fixture scope narrowing:** In addition to `cleanup_memory`, a number of analysis-oriented fixtures
(SAE backend parity analysis sessions in `FIXTURE_CFGS`) have been narrowed from `session`/`class`
scope to `function` scope. This further reduces peak resource usage at the cost of fixture reuse —
each test method gets a fresh fixture instance rather than sharing one across the test session.
This tradeoff intentionally prioritises cross-platform validation in the normal CI suite over
runtime efficiency; the self-hosted standalone runner is reserved for tests that require subprocess
isolation for correctness reasons (e.g., `sys.settrace` / coverage conflicts).

**Tradeoff summary:** Using `cleanup_memory` at the function level means some fixtures with wider
scope are still shared across test methods — fixture teardown is not accelerated by the gc sweep
alone. The combination of `cleanup_memory` + `function`-scoped fixtures provides the strongest
memory containment short of standalone isolation.

**When standalone isolation is still warranted:** Some tests genuinely require subprocess isolation,
e.g., when a library uses `sys.settrace()` (which interferes with pytest-cov). The current canonical
example is `test_nnsight_trace_context` in `tests/core/test_adapters_nnsight.py`, which must stay
`@RunIf(standalone=True)`.

### Shared Low-Memory Extraction Helpers

Heavy analysis-parity tests should prefer the shared helpers in `tests/analysis_resource_utils.py`
over ad hoc lightweight test dataclasses. `ExtractedAnalysisStore` and
`extract_analysis_store_fields(...)` allow tests to copy only the `AnalysisStore` fields they need
while still preserving `by_latent_model(...)` for downstream helpers like `compute_correct()`.

When a test class needs to reuse a lightweight projection of the same heavyweight fixture across
multiple test methods, prefer the generalized cached-payload helpers:

- `AnalysisFixtureSpec` lets a test class declare, at the class level, which fixture aliases it
  needs, whether each alias should materialize a full result or a lightweight payload, which
  `AnalysisStore` fields should be copied, and whether dataset metadata should be captured.
- `build_analysis_fixture_payload_extractor(...)` builds a reusable extractor that can bundle
-  selected `AnalysisStore` fields, dataset metadata, optional full-result copies, and custom
  payloads.
- `AnalysisExtractionMixin.extract_values(...)` now falls back to those declarative fixture specs,
  so simple parity classes only need class-level configuration.
- `AnalysisExtractionMixin.extract_field_store(...)` and
  `AnalysisExtractionMixin.extract_dataset_metadata(...)` provide lazy, per-alias access to cached
  payloads without re-introducing class-local fixture maps and helper wrappers.
- `AnalysisExtractionMixin.extract_cached_fixture_data(...)` remains available for truly custom
  cases, but should no longer be the default pattern for new parity classes.
- `ExtractedFixturePayload` provides a minimal attribute-style wrapper for the cached payload.

This pattern is now used by `TestBackendParityEdgeCases` and the simpler Bridge/NNsight parity
classes to keep GitHub-hosted runner memory usage bounded while removing parity-local extraction
bookkeeping.

The current low-memory threshold in `analysis_resource_utils.py` is `32 GB`, which intentionally
forces GitHub-hosted Ubuntu and Windows runners down the low-memory fixture path. This is more
conservative than the raw hosted-runner memory spec and is intended to leave margin for pytest,
coverage, dataset serialization, and worker-thread overhead.

### Current Local Memory Baselines

Using the local reproduction knobs above, the NNsight attr-ablation parity path has been measured on
CPU-only runs with mocked CI-like RAM constraints:

- `IT_NNSIGHT_CONFIGS_PER_PASS=8`: peak RSS about `11.28 GB`
- `IT_NNSIGHT_CONFIGS_PER_PASS=4`: peak RSS about `10.67 GB`
- `IT_NNSIGHT_CONFIGS_PER_PASS=2`: peak RSS about `10.43 GB`

The adaptive default now uses `4` on CPU Linux and `2` on CPU Windows, while retaining `32` when
CUDA is available. Under `CUDA_VISIBLE_DEVICES=''` and `IT_MOCK_RUNNER_RAM_GB=32`, the attr-ablation
parity test completed locally with peak RSS about `10.81 GB`, which is below the `16 GB` memory on
standard GitHub-hosted Ubuntu runners.

### Successful Linux Monitor Baseline

The Linux `ci_resource_monitor` artifact from successful run `22870543454` is useful as a coarse
runner-health baseline, but not as a precise substitute for inline RSS markers:

- The hosted Ubuntu runner reported roughly `15 GiB` total memory at startup.
- The sampled `python -m pytest` process reached about `94.3%` of system memory during the run.
- The artifact reports top-process `%MEM` snapshots and whole-disk usage, not per-test RSS. Keep
  using `log_resource_snapshot(...)` for precise analysis-test instrumentation.

### Flaky Ubuntu Hosted-Runner Shutdowns

We have also seen a distinct Linux CI failure mode that does not present as a normal pytest failure.
On GitHub Actions run `22924129626`, job `66530036090` (`cpu (ubuntu-22.04, 3.13)`), the pytest
coverage step was marked failed even though the raw job log showed normal test progress and no
assertion failure. The final lines were:

- `##[error]The runner has received a shutdown signal.`
- `##[error]The operation was canceled.`

Key characteristics of this failure mode:

- artifact upload steps are usually skipped, so the raw job log is the primary source of truth
- GitHub may report the step as `failure` even when the underlying cause is hosted-runner loss of
  communication rather than a deterministic test regression
- the surviving raw log may end mid-test without a pytest summary or traceback
- other OS jobs, or the Azure self-hosted GPU workflow, may pass on the same commit

Recommended triage flow:

```bash
# Inspect the failed job summary first
gh run view --repo speediedan/interpretune --job <job_id>

# Download the raw job log when artifacts are missing or upload steps were skipped
gh api "repos/speediedan/interpretune/actions/jobs/<job_id>/logs" > /tmp/ci_job_<job_id>.log

# Search for runner-loss markers
grep -nE "shutdown signal|lost communication|operation was canceled|exit code 143" /tmp/ci_job_<job_id>.log

# If the log survived, inspect the last emitted lines
tail -80 /tmp/ci_job_<job_id>.log
```

Interpretation guidance:

- if the last surviving log lines show normal test execution and then a runner shutdown/cancel,
  treat the event as probable runner instability first
- if the rerun on the same commit fails again with a real traceback, switch from infrastructure
  triage to a code/test fix
- compare against a successful Linux `ci_resource_monitor` artifact only as a coarse baseline;
  continue to rely on inline `IT_RESOURCE_DEBUG=1` snapshots for precise per-test memory analysis
- for heavy analysis cases, combine the resource flags with `IT_MOCK_RUNNER_RAM_GB` and, when
  relevant, `IT_NNSIGHT_CONFIGS_PER_PASS` to reproduce GitHub-hosted memory conditions locally

### Known Bug: Class-Level Standalone Marks Are Silently Ignored

`pytest_collection_modifyitems` in `tests/conftest.py` uses `item.own_markers`, which only contains
markers directly on the test *function* — not those inherited from a parent class. Class-level
`@RunIf(standalone=True)` decorators are invisible to the standalone collection filter, causing those
tests to be silently excluded from standalone runs.

**Fix (TODO):** Change `item.own_markers` → `item.iter_markers()` (or equivalent) in
`pytest_collection_modifyitems`. Until this is fixed, **always apply standalone marks at the
individual test method level, not at the class level**.
