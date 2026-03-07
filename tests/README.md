# tests/README.md

## CI Environment Variables

- **IT_CI_LOG_LEVEL**: Defaults to `INFO` at the repository level. You can override this variable at the workflow or step levels in GitHub Actions to enable more detailed logging (e.g., set to `DEBUG` for verbose output).

- **CI_RESOURCE_MONITOR**: Defaults to `0` at the repository level. You can override this variable at the workflow or step levels to enable basic CI resource logging. This is useful for debugging resource exhaustion issues on default GitHub Actions runners.

For more details, see the main CI workflow configuration in `.github/workflows/ci_test-full.yml`.

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

## Memory Management for NNsight Tests

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

### Known Bug: Class-Level Standalone Marks Are Silently Ignored

`pytest_collection_modifyitems` in `tests/conftest.py` uses `item.own_markers`, which only contains
markers directly on the test *function* — not those inherited from a parent class. Class-level
`@RunIf(standalone=True)` decorators are invisible to the standalone collection filter, causing those
tests to be silently excluded from standalone runs.

**Fix (TODO):** Change `item.own_markers` → `item.iter_markers()` (or equivalent) in
`pytest_collection_modifyitems`. Until this is fixed, **always apply standalone marks at the
individual test method level, not at the class level**.
