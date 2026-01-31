# tests/README.md

## CI Environment Variables

- **IT_CI_LOG_LEVEL**: Defaults to `INFO` at the repository level. You can override this variable at the workflow or step levels in GitHub Actions to enable more detailed logging (e.g., set to `DEBUG` for verbose output).

- **CI_RESOURCE_MONITOR**: Defaults to `0` at the repository level. You can override this variable at the workflow or step levels to enable basic CI resource logging. This is useful for debugging resource exhaustion issues on default GitHub Actions runners.

For more details, see the main CI workflow configuration in `.github/workflows/ci_test-full.yml`.

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
