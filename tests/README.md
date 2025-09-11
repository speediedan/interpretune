# tests/README.md

## CI Environment Variables

- **IT_CI_LOG_LEVEL**: Defaults to `INFO` at the repository level. You can override this variable at the workflow or step levels in GitHub Actions to enable more detailed logging (e.g., set to `DEBUG` for verbose output).

- **CI_RESOURCE_MONITOR**: Defaults to `0` at the repository level. You can override this variable at the workflow or step levels to enable basic CI resource logging. This is useful for debugging resource exhaustion issues on default GitHub Actions runners.

For more details, see the main CI workflow configuration in `.github/workflows/ci_test-full.yml`.

## Profiling & Analysis

We include a couple of developer-facing documents under `tests/` for profiling and fixture analysis:

- `tests/FIXTURE_ANALYSIS.md` — Dynamic fixture benchmark & analysis report (recent export).
- `tests/PROFILING.md` — How-to for generating pytest flamegraphs, speedscope captures, and using `scripts/speedscope_top_packages.py` to parse and sample top stacks by package.

These are intended for maintainers and contributors performing performance investigations and are particularly useful for managing the complexity of our numerous package integrations.
