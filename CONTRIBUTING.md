# Contributing to Interpretune

> **Note — pre‑MVP:** Interpretune is currently in pre‑MVP stage. We expect to reach an MVP soon and begin accepting external PRs. If you're preparing contributions, please open issues or draft PRs now; maintainers will review them as we transition to the public MVP.

Thank you for contributing to Interpretune.

## Import-time policy

Policy summary
- The top-level package import (`import interpretune`) should only import small, essential core modules and must not eagerly import heavy adapter/extension packages (for example: `transformer_lens`, `lightning`, `neuronpedia`, `circuit_tracer`, etc.) that would otherwise dominate cold startup time.

How to follow this policy
1. Do not add top-level imports of heavy third-party adapter/extension packages inside `src/interpretune/__init__.py` or other package-level modules.
2. Prefer centralized helpers for minimal registration and deferred import logic. This repository includes `src/interpretune/adapters/_light_register.py` as a small, centralized place for minimal, low-overhead registration logic. If you need more general deferred import behavior, implement a single deferred-import helper (import-proxy) rather than sprinkling localized imports across many files.


Profiling and diagnosing import-time issues
- We have profiling and fixture-analysis tooling to make it easy to diagnose testing hotspots (e.g. including import-time analysis):
  - `tests/PROFILING.md` documents capturing CPU profiles with `py-spy` and viewing results in `speedscope`.
  - `scripts/speedscope_top_packages.py` helps summarize CPU time by package from speedscope JSON files.
  - `tests/dynamic_fixture_benchmark.py` enumerates pytest fixtures and can be used to micro-benchmark fixture setup costs.
- Use these tools if you think an import is unexpectedly expensive. A typical workflow is:
  1. Reproduce the import path or fixture that looks slow.
  2. Run `py-spy` to capture a short CPU profile while the import runs.
  3. Open the resulting speedscope JSON (or use the helper script) to find the heavy package call stacks.

- We also provide an automated test that guards against accidental eager imports and enforces a fast package import time: `tests/core/test_import_time_and_adapters.py`.

Recommended process for larger deferred-import work
1. Prototype a centralized deferred-importer (import-proxy) in a feature branch.
2. Add unit and integration tests that confirm both lazy and eager code paths behave as expected.
3. Run fixture benchmarks and a py-spy capture to ensure the change actually improves import time and does not introduce ordering regressions.
4. Keep the implementation small and well-documented; prefer explicitness over surprising implicit side-effects.

Contact
- If you're unsure whether an import should be deferred, or how to centralize it safely, open an issue or a draft PR and tag a maintainer for guidance.

Thanks for helping keep Interpretune fast and maintainable!
