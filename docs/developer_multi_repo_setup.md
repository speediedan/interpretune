# Developer Guide: Unified Multi-Repo Editable Environment

Interpretune development frequently spans several tightly-coupled repositories (SAELens, SAEDashboard,
circuit-tracer, TransformerLens, nnsight). This guide documents the supported way to build a single
virtualenv where any subset of those dependencies is installed **editable from a local checkout**, layered
on top of interpretune's locked CI requirements. The builder is `scripts/build_it_env.sh`.

## Prerequisites

- **git** (with `git-lfs` available; the builder runs `git lfs install`)
- **uv** — a recent release (the environment is routinely built with uv >= 0.10; `UV_EXCLUDE`/`UV_OVERRIDE`
  support is required)
- **Python 3.13** (default; any `>= 3.10` interpreter can be selected with `--python-version`). uv-managed
  interpreters work out of the box.
- **bash >= 4.3** to run the builder itself
  - *Linux*: any modern distro bash qualifies.
  - *macOS*: the system `/bin/bash` is 3.2 and will be rejected with a clear error. Install a modern bash
    with `brew install bash`; the script resolves bash via `/usr/bin/env bash`, which picks up the Homebrew
    bash from `PATH`.
- **Torch backend**
  - *Linux + NVIDIA GPU*: the default (`--torch-backend=cu128`) installs stable torch CUDA 12.8 wheels.
  - *macOS / CPU-only hosts*: pass `--torch-backend=cpu` (or `--torch-backend=auto`). See
    [CPU-only builds](#cpu-only-builds-ci-macos) for how CI pins CPU torch.

Recommended repo layout is sibling checkouts under a common directory (defaults assume `~/repos/<repo>`):

```
~/repos/interpretune
~/repos/SAELens
~/repos/SAEDashboard
~/repos/circuit-tracer
~/repos/TransformerLens
~/repos/nnsight
```

## Canonical full build (one command)

With `IT_REPO_DIR` pointing at your interpretune checkout and `IT_VENV_BASE` at your venv base directory
(place it on the same filesystem as the uv cache to keep hardlinking fast):

```bash
./scripts/build_it_env.sh --repo-home=${IT_REPO_DIR} --target-env-name=it_latest --venv-dir=${IT_VENV_BASE} \
  --from-source="sae_dashboard:${HOME}/repos/SAEDashboard:dev:UV_EXCLUDE=${IT_REPO_DIR}/requirements/ci/excludes.txt:UV_OVERRIDE=${IT_REPO_DIR}/requirements/ci/overrides.txt" \
  --from-source="sae_lens:${HOME}/repos/SAELens:dev:UV_OVERRIDE=${IT_REPO_DIR}/requirements/ci/overrides.txt:FLAGS=-r ${IT_REPO_DIR}/requirements/ci/sl_uv_requirements.txt" \
  --from-source="circuit_tracer:${HOME}/repos/circuit-tracer:dev:UV_EXCLUDE=${IT_REPO_DIR}/requirements/ci/excludes.txt:UV_OVERRIDE=${IT_REPO_DIR}/requirements/ci/overrides.txt" \
  --from-source="transformer-lens:${HOME}/repos/TransformerLens" \
  --from-source="nnsight:${HOME}/repos/nnsight:all:UV_OVERRIDE=${IT_REPO_DIR}/requirements/ci/overrides.txt"
```

The build proceeds in a fixed order:

1. Create/clear the venv and install torch (CUDA/CPU/prerelease per `--torch-backend` and
   `requirements/ci/torch-pre.txt`).
2. Install interpretune editable plus its `git-deps` dependency group (git-pinned circuit-tracer,
   transformer-lens, finetuning-scheduler, plus a PyPI `sae_lens` floor).
3. Install the locked CI requirements (`requirements/ci/requirements.txt`, a universal lock — torch is
   deliberately excluded from it).
4. Install each `--from-source` package **last**, editable, so local checkouts override the PyPI/git
   versions installed in steps 2–3.
5. Set up git hooks, run a non-blocking pyright pass, and print key package versions.

## `--from-source` semantics

Each directive has the shape `package:path[:extras][:ENV_VAR=value...]` (repeat the flag, or separate
specs with semicolons). Package names may use underscores or hyphens. Env vars are exported only for the
duration of that package's install. Three keywords matter in practice:

- **`UV_OVERRIDE=<file>`** — uv override file applied while resolving that package's dependencies, used to
  stop a from-source package from downgrading shared pins. Two maintained files:
  - `requirements/ci/torch-override.txt`: auto-generated, pins only torch. Use for simple cases.
  - `requirements/ci/overrides.txt`: manually maintained; pins torch **and** triton/torchvision plus the
    transformer-lens git SHA and several ecosystem floors. Use for packages with aggressive constraints
    (nnsight, SAEDashboard's `<2.8` torch pin, etc.). If the override file mentions the package currently
    being installed, the builder automatically filters that line out so the editable install still wins.
- **`UV_EXCLUDE=<file>`** — uv excludes file (packages removed from resolution entirely). **Required** when
  interpretune, circuit-tracer, AND transformer-lens are all from source, so exactly one directive controls
  the transformer-lens install; `UV_OVERRIDE` alone suffices when only two of the three are from source.
- **`FLAGS=<extra uv pip install flags>`** — appended to that package's `uv pip install ... -e .[extras]`
  invocation. Needed for SAELens, which is Poetry-legacy (not PEP 621): its dev/test dependency groups are
  not expressible as extras, so uv is given an exported requirements file via
  `FLAGS=-r ${IT_REPO_DIR}/requirements/ci/sl_uv_requirements.txt`. That file is vendored; regenerate it
  whenever SAELens' `pyproject.toml`/lock changes (`poetry export --all-groups --all-extras` → post-process).

Additional rules of thumb:

- nnsight should be installed first among the from-source set (its vllm-era pins occasionally need the
  override file). Note the builder iterates from-source packages in unspecified order; for the canonical
  set above this has not mattered in practice, but if you hit a resolution conflict, build nnsight in a
  first pass (its own `--from-source` invocation) and the rest in a second.
- After every rebuild, validate with `python requirements/utils/collect_env_details.py` (see below).

## CPU-only builds (CI, macOS)

Two supported mechanisms, no interface changes required:

- `--torch-backend=cpu` installs the latest stable CPU torch.
- For a **pinned** CPU torch that stays consistent with `overrides.txt` (recommended for CI), write
  `requirements/ci/torch-pre.txt` with the stable channel and cpu target before building:

  ```bash
  printf '%s\ncpu\nstable\n' "$(grep -E '^torch==' requirements/ci/torch-override.txt | sed 's/^torch==//')" \
    > requirements/ci/torch-pre.txt
  ```

  The installed `<pin>+cpu` build satisfies the `torch==<pin>` override line, so from-source installs never
  re-resolve torch against CUDA wheels. This is exactly what the standalone
  [`env-build-smoke`](../.github/workflows/env-build-smoke.yml) workflow does on `ubuntu-latest` and
  `macos-latest` (it also prunes the `nvidia-*`/`triton` pins from the SAELens exported requirements, which
  are CUDA-resolution artifacts and dead weight on CPU runners).

  Two Linux-specific caveats the smoke workflow also handles (macOS wheels are single-variant and
  unaffected): `torchvision`/`torchaudio` pins resolve to CUDA-built PyPI linux wheels that are
  ABI-incompatible with `+cpu` torch (`operator torchvision::nms does not exist` /
  `_torchaudio.abi3.so` load failure at import time) — after the build, reinstall the same pinned
  versions from `https://download.pytorch.org/whl/cpu`; and the exported `jaxtyping`/`transformer-lens`
  pins must yield to the `overrides.txt` transformer-lens git pin in a joint resolution (see the
  workflow's requirements-pruning step).

## Environment variable contract (dashboard pipeline)

The builder itself honors `IT_VENV_BASE` (venv base directory when `--venv-dir` is not passed; default
`~/.venvs`). The Neuronpedia dashboard pipeline and benchmark tooling resolve everything else from the
environment with portable defaults:

| Variable | Default | Purpose |
| --- | --- | --- |
| `IT_NP_CACHE` | `$HF_HOME/interpretune/neuronpedia` | Neuronpedia cache root (dashboard runs, pretokenized prompt caches, activation caches). Only set it when the cache should live outside the HuggingFace cache tree; `HF_DATASETS_CACHE`/`HF_HUB_CACHE` derive from `HF_HOME` as usual. |
| `IT_BENCH_PYTHON` | the invoking interpreter | Interpreter used by the benchmark/profiling harnesses (point it at your built venv's `python`). |
| `IT_BENCH_PY_SPY` | `py-spy` on `PATH` (or next to `IT_BENCH_PYTHON`) | py-spy binary used for profiling waves. |
| `SAEDASHBOARD_REPO_ROOT` | `~/repos/SAEDashboard` | Local SAEDashboard checkout used by pipeline tooling. |
| `SAELENS_REPO_ROOT` | `~/repos/SAELens` | Local SAELens checkout used by pipeline tooling. |
| `NEURONPEDIA_UTILS_ROOT` | `~/repos/neuronpedia/utils/neuronpedia-utils` | neuronpedia-utils checkout (columnar import tooling). |
| `IT_ENV_FILE` | repo-local `.env` when present | Overrides the Neuronpedia env-file path used to resolve the local DB URL for imports. |

See `docs/neuronpedia_dashboard_pipeline.md` ("Required environment") for how the pipeline consumes these.

## Post-build validation

Always validate a rebuild from the activated venv:

```bash
source ${IT_VENV_BASE}/it_latest/bin/activate
python requirements/utils/collect_env_details.py            # full report (system, CUDA, packages)
python requirements/utils/collect_env_details.py --packages-only
```

For editable and git-backed installs the report appends provenance such as
`(fork:speediedan/SAELens, branch:<branch>, sha:<short-sha>)` — confirm each from-source package points at
the expected checkout/commit. This output also feeds the `salient_pkg_versions` provenance used by the
benchmark registry.

Sibling-suite/benchmark dependencies are now installed automatically by every
`build_it_env.sh` run — no post-build extras remain:

- `syrupy` (SAEDashboard snapshot tests) and `pgpq` (the neuronpedia-utils columnar local-DB
  import encoder) live in interpretune's `examples` extra and the CI lock.
- `neuronpedia-utils` itself installs `--no-deps` from the pinned neuronpedia fork SHA via
  `requirements/ci/nodeps_git_requirements.txt` (a dedicated `build_it_env.sh` step). Its use in
  interpretune is deliberately circumscribed — only the dashboard local-DB import/benchmark lanes
  import it — and while its `sae-dashboard` constraint is now compatible (`>=0.6.8,<0.9`), a
  full-dependency install would drag in its unused autointerp/cloud chain
  (`automated-interpretability` → `blobfile` → `lxml` 4.x source build, `openai`, `google-genai`,
  `boto3`); the runtime deps the import lane needs are already in the lock. For plain-uv installs
  (outside `build_it_env.sh`): `uv pip install --no-deps -r requirements/ci/nodeps_git_requirements.txt`.
  Update the pin alongside the `git-deps` group pins.

(`polars` is optional: only the neuronpedia-utils converter's opt-in `--emit-arrow` mode and its tests
use it — those tests skip cleanly when it is absent. The production columnar lane uses pyarrow.)

Neuronpedia subproject tooling note (upstream `b6156f70`, 2026-07): `apps/inference` is **uv-managed**
(PEP-621 `pyproject.toml` + `uv.lock`; the old `poetry.lock` is gone — `uv sync` / `uv run pytest` from
`apps/inference`, with our sae-lens fork pin under `[tool.uv.sources]`). `utils/neuronpedia-utils`
remains Poetry-managed; the SAELens Poetry-legacy notes above are unchanged.

## Minimal quickstart (reduced from-source set)

A lighter build that only takes SAELens and SAEDashboard from source (circuit-tracer/TransformerLens come
from their git pins) — the same shape the CI smoke workflow exercises:

```bash
export IT_REPO_DIR=${HOME}/repos/interpretune
export IT_VENV_BASE=${HOME}/.venvs

cd ${IT_REPO_DIR}
./scripts/build_it_env.sh --repo-home=${IT_REPO_DIR} --target-env-name=it_smoke --venv-dir=${IT_VENV_BASE} \
  --from-source="sae_lens:${HOME}/repos/SAELens:dev:UV_OVERRIDE=${IT_REPO_DIR}/requirements/ci/overrides.txt:FLAGS=-r ${IT_REPO_DIR}/requirements/ci/sl_uv_requirements.txt" \
  --from-source="sae_dashboard:${HOME}/repos/SAEDashboard:dev:UV_OVERRIDE=${IT_REPO_DIR}/requirements/ci/overrides.txt"

source ${IT_VENV_BASE}/it_smoke/bin/activate
python -c "import interpretune, sae_lens, sae_dashboard; print('ok')"
python requirements/utils/collect_env_details.py --packages-only
```

On a CPU-only host, add `--torch-backend=cpu` (or the pinned `torch-pre.txt` mechanism above).
