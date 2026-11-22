---
name: it-release
description: Prepare, gate, publish and verify an interpretune release - the release candidate, every pre-publish gate that already exists (benchmark refresh, full coverage run, stubs, notebooks, hosted matrix, the gated GPU pipeline, the two-vendor release pipeline, the README-flow tiers, build hygiene), the maintainer authorization gate, the tag and the PyPI workflow, and the post-publish checks. Use when preparing a release candidate, checking release readiness, or publishing a release.
license: Apache-2.0
metadata:
  author: speediedan
  version: '0.1'
---

# interpretune release: prepare, gate, publish, verify

**No release has been cut yet.** The first is the alpha, `0.1.0a0`. Three decisions it needs are still open on
[#259](https://github.com/speediedan/interpretune/issues/259), and this skill does not make them: the tag and branch
policy, the extras story for the git-installed dependencies (#1018), and what the alpha explicitly leaves out. Steps
that depend on them are marked **Pending #259**. When #259 rules, replace each marker with the ruling.

## The authorization gate (read first)

**Creating a `v*` tag or a GitHub Release needs the maintainer's authorization for that release, naming the
version.** A tag is irreversible in practice: the package version is derived from it, downstream installs pin it, and a
published tag is never deleted or force-pushed. A mistake is corrected with a new release, not by moving the tag.
Phases 1 to 3 may proceed without asking. Then stop, report the evidence table from Phase 2, and wait. Approval for one
release does not carry over to the next.

## How the version is set

There is no version string in the source tree. `[tool.setuptools_scm]` in `pyproject.toml` derives it from the most
recent tag matching `tag_regex = '^v(?P<version>\d.*)$'` (`git describe --match v[0-9]*`), with
`fallback_version = "0.1.0.dev0"` when none is reachable. So the released version is exactly the tag without its `v`,
and tags without the prefix (evidence tags, archive tags) never become versions. `interpretune.__version__` reads the
installed distribution's metadata.

Two consequences worth checking rather than assuming: a build that cannot see the tag silently produces the fallback
version, and PEP 440 sorts `0.1.0.devN` before `0.1.0a0`, so every floor written against the dev series has to be
looked at once a real version exists (Phase 1).

## Phase 1: prepare the candidate

1. **Prerequisites.** finetuning-scheduler 2.15.0 is on PyPI and #1006 has moved the conformance extra onto it, so the
   release depends on a published version rather than a commit. #835 tracks the GPU-side readiness audit.
1. **Branch.** **Pending #259** (a long-lived `release/X.Y.x` branch, as finetuning-scheduler uses, or `main`). The
   gated GPU pipeline triggers on pushes to `release/*` and on tags; the hosted matrix on pushes to `main` and
   `release/*`, not on tags. Both filter by path, so a push that changes only Markdown (the CHANGELOG included) runs
   neither: gate the candidate revision itself, not a later documentation commit. `~/repos/it-release` with the
   `it_release` venv is the release worktree.
1. **CHANGELOG.** Turn `## [Unreleased]` into `## [X.Y.Z] - YYYY-MM-DD`. Every breaking change sits under a
   `### Breaking` heading naming the old and new form (`docs/source/stability.md`). Nothing checks that heading
   automatically yet, so read the section by hand.
1. **Component floors.** The in-tree component manifests (`src/it_examples/examples/*/it_component.yaml`) declare
   `requires: interpretune: '>=0.1.dev0'`, with a note to tighten it to `'>=0.1a0'` at release. Tightening changes
   what the Hub components accept, so it ships as a component republish, and each Hub write needs its own
   maintainer approval naming the repository.
1. **One commit.** The CHANGELOG and any release-only edits go in one commit: every push to a branch the gated
   pipeline watches spends another GPU slot.

## Phase 2: gate the candidate

Run every gate on the exact candidate revision, and record each result in an evidence table: the gate, the
revision it ran on (the full sha) and the result, with `collected` beside `passed` for every test run.

| # | Gate | How | Notes |
| --- | --- | --- | --- |
| 1 | Benchmark registry refresh | `python tests/benchmarks/run_benchmarks.py --all --update-registry` from a clean tree | Commit the registry diff alone. Must precede gate 2. |
| 2 | Full local coverage run | `scripts/gen_it_coverage.sh ... --run-all-and-examples` (see `AGENTS.md`, Testing) | The final run is free of real failures. Use `--allow-failures` only for diagnosing runs. |
| 3 | Stub freshness | `python scripts/generate_op_stubs.py`, then `git diff --exit-code` | |
| 4 | Lint and types | `pre-commit run --all-files`, `pyright` | |
| 5 | Notebooks | `nbpress-publish --check-only`, `nbpress-render --check-stale` | A stale or missing docs artifact fails here. |
| 6 | Hosted matrix | `ci_test-full` on Linux, Windows and macOS, `type-check`, `docs-build`, `lint` | A push to `release/*` runs macOS; on a pull request add the `ci:macos` label. |
| 7 | Gated GPU pipeline | `.azure-pipelines/gpu-tests.yml`, every phase | A pull request gate runs only the tests its change selects; add `ci:gpu-full` (or `mode=full`) for the full set. |
| 8 | Release pipeline, both vendors | `.azure-pipelines/gpu-release.yml`, queued by hand on the candidate | Check both agents carry the `ci-release` capability first. Its verdict, `it-release-eligibility`, is the GPU gate; a skip needs a reason in `tests/gpu_ci/reviewed_exclusions.json`. |
| 9 | README install flow | Tier 1 `.github/workflows/readme-flow-smoke.yml` (dispatch it), Tier 2 `.azure-pipelines/readme-flow-gpu.yml` (runs on `release/*` and tags) | The pre-release full-suite half of the install-flow cadence: the documented install, not the CI lock. |
| 10 | Build hygiene | `python -m build`, `twine check dist/*` | The wheel's version is the intended one, and its `METADATA` has no `@ git+` URL in `Requires-Dist` (PyPI rejects the upload). |

Gate 10 is where the extras decision shows: circuit-tracer and sae-dashboard are a `git-deps` dependency group, which
a published wheel does not carry, so a plain install lacks them. **Pending #259** for how the release presents that.

## Phase 3: ask for authorization

Report the candidate sha, the evidence table and anything skipped with its reason, then ask the maintainer for the
release, naming the version. Do not create the tag until the answer names it.

## Phase 4: publish

1. Create the tag `vX.Y.Z` on the candidate and a GitHub Release from it. The release event runs
   `.github/workflows/release_pypi.yml`: `build-package`, `upload-package` (the built files attached to the Release)
   and `publish-package-testpypi` (TestPyPI, after a three-minute delay, by trusted publishing).
1. **The PyPI job is commented out** in that workflow (`publish-package-pypi`). Enabling it, after a TestPyPI
   rehearsal, is the publish-workflow row of #259. **Pending #259.**

## Phase 5: verify

1. **The installed package is the released one.** In a new virtual environment, from an empty directory outside
   any checkout, install the release (from TestPyPI with PyPI as the extra index until the PyPI job is on), then print
   `interpretune.__file__` and `interpretune.__version__`. A matching version from the wrong file is not a pass.
1. **Post-publish smoke test** (a #259 row, not automated yet): the quickstart (#254), unauthenticated and CPU-only,
   in that environment.
1. **Docs.** Read the Docs builds the tag, and the stable version points at it.
1. **Close the loop.** Close or update the issues the release delivers, open a new `## [Unreleased]` section in the
   CHANGELOG, and record in #259 what the release did and did not include.
