---
name: it-release
description: Prepare, gate, publish and verify an interpretune release - the release candidate, every pre-publish gate that already exists (benchmark refresh, full coverage run, stubs, notebooks, hosted matrix, the gated GPU pipeline, the two-vendor release pipeline, the README-flow tiers, build hygiene), the maintainer authorization gate, the tag and the PyPI workflow, and the post-publish checks. Use when preparing a release candidate, checking release readiness, or publishing a release.
license: Apache-2.0
metadata:
  author: speediedan
  version: '0.1'
---

# interpretune release: prepare, gate, publish, verify

**No release has been cut yet.** The first is the alpha, `0.1.0a0`. Its three decisions are recorded on
[#259](https://github.com/interpretune/interpretune/issues/259):

- **Release model (ruled 2026-11-22): tags on `main`.** A release is a tag on a gated `main` commit, and the tag is the
  version bump. A `release/X.Y.x` branch is cut from a tag only when a published release needs a fix that cannot wait
  for `main` (see Phase 1). A long-lived release branch per line is the expected model from the first non-alpha
  release, when the stability policy adds deprecation notice.
- **Extras for the git-installed dependencies: deferred.** The contingency paths for circuit-tracer are on #1018;
  sae-dashboard is expected to stay a git dependency. Steps that depend on it are marked **Pending #1018**.
- **What the release leaves out:** the reviewed list is on #259. Copy it into the release notes (Phase 4).

## The authorization gate (read first)

**Creating a `v*` tag or a GitHub Release needs the maintainer's authorization for that release, naming the
version.** A tag is irreversible in practice: the package version is derived from it, downstream installs pin it, and a
published tag is never deleted or force-pushed. A mistake is corrected with a new release, not by moving the tag.
Phases 1 to 3 may proceed without asking. Then stop, report the evidence table from Phase 2, and wait. Approval for one
release does not carry over to the next.

## How the version is set

There is no version string in the source tree. `[tool.setuptools_scm]` in `pyproject.toml` derives it from the most
recent tag matching `tag_regex = '^v(?P<version>\d.*)$'` (`git describe --match v[0-9]*`), with
`fallback_version = "0.1.0.dev0"` when there is no SCM metadata at all (a source tree without its `.git`). So the
released version is exactly the tag without its `v`, and tags without the prefix (evidence tags, archive tags) never
become versions. `interpretune.__version__` reads the installed distribution's metadata.

Around a tag on `main` (measured with setuptools-scm on a scratch repository): a clone with no `v` tag reachable reads
`0.1.devN+g<sha>`, the tagged commit reads `0.1.0a0`, and the next commit on `main` reads `0.1.0a1.dev1+g<sha>`. So
`main` sorts above the published alpha and below the next one with no version edit anywhere. A tag on a side branch
is not reachable from `main`, which is why releases are tagged on `main` itself.

Two consequences worth checking rather than assuming: a build that cannot see the tag silently produces the fallback
version, and PEP 440 sorts `0.1.0.devN` before `0.1.0a0`, so every floor written against the dev series has to be
looked at once a real version exists (Phase 1).

## Phase 1: prepare the candidate

1. **Prerequisites.** finetuning-scheduler 2.15.0 is on PyPI and #1006 has moved the conformance extra onto it, so the
   release depends on a published version rather than a commit. #835 tracks the GPU-side readiness audit.
1. **The candidate is a commit on `main`.** The CHANGELOG commit (below) lands on `main` by pull request, and the
   candidate is the `main` commit that carries it. Record its full sha: `main` may move while the gates run, and the
   tag goes on the gated sha, never on whatever `main` has reached by then. The hosted matrix triggers on pushes to
   `main` and the gated GPU pipeline on tags, and both filter by path, so a merge that changes only Markdown (the
   CHANGELOG included) runs neither: start each gate on the candidate explicitly (Phase 2). `~/repos/it-release` with
   the `it_release` venv is the release worktree.
1. **A fix for a published release** that cannot wait for `main`: cut `release/X.Y.x` from that release's tag,
   cherry-pick the fix, and tag the next version there (`vX.Y.Za1` after `vX.Y.Za0`, for example). Pushes to
   `release/*` trigger both the hosted matrix and the gated GPU pipeline. Land the fix on `main` too.
1. **CHANGELOG.** Turn `## [Unreleased]` into `## [X.Y.Z] - YYYY-MM-DD`. Every breaking change sits under a
   `### Breaking` heading naming the old and new form (`docs/source/stability.md`). Nothing checks that heading
   automatically yet, so read the section by hand.
1. **Component floors.** The in-tree component manifests (`src/it_examples/examples/*/it_component.yaml`) declare
   `requires: interpretune: '>=0.1.dev0'`, with a note to tighten it to `'>=0.1a0'` at release. Tightening changes
   what the Hub components accept, so it ships as a component republish, and each Hub write needs its own
   maintainer approval naming the repository.
1. **One commit.** The CHANGELOG and any release-only edits go in one commit, so one candidate carries all of them
   and the gates run once.

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
| 6 | Hosted matrix | `ci_test-full` on Linux, Windows and macOS, `type-check`, `docs-build`, `lint` | Dispatch `ci_test-full` on `main` while `main` is still at the candidate (a dispatch runs all three operating systems), or run it on a `release/*` push. Check the run's head sha is the candidate. |
| 7 | Gated GPU pipeline | `.azure-pipelines/gpu-tests.yml`, every phase | Queue a manual run on the candidate sha with `mode=full`: a pull request gate runs only the tests its change selects, and `main` has no push trigger. |
| 8 | Release pipeline, both vendors | `.azure-pipelines/gpu-release.yml`, queued by hand on the candidate | Check both agents carry the `ci-release` capability first. Its verdict, `it-release-eligibility`, is the GPU gate; a skip needs a reason in `tests/gpu_ci/reviewed_exclusions.json`. |
| 9 | README install flow | Tier 1 `.github/workflows/readme-flow-smoke.yml` (dispatch it), Tier 2 `.azure-pipelines/readme-flow-gpu.yml` (queue it by hand on the candidate) | The pre-release full-suite half of the install-flow cadence: the documented install, not the CI lock. Tier 2 also triggers on tags and `release/*`, but a tag run comes after the release, too late to gate it. |
| 10 | Build hygiene | `python -m build`, `twine check dist/*` | The wheel's version is the intended one, and its `METADATA` has no `@ git+` URL in `Requires-Dist` (PyPI rejects the upload). |

Gate 10 is where the extras decision shows: circuit-tracer and sae-dashboard are a `git-deps` dependency group, which
a published wheel does not carry, so a plain install lacks them. **Pending #1018** for how the release presents
circuit-tracer; sae-dashboard stays a git dependency.

## Phase 3: ask for authorization

Report the candidate sha, the evidence table and anything skipped with its reason, then ask the maintainer for the
release, naming the version. Do not create the tag until the answer names it.

## Phase 4: publish

1. Create the tag `vX.Y.Z` on the candidate sha (on `main`, or on `release/X.Y.x` for a fix release) and a GitHub
   Release from it, with the release notes stating what the release leaves out (#259's reviewed list). The release
   event runs
   `.github/workflows/release_pypi.yml`: `build-package`, `upload-package` (the built files attached to the Release)
   and `publish-package-testpypi` (TestPyPI, after a three-minute delay, by trusted publishing).
1. **The PyPI job is commented out** in that workflow (`publish-package-pypi`). Enabling it, after a TestPyPI
   rehearsal, is the publish-workflow row of #259, still open.

## Phase 5: verify

1. **The installed package is the released one.** In a new virtual environment, from an empty directory outside
   any checkout, install the release (from TestPyPI with PyPI as the extra index until the PyPI job is on), then print
   `interpretune.__file__` and `interpretune.__version__`. A matching version from the wrong file is not a pass.
1. **Post-publish smoke test** (a #259 row, not automated yet): the quickstart (#254), unauthenticated and CPU-only,
   in that environment.
1. **Docs.** Read the Docs builds the tag, and the stable version points at it.
1. **Close the loop.** Close or update the issues the release delivers, open a new `## [Unreleased]` section in the
   CHANGELOG, and record in #259 what the release did and did not include.
