---
name: az-pipelines-failure-triage
description: Diagnose a FAILED or memory-suspect run of the interpretune self-hosted Azure GPU pipeline - failure-class taxonomy across the four phase-split Testing tasks, local reproduction commands per phase, cuda-marked selector semantics, and fixture/CUDA scope narrowing for OOM-prone tests. Use when a build is red or exiting 137, not for approving or dispatching runs.
license: Apache-2.0
metadata:
  author: speediedan
  version: '2.0'
---

# Azure Pipelines Failure Triage

Use this skill when an interpretune self-hosted GPU Azure DevOps run has FAILED, or is suspected of
exhausting memory. Driving the pipeline itself (auth, releasing or rejecting gated runs, queue and
dispatch problems, "why has it not started") is the vendored `az-pipelines-ops` skill; GPU
serialization is the vendored `gpu-lease` skill. This file carries what neither can: the failure
taxonomy and reproduction paths specific to THIS repo's pipeline shape.

Both halves number their steps from 1, so a bare "Step N" below always means a step of THIS skill;
any reference to the other half is qualified by skill name.

## When to Use This Skill

- A GPU Azure build completed `failed` and you have (or can fetch) its logs
- A GPU Azure step exits `137`, receives a shutdown signal, or otherwise fails under load
- You need to narrow fixture retention or split test slices to stabilize CUDA-involved phases

## Constraints and Ground Truth

- The pipeline is `.azure-pipelines/gpu-tests.yml`
- Runner constraints (**host-specific — see the note below; values here are illustrative**):
  - RAM on the order of tens of GiB, with **swap much smaller than RAM** (the current host is an example:
    roughly 62 GiB RAM against 2 GiB swap unless explicitly expanded). This ratio is why memory pressure,
    not GPU memory, is the usual cause of an exit `137`.
  - Agent service sets a low `OOMScoreAdjust` (-900). It does **not** set `MemoryMax`/`MemoryHigh` —
    an earlier version of this file claimed it did; verified absent 2026-07-29 (no systemd drop-in).
  - GPU jobs take the host GPU lease (`/tmp/di_leases` bind-mounted to `/gpu_leases`); see the
    'Acquire host GPU lease' step. It fails open, and container teardown always frees the lease, so
    cancel a run rather than force-resetting a lease held by CI.
  - Rootless Docker and cgroups v2 are in use

> **Host-specific values live in `CLAUDE.local.md`, not here.** This skill is deliberately host-independent.
> Agent hostname, RAM/swap, GPU models, the agent install directory and the agent's uid all vary by machine,
> so the commands below use `$AGENT_HOME` / `$AGENT_UID` with illustrative defaults. Substitute the real
> values for the machine you are on:
>
> ```bash
> AGENT_HOME=${AGENT_HOME:-/opt/az_pipeline_agent}   # example default
> AGENT_UID=${AGENT_UID:-998}                        # example; the uid the agent runs as
> ```
- The GPU test flow is phase-split to reduce peak memory:
  1. `Testing: standard` is CPU-only with `CUDA_VISIBLE_DEVICES=''`
  2. `Testing: standard gpu cuda-marked` runs regular CUDA-gated tests under `IT_RUN_CUDA_TESTS=1`
  3. `Testing: standalone gpu` runs standalone GPU tests
  4. `Testing: CI Profiling` runs profiling GPU tests

## Step 1: Triage Failure Class

### Queue / approval failures

- Build stays `notStarted`
- No worker log is created
- Approval query returns a pending approval

Action:

- Approve the run first
- Only restart the agent after the build TIMELINE shows neither a pending approval nor an
  in-progress authorization checkpoint; a build can have clear approvals and still be blocked on
  resource authorization, which the approvals API cannot see (see the vendored `az-pipelines-ops`
  skill, Step 1)

### Infrastructure or runner failures

- Worker log starts, then dies before pytest output
- Agent log shows docker socket, SSL, or shutdown issues

Action:

- Restart the whole agent stack (rootless docker + agent service) via the operator's authorized
  wrapper script. Whether an agent may invoke it, and how it is authorized, is machine-specific and
  recorded in the local instructions file rather than here:

  ```bash
  sudo "$AGENT_HOME"/restart-stack.sh
  ```

- Recheck `/var/run/docker.sock` symlink handling (`/var/run/docker.sock ->
  /run/user/$AGENT_UID/docker.sock`, e.g. uid `998`; the symlink lives on tmpfs and is lost on reboot) and
  agent service
  health; the restart script covers the standard recovery, with the manual flow documented in
  `distributed-insight/cmdref/ml_engineering/ref_azure_pipelines.md`

### Test-memory failures

- Step exits `137`
- Job receives shutdown signal during a heavy pytest phase
- GPU or host memory ramps sharply during a single phase

Action:

- Confirm the phase split is intact before changing memory limits
- Keep baseline coverage CPU-only
- Prefer isolating CUDA-gated tests into `IT_RUN_CUDA_TESTS=1`, standalone, and `profile_ci` slices
- Increase swap only after verifying the phase split and fixture retention are not the main cause

## Step 2: Local Reproduction Strategy

Use the local Azure reproduction flow in `distributed-insight` to recreate the containerized runner context. Start with the same phase that failed remotely.

Useful commands:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest --cov=src/interpretune --cov-append --cov-report= tests src/it_examples/tests -v --reruns 2 --reruns-delay 5
IT_RUN_CUDA_TESTS=1 python -m pytest --cov=src/interpretune --cov-append --cov-report= tests -v --durations=50 --reruns 2 --reruns-delay 5
bash ./tests/special_tests.sh --mark_type=standalone
bash ./tests/special_tests.sh --mark_type=profile_ci
```

Fast-iteration guidance (validated 2026-07; local runs are for DEBUG ONLY — the
gated pipeline must still pass on push/merge so coverage stays updated):

- **Run the failing phase, not the pipeline.** A single phase locally gives minutes-scale signal vs
  the ~50-minute full-pipeline round trip; narrow further with `-k`/node ids once the phase-level
  failure set is known. `special_tests.sh` accepts `--filter_pattern` for the standalone/profile
  phases.
- **Cuda-marked tests hide inside the "skipped" count** without `IT_RUN_CUDA_TESTS=1` — a locally
  "green" standard run does NOT imply the `standard gpu cuda-marked` phase is green (this exact gap
  let the 2026-07-20 circuit-tracer merge land red). The phase also needs `HF_GATED_PUBLIC_REPO_AUTH_KEY`/
  `HF_TOKEN` in the environment (gated-model + notebook tests).
- **Debug another branch without disturbing your working tree**: detached `git worktree` + an
  overlay venv — `python -m venv <env>`, `pip install -e <worktree> --no-deps`, then an executable
  `.pth` bridge (`import site; site.addsitedir('<base env site-packages>')`; a plain-directory
  `.pth` does NOT propagate the base env's editable installs). Put the overlay env's `bin` on
  `PATH` for tests that spawn the `interpretune` console script (`test_it_cli.py`). See
  `CLAUDE.md` "Worktrees & Parallel Environments"; the long-lived `~/repos/it-release` worktree +
  `it_release` venv can serve as the second pair when isolation from its state isn't required.
- **The agent shares this machine's GPUs**: never run local GPU phases while a gated build is
  executing (build 607's first attempt OOMed on a leftover 7.6 GiB kernel), and HOLD pending gate
  approvals until local GPU work finishes — approve after, with the hold noted in the approval
  comment.

## Step 3: Fixture and Scope Triage for CUDA-Involved Tests

When CUDA tests still carry too much memory:

- Read `fixture_usage.instructions.md` before changing test fixtures
- Prefer existing generated fixtures over bespoke fixture builders
- Do not add application-code workarounds for test-environment problems
- Reduce fixture retention in tests before changing product code
- For analysis-heavy classes, prefer the shared helpers in `tests/analysis_resource_utils.py`
- Use `analysis_fixture_scope()` so low-RAM runners can fall back to function scope while higher-RAM runners keep class reuse
- Prefer `AnalysisExtractionMixin` and declarative `AnalysisFixtureSpec` entries over parity-local extraction helpers
- Narrow heavyweight fixture scope only for the classes or aliases that are actually forcing retention across methods

## Step 4: What Not to Do

- Do not permanently disable GPU coverage for the self-hosted pipeline just to avoid OOMs
- Do not degrade application code to compensate for fixture or CI-environment issues

## Expected Outcome

After following this skill you should know:

- Whether the run was blocked by approval, queueing, agent health, or test memory
- Which Azure worker log corresponds to the active run
- Whether the correct fix belongs in pipeline structure, agent operations, or test fixture scope
