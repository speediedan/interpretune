---
name: az-pipelines-debug
description: Debug and operate the interpretune self-hosted Azure GPU pipeline, including PAT-backed approval release, queue triage, worker dispatch checks, phase-split test diagnosis, and memory-aware fixture narrowing.
license: Apache-2.0
metadata:
  author: speediedan
  version: '1.0'
compatibility: Requires bash, az CLI with azure-devops extension, curl, Python 3.10+, access to the interpretune Azure DevOps project, and AZURE_DEVOPS_EXT_PAT in the shell environment.
---

# Azure Pipelines Debug Skill

Use this skill when an interpretune self-hosted GPU Azure DevOps run is queued, failing, or suspected to be exhausting memory.

## GPU lease: how this pipeline interacts with local GPU work

The self-hosted agent runs **one Azure job at a time**, so two pipeline runs never collide. The real risk is
a pipeline job landing on top of a **local** multi-GPU run — this host is shared with finetuning-scheduler
and both projects are worked on interactively.

The job bind-mounts `/tmp/di_leases:/gpu_leases` and holds the host `gpu` lease for its duration
(`Acquire host GPU lease` step; released in the `always()` cleanup step). `flock` works on the inode, so the
mounted lock file interlocks with host processes — **no change to the agent installation** is required. It
**fails open**: if the directory is not mounted the job runs unserialized rather than failing.

### ⛔ Never reset a lease held by CI — either project's CI

`gpu_lease.sh --reset --force` **kills the holder process**. That is the right escape hatch for a wedged
*local* run and the wrong tool for a pipeline job, for two reasons:

1. **The holder pid is meaningless on the host.** A CI holder lives in the job container's PID namespace,
   so `--force` either fails to kill it or, worse, kills an unrelated host process that happens to share
   that pid number. `--status` marks these holders with a `[container]` tag and `project=azure-<buildId>`
   (interpretune: `azure-it-<buildId>`) — treat either as read-only.
2. **The lease is already self-healing for CI.** Container teardown kills every process inside the job, and
   the kernel releases the lease. There is no stale-lock path to clean up.

**The host and pool are shared between finetuning-scheduler and interpretune**, so a lease you did not
expect may legitimately belong to the *other* project's pipeline job or local suite. Check `project=`
before assuming it is stale.

Correct responses:

| Situation | Do this |
| --- | --- |
| Lease held by a CI job you want to stop | **Cancel the pipeline run.** Teardown frees the lease. |
| Lease held by the other project | Leave it. Wait, or coordinate — do not reset. |
| CI job timed out waiting for the lease | A genuine conflict. Let the local run finish and re-queue. |
| Lease looks stale (`--status` flags an anomaly) | `gpu_lease.sh --doctor`, then plain `--reset` (free leases only). |
| Genuinely wedged **local** run | `--reset --force` is appropriate here. |

**Never kill or restart the agent to free a lease.** `restart-stack.sh` is for a wedged agent, not for lease
recovery, and restarting it mid-job strands the run without releasing anything the kernel would not have
released anyway.

## When to Use This Skill

- A GPU Azure build remains `notStarted` after a PR becomes ready for review
- The self-hosted agent is online, but no worker log is created for the queued build
- A GPU Azure step exits `137`, receives a shutdown signal, or otherwise fails under load
- You need to approve, monitor, or re-triage the interpretune GPU pipeline from the shell
- You need to narrow fixture retention or split test slices to stabilize CUDA-involved phases

## Constraints and Ground Truth

- The pipeline is `.azure-pipelines/gpu-tests.yml`
- The GPU runner uses the self-hosted `Default` pool, but a queued build may still show `queue.name = Azure Pipelines` at the build level
- PR-triggered GPU runs require explicit Azure approval before the job is dispatched to the self-hosted runner
- **Path filters do not stop a docs-only PUSH from queueing a build.** For a `pr:` trigger the filters
  are evaluated against the pull request's **cumulative diff**, not the delta of the push that fired
  it — a pipeline validates the merge commit. So once a PR touches `src/**`, every later push
  re-triggers it, including one that only edits `docs/`. Verified 2026-08-05: a docs-only commit
  queued build 709 on PR #240 even though `docs/**` is absent from the include list.
  - `[skip ci]` does **not** rescue this. Microsoft documents that PR pipelines run on the merge
    commit "regardless if there exist pushed commits whose messages or descriptions contain
    `[skip ci]`" — the skip tokens (`[skip ci]`, `[azurepipelines skip]`, `***NO_CI***`, …) suppress
    only CI/branch triggers.
  - What path filters DO buy is skipping a PR whose *entire* diff is documentation, so keep
    `docs/**` and `*.md` in the exclude lists.
  - Practical consequence when managing the queue: batch documentation pushes with the code they
    describe, or expect to dispose of a gate per docs push. Approving a gate and then pushing again
    is worse — Azure cancels the superseded run, so the approval buys nothing.
- `AZURE_DEVOPS_EXT_PAT` is the preferred non-interactive authentication path for `az devops` and Azure DevOps REST calls
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

## Step 1: Verify Auth, Build State, and Approval Gate

```bash
printenv AZURE_DEVOPS_EXT_PAT | wc -c
az pipelines build show --id <build_id> --organization https://dev.azure.com/speediedan --project interpretune -o table
curl -sS -u ":${AZURE_DEVOPS_EXT_PAT}" \
  "https://dev.azure.com/speediedan/interpretune/_apis/pipelines/approvals?state=pending&api-version=7.1-preview.1"
```

Interpretation:

- If the build is `notStarted` and approvals are pending, approve the run before touching the agent
- If no approval is pending, then inspect queue backlog, worker dispatch, and agent availability next

## Step 2: Release (or Reject) the Queued Run

Preferred: use the multi-mode approvals helper script (distributed-insight repo,
`project_admin/shared_admin_scripts/az_pipeline_agent_scripts/manage-approvals.sh`). It reads
`ADO_MCP_AUTH_TOKEN` or `AZURE_DEVOPS_EXT_PAT` from the environment and supports
`list`, `approve`, `reject`, `approve-all`, and `reject-all` modes:

```bash
cd /home/speediedan/repos/distributed-insight/project_admin/shared_admin_scripts/az_pipeline_agent_scripts
./manage-approvals.sh -o speediedan -p interpretune -m list
./manage-approvals.sh -o speediedan -p interpretune -m approve -i "<approval_id>" -c "Approved via CLI for self-hosted GPU validation."
./manage-approvals.sh -o speediedan -p interpretune -m reject -i "<approval_id>"   # terminates the gated build
./manage-approvals.sh -o speediedan -p interpretune -m reject-all                   # dispose all stale pending gates
```

Notes: pending approvals for PR-gated builds are only visible via the pipelines approvals API
(`state=pending`) — gated builds sit `notStarted`, so scanning in-progress build timelines finds
nothing. Rejecting a gate completes the build as `failed` (that is the terminal state for a
rejected approval, not an error in the script).

Fallback: approve the pending gate directly with curl:

```bash
curl -sS -X PATCH -u ":${AZURE_DEVOPS_EXT_PAT}" \
  -H "Content-Type: application/json" \
  -d '[{"approvalId":"<approval_id>","status":"approved","comment":"Approved via CLI for self-hosted GPU validation."}]' \
  "https://dev.azure.com/speediedan/interpretune/_apis/pipelines/approvals?api-version=7.1-preview.1"
```

## Step 3: Monitor Job Dispatch and Runner Activity

```bash
watch -n 30 'az pipelines build show --id <build_id> --organization https://dev.azure.com/speediedan --project interpretune --query "{status:status,result:result,startTime:startTime,finishTime:finishTime}" -o json'
tail -f "$AGENT_HOME"/_diag/Agent_*.log
ls -1t "$AGENT_HOME"/_diag/Worker_*.log | head
az pipelines agent list --organization https://dev.azure.com/speediedan --pool-id 1 -o table
```

Interpretation:

- If the agent log only shows keepalive polling and no new worker log appears, the run is still blocked upstream
- If a new worker log appears, switch to that log immediately for step-level failure details
- If the agent is offline or disabled, fix that before editing the pipeline or tests

## Step 4: Triage Failure Class

### Queue / approval failures

- Build stays `notStarted`
- No worker log is created
- Approval query returns a pending approval

Action:

- Approve the run first
- Only restart the agent if approvals are clear and the pool still is not dispatching work

### Infrastructure or runner failures

- Worker log starts, then dies before pytest output
- Agent log shows docker socket, SSL, or shutdown issues

Action:

- Restart the whole agent stack (rootless docker + agent service) with the NOPASSWD-sudoers
  one-liner — agents are explicitly authorized to run this:

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

## Step 5: Local Reproduction Strategy

Use the local Azure reproduction flow in `distributed-insight` to recreate the containerized runner context. Start with the same phase that failed remotely.

Useful commands:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest --cov=src/interpretune --cov-append --cov-report= src/interpretune tests -v --reruns 2 --reruns-delay 5
IT_RUN_CUDA_TESTS=1 python -m pytest --cov=src/interpretune --cov-append --cov-report= tests -v --durations=50 --reruns 2 --reruns-delay 5
bash ./tests/special_tests.sh --mark_type=standalone
bash ./tests/special_tests.sh --mark_type=profile_ci
```

Fast-iteration guidance (validated Sessions 29-31, 2026-07; local runs are for DEBUG ONLY — the
gated pipeline must still pass on push/merge so coverage stays updated):

- **Run the failing phase, not the pipeline.** A single phase locally gives minutes-scale signal vs
  the ~50-minute full-pipeline round trip; narrow further with `-k`/node ids once the phase-level
  failure set is known. `special_tests.sh` accepts `--filter_pattern` for the standalone/profile
  phases.
- **Cuda-marked tests hide inside the "skipped" count** without `IT_RUN_CUDA_TESTS=1` — a locally
  "green" standard run does NOT imply the `standard gpu cuda-marked` phase is green (this exact gap
  let the Session-31 CT merge land red). The phase also needs `HF_GATED_PUBLIC_REPO_AUTH_KEY`/
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

## Step 6: Fixture and Scope Triage for CUDA-Involved Tests

When CUDA tests still carry too much memory:

- Read `fixture_usage.instructions.md` before changing test fixtures
- Prefer existing generated fixtures over bespoke fixture builders
- Do not add application-code workarounds for test-environment problems
- Reduce fixture retention in tests before changing product code
- For analysis-heavy classes, prefer the shared helpers in `tests/analysis_resource_utils.py`
- Use `analysis_fixture_scope()` so low-RAM runners can fall back to function scope while higher-RAM runners keep class reuse
- Prefer `AnalysisExtractionMixin` and declarative `AnalysisFixtureSpec` entries over parity-local extraction helpers
- Narrow heavyweight fixture scope only for the classes or aliases that are actually forcing retention across methods

## Step 7: What Not to Do

- Do not permanently disable GPU coverage for the self-hosted pipeline just to avoid OOMs
- Do not assume a build-level `queue.name` of `Azure Pipelines` means the YAML job pool changed
- Do not restart the agent before checking for a pending approval
- Do not degrade application code to compensate for fixture or CI-environment issues

## Expected Outcome

After following this skill you should know:

- Whether the run was blocked by approval, queueing, agent health, or test memory
- Which Azure worker log corresponds to the active run
- Whether the correct fix belongs in pipeline structure, agent operations, or test fixture scope
