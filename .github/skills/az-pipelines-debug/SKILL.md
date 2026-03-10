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
- `AZURE_DEVOPS_EXT_PAT` is the preferred non-interactive authentication path for `az devops` and Azure DevOps REST calls
- Current runner constraints observed on `speediedl`:
  - RAM is about 62 GiB
  - Swap is only 2 GiB unless explicitly expanded
  - Agent service is already protected with unlimited `MemoryMax`/`MemoryHigh` and low `OOMScoreAdjust`
  - Rootless Docker and cgroups v2 are in use
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

## Step 2: Release the Queued Run

Approve the pending gate directly from the shell:

```bash
curl -sS -X PATCH -u ":${AZURE_DEVOPS_EXT_PAT}" \
  -H "Content-Type: application/json" \
  -d '[{"approvalId":"<approval_id>","status":"approved","comment":"Approved via CLI for self-hosted GPU validation."}]' \
  "https://dev.azure.com/speediedan/interpretune/_apis/pipelines/approvals?api-version=7.1-preview.1"
```

## Step 3: Monitor Job Dispatch and Runner Activity

```bash
watch -n 30 'az pipelines build show --id <build_id> --organization https://dev.azure.com/speediedan --project interpretune --query "{status:status,result:result,startTime:startTime,finishTime:finishTime}" -o json'
tail -f /opt/az_pipeline_agent/_diag/Agent_*.log
ls -1t /opt/az_pipeline_agent/_diag/Worker_*.log | head
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

- Follow the documented rootless Docker and agent restart flow in the Azure pipeline docs
- Recheck `/var/run/docker.sock` symlink handling and agent service health

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
