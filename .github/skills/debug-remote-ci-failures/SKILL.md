---
name: debug-remote-ci-failures
description: Systematic workflow for debugging remote CI test failures across multiple OS runners (Ubuntu, Windows, macOS). Covers artifact analysis, failure categorization, root cause isolation, and iterative fix/push/verify cycles.
license: Apache-2.0
metadata:
  author: speediedan
  version: '1.0'
compatibility: Requires bash, git, gh CLI, Python 3.10+, and access to the Interpretune repository.
---

# Debug Remote CI Failures Skill

This skill documents the systematic process for debugging CI test failures that occur on remote GitHub Actions runners but may not reproduce locally. It was distilled from debugging interpretune PR #197 (nnsight-support branch) across Ubuntu 22.04, Windows 2022, and macOS 14 runners.

## When to Use This Skill

Use this skill when:

- CI tests fail on one or more OS runners but pass locally
- You need to analyze failures across multiple platforms systematically
- Failures span multiple categories (encoding, dependency versions, numerical precision, etc.)
- You need to decide between fixing application code vs test infrastructure vs dependency pins

## Prerequisites

- `gh` CLI authenticated with repo access
- Local development environment matching CI Python version (3.13)
- Access to push to the branch under test

## Workflow Overview

1. **Download and parse CI artifacts** to see exact failure output
2. **Categorize failures** by root cause pattern
3. **Prioritize fixes** — some categories may resolve others
4. **Fix, commit, push, verify** — iterative loop
5. **Document findings** for future reference

---

## Step 1: Download CI Artifacts

```bash
# Find the failed workflow run
gh run list --branch <branch-name> --limit 10

# Download all artifacts from the run
mkdir -p /tmp/ci_artifacts_<run_id>
gh run download <run_id> --dir /tmp/ci_artifacts_<run_id>

# If artifacts aren't available, use run logs directly
gh run view <run_id> --log > /tmp/ci_run_<run_id>.log 2>&1
```

### Extracting Test Results from Logs

When artifacts contain raw pytest output, extract the failure summary:

```bash
# Get just the FAILURES section
grep -A 200 "^FAILURES" /tmp/ci_artifacts_<run_id>/<artifact>/output.txt

# Count failures per OS
grep -c "FAILED" /tmp/ci_artifacts_<run_id>/<os>-*/output.txt

# Get unique test names that failed
grep "^FAILED" /tmp/ci_artifacts_<run_id>/<os>-*/output.txt | sort -u
```

---

## Step 2: Categorize Failures

Group failures by root cause pattern, not by test name. Common categories:

### Category A: Platform-Specific Encoding Issues

**Pattern**: `UnicodeEncodeError: 'charmap' codec can't encode character`

**Typical OS**: Windows only (cp1252 encoding)

**Root Cause**: Source files contain Unicode characters (arrows, checkmarks, emojis) that Windows default encoding cannot handle.

**Fix**: Replace Unicode symbols with ASCII equivalents in test output strings:
- `\u2713` (checkmark) -> `[OK]`
- `\u2717` (cross) -> `[FAIL]`
- `\u2705` (green check) -> `[PASS]`
- `\u26A0` (warning) -> `[WARN]`
- `\u2192` (arrow) -> `->`

**Example from PR #197**:
```python
# Before (Windows fails)
summary.append(f"  \u2713 {name}: validated")

# After (cross-platform)
summary.append(f"  [OK] {name}: validated")
```

### Category B: Dependency Version Mismatch

**Pattern**: Shape mismatches, missing methods/attributes, unexpected API behavior

**Typical OS**: All platforms

**Root Cause**: CI installs different dependency versions than local development. Common when:
- `pyproject.toml` pins point to old commits
- Override files (`overrides.txt`) are out of sync with `pyproject.toml`
- Local env has from-source installs that haven't been pushed as CI dependency pins

**Diagnosis**:
```bash
# Compare local vs CI package versions
# Local:
pip show transformer-lens sae-lens circuit-tracer nnsight

# CI (from log artifacts):
grep "transformer.lens\|sae.lens\|circuit.tracer\|nnsight" /tmp/ci_artifacts_<run_id>/*/output.txt
```

**Fix**: Update dependency pins in `pyproject.toml` and ensure override-dependencies use the same fork/commit:

```bash
# Check for discrepancies between pyproject.toml pins and override files
diff <(grep "TransformerLens\|SAELens" pyproject.toml) <(cat requirements/ci/overrides.txt)

# Regenerate lock file after updating pins
./requirements/utils/lock_ci_requirements.sh
```

**Key Lesson from PR #197**: The `override-dependencies` section in `pyproject.toml` must match the actual dependency pins. Having `TransformerLensOrg` in overrides while the main dependency points to `speediedan` fork causes CI to install the wrong version, even though both forks share commit history.

### Category C: Numerical Precision / Backend Parity

**Pattern**: `AssertionError: Tensor close check failed: max diff = 0.09...`

**Typical OS**: All platforms (may vary by CPU architecture)

**Root Cause**: Different backends (TransformerBridge vs NNsight vs HookedTransformer) produce slightly different numerical results due to:
- Different computation order
- Float32 accumulation differences
- Bridge vs HookedTransformer weight conversion path differences

**Fix Options**:
1. Relax tolerances if the difference is mathematically acceptable
2. Fix the computation path if there's a genuine bug
3. Accept platform-specific tolerances via parametrize marks

### Category D: Test Infrastructure Issues

**Pattern**: Varied — fixture resolution failures, import errors, mock failures

**Root Cause**: Test helpers or fixtures that make assumptions about the environment.

**Fix**: Address the specific infrastructure issue. See the `fixture_usage.instructions.md` for fixture patterns.

### Category E: Memory Exhaustion During Dataset Fingerprinting

**Pattern**: `MemoryError` during `dill.Pickler` → `_save_torchTensor` → `write_large_bytes`

**Typical OS**: Windows (most memory-constrained CI runners), potentially Ubuntu

**Root Cause**: `Dataset.from_generator()` (HuggingFace `datasets` library) hashes all `gen_kwargs` via `dill` serialization to build a fingerprint for caching. When `gen_kwargs` contains a full model module (with all weights), dill tries to serialize the entire model into memory — causing `MemoryError` on memory-constrained CI runners.

**Fix**: Replace `Dataset.from_generator()` with `Dataset.from_list()`:
```python
# Before (OOMs on CI):
dataset = Dataset.from_generator(generator_fn, gen_kwargs=gen_kwargs, ...)

# After (no dill serialization):
records = list(generator_fn(**gen_kwargs))
dataset = Dataset.from_list(records, ...)
```

This works when the dataset is immediately saved to disk after creation (so lazy generation provides no benefit). The fix also avoids pulling in HF datasets caching infrastructure which is typically disabled in test environments anyway.

**Example from PR #197**: `generate_analysis_dataset()` in `analysis.py` passed `gen_kwargs` containing the entire ITModule (with model weights). Switching to `Dataset.from_list()` eliminated the MemoryError.

### Category F: Unexplained Step Cancellation (No Log Output)

**Pattern**: A CI step shows `conclusion: "cancelled"` with `startedAt: null` and zero log output

**Typical OS**: Ubuntu (observed), potentially any

**Root Cause**: Usually OOM-kill by the Linux kernel or a transient runner infrastructure issue. The process is killed before producing any output. Key indicators:
- Job ran for a round number of minutes (e.g., exactly 60m)
- No error artifacts uploaded
- Install step succeeded but the test step produced zero lines of output
- Other OS jobs on the same run completed normally

**Diagnosis**:
```bash
# Check step conclusions
gh run view <run_id> --json jobs --jq '.jobs[] | select(.name | contains("ubuntu")) | .steps[] | {name: .name, conclusion: .conclusion}'

# Compare job duration (suspiciously round = likely killed)
gh run view <run_id> --json jobs --jq '.jobs[] | {name: .name, started: .startedAt, completed: .completedAt}'
```

**Fix**: Often transient — push a new commit and re-run. If persistent, investigate memory usage during test collection (may need to add resource monitoring or reduce parallel test load).

---

## Step 3: Prioritize Fixes

Apply fixes in dependency order:

1. **Dependency pins first** — Updating dependency versions may resolve many failures at once (Category B often triggers symptoms that look like Category C or D)
2. **Encoding/platform fixes** — Quick wins that are clearly correct
3. **Refactors** — Changes that improve code organization (e.g., moving test helpers to adapter methods)
4. **Precision/tolerance fixes** — Only after dependencies are correct

### Decision Framework

```
Is the failure caused by wrong dependency version?
  Yes -> Update pins, push, wait for CI before fixing other failures
  No  -> Is it a platform encoding issue?
    Yes -> Replace Unicode with ASCII, commit with other changes
    No  -> Is it a test infrastructure issue?
      Yes -> Fix test code, not application code
      No  -> Investigate application code
```

---

## Step 4: Fix, Commit, Push, Verify Loop

### Efficient CI Iteration

```bash
# Commit and push
git add -A && git commit -m "fix: <descriptive message>"
git push origin <branch>

# Find the new Test full run
gh run list --branch <branch> --limit 5

# Monitor progress
gh run view <run_id> --json jobs --jq '.jobs[] | {name: .name, status: .status, conclusion: .conclusion}'

# Once complete, check results
gh run view <run_id> --json jobs --jq '.jobs[] | select(.conclusion == "failure") | .name'
```

### Pre-commit Hook Awareness

The interpretune pre-commit hooks include `end-of-file-fixer` and `trailing-whitespace`. These may modify files during the first commit attempt:

```bash
# First attempt may fail
git add -A && git commit -m "fix: ..."
# Output: "fix end of files... Failed"

# Files were auto-fixed — just re-add and commit
git add -A && git commit -m "fix: ..."
```

### Handling Auto-Cancellation

GitHub Actions auto-cancels in-progress runs when a new push arrives on the same branch (if configured with `concurrency` groups). If you need results from a specific run:
- Wait for it to complete before pushing again, OR
- Push only non-test changes (like type annotations) that won't trigger cancellation of the test workflow

---

## Step 5: Analyzing CI Results

### Quick Status Check

```bash
# Get pass/fail counts from a completed run
gh run view <run_id> --log 2>/dev/null | grep -E "passed|failed|error" | tail -5
```

### Downloading Failure Artifacts

```bash
# Download artifacts from failed run
gh run download <run_id> --dir /tmp/ci_artifacts_<run_id>

# Parse test results
for f in /tmp/ci_artifacts_<run_id>/*/output.txt; do
  echo "=== $(dirname $f | xargs basename) ==="
  grep "^FAILED" "$f" | wc -l
  grep "^FAILED" "$f"
done
```

### Cross-Platform Failure Comparison

```bash
# Compare failures across OS runners
diff <(grep "^FAILED" /tmp/ci_artifacts/windows-*/output.txt | sed 's/.*FAILED //' | sort) \
     <(grep "^FAILED" /tmp/ci_artifacts/macos-*/output.txt | sed 's/.*FAILED //' | sort)
```

---

## Lessons Learned from PR #197

### 1. Dependency Pin Consistency is Critical

The `override-dependencies` section in `pyproject.toml` serves a specific purpose: it replaces git URL dependencies with version constraints so that editable installations work correctly. If these overrides point to a different fork than the main dependency pins, CI will install the wrong package version silently.

**Always verify**: `pyproject.toml` dependency URLs, `requirements/ci/overrides.txt`, and `override-dependencies` all reference the same fork and commit.

### 2. Don't Chase Symptoms Before Fixing Dependencies

When dependency versions are wrong, failures cascade. A single wrong TransformerLens version can cause 20+ tensor shape failures, 3 cache key failures, and 3 numerical parity failures — all from the same root cause. Fix dependency pins first and re-run CI before investigating individual test failures.

### 3. Unicode in Test Output is a Windows Landmine

Even when tests pass on Linux and macOS, Windows CI runners use `cp1252` encoding by default. Any Unicode characters in test assertion messages, logging output, or summary strings will cause `UnicodeEncodeError`. Use ASCII-only characters in all test output.

### 4. Type Checking is a Separate CI Job

The interpretune CI runs `pyright` type checking as a separate job from pytest. Adding new methods or modifying existing ones may trigger type errors even if tests pass. Check the "Stale Stubs and Type Checks" workflow alongside "Test full".

### 5. Pre-commit Hooks May Modify Files

The `end-of-file-fixer` and `trailing-whitespace` hooks can modify files during commit, causing the first `git commit` to fail. This is expected — just re-stage and commit again.

### 6. SAE Hook Name Resolution Varies by Model Backend

TransformerBridge resolves hook aliases to canonical names (e.g., `blocks.0.hook_resid_pre` -> `blocks.0.hook_in`), while HookedTransformer uses aliases directly. Any code that constructs SAE hook paths must account for this via `SAELensTLModuleMixin.resolve_sae_hook_name()` (or equivalent) rather than hardcoding the alias + suffix pattern.

### 7. Avoid Passing Full Modules Through HF Datasets Fingerprinting

`Dataset.from_generator()` uses `dill` to serialize all `gen_kwargs` for cache fingerprinting. If `gen_kwargs` contains a PyTorch module with model weights, this serializes the entire model into memory — an instant OOM on CI runners. Prefer `Dataset.from_list()` when the dataset will be saved to disk immediately after creation.

### 8. "Cancelled" Steps With No Output Usually Mean OOM Kill

When a CI step shows `conclusion: "cancelled"` with zero log lines, the process was likely killed by the OS (e.g., Linux OOM killer). This is different from GitHub Actions "cancel-in-progress" which cancels the entire run. Look for round job durations (exactly 60m, 90m) and check if other OS jobs in the same run completed normally.

---

## Related Files

- `.github/workflows/ci_test-full.yml` — Main CI workflow
- `.github/workflows/ci_stubs_types.yml` — Type checking workflow
- `pyproject.toml` — Dependency pins and override-dependencies
- `requirements/ci/overrides.txt` — CI dependency overrides
- `requirements/ci/requirements.txt` — Locked CI requirements
- `.github/instructions/fixture_usage.instructions.md` — Test fixture patterns
