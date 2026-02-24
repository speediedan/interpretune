---
name: update-analysis-injection-notebook
description: Updates an existing Interpretune analysis injection notebook and its hook configuration when a target upstream package changes versions. Covers diff analysis, regex pattern validation, config/notebook updates, and test validation. Generalizable to any notebook using the analysis injection framework.
license: Apache-2.0
metadata:
  author: speediedan
  version: '1.0'
compatibility: Requires bash, git, Python 3.10+, uv, and access to the Interpretune repository. GPU required for notebook test execution (standalone tests).
---

# Update Analysis Injection Notebook Skill

This skill guides the process of updating an Interpretune analysis injection notebook and its supporting files when an upstream target package (e.g., `circuit-tracer`) changes versions. It covers identifying breaking changes, fixing hook regex patterns, updating documentation, and validating everything end-to-end.

## When to Use This Skill

Use this skill when:

- An upstream package targeted by analysis injection hooks has been updated to a new version
- Hook regex patterns in `analysis_injection_config.yaml` may no longer match current package code
- The notebook's inline documentation references outdated API signatures
- Notebook tests are failing due to hook validation errors (e.g., `RuntimeError: Hook validation failed: MISSING`)

## Reference Instructions

This skill builds on the analysis injection framework documented in:

- `.github/instructions/analysis_injection.instructions.md` — Architecture overview, config format, debugging workflow, common failure patterns
- Read this file first if unfamiliar with the analysis injection system

## Required User Inputs

Before running this skill, gather the following:

1. **Target notebook name** — The notebook directory name (e.g., `attribution_analysis`)
2. **Target package name** — The upstream package being updated (e.g., `circuit-tracer`)
3. **Old version/SHA** — The previous package version and git commit SHA (e.g., `0.1.0` / `fe1743f`)
4. **New version/SHA** — The new package version and git commit SHA (e.g., `0.4.0` / `14cc3e8`)
5. **Test path and parameterization** — The pytest path to the notebook test (e.g., `tests/examples/test_notebooks.py::test_attribution_analysis_notebook[analysis_inj_salient_logits_SLT]`)
6. **Venv name** — The development environment to use (e.g., `it_latest`)
7. **Local package repo path** — Path to the local clone of the target package (e.g., `~/repos/circuit-tracer`)
8. **[Optional] Report output location** — Defaults to `~/repos/distributed-insight/project_admin/interpretune/handoff_docs/` or `/tmp/`

## Prerequisites

- Interpretune repository checked out locally
- Development venv built and activated (see `copilot-instructions.md` for build instructions)
- Target package installed at the new version in the venv (verify with `pip show <package>`)
- Local clone of the target package repo available for git diff analysis
- GPU available for running standalone notebook tests

## Step-by-Step Process

### Phase 1: Analyze Upstream Changes

**Goal:** Identify which code changes in the target package affect analysis injection hooks.

1. **Identify the files referenced by hooks**:

   Read the notebook's `analysis_injection_config.yaml` and extract the unique `file_path` values from `file_hooks`:

   ```bash
   grep 'file_path:' src/it_examples/notebooks/dev/<notebook>/analysis_injection_config.yaml | sort -u
   ```

2. **Diff the target package between old and new versions**:

   ```bash
   cd <local_package_repo>
   git log --oneline <old_sha>..<new_sha> -- <file_paths_from_step_1>
   git diff <old_sha>..<new_sha> -- <file_paths_from_step_1>
   ```

   Focus on changes that affect:
   - Lines matched by existing regex patterns
   - Function signatures or call patterns referenced in hooks
   - Module reorganization (moved/renamed files)

3. **Categorize changes by impact**:

   - **No impact**: Changes in unrelated functions or files not referenced by hooks
   - **Regex breakage**: Lines matched by regex patterns were modified (single-line → multi-line, renamed variables, etc.)
   - **File path changes**: Target files were moved or renamed
   - **Variable availability changes**: Local variables captured by `analysis_points.py` were renamed or removed

### Phase 2: Validate All Regex Patterns

**Goal:** Systematically verify every hook regex pattern matches the new package code.

1. **Write and run a validation script** that tests all regex patterns from the config against the installed package code:

   ```python
   import re, yaml, inspect, importlib

   with open('src/it_examples/notebooks/dev/<notebook>/analysis_injection_config.yaml') as f:
       config = yaml.safe_load(f)

   package_root = "<package_import_name>"  # e.g., "circuit_tracer"

   for hook_id, hook_cfg in config['file_hooks'].items():
       if not hook_cfg.get('enable', True):
           continue

       # Resolve the module from file_path
       # e.g., "replacement_model/replacement_model_transformerlens.py"
       # → "circuit_tracer.replacement_model.replacement_model_transformerlens"
       module_path = hook_cfg['file_path'].replace('/', '.').replace('.py', '')
       full_module = f"{package_root}.{module_path}"

       try:
           mod = importlib.import_module(full_module)
           source = inspect.getsource(mod)
       except Exception as e:
           print(f"FAIL {hook_id}: Could not load module {full_module}: {e}")
           continue

       pattern = hook_cfg['regex_pattern']
       matched = False
       for line_num, line in enumerate(source.split('\n'), 1):
           if re.search(pattern, line):
               print(f"OK   {hook_id}: matched at line {line_num}")
               matched = True
               break
       if not matched:
           print(f"FAIL {hook_id}: NO MATCH for pattern: {pattern}")
   ```

2. **Record which patterns fail** — these need fixing in Phase 3.

3. **Also check notebook inline overrides** — some notebooks define `config_overrides` dicts in code cells that contain additional regex patterns. Search for these:

   ```bash
   grep -n "regex_pattern\|config_overrides" src/it_examples/notebooks/dev/<notebook>/<notebook>.ipynb
   ```

### Phase 3: Update Configuration Files

**Goal:** Fix broken regex patterns and update version metadata.

#### For each broken regex pattern:

1. **Identify an alternative anchor line** near the original match point:

   - If a single-line statement became multi-line, find the next stable single-line anchor
   - Prefer lines that are unlikely to change (assignments to well-known variables, logger calls, return statements)
   - Consider `insert_after` semantics: if changing the target line, you may need to flip `insert_after` to maintain correct insertion position

2. **Critical: Verify insertion point correctness**:

   Before changing `insert_after`, verify that the analysis point function in `analysis_points.py` will still have access to the variables it needs at the new insertion point:

   ```python
   # Check what variables the analysis point function accesses
   grep "local_keys\|context_keys" src/it_examples/notebooks/dev/<notebook>/analysis_points.py
   ```

   - If using `insert_after: true`, variables computed on the matched line ARE available
   - If using `insert_after: false`, variables computed on the matched line are NOT yet available
   - Ensure all required variables exist at the new insertion point

3. **Update the dev config**:

   Edit `src/it_examples/notebooks/dev/<notebook>/analysis_injection_config.yaml`:
   - Update `settings.target_package_version` to the new version
   - Update broken `regex_pattern` values
   - Flip `insert_after` if the insertion semantics changed

4. **Verify the new regex matches**:

   Re-run the validation script from Phase 2 to confirm all patterns now match.

5. **Update the publish config**:

   Edit `src/it_examples/notebooks/publish/<notebook>/analysis_injection_config.yaml` with the same changes. Verify dev and publish configs are consistent:

   ```bash
   diff src/it_examples/notebooks/dev/<notebook>/analysis_injection_config.yaml \
        src/it_examples/notebooks/publish/<notebook>/analysis_injection_config.yaml
   ```

   The only expected differences should be notebook-specific (e.g., Colab-specific overrides). Version numbers and regex patterns should match.

### Phase 4: Update Notebook Content

**Goal:** Update markdown documentation and clear stale outputs.

1. **Update markdown code snippets** that quote old API signatures:

   Search for old API patterns in the notebook:

   ```bash
   grep -n "<old_function_signature>" src/it_examples/notebooks/dev/<notebook>/<notebook>.ipynb
   ```

   For each match in a markdown cell, update the code snippet to reflect the new API. Since notebooks are JSON, use a Python script for reliable editing:

   ```python
   import json

   for nb_path in ['dev/<notebook>/<notebook>.ipynb', 'publish/<notebook>/<notebook>.ipynb']:
       full_path = f'src/it_examples/notebooks/{nb_path}'
       with open(full_path) as f:
           nb = json.load(f)

       for cell in nb['cells']:
           for j, line in enumerate(cell.get('source', [])):
               if '<old_pattern>' in line:
                   cell['source'][j] = line.replace('<old_pattern>', '<new_pattern>')

       with open(full_path, 'w') as f:
           json.dump(nb, f, indent=1, ensure_ascii=False)
           f.write('\n')
   ```

2. **Clear stale error outputs from the dev notebook** (if any exist from previous failed runs):

   ```python
   import json

   nb_path = 'src/it_examples/notebooks/dev/<notebook>/<notebook>.ipynb'
   with open(nb_path) as f:
       nb = json.load(f)

   for cell in nb['cells']:
       if cell['cell_type'] == 'code':
           for output in cell.get('outputs', []):
               if output.get('output_type') == 'error':
                   cell['outputs'] = []
                   break

   with open(nb_path, 'w') as f:
       json.dump(nb, f, indent=1, ensure_ascii=False)
       f.write('\n')
   ```

3. **Verify the publish notebook has no outputs** (it should always be clean):

   ```python
   import json

   nb_path = 'src/it_examples/notebooks/publish/<notebook>/<notebook>.ipynb'
   with open(nb_path) as f:
       nb = json.load(f)

   for cell in nb['cells']:
       if cell['cell_type'] == 'code' and cell.get('outputs'):
           print(f"WARNING: publish notebook cell has outputs: {''.join(cell['source'][:1])[:60]}")
   ```

### Phase 5: Update Analysis Points (If Needed)

**Goal:** Ensure analysis point functions still work with the new code.

1. **Check if any variables accessed by analysis points were renamed or removed**:

   For each analysis point function in `analysis_points.py`, verify that:
   - All `local_keys` variables exist at the hook insertion point in the new code
   - All `context_keys` are still valid
   - Return data structures still make sense with the new API

2. **If variable names changed**, update the corresponding entries in `analysis_points.py` and verify in both dev and publish copies:

   ```bash
   diff src/it_examples/notebooks/dev/<notebook>/analysis_points.py \
        src/it_examples/notebooks/publish/<notebook>/analysis_points.py
   ```

   These files should be identical (the publish pipeline copies them as-is).

### Phase 6: Run Tests

**Goal:** Validate everything works end-to-end.

1. **Run the specific notebook test(s)**:

   ```bash
   source /mnt/cache/$USER/.venvs/<venv_name>/bin/activate
   cd ~/repos/interpretune

   IT_RUN_STANDALONE_TESTS=1 python -m pytest "<test_path>" -v
   ```

   If the notebook has multiple parameterizations (e.g., SLT and CLT variants), run all of them:

   ```bash
   IT_RUN_STANDALONE_TESTS=1 python -m pytest \
     "tests/examples/test_notebooks.py::test_attribution_analysis_notebook" -v
   ```

2. **Run the basic test suite** to ensure no regressions:

   ```bash
   python -m pytest src/interpretune tests --tb=short -q
   ```

3. **If tests fail**, debug using the analysis injection debugging workflow:

   - Check the analysis log in `/tmp/<analysis_log_prefix>_*.log`
   - Verify regex matches in the log output
   - Check the test's output notebook in `/tmp/pytest-*/test_*/notebook_outputs/`
   - Refer to the "Common Failure Patterns" table in `analysis_injection.instructions.md`

### Phase 7: Run Pre-commit Checks

```bash
pre-commit run ruff-check --all-files
pre-commit run ruff-format --all-files
pre-commit run --all-files
```

### Phase 8: Write Summary Report

Write a summary report documenting:

1. **Package update**: Old version → new version, relevant commit range
2. **Changes identified**: Which upstream code changed and how
3. **Hooks affected**: Which regex patterns broke and how they were fixed
4. **Notebook updates**: What markdown documentation was updated
5. **Test results**: All tests passing, number passed/skipped
6. **Files modified**: Complete list of files changed

Save to the report output location specified by the user.

## Example: circuit-tracer 0.1.0 → 0.4.0

This section documents a concrete application of this skill.

### Inputs

| Input | Value |
|-------|-------|
| Target notebook | `attribution_analysis` |
| Target package | `circuit-tracer` |
| Old version/SHA | `0.1.0` / `fe1743f` |
| New version/SHA | `0.4.0` / `14cc3e8` |
| Test path | `tests/examples/test_notebooks.py::test_attribution_analysis_notebook[analysis_inj_salient_logits_SLT]` |
| Venv | `it_latest` |
| Local repo | `~/repos/circuit-tracer` |

### What Changed

One relevant commit (10b4178) modified `replacement_model/replacement_model_transformerlens.py`:

- **New `zero_positions` attribute**: `self.zero_positions = slice(0, 4) if gemma_3_it else slice(0, 1)` (for Gemma 3 IT model support)
- **`compute_attribution_components` call changed**: Single-line `compute_attribution_components(mlp_in_cache)` became multi-line `compute_attribution_components(mlp_in_cache, self.zero_positions)`
- **Error vector zeroing changed**: `error_vectors[:, 0] = 0` → `error_vectors[:, self.zero_positions] = 0`

### Hooks Affected

Only **1 of 13** hooks broke: `ap_compute_attribution_end`

The old regex targeted the single-line `compute_attribution_components(mlp_in_cache)` call. After the refactor to a multi-line call with added `self.zero_positions` argument, the regex no longer matched.

**Fix**: Changed the regex target from the function call to the next stable line (`error_vectors = mlp_out_cache - attribution_data["reconstruction"]`) and flipped `insert_after` from `true` to `false`. This maintains the same insertion semantics — the hook fires after `attribution_data` is populated. This is safe because the `ap_compute_attribution_end` analysis point function only accesses `attribution_data`, not `error_vectors`.

### Files Modified

1. `src/it_examples/notebooks/dev/attribution_analysis/analysis_injection_config.yaml` — regex + insert_after for `ap_compute_attribution_end`
2. `src/it_examples/notebooks/publish/attribution_analysis/analysis_injection_config.yaml` — same regex fix + version bump (`0.1.0` → `0.4.0`)
3. `src/it_examples/notebooks/dev/attribution_analysis/attribution_analysis.ipynb` — markdown doc update (`compute_attribution_components(mlp_in_cache)` → `compute_attribution_components(mlp_in_cache, self.zero_positions)`), cleared stale error output from cell 18
4. `src/it_examples/notebooks/publish/attribution_analysis/attribution_analysis.ipynb` — same markdown doc update

### Test Results

- `analysis_inj_salient_logits_SLT`: PASSED (67.60s)
- `analysis_inj_salient_logits_CLT`: PASSED (65.37s)
- Full basic test suite: 814 passed, 56 skipped, 0 failed (812.88s)

## Common Pitfalls

### Multi-line refactors breaking insert_after semantics

When upstream code refactors a single-line expression to multi-line (e.g., adding arguments), using `insert_after: true` on the first line of the multi-line statement will insert code in the middle of the statement, causing `SyntaxError`. Find an alternative anchor line nearby and adjust `insert_after` accordingly.

### Forgetting the publish config

The publish directory (`src/it_examples/notebooks/publish/<notebook>/`) contains its own copy of `analysis_injection_config.yaml`. Both dev and publish configs must be updated. The `analysis_points.py` file is shared between dev and publish (identical copies).

### Variable availability at new insertion points

When changing the target line for a hook, always verify that the analysis point function's required variables are available at the new insertion point. Check `local_keys` in the `get_analysis_vars()` call within the analysis point function.

### Notebook JSON formatting

Notebooks are JSON files. Direct string replacement may fail due to JSON escaping. Use Python's `json` module for reliable notebook editing.
