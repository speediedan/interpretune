---
applyTo: "**/src/it_examples/**"
---

# Analysis Injection Framework - Development Instructions

## Overview

The analysis injection framework allows runtime instrumentation of target package code (e.g., `circuit_tracer`) by inserting hook calls at specific code locations identified by regex patterns. This enables detailed analysis of internal data flow without modifying the upstream package.

## Architecture

### Key Components

1. **`analysis_injection_config.yaml`** - Defines hooks: target files, regex patterns, hook IDs, and insertion positions
2. **`analysis_hook_patcher.py`** - Patches target modules at runtime with hook calls
3. **`analysis_points.py`** - Contains the analysis functions executed when hooks fire
4. **`orchestrator.py`** - Coordinates setup, manages hook registry, and provides data access

### How Patching Works

1. Config is loaded specifying target files and regex patterns
2. For each target file, a patched copy is created in a temp directory with hook calls inserted
3. Patched modules are loaded into `sys.modules`, replacing originals
4. References to functions AND classes from patched modules are updated in all importing modules
5. When patched code executes, hooks fire and analysis functions capture data

## Config File Structure

The `analysis_injection_config.yaml` uses this structure (note: `file_hooks` is a dict, not a list):

```yaml
settings:
  enabled: true
  target_package_version: "0.1.0"
  log_to_console: false
  log_to_file: true
  log_dir: /tmp
  analysis_log_prefix: attribution_flow_analysis
  analysis_points_module_path: ./analysis_points.py

shared_context:
  target_tokens:
    - "▁Dallas"
    - "▁Austin"

file_hooks:
  ap_forward_pass_end:  # Hook ID is the dict key
    file_path: attribution/attribute_transformerlens.py
    enable: true
    regex_pattern: '^\s*ctx\._resid_activations\[-1\]\s*=\s*model\.ln_final'
    insert_after: true
    description: "After forward pass on original replacement model"
```

## Analysis Point Function Signature

Analysis point functions receive `local_vars: dict[str, Any]` containing the local variables at the hook insertion point:

```python
def ap_forward_pass_end(local_vars: dict[str, Any]) -> None:
    # Use get_analysis_vars to access local vars and shared context
    v = get_analysis_vars(
        context_keys=["target_token_analysis"],
        local_keys=["ctx", "model"],
        local_vars=local_vars
    )
    # Access variables via v["ctx"], v["model"], v["target_token_analysis"]
    data = {"target_logits": v["ctx"].logits}
    analysis_log_point("after forward pass", data)
```

## Common Update Scenarios

### When Upstream Module Paths Change

If the target package reorganizes its code (e.g., `attribution/attribute.py` → `attribution/attribute_transformerlens.py`):

**Files to update:**
1. `analysis_injection_config.yaml` - Update `file_path` for affected hooks
2. Notebook inline config overrides - Search notebooks for `config_overrides` variables
3. Analysis points module - Update any hardcoded module references

### When Upstream Code Structure Changes

If regex patterns no longer match due to code changes:

1. Check the installed package code: `grep -n "pattern" /path/to/site-packages/package/file.py`
2. Update regex patterns in `analysis_injection_config.yaml`
3. Verify with: Look at analysis log for "Regex matched" messages during setup

### When Analysis Point Functions Fail

If hooks fire but analysis functions throw exceptions:

1. Check variable names in `analysis_points.py` against current upstream code
2. Variables accessed via `local_vars` must exist at hook insertion point
3. Use `get_analysis_vars()` helper for safe access with context keys

## Debugging Workflow

### Setting Up a Baseline Environment

Maintain a separate working tree with known-working commits for comparison:

```bash
# Create baseline environment
git worktree add ~/repos/it-release release/0.1.x
./scripts/build_it_env.sh --repo_home=~/repos/it-release --target_env_name=it_release

# Compare behavior between environments
source /mnt/cache/$USER/.venvs/it_release/bin/activate  # baseline
source /mnt/cache/$USER/.venvs/it_latest/bin/activate   # development
```

### Debugging Hook Failures

1. **Check the analysis log** - Created during setup, logs which hooks matched:
   ```
   📝 Analysis output will be logged to: /tmp/attribution_flow_analysis_*.log
   ```

2. **Verify regex patterns match** - The log shows each successful match:
   ```
   Regex matched in /path/to/file.py: pattern='...' at line N
   ```

3. **Check notebook output cells** - Failed cells show error messages and tracebacks

4. **Examine the output notebook** - Tests write to `/tmp/pytest-*/test_*/notebook_outputs/`

### Common Failure Patterns

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `analysis_injector["hook_id"]` returns `None` | Hook didn't fire | Check regex patterns, verify code path executed |
| `SyntaxError` after patching | Import insertion failed | Check for docstrings, multi-line imports |
| `isinstance()` check fails | Class reference not updated | Ensure `inspect.isclass()` check in patcher |
| `'X' object has no attribute 'seek'` | Type mismatch due to class identity | Class from original module vs patched module |

### Verifying Patches Applied Correctly

```python
# In notebook or test, after setup:
from it_examples.utils.analysis_injection import orchestrator

print("Hook registry status:")
print(f"  Enabled: {orchestrator.HOOK_REGISTRY._enabled}")
print(f"  Registered hooks: {len(orchestrator.HOOK_REGISTRY._hooks)}")

# Check specific module was patched
import sys
module = sys.modules.get('circuit_tracer.graph')
print(f"Module file: {module.__file__}")  # Should show temp patched path
```

## Import Insertion Rules

The patcher inserts imports after finding a valid insertion point. Current logic handles:

- Multi-line docstrings (single `'''` or `"""` and triple-quoted)
- Multi-line imports with parentheses: `from x import (\n    a,\n    b\n)`
- Comments and blank lines at file start
- `from __future__ import` statements (must remain first)

**Test import insertion** by examining patched files in the temp directory logged during setup.

## Testing Analysis Injection Changes

```bash
# Run specific notebook test
IT_RUN_STANDALONE_TESTS=1 python -m pytest \
  "tests/examples/test_notebooks.py::test_attribution_analysis_notebook[analysis_inj_salient_logits_SLT]" -v

# Run all circuit-tracer notebook tests
IT_RUN_STANDALONE_TESTS=1 python -m pytest \
  "tests/examples/test_notebooks.py" -k "circuit_tracer or attribution_analysis" -v
```

## Checklist for Analysis Injection Updates

- [ ] Verify target package version and code structure
- [ ] Update `file_path` in config if module paths changed
- [ ] Update regex patterns if code structure changed
- [ ] Check notebook inline `config_overrides` variables
- [ ] Update `analysis_points.py` variable references if needed
- [ ] Run notebook tests and check analysis log for hook matches
- [ ] Compare behavior against baseline environment if issues persist
