# Analysis Injection Framework Usage Guide

The analysis injection framework enables detailed inspection of internal data flow in external packages (like `circuit_tracer`) without modifying the upstream code. It works by dynamically patching target modules with hook calls that capture intermediate values during execution.

## Quick Start

### 1. Configure Your Analysis Points

Create or modify an `analysis_injection_config.yaml` file that specifies:

- **Settings** including logging and module paths
- **Shared context** for passing data between hooks
- **File hooks** as a dictionary mapping hook IDs to their configurations

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

### 2. Set Up Analysis Injection in Your Notebook

```python
from pathlib import Path
from it_examples.utils.analysis_injection import orchestrator

# Path to your config file
config_path = Path.cwd() / "analysis_injection_config.yaml"

# Optional: YAML string to override specific config values
config_overrides = None  # Or a YAML string like "file_hooks:\n  ap_hook:\n    enable: false"

# Optional: Additional analysis functions defined in notebook
NOTEBOOK_ANALYSIS_FUNCTIONS = {}  # Or {"hook_id": my_function, ...}

# Initialize the analysis injector
analysis_injector = orchestrator.setup_analysis_injection(
    config_path=config_path,
    target_package="circuit_tracer",
    config_overrides=config_overrides,
    analysis_functions=NOTEBOOK_ANALYSIS_FUNCTIONS,
)

print(f"Enabled: {orchestrator.HOOK_REGISTRY._enabled}")
print(f"Registered hooks: {len(orchestrator.HOOK_REGISTRY._hooks)}")
```

### 3. Run Your Analysis Code

Execute the code that triggers the instrumented functions. The hooks will automatically capture data.

```python
# Run the analysis (hooks fire automatically)
graph = ct_module.generate_graph(prompt="The capital of Texas is")
```

### 4. Access Captured Data

```python
# Get output from a specific hook
forward_pass_data = analysis_injector.get_output("ap_forward_pass_end")

# Access specific captured values (returns None if hook didn't fire)
hook_data = analysis_injector["ap_build_input_vectors_end"]
if hook_data:
    edge_matrix = hook_data["edge_matrix"]

# Get all captured data
all_data = analysis_injector.get_analysis_data()
```

### 5. Teardown

```python
# Clean up when done
analysis_injector.teardown()
```

## Configuration Reference

### Hook Definition Fields

| Field | Description | Required |
|-------|-------------|----------|
| `file_path` | Path to target file relative to package root | Yes |
| `regex_pattern` | Python regex matching the insertion line | Yes |
| `insert_after` | `true` to insert after matched line, `false` for before | Yes |
| `enable` | Whether hook is active (default: `true`) | No |
| `description` | Human-readable description of the hook | No |

### Regex Pattern Tips

- Use `^\s*` to match any leading whitespace
- Escape special regex characters: `\.`, `\(`, `\[`
- Test patterns against the installed package:
  ```bash
  grep -n "your_pattern" /path/to/site-packages/package/file.py
  ```


### Config Overrides

Override config values at runtime via inline YAML string or path to a YAML file:

```python
# Inline YAML string
config_overrides = """
file_hooks:
  ap_forward_pass_end:
    file_path: attribution/attribute_transformerlens.py  # Updated path
"""

analysis_injector = orchestrator.setup_analysis_injection(
    config_path=config_path,
    target_package="circuit_tracer",
    config_overrides=config_overrides,
)
```

## Writing Analysis Point Functions

Analysis point functions receive `local_vars: dict[str, Any]` containing local variables at the hook insertion point. Use the `get_analysis_vars` helper to access both local variables and shared context:

```python
# In analysis_points.py
from typing import Any
from it_examples.utils.analysis_injection.analysis_hook_patcher import get_analysis_vars
from it_examples.utils.analysis_injection.orchestrator import analysis_log_point

def ap_forward_pass_end(local_vars: dict[str, Any]) -> None:
    """Capture data after forward pass completes."""
    # Access local vars and shared context via helper
    v = get_analysis_vars(
        context_keys=["target_token_analysis"],  # From shared_context
        local_keys=["ctx", "model"],              # From function locals
        local_vars=local_vars
    )

    target_logits = v["ctx"].logits[0, -1, v["target_token_analysis"].token_ids]
    data = {
        "target_token_ids": v["target_token_analysis"].token_ids,
        "target_logits": target_logits,
    }
    analysis_log_point("after forward pass", data)
```

### Exporting Analysis Functions

Define an `AP_FUNCTIONS` dict at module level to auto-register functions:

```python
AP_FUNCTIONS = {
    "ap_forward_pass_end": ap_forward_pass_end,
    "ap_build_input_vectors_end": ap_build_input_vectors_end,
    # ... more hooks
}

__all__ = ["AP_FUNCTIONS"]
```


## Debugging

### Check the Analysis Log

During setup, a log file is created that shows which hooks matched:

```
📝 Analysis output will be logged to: /tmp/attribution_flow_analysis_20260116_114129.log
```

Examine this log to verify:
- Which regex patterns matched (and at which lines)
- Which hooks were registered
- Any errors during hook execution

### Verify Hook Registration

```python
from it_examples.utils.analysis_injection import orchestrator

print(f"Enabled: {orchestrator.HOOK_REGISTRY._enabled}")
print(f"Registered hooks: {list(orchestrator.HOOK_REGISTRY._hooks.keys())}")
```

### Common Issues

**Hook returns `None`:**
- The regex pattern didn't match any line in the target file
- The code path containing the hook wasn't executed
- Check the analysis log for "Regex matched" messages

**`SyntaxError` after setup:**
- The import insertion logic failed to find a valid insertion point
- Check for unusual file structure (docstrings, multi-line imports)

**`TypeError: 'NoneType' object is not subscriptable`:**
- Accessing a hook that didn't capture data
- Use `analysis_injector.get_output("hook_id")` which handles None gracefully

## Example: Attribution Analysis Notebook

See `src/it_examples/notebooks/publish/attribution_analysis/attribution_analysis.ipynb` for a complete example that:

1. Sets up analysis injection for `circuit_tracer`
2. Captures intermediate values during graph generation
3. Visualizes the captured data (edge matrices, influence scores, etc.)
4. Demonstrates hook chaining and context sharing

## Maintaining Analysis Injection Configs

When the target package updates:

1. **Check if module paths changed** - Update `file_path` in config
2. **Verify regex patterns still match** - Test against installed package
3. **Update variable references** - If upstream renamed variables
4. **Test with baseline environment** - Compare against known-working version

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
