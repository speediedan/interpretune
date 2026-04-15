# Resource Management — Developer Notes

This document provides developer-level detail on memory management for the
concept-direction experiment framework in `tests/concept_direction_approach_parity/`.

## Architecture

```
YAML config (BATCH_SIZE, MAX_FEATURE_NODES)
  └─ papermill → notebook parameters cell
       └─ NotebookHarnessConfig(batch_size=…, max_feature_nodes=…)
            └─ cfg.session_kwargs  (property, bundles all kwargs)
                 └─ experiment_session(batch_size=…, max_feature_nodes=…)
                      └─ build_test_cfg(batch_size=…, max_feature_nodes=…)
                           └─ cfg.circuit_tracer_cfg.batch_size = batch_size
```

## Key Files

| File | Role |
|------|------|
| `../nb_experiment_harness/session.py` | shared model spec registry, `build_test_cfg()`, `experiment_session()` |
| `concept_direction.py` | `NotebookHarnessConfig`, concept-pair loading, direction computation |
| `../nb_experiment_harness/nb_harness_utils.py` | shared prompt/token/resource helpers |
| `../nb_experiment_harness/pipeline_patterns.py` | shared notebook phase runners |
| `../nb_experiment_harness/nb_experiment_launcher.py` | Papermill driver with inter-notebook `gc.collect()` + CUDA cleanup |
| `concept_direction_template.ipynb` | Notebook template; params cell provides `BATCH_SIZE`, `MAX_FEATURE_NODES` |
| `src/interpretune/utils/resource_mgmt.py` | `cleanup_python_cuda()`, `safe_clean_cuda()` |
| `src/it_examples/utils/nb_ui_utils.py` | `display_ablation_chart()` with `plt.close(fig)` for figure cleanup |

## OOM Root Cause (gemma2-it / V4 Wave)

The upstream `circuit_tracer/attribution/attribute_nnsight.py` line 150
calls `tracer.invoke(input_ids.expand(batch_size, -1))`.  When
`batch_size=256` (default), instruction-tuned models whose chat-template
prompts are ~25 tokens produce a batch of ~256 × 25 = 6400 token positions
per attribution pass.  On a 24 GiB GPU this exceeds available VRAM.

**Fix:** YAML configs for IT models set `BATCH_SIZE: 128`.  The override
flows through `NotebookHarnessConfig.session_kwargs` → `experiment_session()`
→ `build_test_cfg()` → `CircuitTracerConfig.batch_size`.

## Adding a New Model

1. Add a `ModelSpec` entry in `tests/nb_experiment_harness/configs/model_specs.yaml`.
2. Add a cfg_aliases class in `tests/core/cfg_aliases.py`.
3. Register the example module in `src/it_examples/example_module_registry.yaml`
   if the adapter combination is new.
4. Create a YAML config under `tests/concept_direction_approach_parity/configs/`.
5. Set `BATCH_SIZE` conservatively for the model size and prompt style:
   - 2B base → 256, 2B IT → 128, 4B IT → 64.

## Debugging VRAM Issues

1. Set `IT_RESOURCE_DEBUG=1` for per-fixture/per-test snapshots.
2. Run a single config through the launcher:
   ```bash
    python ../nb_experiment_harness/nb_experiment_launcher.py \
       --notebook concept_direction_template.ipynb \
       gemma2_it_capitals_states.yaml \
       --continue-on-error
   ```
3. Look for "Tried to allocate N MiB" in the traceback — this indicates
   peak VRAM at the allocation site.
4. Reduce `BATCH_SIZE` (halve it) and re-run.

## `PromptRenderMode` Values

The `PromptRenderMode` Literal accepts: `"plain"`, `"apply_chat_template"`,
`"gemma_dataclass"`.  The `render_prompt()` function treats any non-plain,
non-gemma_dataclass mode as `apply_chat_template` — but YAML configs should
use the exact Literal values for clarity.

## PYTORCH_CUDA_ALLOC_CONF

The notebook imports cell sets `expandable_segments:True` to reduce
fragmentation.  This is set via `os.environ.setdefault()` so it does not
override a user's existing setting.
