# Resource Management for Circuit-Tracer Experiments

This document describes the resource management controls available when running
concept-direction experiment notebooks via the shared
`tests/nb_experiment_harness/nb_experiment_launcher.py` launcher and the
`concept_direction_template.ipynb` notebook template.

## Background

Circuit-tracer attribution is VRAM-intensive.  The upstream
`attribute_nnsight.py` expands every prompt into a batch of size
`CircuitTracerConfig.batch_size` (default 256) via
`input_ids.expand(batch_size, -1)`.  For instruction-tuned models whose
chat-template prompts are longer than base-model prompts, this expansion
can exceed available GPU memory.

## Per-Experiment YAML Overrides

Each flat YAML config passed to the launcher can include:

| Key                | Default | Effect |
|--------------------|---------|--------|
| `BATCH_SIZE`       | `None` (uses cfg_aliases default of 256) | Reduces the per-attribution batch size.  Use 128 for 2B-IT models, 64 for 4B-IT models on a 24 GiB GPU. |
| `MAX_FEATURE_NODES`| `None` (uses cfg_aliases default of 8192) | Limits the number of feature nodes considered.  Smaller values reduce peak VRAM at the cost of circuit resolution. |

### Example

```yaml
# example config fragment
MODEL_FAMILY: gemma2
MODEL_NAME: google/gemma-2-2b-it
TRANSCODER_SET: gemma
BATCH_SIZE: 128           # halved from default 256 due to longer chat-template prompts
```

## How Overrides Flow

1. Papermill injects YAML values into the notebook parameters cell.
2. `NotebookHarnessConfig` stores `batch_size` and `max_feature_nodes`.
3. Every experiment function receives them via `cfg.session_kwargs`.
4. `experiment_session()` → `build_test_cfg()` applies the overrides to
   `CircuitTracerConfig` after the base config is constructed.

## PyTorch CUDA Allocator

The notebook imports cell sets:

```python
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
```

This enables the expandable-segments allocator strategy, reducing
fragmentation-related OOM on GPUs with ≤ 24 GiB VRAM.

## Inter-Notebook Cleanup

`nb_experiment_launcher.py` calls `gc.collect()` and
`torch.cuda.empty_cache()` between every notebook execution to reclaim
VRAM before the next experiment starts.

## Matplotlib Figure Cleanup

`display_ablation_chart()` in `nb_ui_utils.py` closes figures with
`plt.close(fig)` after `plt.show()` to prevent figure accumulation
across long notebook sequences.

## Utilities

`src/interpretune/utils/resource_mgmt.py` provides:

- **`cleanup_python_cuda()`** — runs `gc.collect()` + `torch.cuda.empty_cache()`.
- **`safe_clean_cuda(model)`** — context manager that moves a model to CUDA,
  tracks new tensors, and frees transient allocations on exit before moving
  the model back to CPU.

## Choosing `batch_size` Values

| Model class | Prompt style | Recommended `BATCH_SIZE` | Notes |
|-------------|-------------|--------------------------|-------|
| 2B base     | plain       | 256 (default)            | Short prompts fit comfortably |
| 2B IT       | chat template | 128                    | ~25-token prompts × 256 exceeds 24 GiB |
| 4B IT       | chat template | 64                     | Larger model + longer prompts |
