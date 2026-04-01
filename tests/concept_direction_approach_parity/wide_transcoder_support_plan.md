# Wide Transcoder (262k) Support Plan

## Status: De-prioritised — runtime bottleneck identified, no memory constraint

This document captures the analysis and future improvement plan for 262k-width transcoder
support in the concept-direction experiment pipeline. It is intended as the basis for a
GitHub issue to track the work.

---

## 1. Background

### What are wide transcoders?

GemmaScope provides transcoders at multiple widths. For Gemma 3 4B models two are relevant:

| Width | d_transcoder | Per-layer safetensors file | Layers | HuggingFace path suffix |
|-------|-------------|--------------------------|--------|------------------------|
| 16k   | 16,384      | ~362 MB                  | 34     | `width_16k_l0_small_affine` |
| 262k  | 262,144     | ~5.4 GB                  | 34     | `width_262k_l0_small_affine` |

The 262k transcoders are 14.9× larger per layer. Each safetensors file contains:
- `W_enc` (d_transcoder × d_model, bf16): **1.34 GB** per layer
- `W_dec` (d_model × d_transcoder, bf16): **1.34 GB** per layer
- `b_enc`, `b_dec`, skip-connection weights, activation thresholds, metadata

### Why support them?

Wider transcoders may provide finer-grained feature decomposition for interpretability
work. However, V9 experiments show that **core pipeline metrics (cosine_sim, Jaccard) are
identical between 16k and 262k** for the same model+concept pair, since these depend on
model internals rather than transcoder width. Gap deltas differ only marginally:

| Model+Concept | 16k embed Δ | 262k embed Δ | 16k store Δ | 262k store Δ |
|---------------|-------------|-------------|-------------|-------------|
| gemma-3-4b-pt capitals_states | +3.25 | +3.13 | −0.88 | −0.75 |
| gemma-3-4b-it capitals_states | −7.56 | −6.84 | −15.13 | −12.58 |
| gemma-3-4b-pt dog_cat | +2.25 | +2.81 | +4.06 | +6.25 |
| gemma-3-4b-it dog_cat | −1.25 | +0.25 | −0.63 | −0.88 |

The wider transcoder captures similar concept structure with marginal quantitative
differences. The value proposition will increase when experiments demand finer feature
granularity — for example, sign-aware feature subsets or causal feature density analysis.

### Neuronpedia dashboard availability

262k transcoders have **full layer coverage** (layers 0–33) on Neuronpedia for
gemma-3-4b-it via `gemmascope-2-transcoder-262k`, unlike `gemmascope-2-transcoder-16k`
which only covers the first 12 layers. This makes 262k the only option for full-layer
feature dashboard inspection on gemma-3-4b-it.

---

## 2. Current State

### What works

262k experiments run end-to-end via the same notebook harness as 16k:

```bash
python nb_experiment_launcher.py --config-pattern "262k" --continue-on-error
```

Configuration overrides in 262k YAML configs:
- `BATCH_SIZE: 64` (vs default 256) to prevent GPU OOM
- `MODEL_VARIANT: 4b_262k_pt` / `4b_262k_it` (selects 262k ModelSpec)

The `experiment_resource_utils.py` MODEL_SPECS registry includes both 262k variants:
- `("gemma3", "4b_262k_pt")` → `width_262k_l0_small_affine` (PT)
- `("gemma3", "4b_262k_it")` → `width_262k_l0_small_affine` (IT)

### Runtime bottleneck (~51 min vs ~4 min for 16k)

The runtime is dominated by `compute_attribution_components()` in circuit-tracer's
`SingleLayerTranscoder`. This function iterates all 34 layers, and for each layer:

1. **`encode_sparse()`** accesses `self.W_enc` → triggers `__getattr__` lazy load →
   `safe_open()` reads the full 1.34 GB encoder matrix from safetensors on disk into GPU
2. **`decode_sparse()`** calls `_get_decoder_vectors(feat_idx)` → `safe_open()` slice-reads
   only the active decoder rows (much smaller, but still a disk open per layer)

That's 68+ separate safetensors file open/read cycles during attribution setup alone,
plus additional reads during the backward passes in Phase 4 (feature attribution,
processed in batch_size=64 chunks up to max_feature_nodes).

### Memory is NOT a constraint

| Metric | Value | Interpretation |
|--------|-------|----------------|
| VMPeak | 93.6 GB | mmap virtual address space from safetensors `safe_open()` — **NOT physical memory** |
| VmHWM | 12.7 GB | Actual peak physical memory — well within 62 GiB system RAM |
| VmRSS | 6.1 GB | Resident set at measurement time |

The 93 GB VMPeak is from safetensors mmapping the 34 per-layer files (34 × 5.4 GB ≈ 184 GB
total on disk, but mmap only allocates virtual address space). Physical memory stays low
because `lazy_encoder` + `lazy_decoder` keep only biases resident (~18 MB total for 34
layers). The 4B model itself is ~8 GB in bf16.

### What's already well-optimised

- **`lazy_encoder=True` + `lazy_decoder=True`**: Auto-enabled when `offload='cpu'`. Only
  biases and activation thresholds are allocated as nn.Parameters; W_enc/W_dec are loaded
  on demand via `__getattr__` → `safe_open()`.
- **`batch_size=64`**: Prevents GPU OOM for 262k feature attribution backward passes.
- **CPU offload**: Transcoders offloaded to CPU after forward pass, MLP layers offloaded
  after forward, freeing GPU for backward passes.
- **Slice decoder reads**: `_get_decoder_vectors` reads only the rows for active features,
  not the full 1.34 GB W_dec matrix.
- **NVMe SSD cache**: All transcoder weights pre-cached on local NVMe at
  `/mnt/cache_extended/speediedan/.cache/huggingface/hub/` (sequential read throughput
  ~3.5 GB/s, but per-file open overhead dominates at 68+ operations).

---

## 3. Improvement Plan

### Priority 1: CPU tensor cache for encoder weights (HIGH — biggest win)

**Problem**: Every `encode_sparse()` call re-opens the safetensors file and reads the full
1.34 GB W_enc for that layer. During `compute_attribution_components()`, each of the 34
layers triggers at least one full W_enc read. Subsequent backward passes may trigger more.

**Proposal**: Add an optional CPU-resident LRU cache in `SingleLayerTranscoder` that retains
W_enc tensors after the first load. When `lazy_encoder=True` and `offload='cpu'`:
1. First access: `safe_open()` → read W_enc → store in CPU cache → copy to GPU
2. Subsequent access: copy from CPU cache to GPU (skip disk I/O)

**Expected speedup**: Eliminates disk I/O for repeated encoder accesses. CPU→GPU transfer
for 1.34 GB at PCIe 4.0 x16 (~25 GB/s) ≈ 54ms, vs safetensors read from NVMe (~400ms+
including file open overhead). For 34 layers × multiple passes, this could reduce total
runtime by 50-70%.

**Memory cost**: 34 layers × 1.34 GB = ~45 GB CPU resident. Fits within 62 GiB system RAM
if the 4B model (~8 GB) is the only other resident allocation.

**Implementation location**: `circuit_tracer/transcoder/single_layer_transcoder.py` in
`__getattr__` and/or a new `_cached_load()` method. Could be gated by a new
`cache_encoder_on_cpu: bool` config field in `CircuitTracerConfig`.

### Priority 2: Layer subsetting (MEDIUM — reduce unnecessary work)

**Problem**: `compute_attribution_components()` iterates all 34 layers even if only a
subset is analytically interesting.

**Proposal**: Add `layers: list[int] | None` parameter to `compute_attribution_components()`
(and surface it in `CircuitTracerConfig`). When specified, only the listed layers would
contribute encoder/decoder vectors to the attribution context.

**Expected speedup**: Linear reduction — if only 12 layers are analysed, runtime drops ~65%.

**Implementation location**: `circuit_tracer/transcoder/single_layer_transcoder.py` in
`compute_attribution_components()` loop.

### Priority 3: Prefetch/streaming encoder loads (LOW — complex, marginal gain if P1 done)

**Problem**: Even with P1 (CPU cache), the first-pass load of all 34 layers is sequential.

**Proposal**: Use a background thread to prefetch the next layer's W_enc while the current
layer's encode_sparse() is running on GPU. This overlaps disk I/O with GPU computation.

**Expected speedup**: ~30-40% on first-pass only (subsequent passes hit CPU cache if P1 is
implemented). Diminishing returns if P1 is already in place.

**Implementation complexity**: Moderate — requires thread-safe access to safetensors files
and careful GPU memory management.

### Priority 4: Reduce max_feature_nodes (LOW — user-configurable already)

The `max_feature_nodes` parameter already limits how many features go through the backward
pass in Phase 4. For exploration (vs production analysis), setting this to e.g. 4096
instead of 8192 halves the backward-pass work. Already controllable via YAML config;
no code change needed.

---

## 4. Decision Criteria for Activation

Current priority is low because:
1. **16k and 262k produce equivalent results** for the current experiment suite (§1)
2. **16k runs are 12× faster** (~4 min vs ~51 min)
3. **262k is only needed** for full-layer Neuronpedia dashboard access on gemma-3-4b-it or
   when finer-grained feature decomposition becomes experimentally necessary

Activate this work when:
- Sign-aware feature selection (OQ-B) or causal feature density analysis (OQ-G) requires
  the 262k feature granularity
- New concept pairs show meaningful 16k vs 262k divergence in gap deltas
- Neuronpedia integration requires full-layer 262k transcoders for dashboard links

---

## 5. References

- **experiment_summary.md point 9**: V9 262k transcoder comparison results
- **experiment_summary.md priority 6**: De-prioritised 262k optimisation rationale
- **V9_experimental_summary.md §Infrastructure Notes**: Runtime, GPU memory, batch size details
- **circuit_tracer/transcoder/single_layer_transcoder.py**: Lazy loading implementation
  (`__getattr__`, `encode_sparse`, `decode_sparse`, `compute_attribution_components`)
- **circuit_tracer/attribution/attribute_nnsight.py**: Full attribution pipeline phases
- **circuit_tracer/config/circuit_tracer.py**: `CircuitTracerConfig` fields (offload,
  lazy_encoder, lazy_decoder, batch_size)
- **circuit_tracer/utils/disk_offload.py**: Offload utilities (`disk_offload_module`,
  `cpu_offload_module`)
