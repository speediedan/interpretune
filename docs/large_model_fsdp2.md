# Large models on two cards: FSDP2 recipes

This page records how a model too large to load whole on one card (gemma-3-27b-it, with Gemma Scope 2 transcoders
spliced in) was fitted for latent-space tuning on two 32 GB cards, which settings mattered and why, and a working
tensor-parallel variant that was measured and not adopted. The settings are packaged as a standalone two-rank CLI,
`scripts/fsdp2_large_model_recipe.py`, which measures capacity (memory per card, host memory, load and step time) for
one batch shape. Wiring the same settings into the core loop's `fsdp2` placement
({doc}`core_loop_data_parallel`) is tracked in
[interpretune#852](https://github.com/speediedan/interpretune/issues/852), and the 27B capacity work as a whole
(including a quantized arm, still blocked) in [interpretune#833](https://github.com/speediedan/interpretune/issues/833).

All measurements below come from two AMD Radeon AI PRO R9700 (32 GB each) on PCIe with no peer link, 123 GiB of usable
host memory, ROCm and PyTorch 2.15, transformers 5.17. The 4B checks also ran on an NVIDIA RTX 4090 with an RTX 2070
SUPER. The trainables are rsLoRA adapters (rank 8) on the first 10 decoder layers, plus, in the latent phase, the
decoders of two Gemma Scope 2 16k affine transcoders (layers 31 and 40 at 27B).

## The recipe

```bash
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True   # PYTORCH_CUDA_ALLOC_CONF on NVIDIA
torchrun --nproc-per-node 2 scripts/fsdp2_large_model_recipe.py --model google/gemma-3-27b-it --out OUT \
  --tc-layers 31,40 --tc-repo google/gemma-scope-2-27b-it --batch 4x4096
```

| setting | what it does | flag (default) |
| --- | --- | --- |
| meta-device build | every rank builds the model from its config with parameters on the meta device, so no rank ever materializes the whole model | always |
| one FSDP2 unit per decoder layer | each spliced transcoder is its own unit, sharded before its layer, so the layer's all-gather does not carry it | always |
| `CPUOffloadPolicy` | the bf16 parameter shards and the optimizer state live pinned on the host; one unit at a time is gathered onto the card | always |
| DCP loading | `torch.distributed.checkpoint` with `HuggingFaceStorageReader`: each rank reads its own slices of the safetensors files | always |
| per-layer activation checkpointing | composable `checkpoint` on every decoder layer, applied before `fully_shard` | always |
| fp32 islands | the base stays bf16-stored; adapters, transcoders and norms are stored (and so computed) in fp32 | `--fp32-islands lora,tc,norm` |
| target-position loss | logits only for the last $K$ positions (`logits_to_keep`), so the vocabulary-by-sequence logits never exist | `--loss-positions 64` |

On the cards above this run loads in 21 s, passes the FVU gate (0.63 and 0.42 at layers 31 and 40), and trains at
21.5 GiB per card: 41.7 s per LoRA step (393 tokens/s per rank) and 43.5 s per latent step (185M trainable parameters,
377 tokens/s per rank), with 37.6 GiB peak RSS per rank and the host's MemAvailable never below 45 GiB.

The 4B model checks the same path in seconds:

```bash
torchrun --nproc-per-node 2 scripts/fsdp2_large_model_recipe.py --model google/gemma-3-4b-it --out OUT --reference
```

### Controls

- `--reference` compares the sharded model's logits on a fixed prompt with a plain `from_pretrained` copy on rank 0's
  card. FSDP2 alone runs the same kernels as the unsharded model, so in bf16 the default `--ref-atol 0` holds: the
  logits match bit for bit (measured on ROCm and CUDA). A perturbed norm weight fails it by about 28. Under `--tp 2`
  the same bf16 check fails by about 23 without any defect (see below); in fp32 TP passes at `--ref-atol 0.05`
  (0.0096 measured).
- With `--tc-layers`, every spliced layer's fraction of variance unexplained must be below `--max-fvu`. A transcoder
  whose decoder and skip connection are zeroed fails it.
- Refusals by name rather than silent defaults: a checkpoint key the model does not have, a model parameter that no
  checkpoint file supplies (DCP's planner refuses a missing file the same way), a weight converter transformers would
  apply (this loader only renames, by transformers' own conversion mapping), and any tensor-parallel plan entry or
  style the recipe does not place.

## Capacity at 27B

Per rank, two ranks. With FSDP2 alone each rank trains on its own batch, so the global batch is twice the per-rank one.
The first five rows predate DCP loading and fp32 islands; neither changed GPU memory or step time measurably (below).

| configuration | tokens per rank per step | peak per card | step | host |
| --- | --- | --- | --- | --- |
| LoRA phase at sequence 256 | 256 | 9.9 GiB | 5.5 s | RSS 37 GiB, MemAvailable low 45 GiB |
| latent phase, both transcoder decoders trainable (185M) | 256 | 9.8 GiB | 5.7 s | RSS 37 to 40 GiB |
| full logits, no checkpointing | 1x1024 | out of memory | | |
| full logits, per-layer checkpointing | 2x2048 | 21.1 GiB | 11.3 s | |
| full logits, per-layer checkpointing | 4x2048 | out of memory (fp32 logits alone are 8 GiB) | | |
| target-position loss (last 64), per-layer checkpointing | 4x2048 | 13.3 GiB | 21.5 s | RSS 37.6 GiB |
| **target-position loss (last 64), per-layer checkpointing** | **4x4096** | **21.5 GiB** | **44.4 s** (369 tok/s per rank) | RSS 37.5 GiB, MemAvailable low 45.6 GiB |

## What each setting bought, and what was measured and not taken

- **DCP loading** against a rank-0 read and broadcast per safetensors file: identical logits, load 31 s instead of
  42 s, and less host pressure (MemAvailable low 52 GiB instead of 47 GiB), with GPU memory and step time unchanged.
- **fp32 islands** cost 1.5% (LoRA phase) and 4% (latent phase) step time at unchanged peak GPU memory and 0.9 GiB
  more host memory. With a bf16-stored base, FSDP2 in PyTorch 2.15 accepts mixed stored dtypes in one unit and each
  tensor computes in its stored dtype, so `MixedPrecisionPolicy(param_dtype_override_fn=...)` adds nothing here. It
  matters for an fp32-stored master computing in bf16, which a 123 GiB host cannot hold at 27B (about 108 GB of fp32
  weights alone).
- **Unit grain** is the only wrapping choice that moved anything: sub-layer units saved 0.77 GiB for 19% more step
  time, and layer pairs cost 0.77 GiB for no gain. Last-layer no-reshard, explicit prefetch, and placing a transcoder
  inside its layer's unit were within the run-to-run spread (about 10%, which a single run per configuration misreads
  as an effect; interleave replicates).
- **Activation memory**: checkpointing every other layer and activation offload (`save_on_cpu`) do not fit at 27B.
  Selective checkpointing under composable `checkpoint` with FSDP2 fails on a replayed profiler op;
  `checkpoint_wrapper` would rename parameters, which the checkpoint key mapping then has to strip.
- **Chunked loss** (per-chunk logits recomputed in backward) works, but target-position loss removes the logits limit
  for tasks that score an answer or a short target rather than every token.

## Tensor parallelism: a working recipe, and why it did not save memory here

`--tp 2` applies transformers' own tensor-parallel plan for the text model (`base_model_tp_plan`: colwise q/k/v and
gate/up, rowwise o and down) with `parallelize_module` while the model is still on the meta device, then runs FSDP2
with offload on the data-parallel submesh of a `(dp=1, tp=2)` mesh. A size-1 data-parallel dimension keeps offload and
the fp32 islands working on the tensor-parallel DTensor parameters, and DCP loads them unchanged. The customizations:

- a LoRA-wrapped linear splits to match its base: colwise shards `base` and `lora_B` (whose colwise input sums the
  gradient reaching the replicated `lora_A` over the ranks), rowwise shards `base` and `lora_A`;
- the tied embedding stays replicated: sharding it re-registers the embedding's parameter and leaves the tied head
  holding the original, which nothing then loads;
- frozen q/k norms stay plain, replicated tensors, and so do the transcoders.

**Correctness is judged in fp32.** Tensor parallelism changes the matmul shapes and the partial-sum order, so a bf16 run
cannot match bf16 `from_pretrained` bit for bit. At 4B, bf16 TP differs from bf16 `from_pretrained` by up to 29 in a
logit, which looks like a defect, but measured against an fp32 reference it is as close as bf16 `from_pretrained`
itself:

| 4B, against fp32 `from_pretrained` | max $\lvert \Delta \text{logit} \rvert$ | mean | argmax agreement |
| --- | --- | --- | --- |
| `from_pretrained`, bf16 | 28.4 | 0.167 | 0.996 |
| TP=2, bf16 | 22.7 | 0.167 | 0.996 |
| TP=2, fp32 | 0.0065 | 0.00002 | 1.000 |

So check a TP run with `--tp 2 --dtype float32 --reference --ref-atol 0.05`. Every trainable's first-step gradient also
matched FSDP2 alone to a relative $3.5 \times 10^{-4}$ in fp32, and halving the gradient that reaches `lora_A` failed
that comparison at 0.76. Two traps in building that comparison: with $B = 0$, `lora_A`'s first gradient is exactly zero,
so the comparison cannot fail; and comparing a later phase at a nonzero learning rate compares different weights,
since Adam's first update is about $\eta \cdot \mathrm{sign}(g)$ and near-zero gradients that differ in sign move apart.

**At 27B, on these cards, TP lost on both time and memory.** With $dp = 1$ both ranks process one global batch, so the
comparison is at equal global batch:

| global batch | TP=2 | FSDP2 alone |
| --- | --- | --- |
| 4x4096 | 52.8 s, 311 tok/s, 18.9 GiB per card | 2x4096 per rank: 23.3 s, 703 tok/s, 13.4 GiB per card |
| 8x4096 | out of memory | 4x4096 per rank: 738 tok/s, 21.5 GiB per card |

The time goes to two activation all-reduces per layer, plus their backward and the recomputation, over PCIe without
a peer link. The memory is explained by `FSDP2MemTracker` (below) on one steady-state step at equal global batch
4x4096, in GiB per card:

| | FSDP2 alone | TP=2 |
| --- | --- | --- |
| end of forward (only the saved checkpoints are live): activations | 5.33 | 10.65 |
| backward peak: activations | 7.25 | 13.05 |
| backward peak: temporaries | 0.90 | 1.80 |
| backward peak: gathered parameters | 4.18 (plus a 0.77 all-gather buffer) | 3.80 |
| backward peak: total | 13.10 | 18.66 |

1. **Per-layer checkpointing keeps what lies between layers, and TP shards only what lies inside one.** The
   activations held across the step are each layer's saved input, the residual stream:
   $62 \times B L \times 5376 \times 2$ bytes, which predicts 5.08 GiB for FSDP2 alone at $B L = 2 \times 4096$ per rank
   (5.33 measured). Without sequence parallelism the residual stream is replicated on every TP rank, so each card holds
   the whole global batch's checkpoints: exactly twice as much. The tensors TP does split (per-head q/k/v and the MLP
   intermediate) exist only for the one layer being recomputed.
2. **Offload had already taken what TP shards.** With the parameters on the host, only the unit being computed is on the
   card, and TP halves that one unit (0.38 GiB here). Against FSDP2 alone TP never saves weight memory beyond that
   transient gather, offload or not: both hold half of every weight per rank. Its usually quoted weight saving is
   against replication.
3. **Host memory goes up** (offloaded parameters +2.5 GiB, gradients and optimizer state +1.3 GiB per rank), because the
   tied embedding, the norms, the replicated LoRA factors and the transcoders are whole on each rank.

Sequence parallelism would shard exactly the replicated term (predicted backward peak about 12.7 GiB here, slightly
below FSDP2 alone) without changing the communication volume, so it does not change the time. A trainable that sees
only a sequence shard then gets a per-rank partial gradient, which has to be reduced. TP is worth revisiting with a
peer link (at sequence 256 it was already faster: 3.7 and 5.5 s per LoRA and latent step against 5.3 and 6.1 s), with
sequence parallelism, or as the intra-node dimension of a larger FSDP2 by TP mesh.

## Measuring memory

`--mem-track` wraps the last step of each phase (after the optimizer state exists) in
`torch.distributed._tools.fsdp2_mem_tracker.FSDPMemTracker` and writes, per rank, the peak by category
(`mem_peak_rank<N>.json`: sharded and unsharded parameters, gradients, activations, temporaries, all-gather and
reduce-scatter buffers, optimizer state, per device) and a module-wise report through the step
(`mem_modulewise_rank<N>.txt`: before and after each pre-forward, post-forward, pre-backward and post-backward, and
the forward and backward peaks). The end-of-forward state is the cleanest read of the activation checkpoints, since
nothing else from the step is live. The tracker works on ROCm builds of PyTorch, whose device type is `cuda`. For a
timeline, `torch.cuda.memory._record_memory_history()` followed by `torch.cuda.memory._dump_snapshot()` around the same
step produces a snapshot that [pytorch.org/memory_viz](https://pytorch.org/memory_viz) renders, on ROCm as on CUDA.

## Pitfalls found on the way

- **Build the checkpoint-key renamer before `fully_shard`.** transformers selects its renames by the model's class,
  which `fully_shard` replaces, after which it falls back to a legacy mapping.
- **Turn `use_cache` off on every sub-config.** Gemma 3's composite config keeps it on in the text config; under
  activation checkpointing the recomputed forward then sees doubled key lengths.
- **Re-create non-persistent buffers after `to_empty`** (rotary frequencies, the embedding scale): they are in no state
  dict, and `to_empty` leaves them uninitialized.
- **Run one configuration per process when probing capacity.** An out-of-memory error inside an FSDP2 collective on one
  rank leaves the other blocked in the collective until the process group times out, so it cannot be caught and
  retried in-process.
- **Use `expandable_segments`** in the allocator configuration: the first 27B out-of-memory error, without it, left
  4.8 GiB reserved but unallocated.
- **An offloaded gradient lives on the host**, so gathering it with `DTensor.full_tensor()` over NCCL fails; move the
  local shard to the device first. A one-rank run does not show this, since it needs no communication.
