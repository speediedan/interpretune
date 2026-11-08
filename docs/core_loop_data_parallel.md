# Data-parallel training on the core loop

The core training and test loops run on one process by default. `SessionRunnerCfg.placement` spreads them across the
ranks `torchrun` launches:

| placement | each rank holds | gradients |
| --- | --- | --- |
| `single` (default) | the whole model, one process | local |
| `ddp` | the whole model | averaged across ranks |
| `fsdp2` | a shard of each parameter, its gradient and its optimizer state | reduce-scattered across ranks |

Both data-parallel placements use FSDP2's `fully_shard`. For `ddp` it runs over a `(replicate, shard)` mesh whose shard
dimension has size one, so every rank keeps whole, replicated parameters. Either way the module is modified in place
rather than wrapped, so `module.model`, hooks and a tunable view's names stay as they were.

## Launching

From the CLI:

```bash
torchrun --nproc-per-node 2 --no-python interpretune --run_command train --config <config.yaml> --run_cfg.placement ddp
```

or in code, under `torchrun --nproc-per-node 2 train.py`:

```python
from interpretune.config import SessionRunnerCfg
from interpretune.runners import SessionRunner

runner = SessionRunner(SessionRunnerCfg(it_session=session, max_epochs=2, placement="fsdp2",
                                        fsdp_plan={"class:GPT2Block": {}}))
runner.train()
```

## The FSDP2 plan

`fsdp2` needs `fsdp_plan`: the modules to shard as their own units, each with its `fully_shard` keyword arguments. It
is Fine-Tuning Scheduler's FSDP2 plan (`finetuning_scheduler.strategies.plan.shard_model`), so the core loop takes the
same grammar as Fine-Tuning Scheduler's own interfaces:

| key | selects |
| --- | --- |
| `model.layers.3` | the module with that name |
| `model\.layers\.\d+$` | every module whose name the regular expression matches from the start |
| `class:GPT2Block` (or `class:<module>.<qualname>`) | every instance of that class or of a subclass |

Each value is a mapping of `fully_shard` keyword arguments (or `null`), plus two aliases: `act_ckpt: composable`
applies composable activation checkpointing to the unit before it is sharded, and `cpu_offload_policy` (a mapping of
`CPUOffloadPolicy` arguments) becomes its `offload_policy`. `fsdp_default_kwargs` gives every planned unit keyword
arguments its entry does not override; all but `act_ckpt` also apply to the root unit.

```yaml
run_cfg:
  placement: fsdp2
  fsdp_plan:
    class:GPT2Block: {act_ckpt: composable}
  fsdp_default_kwargs:
    cpu_offload_policy: {pin_memory: true}
```

Pinned offload lets the copies between host and accelerator run asynchronously, but PyTorch's pinned host allocator
rounds every allocation up to a power of two by default, so offloaded shards can take up to twice their size in host
memory. Setting `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:0` in the launch environment turns the rounding off
and keeps the shards pinned. On the circuit-tracer walkthrough below, it brings each rank's pinned shards from about
10.3 to about 7.0 GiB, about the host memory the same shards take unpinned, on both CUDA and ROCm. That was measured
for a forward-only run, whose offloaded shards are allocated once; whether a training run's repeated pinned
allocations reuse memory less well without the rounding has not been measured.

A plan is usually the model's block class, plus each spliced instrument's class, so that an instrument is gathered on
its own rather than with the block around it. Units are sharded deepest first whatever the plan's key order, so an
instrument spliced inside a block can be listed after the block, and then the Interpretune module itself is sharded as
the root unit. An empty plan (`{}`) shards the whole module as one unit, which gathers every parameter for the entire
forward pass. Under `ddp` a plan is optional; without one the module is the only unit.

A `class:` key matches subclasses too, and a regex matches every module name it is a prefix of (anchor it with `$` to
select the blocks but not their children). A plan under the `single` placement, a key that selects no module, and a plan
without Fine-Tuning Scheduler installed are each refused by name.

FSDP2 gathers a unit's parameters in hooks on its `forward`, so a module called some other way computes with sharded
parameters. Some compositions do exactly that: nnsight calls the Hugging Face model directly rather than through the
Interpretune module, and circuit-tracer reads transcoders through methods such as `encode` and calls their activation
functions directly. Such a composition declares these calls (`fsdp_direct_calls`), the placement registers the methods
as FSDP2 forward methods, and each called module must be its own unit. A circuit-tracer composition over nnsight with
Gemma 2 transcoders, for one, needs:

```yaml
run_cfg:
  placement: fsdp2
  fsdp_plan:
    class:Gemma2DecoderLayer: {}
    class:SingleLayerTranscoder: {}
    class:JumpReLU: {}
    class:Gemma2ForCausalLM: {}
```

A plan that leaves a called module to the root unit is refused, naming the `class:` keys to add. FSDP2 cannot shard a
0-d parameter, which it refuses by name too.

## Models larger than one rank's accelerator

By default every rank builds the whole model before it is sharded, so each rank must fit it once. For a model loaded
with `from_pretrained` (`hf_from_pretrained_cfg`), `meta_init: true` builds it on the meta device instead:

```yaml
module_cfg:
  init_args:
    hf_from_pretrained_cfg:
      init_args:
        meta_init: true
```

Every rank then builds the model from its configuration without any weights, and the launcher's rank 0 also loads the
checkpoint on the CPU. The placement shards the empty model with the plan, materializes each rank's shards on its
device, and loads them from rank 0's copy (`torch.distributed.checkpoint.state_dict.set_model_state_dict` with
`broadcast_from_rank0`), broadcasting the buffers too, since non-persistent ones such as rotary frequencies are in no
state dict. No accelerator ever holds the whole model; rank 0's host holds one copy of it while loading.

When even one host copy is too much, `meta_init_load: dcp` loads each rank's shards straight from the checkpoint's
safetensors files with distributed checkpointing (`torch.distributed.checkpoint` with `HuggingFaceStorageReader`), so
every rank reads only the slices it holds and no host ever holds the whole model:

```yaml
module_cfg:
  init_args:
    hf_from_pretrained_cfg:
      init_args:
        meta_init: true
        meta_init_load: dcp
```

Only the parameters are built on the meta device then, while buffers such as rotary frequencies, which no checkpoint
holds, are computed on the CPU as the model is built and kept through the placement. Checkpoint keys are renamed the
way `from_pretrained` renames them, and a parameter tied to one the checkpoint supplies (an LM head tied to the
embedding) needs no key of its own. Before reading anything, every rank refuses a checkpoint key that matches no model
key and a parameter no key supplies, naming them, and a model whose checkpoint needs transformers' weight converters
(which merge or reshape tensors rather than rename them) is refused in favor of the default `rank0` load. A Hub model's
safetensors files are downloaded to the local cache first, on every rank.

The load is checked rather than trusted: when rank 0's copy is missing a key of the model, has one the model lacks, or
differs in a shape (embeddings resized after the build, for one), every rank refuses, naming the keys. With
`lora_cfg`, rank 0's copy takes the same LoRA adapters as the model, so their initial values reach every rank with the
base weights. `meta_init` cannot be combined with `bitsandbytesconfig` (FSDP2 does not shard quantized parameters), a
module built only partly on the meta device is refused, and so is `meta_init` on a composition whose model init does
not build on the meta device (a TransformerLens bridge, a plain nnsight model, or `defer_model_init`), rather than being
ignored.

A circuit-tracer composition over nnsight builds on the meta device too: its replacement model is built from the
model's config with every Hugging Face and transcoder parameter on the meta device, and rank 0 also loads the model and
its transcoders on the CPU, which the placement loads every rank from into the Hugging Face module nnsight wraps (the
transcoders are registered inside it). The transcoders must be loaded eagerly (`lazy_encoder: false`,
`lazy_decoder: false`), since a lazily loaded weight is read from disk rather than held as a parameter, and the
composition needs an `nnsight_cfg` without `device_map` or remote execution; `meta_init_load: dcp` is refused, since the
transcoders are not in the model's checkpoint. Each of these is refused by name.

## Walkthrough: a sharded circuit-tracer composition from the CLI

The `rte` component ships a run that puts the pieces above together: the RTE test loop of circuit-tracer over nnsight
on gemma-3-4b-it, with every layer's Gemma Scope 2 transcoder (16k features), built on the meta device and sharded with
FSDP2 across two ranks with CPU offload. The configuration it runs is
`rte_demo.gemma3.circuit_tracer+nnsight.fsdp2`, and the run itself is
`runs/rte_test.gemma3_4b.circuit_tracer.fsdp2.yaml` in the same component.

### Running it

The model is gated, so the launching account needs access to `google/gemma-3-4b-it` (through
`HF_GATED_PUBLIC_REPO_AUTH_KEY` or a cached Hugging Face login). The component ships its own module code, which
Interpretune runs only when you opt in (see [the Hub trust posture](hub_trust_posture.md)), and loading a component
never downloads, so fetch it first:

```bash
python -c "import interpretune.hub; interpretune.hub.pull('speediedan/rte')"
IT_TRUST_REMOTE_CODE=1 torchrun --standalone --nproc-per-node 2 --no-python interpretune --run_command test \
  --config hub://speediedan/rte/runs/rte_test.gemma3_4b.circuit_tracer.fsdp2.yaml
```

The run configuration names the composition by `session_ref` and adds only the sharding: `meta_init: true` under
`hf_from_pretrained_cfg`, `placement: fsdp2`, the plan, `cpu_offload_policy: {pin_memory: true}` for every unit, and
`limit_test_batches: 16`. Every rank builds the replacement model, transcoders included, on the meta device; rank 0
also loads the model and its transcoders on the CPU, and the placement loads each rank's shards from that copy.

### The plan's units, and why each is there

| plan key | why it is a unit |
| --- | --- |
| `class:Gemma3DecoderLayer` | the transformer blocks, so that one block is gathered at a time |
| `class:SingleLayerTranscoder` | circuit-tracer calls each transcoder's `encode` and `decode` directly, outside the module's forward, so each transcoder must be its own unit with those methods registered |
| `class:JumpReLU` | circuit-tracer calls the transcoders' activation functions directly too |
| `class:Gemma3ForConditionalGeneration` | nnsight calls the Hugging Face model directly rather than through the Interpretune module, so the parameters left outside the other units (the embeddings among them) would otherwise never be gathered |
| `class:SiglipEncoderLayer` | the vision tower's layers; the RTE task never calls the tower, and as units they stay sharded and offloaded instead of being gathered with the Hugging Face model's unit |

Leaving out a transcoder, `JumpReLU` or the Hugging Face model, the modules called outside the forward, is refused
by name, with the `class:` keys to add. The Interpretune module itself is
sharded last, as the root unit.

### Memory

Measured with this plan and offload policy at 4B in bf16, against the same composition in one process on one GPU:

| | accelerator peak | host |
| --- | --- | --- |
| one process, one GPU | 14.0 GiB | |
| two ranks, sharded and offloaded | 2.80 GiB per rank | each rank's pinned shards: about 10.3 GiB (about 7.0 GiB without the allocator's rounding, see above); rank 0 also holds the CPU source while loading |

So the run fits two cards of unequal size (a 24 GB and an 8 GB card were used). Rank 0's CPU source is mostly the
model's safetensors files mapped into memory (about 8.5 GiB, released after the load), plus the transcoders, which are
loaded eagerly into memory (about 5.8 GiB). When each card can hold half the model with room to spare, drop
`fsdp_default_kwargs`, and the shards stay on the accelerators.

The same run with the composition switched to gemma-3-27b-it and all 62 of its Gemma Scope 2 transcoders fits two
32 GB cards on a 123 GiB host, with the allocator's rounding off: each rank's pinned shards take 37.6 GiB, each card
peaks at about 10 GiB, and the host keeps about 15 GiB available at the peak, while rank 0 still holds the CPU source.
With the rounding on, the pinned shards are projected at 49.2 GiB per rank, which would not fit.

### Comparing with one process

The same configuration runs in one process when the run drops the plan and the offload policy, places nnsight's model
on the GPU, and uses the `single` placement:

```yaml
session_ref: speediedan/rte:rte_demo.gemma3.circuit_tracer+nnsight.fsdp2
seed_everything: 0
registered_cfg:
  module_cfg:
    init_args:
      nnsight_cfg:
        init_args:
          device_map: cuda
run_cfg:
  placement: single
  limit_test_batches: 32
```

`limit_test_batches` counts each rank's batches, so the one-process run takes twice the sharded run's limit to see the
same samples. With the sharded run at `limit_test_batches: 8` (64 samples across the two ranks) and the one-process
run at 16, both report an accuracy of 0.7812 (50 of 64). Below the predictions, the sharded
readings were compared with one process directly: plain logits, an nnsight trace of `lm_head`, the transcoder
activations, and the logits and activations of a feature intervention are bitwise equal on both ranks, with an
intervention that moves the logits by up to 4.86, so the comparison could fail. Pin the component revision in both
runs (`speediedan/rte:<config>@<revision>`) so that both read the same configuration.

## What the runner does

1. **Joins the process group** from the launcher's `WORLD_SIZE`, `RANK` and `LOCAL_RANK`. When accelerators are
   present, `cuda:LOCAL_RANK` becomes the current device first, so a module placed on `"cuda"` lands on its rank's
   device.
2. **Builds the session one rank at a time**, rank 0 first. Only each host's local rank 0 prepares data: concurrent
   builds race on the prepared-dataset cache, and a later rank re-preparing would rewrite files an earlier rank has
   already memory-mapped.
3. **Places the module** before its optimizers are built (sharding replaces the parameters). It records each rank's
   device, broadcasts rank 0's parameters and buffers (`fully_shard`, unlike `DistributedDataParallel`, keeps each
   rank's own values, so differently initialized ranks would otherwise train different models with no error), then
   shards.
4. **Shards the data.** Each loader the datamodule returns is rebuilt over a `DistributedSampler`, with the same batch
   size, collate function and worker settings. A shuffled loader stays shuffled, using the same seed on every rank, and
   the train sampler's epoch is set each epoch. The global batch is the per-rank batch times the number of ranks.
5. **Averages metrics across ranks** before callbacks, lr schedulers and transition policies see them, and prints them
   on rank 0 only. Ranks that disagreed on a metric would disagree on a phase transition, and then deadlock at the next
   collective.

## What is refused

Each of these fails with a message naming what was found and what would be accepted, rather than running as something
else:

- `single` launched as one of several ranks (every rank would train its own copy on the whole dataset), or a
  data-parallel placement with fewer than two ranks.
- A rank whose module spans several devices (a layer split combined with data parallelism), two ranks on one
  accelerator, or ranks on different device types.
- A parameter `fully_shard` cannot shard: a non-contiguous one, such as TransformerLens's split q/k/v weights, which are
  transposed views of the packed attention weight.
- A loader that cannot be rebuilt faithfully: not a `DataLoader`, an `IterableDataset`, or a sampler other than
  PyTorch's sequential sampler and its random sampler without replacement (shard such data by rank in the datamodule).
- Ranks that logged different metric names in an epoch.
- A data-parallel placement on the analysis runner, since attribution and analysis ops are not yet designed for it.

## Limits

- Without `drop_last`, `DistributedSampler` pads the last round with repeated samples so that every rank takes the same
  number of batches. A metric averaged over a dataset that does not divide evenly therefore counts a few samples twice.
- The core loop's `checkpoint_state` is not distributed-aware.
- Phased tuning with `FTSCoreCallback` is tested under `fsdp2` with a plan on two CPU ranks (two phases, matching one
  process on the combined batch step for step), not yet under `ddp` or on accelerators.
- FSDP2's `MixedPrecisionPolicy` can be passed as a unit's `mp_policy` from Python, but has no configuration alias yet,
  so a YAML run uses the loop's `torch.autocast`.
- Without `meta_init`, the session is built fully on each rank before it is sharded, so each rank must fit the whole
  model once. `meta_init` covers the core `from_pretrained` path and circuit-tracer over nnsight (with `rank0` loading
  only): TransformerLens and plain nnsight compositions build their models their own way and refuse it at construction
  (tracked in [interpretune#882](https://github.com/speediedan/interpretune/issues/882)).

The design, and the work that remains for attribution and J++ ops, is tracked in
[interpretune#803](https://github.com/speediedan/interpretune/issues/803).
