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
is finetuning-scheduler's FSDP2 plan (`finetuning_scheduler.strategies.plan.shard_model`), so the core loop takes the
same grammar as finetuning-scheduler's own interfaces:

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

A plan is usually the model's block class, plus each spliced instrument's class, so that an instrument is gathered on
its own rather than with the block around it. Units are sharded deepest first whatever the plan's key order, so an
instrument spliced inside a block can be listed after the block, and then the interpretune module itself is sharded as
the root unit. An empty plan (`{}`) shards the whole module as one unit, which gathers every parameter for the entire
forward pass. Under `ddp` a plan is optional; without one the module is the only unit.

A `class:` key matches subclasses too, and a regex matches every module name it is a prefix of (anchor it with `$` to
select the blocks but not their children). A plan under the `single` placement, a key that selects no module, and a plan
without finetuning-scheduler installed are each refused by name.

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
differs in a shape (embeddings resized after the build, for one), every rank refuses, naming the keys. `meta_init`
cannot be combined with `lora_cfg` or `bitsandbytesconfig`, a module built only partly on the meta device is
refused, and so is `meta_init` on a composition whose model init does not build on the meta device (a
TransformerLens bridge, an nnsight or circuit-tracer model, or `defer_model_init`), rather than being ignored.

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
  model once. `meta_init` covers the core `from_pretrained` path only: TransformerLens, nnsight and circuit-tracer
  compositions build their models their own way and refuse it at construction (tracked in
  [interpretune#882](https://github.com/speediedan/interpretune/issues/882)).

The design, and the work that remains for attribution and J++ ops, is tracked in
[interpretune#803](https://github.com/speediedan/interpretune/issues/803).
