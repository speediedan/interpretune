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
                                        shard_units=["GPT2Block"]))
runner.train()
```

`fsdp2` needs `shard_units`: the class names of the modules to shard as their own units. That is usually the model's
block class, plus each spliced instrument's class, so that an instrument is gathered on its own rather than with the
block around it. `fully_shard` is applied deepest first, then to the interpretune module itself as the root unit. An
empty list shards the whole module as one unit, which gathers every parameter for the entire forward pass.

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
- Phased tuning with `FTSCoreCallback` under `ddp` or `fsdp2` needs finetuning-scheduler's FSDP2 adapter, which is not
  in a release yet.
- Mixed precision still uses the loop's `torch.autocast` rather than FSDP2's `MixedPrecisionPolicy`.
- The session is built fully on each rank before it is sharded, so each rank must fit the whole model once. Building on
  the meta device or the CPU and sharding before materializing is the next step for 12B-to-27B models.

The design, and the work that remains for attribution and J++ ops, is tracked in
[interpretune#803](https://github.com/speediedan/interpretune/issues/803).
