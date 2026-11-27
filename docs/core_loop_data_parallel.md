# Data-parallel training on the core loop

The core training and test loops run on one process by default. `SessionRunnerCfg.placement` spreads them across the
ranks `torchrun` launches:

| placement | each rank holds | gradients |
| --- | --- | --- |
| `single` (default) | the whole model, one process | local |
| `ddp` | the whole model | averaged across ranks |
| `fsdp2` | a shard of each parameter, its gradient and its optimizer state | reduce-scattered across ranks |
| `tp` | a slice of each layer `tp_plan` selects; every rank sees the same batch | computed together |

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

Each value is a mapping of `fully_shard` keyword arguments (or `null`), plus three aliases: `act_ckpt: composable`
applies composable activation checkpointing to the unit before it is sharded, `cpu_offload_policy` (a mapping of
`CPUOffloadPolicy` arguments) becomes its `offload_policy`, and `mp_policy` (a mapping of `MixedPrecisionPolicy`
arguments, dtypes by name) becomes its mixed precision policy. `fsdp_default_kwargs` gives every planned unit keyword
arguments its entry does not override; all but `act_ckpt` also apply to the root unit.

```yaml
run_cfg:
  placement: fsdp2
  fsdp_plan:
    class:GPT2Block: {act_ckpt: composable}
  fsdp_default_kwargs:
    cpu_offload_policy: {pin_memory: true}
    mp_policy: {reduce_dtype: float32}
```

Without a `param_dtype` every tensor computes in the dtype it is stored in, so a bf16 model keeps fp32 islands (its
trainable parameters, say) and `reduce_dtype: float32` reduces their gradients in fp32.

Pinned offload lets the copies between host and accelerator run asynchronously, but PyTorch's pinned host allocator
rounds every allocation up to a power of two by default, so offloaded shards can take up to twice their size in host
memory. Setting `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:0` in the launch environment turns the rounding off
and keeps the shards pinned. On the circuit-tracer walkthrough below, it brings each rank's pinned shards from about
10.3 to about 7.0 GiB, about the host memory the same shards take unpinned, on both CUDA and ROCm. That was measured
for a forward-only run, whose offloaded shards are allocated once; whether a training run's repeated pinned
allocations reuse memory less well without the rounding has not been measured.

A plan is usually the model's block class, plus each spliced latent model's class, so that a latent model is gathered
on its own rather than with the block around it. Units are sharded deepest first whatever the plan's key order, so a
latent model spliced inside a block can be listed after the block, and then the Interpretune module itself is sharded as
the root unit. An empty plan (`{}`) shards the whole module as one unit, which gathers every parameter for the entire
forward pass. Under `ddp` a plan is optional; without one the module is the only unit.

A `class:` key matches subclasses too, and a regex matches every module name it is a prefix of (anchor it with `$` to
select the blocks but not their children). A plan under the `single` placement, a key that selects no module, and a plan
without Fine-Tuning Scheduler installed are each refused.

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
0-d parameter, which it refuses too.

## Tensor parallelism

`tp` splits layers across ranks instead of data: the layers `tp_plan` selects are split with `parallelize_module`
(Fine-Tuning Scheduler's `parallelize_model`), so a model whose weights do not fit on one device can be tuned: built
with `meta_init` (below), it is split before it holds any weights, and each rank holds its slices of the split layers
and the whole of the rest. Without it every rank builds the whole model first. Every rank sees the same batch in the same order, since the ranks compute one forward together. `tp_plan` takes the same keys
as `fsdp_plan`, each mapped to a parallel style (`{style: colwise}`, `rowwise`, `sequence`, `prepare_input`, ...) and
its layouts. Without one, the module's own default is used: a Hugging Face model's declared plan on the core and nnsight
paths (TransformerLens has none). A `tp_plan` under any other placement is refused, and so are `fsdp_plan` and
`fsdp_default_kwargs` under `tp`.

The embedding and the LM head stay whole, and so do query and key norms (Gemma 3, Qwen3): they sit between split layers
and normalize only the heads their rank computes. Each rank's gradient of such a norm covers its own heads, so the
placement sums it across ranks, whether the plan is the model's own or given, and whether the norm is trainable at
placement or thawed later. It takes the modules from the model's own declaration
(`replicated_with_grad_allreduce` in transformers), so a whole module of yours between split layers is not covered.

A latent model spliced with `tuning.latents`, and an SAE that SAE Lens splices into a TransformerLens bridge
(`add_saes_on_init`), read and rewrite a whole activation. To splice one inside a layer the plan splits, set
`use_local_output: false` on that layer's entry, so that the activation reaches the splice split across ranks: the
splice gathers it whole and lays its output out as the activation it replaces, and two ranks tune it as one process
does. For an SAE at `blocks.0.mlp.hook_post`, between a bridge MLP's two projections:

```yaml
run_cfg:
  placement: tp
  tp_plan:
    model\.blocks\.\d+\._original_component\.mlp\._original_component\.fc_in\._original_component:
      style: colwise
      use_local_output: false
    model\.blocks\.\d+\._original_component\.mlp\._original_component\.fc_out\._original_component:
      style: rowwise
```

Where a split layer hands on each rank's local shard instead (the default), the splice is refused, naming the site and
this fix: a latent model's when the module is placed, an SAE's at the first forward pass.

### Circuit-tracer attribution under `tp`

A circuit-tracer composition over nnsight attributes a split model as it does a whole one. The plan transformers
declares splits each attention and MLP block inside and makes the activation whole again at the block's output, and
attribution reads, replaces and injects gradients only there. `generate_attribution_graph` (with live or frozen
attention, and with `offload='cpu'`) and `feature_intervention` therefore need nothing from the caller, and every rank
gets the same graph. Every rank makes every call, since the forward and backward passes are shared.

- Set `circuit_tracer_cfg.tune_base: true`. It registers the base model, whose layers the plan selects; a composition
  without it is refused.
- Every rank holds the whole transcoders. A composition over nnsight can build the base model on the meta device
  (`hf_from_pretrained_cfg.meta_init`), so that no rank holds the whole model on its accelerator; its transcoders are
  then loaded eagerly, and rank 0 loads both on its host.
- `offload='disk'`, head loading reports and head conductance reports are refused on a split model: they read a whole
  weight, or a head inside a split layer.
- The analysis runner takes `placement: tp` too (next section), so an analysis op can attribute a split model.

### The analysis runner under `tp`

`AnalysisRunnerCfg` accepts `placement: tp` beside `single`, so a model too large for one device can be analyzed. The
runner places the module as the core loops do, and then:

- every rank runs every step on the same batches, in the same order (a shuffled loader shuffles alike on every rank,
  and again each epoch), since the split layers' forward and backward passes are shared;
- only rank 0 writes the store, its generator cache and its files; the other ranks run the same steps, discard their
  rows, and then load rank 0's saved store, so every rank returns the same store. Ranks on several hosts need a
  filesystem they share;
- a store served from the generator cache (a deterministic `dataset_fingerprint`) runs no step on any rank: rank 0
  tells the others whether its generation started, so none of them runs a step alone;
- weights an op reads directly rather than through a module call (the J-lens readouts read the unembedding, the final
  norm and the embedding) are gathered whole when the plan splits them, once per run, so a readout on a split model
  equals one process's. Head loading and head conductance reports stay refused on a split model, since they read
  inside split attention layers.

Every rank must take the same data-dependent branches, so seed every rank alike. Transcoders and other latent models
are whole on every rank, as in the core loops.

### The analysis runner under `ddp`

`placement: ddp` (FSDP2 replication: `fully_shard` over a `(replicate, shard)` mesh whose shard dimension has size one)
spreads an analysis over the ranks when the model fits each of them. It adds throughput rather than memory:

- every rank forms the batch sequence one process would (the same order, shuffled from a shared seed each epoch), and
  takes every second, third, ... batch of it, so the shares may differ by one batch and nothing is padded. Steps see
  each batch's index in the whole sequence, and `limit_analysis_batches` counts the whole sequence's batches;
- FSDP2's gradient reduction is off for the run, so a backward pass (attribution, gradient readouts) makes no collective
  and the ranks never wait on each other between batches. A gradient an op reads is its own rank's;
- each rank writes its share, and rank 0 merges the shares by batch into the store one process would write, row for
  row, which every rank then loads. Ranks on several hosts need a filesystem they share;
- an op whose state runs across batches (`op_state`, as `concept_direction` declares) still gives every row its
  one-process value, although each rank sees only its own batches:
  - when it is the composition's last op, each rank records the op's inputs for each of its batches, and after the
    merge rank 0 runs the op again over them in one-process order, with fresh state, and rewrites its columns;
  - when a later op reads its output (`attribution_from_concept` and `intervention_from_concept` attribute and
    intervene with each batch's running direction), every rank first runs the composition up to that op over the whole
    batch sequence, in one-process order, and then runs only the rest on its own batches. That prefix is the cheap part
    when it reads an input store, as `concept_direction` does; the attribution graphs and interventions are shared out;
- a deterministic `dataset_fingerprint` (or `case_key_target_part`) is refused: a share served from the generator cache
  would not say which batches its rows came from.

A module whose submodules the analysis calls directly needs them as units of their own in `fsdp_plan`, as under the
core loops.

## Models larger than one rank's accelerator

By default every rank builds the whole model before it is sharded or split, so each rank must fit it once. For a model
loaded with `from_pretrained` (`hf_from_pretrained_cfg`), `meta_init: true` builds it on the meta device instead:

```yaml
module_cfg:
  init_args:
    hf_from_pretrained_cfg:
      init_args:
        meta_init: true
```

Every rank then builds the model from its configuration without any weights, and the launcher's rank 0 also loads the
checkpoint on the CPU. The placement shards the empty model with the plan (or splits it, under `tp`), materializes each
rank's shards on its device, and loads them from rank 0's copy (`torch.distributed.checkpoint.state_dict.set_model_state_dict` with
`broadcast_from_rank0`), broadcasting the buffers too, since non-persistent ones such as rotary frequencies are in no
state dict. No accelerator ever holds the whole model (under `tp`, each holds the layers the plan leaves whole and its
slices of the rest); rank 0's host holds one copy of it while loading.

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
(which merge or reshape tensors rather than rename them) is refused in favor of the default `rank0` load, naming the
converters that match a key of the checkpoint. A Hub model's safetensors files are downloaded to the local cache first,
on every rank.

A checkpoint already in the model's own layout, such as one written from `model.state_dict()`, loads with its own keys
and no conversion, whatever the model's mapping would do to them. Its keys alone do not show the layout, because a
converter can edit a tensor under its own key, so this holds only for a model whose hub layout has a key the model
does not. For any other model with converters the conversion path decides, and the checkpoint is refused.

The load is checked rather than trusted: when rank 0's copy is missing a key of the model, has one the model lacks, or
differs in a shape (embeddings resized after the build, for one), every rank refuses, naming the keys. With
`lora_cfg`, rank 0's copy takes the same LoRA adapters as the model, so their initial values reach every rank with the
base weights. A latent model or intervention attached beside the model (`interpretune.tuning`) is in no checkpoint of the
model, so it is built with its values on a real device: the placement keeps them, rank 0's on every rank, and refuses one
built on the meta device. `meta_init` cannot be combined with `bitsandbytesconfig` (FSDP2 does not shard quantized
parameters), a model built only partly on the meta device is refused, and so is `meta_init` on a composition whose model init does
not build on the meta device (a TransformerLens bridge, a plain nnsight model, or `defer_model_init`), rather than being
ignored.

A circuit-tracer composition over nnsight builds on the meta device too: its replacement model is built from the
model's config with every Hugging Face and transcoder parameter on the meta device, and rank 0 also loads the model and
its transcoders on the CPU, which the placement loads every rank from into the Hugging Face module nnsight wraps (the
transcoders are registered inside it). The transcoders must be loaded eagerly (`lazy_encoder: false`,
`lazy_decoder: false`), since a lazily loaded weight is read from disk rather than held as a parameter, and the
composition needs an `nnsight_cfg` without `device_map` or remote execution; `meta_init_load: dcp` is refused, since the
transcoders are not in the model's checkpoint. Each of these is refused.

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

Leaving out a transcoder, `JumpReLU` or the Hugging Face model, the modules called outside the forward, is refused, with the `class:` keys to add. The Interpretune module itself is
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

## Walkthrough: staged latent tuning of a sharded Gemma 3 from the CLI

The `rte` component also ships a tuning run: gemma-3-4b-it on the RTE task, with LoRA adapters on its first 10 decoder
layers and two Gemma Scope 2 transcoders (16k features) spliced over the MLPs of layers 17 and 22. The model is built on
the meta device and sharded with FSDP2 across two ranks, and a Fine-Tuning Scheduler schedule tunes it in two stages on
the core loop. The configuration is `rte_latent.gemma3.core.fsdp2` and the run is
`runs/rte_latent.gemma3_4b.core.fsdp2.yaml`.

### Running the stages

As above, the model is gated and the component is fetched first. The two transcoders are downloaded from
`google/gemma-scope-2-4b-it` when the module is built:

```bash
python -c "import interpretune.hub; interpretune.hub.pull('speediedan/rte')"
PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:0 IT_TRUST_REMOTE_CODE=1 \
  torchrun --standalone --nproc-per-node 2 --no-python interpretune --run_command train \
  --config hub://speediedan/rte/runs/rte_latent.gemma3_4b.core.fsdp2.yaml
```

The run trains two epochs, one per stage, at a batch of 2 per rank (a global batch of 4). It keeps no checkpoint.

### The schedule

| stage | entry | what it tunes |
| --- | --- | --- |
| 0 | `role:adapter` | the LoRA adapters; every base weight and transcoder is frozen |
| 1 | `latent_models\.tc\d+\.(w_dec\|b_dec)` | the transcoders' decoders too; their encoders, thresholds and affine skips stay as published |

The schedule is `ft_schedules/rte_latent.gemma3.yaml` in the component. `role:adapter` selects by what a tensor is, so
the schedule does not change when the LoRA targets do. Stage 1 selects by name because `role:latent` would thaw all of a
transcoder, encoder and thresholds included. `max_transition_epoch: 1` ends stage 0 after its first epoch, so each stage
trains for one epoch whatever the validation loss does.

### The units, and why each is there

| plan key | why it is a unit |
| --- | --- |
| `class:Gemma3DecoderLayer` | the transformer blocks, with `act_ckpt: composable`, so that one block is gathered at a time and its activations are recomputed rather than kept |
| `class:GemmaScopeTranscoder` | each spliced transcoder, gathered where it is read; unlisted, it would be gathered with the root unit |
| `class:SiglipEncoderLayer` | the vision tower's layers, which the task never calls; as units they stay sharded and offloaded |

The model is called through the Interpretune module, so neither the Hugging Face model nor a direct call needs a unit of
its own, unlike the circuit-tracer walkthrough above. The base model is held in bf16, and what trains (the adapters and
the transcoders) in fp32. `mp_policy` names only `reduce_dtype: float32`, so each tensor computes in the dtype it is
stored in and the fp32 gradients are reduced in fp32.

### What a run showed

Launched on a 24 GB and an 8 GB card, with eight steps and four validation batches in each stage and otherwise as
shipped:

- Stage 0 trained the 140 adapter tensors and changed nothing else. Stage 1 also trained exactly `w_dec` and `b_dec` of
  `tc17` and `tc22`. Both ranks reported the same validation metrics.
- Each card peaked at 2.8 GiB allocated, in both stages, at about 5 seconds a step.
- Host resident memory peaked at 15.0 GiB on rank 0, which loads the checkpoint, and 8.2 GiB on rank 1. The run completed
  under a 26 GiB limit on the host memory of the whole job, with the allocator's rounding off; an attempt with it on was
  stopped in placement by a 16 GiB limit.

Sixteen steps show that the run works, not that the tuning helps.

### The 27B run

`runs/rte_latent.gemma3_27b.core.fsdp2.yaml` is the same run on gemma-3-27b-it, with transcoders at layers 31 and 40. It
overrides the model, the layers and the transcoder repository, and keeps the plan and the schedule: neither names a
layer. Launched on two 32 GB cards (ROCm), with eight steps and four validation batches in each stage and otherwise as
shipped:

- Each stage trained exactly what its entry selects, as at 4B, and both ranks reported the same validation metrics.
- Each card peaked at 6.47 GiB allocated, in both stages, at about 5.8 seconds a step.
- Host resident memory peaked at 51.8 GiB on rank 0 and 32.8 GiB on rank 1, and the 123 GiB host kept at least 55 GiB
  available.

Leaving the vision layers out of the plan raises each card's peak to 8.28 GiB: they are then gathered with the root
unit for the whole forward pass.

With FSDP2 memory tracking on (`track_fsdp_mem` in `memprofiler_cfg`), each card's peak step holds 2.65 GiB of gathered
parameters, as much again in the all-gather buffer, and about 1.3 GiB of activations; each rank's host holds its
26.3 GiB of offloaded shards. Tracking a plan whose units do not all run, as the vision layers here do not, needs a
Fine-Tuning Scheduler that includes
[finetuning-scheduler#247](https://github.com/speediedan/finetuning-scheduler/pull/247): PyTorch's tracker fails in the
first backward pass on such a unit without it.

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
- `fsdp2` on the analysis runner (`single`, `tp` and `ddp` place it), and under `ddp` a deterministic generator-cache
  key.

## Compositions under `fsdp2`

Which compositions place under `fsdp2` depends on whether their parameters are contiguous and on which module the
backend calls. Each row is a test: placing must keep every parameter's values, and a refusal must name what it found.

| Composition | Under `fsdp2` |
| --- | --- |
| Core (the model is called through the Interpretune module) | Places. The plan names the transformer block class. Training on two ranks matches one process on the combined batch. |
| nnsight, and sae_lens over nnsight | Places, with the Hugging Face model as its own unit beside the blocks (`class:GPT2LMHeadModel` for GPT-2): nnsight calls that model directly, so FSDP2 would otherwise never gather it. A plan without it is refused, naming the unit to add. |
| circuit-tracer over nnsight, with spliced transcoders | Places, with the blocks, each transcoder and its activation functions as their own units, the Hugging Face model among them, and their direct calls registered. On two ranks the logits, the nnsight trace, the transcoder activations and the sparse encode and decode equal one process's. |
| TransformerLens built from a configuration (`ITLensCustomConfig`), sae_lens included | Places. Its blocks hold whole tensors. A schedule that thaws spliced SAEs by role (`role:latent:sae`) tunes on two ranks. |
| TransformerLens bridge over a Hugging Face model, sae_lens included | Refused: the bridge splits the packed attention weight into non-contiguous views (`q`, `k`, `v`), which `fully_shard` cannot shard. The refusal names them. |

The plan for a composition not listed is found the same way: give each block class its own unit, and place any module
the backend calls directly as its own unit too.

An intervention attached to the module trains on two ranks as on one (`ddp` and `fsdp2`): its parameters are the root
unit's, so the hook that applies it inside a block sees the whole operand, and both ranks end with the same value.

A schedule can be written in venngram addresses (`layers:-1: & component:mlp.*`, `role:latent:<name>`) under `ddp` and
`fsdp2`: on a two-rank run each phase thaws the same tensors as in one process, with the same losses and latent model
values. The same holds for a phase that promotes a frozen latent model with `promote` (attached with `frozen=True`, so
its tensors are artifacts until then).

## Limits

- Without `drop_last`, `DistributedSampler` pads the last round with repeated samples so that every rank takes the same
  number of batches. A metric averaged over a dataset that does not divide evenly therefore counts a few samples twice.
- The core loop's `checkpoint_state` is not distributed-aware.
- Phased tuning with `FTSCoreCallback` is tested under `fsdp2` with a plan on two CPU ranks (two phases, matching one
  process on the combined batch step for step) and, with an addressed three-phase schedule, under `ddp`. No test runs it
  on accelerators; the latent tuning walkthrough above was launched on two by hand.
- Without `meta_init`, the session is built fully on each rank before it is sharded, so each rank must fit the whole
  model once. `meta_init` covers the core `from_pretrained` path and circuit-tracer over nnsight (with `rank0` loading
  only): TransformerLens and plain nnsight compositions build their models their own way and refuse it at construction
  (tracked in [interpretune#882](https://github.com/interpretune/interpretune/issues/882)).

The design, and the work that remains for the analysis runner (sharded transcoders and latent models, and FSDP2 shard
dimensions above one), is tracked in
[interpretune#803](https://github.com/interpretune/interpretune/issues/803). Splitting one prompt's attribution across
ranks is [interpretune#1274](https://github.com/interpretune/interpretune/issues/1274).
