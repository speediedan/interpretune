"""Latent-model tuning of a model too large to load whole on one card: a two-rank FSDP2 recipe.

Every rank builds the model from its config on the meta device, applies per-layer activation checkpointing (and, with
``--tp``, transformers' tensor-parallel plan), shards one FSDP2 unit per decoder layer (each spliced transcoder its
own unit) with ``CPUOffloadPolicy``, allocates its shards on the host, and loads the checkpoint with distributed
checkpointing (``HuggingFaceStorageReader``: each rank reads only its own slices of the safetensors files). Trainables
are rsLoRA adapters on the first ``--lora-k`` layers and, with ``--tc-layers``, Gemma Scope 2 transcoders spliced in
place of those layers' MLPs whose decoders train in a second (latent) phase. The base stays bf16; ``--fp32-islands``
stores the trainables, transcoders and norms in fp32, and each tensor computes in its stored dtype. The loss covers
only the last ``--loss-positions`` target positions (``logits_to_keep``), so the full vocabulary-by-sequence logits
are never materialized.

What this recipe measures is capacity: memory per card, host memory, load time and step time for one batch shape
(``--batch BxL``) on a fixed synthetic prompt. It is not a data pipeline; the core loop's placement is the path for
real training. The measured results and the reasons for these settings are recorded on interpretune#833.

Controls:
- ``--reference`` compares the sharded model's logits on a fixed prompt with a plain ``from_pretrained`` copy on rank
  0's device. FSDP alone runs the same kernels, so it matches bit for bit in bf16. Tensor parallelism changes the
  matmul shapes and partial-sum order, so a bf16 TP run differs from bf16 ``from_pretrained`` by about as much as bf16
  itself differs from fp32: judge TP with ``--dtype float32`` and a tight ``--ref-atol``.
- With ``--tc-layers``, every spliced layer's fraction of variance unexplained (FVU) must be below ``--max-fvu``.
- Refusals by name: a checkpoint key the model does not have, a model parameter no checkpoint file supplies, a
  weight converter transformers would apply (this loader only renames), and any tensor-parallel plan entry or style
  this recipe does not place.

Usage (two ranks)::

    torchrun --nproc-per-node 2 scripts/fsdp2_large_model_recipe.py --model google/gemma-3-27b-it --out OUT \\
        --tc-layers 31,40 --tc-repo google/gemma-scope-2-27b-it --batch 4x4096
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import resource
import threading
import time

import torch
import torch.distributed as dist


class LoRALinear(torch.nn.Module):
    """A frozen linear plus a rank-``r`` rsLoRA update scaled by $\\alpha / \\sqrt{r}$.

    Built on the meta device; its initial values (``A`` Kaiming-uniform, ``B`` zero) are broadcast from rank 0 after
    the checkpoint loads.
    """

    def __init__(self, base: torch.nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.lora_A = torch.nn.Linear(base.in_features, r, bias=False, dtype=base.weight.dtype)
        self.lora_B = torch.nn.Linear(r, base.out_features, bias=False, dtype=base.weight.dtype)
        self.scale = alpha / math.sqrt(r)

    def forward(self, x):
        # an fp32-stored adapter inside a bf16 unit computes in fp32 and hands back the caller's dtype
        d = self.lora_A.weight.dtype
        return self.base(x) + (self.lora_B(self.lora_A(x.to(d))) * self.scale).to(x.dtype)


class GemmaScopeTranscoder(torch.nn.Module):
    """A Gemma Scope 2 JumpReLU transcoder (``params.safetensors``), from the pre-feedforward-norm output $x$:

    - $\\hat y = a W_{dec} + b_{dec} + x A$ predicts the post-feedforward-norm output;
    - $a = (x W_{enc} + b_{enc}) \\odot [x W_{enc} + b_{enc} \\gt \\theta]$;
    - $A$ is the affine skip (zero for a non-affine set).
    """

    NAMES = ("w_enc", "b_enc", "threshold", "w_dec", "b_dec", "affine_skip_connection")

    def __init__(self, d_model, width, dtype):
        super().__init__()
        shapes = {
            "w_enc": (d_model, width),
            "b_enc": (width,),
            "threshold": (width,),
            "w_dec": (width, d_model),
            "b_dec": (d_model,),
            "affine_skip_connection": (d_model, d_model),
        }
        for name, shape in shapes.items():
            setattr(self, name, torch.nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False))

    def forward(self, x):
        xd = x.to(self.w_enc.dtype)
        pre = xd @ self.w_enc + self.b_enc
        acts = pre * (pre > self.threshold)
        return (acts @ self.w_dec + self.b_dec + xd @ self.affine_skip_connection).to(x.dtype)


class SplicedMLP(torch.nn.Module):
    """A decoder layer's MLP and post-feedforward norm with a transcoder in their place.

    The real MLP runs only while ``measure`` is set, to accumulate the transcoder's squared error and the real
    output's variance terms (the first position excluded).
    """

    def __init__(self, mlp, post_norm, tc):
        super().__init__()
        self.mlp, self.post_norm, self.tc = mlp, post_norm, tc
        self.measure = False
        self.fit = None

    def forward(self, x):
        y_hat = self.tc(x)
        if self.measure:
            with torch.no_grad():
                y = self.post_norm(self.mlp(x)).double().flatten(0, -2)[1:]
                e = y_hat.detach().double().flatten(0, -2)[1:]
                n = torch.tensor(float(len(y)), device=y.device, dtype=torch.float64)
                self.fit = (torch.stack([((y - e) ** 2).sum(), (y**2).sum(), n]), y.sum(0))
        return y_hat


class HostMemWatch:
    """The host's lowest ``MemAvailable`` while running, sampled every 0.5 s."""

    def __init__(self):
        self.low = math.inf
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        self.low = min(self.low, int(line.split()[1]) / 2**20)
            self._stop.wait(0.5)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def log(msg):
    if dist.get_rank() == 0:
        print(msg, flush=True)


def build_meta(args, dtype):
    """The model on the meta device (buffers real, on the CPU), adapters and transcoders in place, everything
    frozen but the adapters.

    Returns the config, the model, its decoder and a copy of its buffers.
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(args.model)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
    # Gemma 3's composite config keeps use_cache on in its text config; with activation checkpointing the cached keys
    # double the key length in the recomputed forward
    for c in (model.config, cfg.get_text_config()):
        c.use_cache = False
    model.requires_grad_(False)
    decoder = model.get_decoder()
    islands = args.fp32_islands
    for i in range(args.lora_k):
        layer = decoder.layers[i]
        for parent in (layer.self_attn, layer.mlp):
            for name, child in list(parent.named_children()):
                if isinstance(child, torch.nn.Linear):
                    lora = LoRALinear(child, args.lora_r, args.lora_alpha)
                    if "lora" in islands:
                        lora.lora_A.float()
                        lora.lora_B.float()
                    setattr(parent, name, lora)
    d_model = cfg.get_text_config().hidden_size
    for t in args.tc_layers:
        layer = decoder.layers[t]
        tc = GemmaScopeTranscoder(d_model, args.tc_width, torch.float32 if "tc" in islands else dtype)
        layer.mlp = SplicedMLP(layer.mlp, layer.post_feedforward_layernorm, tc)
        layer.post_feedforward_layernorm = torch.nn.Identity()
    if "norm" in islands:
        for m in model.modules():
            if type(m).__name__.endswith("RMSNorm"):
                m.float()
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)
    buffers = {n: b.clone() for n, b in model.named_buffers()}
    return cfg, model, decoder, buffers


def renamer(model):
    """Checkpoint key to model key, by transformers' own conversion mapping for the model's class (build it before
    ``fully_shard``, which replaces the class).

    A mapping that reshapes or merges tensors is refused.
    """
    from transformers import conversion_mapping as cm, core_model_loading as cml

    maps = cm.get_model_conversion_mapping(model)
    ren = [t for t in maps if isinstance(t, cml.WeightRenaming)]
    conv = [t for t in maps if isinstance(t, cml.WeightConverter)]
    if conv:
        raise ValueError(f"{type(model).__name__} needs weight converters {conv}; this loader handles renames only")

    def rename(key):
        out = cml.rename_source_key(key, ren, conv)
        return out[0] if isinstance(out, tuple) else out

    return rename


def checkpoint_name(name):
    """The checkpoint's name for a model key: ``LoRALinear`` moves a wrapped weight to ``<path>.base.weight`` and
    ``SplicedMLP`` moves the MLP to ``mlp.mlp`` and the post-feedforward norm to ``mlp.post_norm``."""
    name = name.replace(".base.weight", ".weight").replace(".base.bias", ".bias")
    return name.replace(".mlp.mlp.", ".mlp.").replace(".mlp.post_norm.", ".post_feedforward_layernorm.")


def apply_tp(decoder, text_cfg, tp_mesh, tied):
    """Tensor parallelism from transformers' plan for the text model (``base_model_tp_plan``), applied while the
    model is on the meta device.

    - A LoRA-wrapped linear splits to match its base. Colwise shards ``base`` and ``lora_B``, and ``lora_B``'s colwise
      input turns the gradient reaching the replicated ``lora_A`` into the sum over ranks. Rowwise shards ``base`` and
      ``lora_A``, with ``lora_B`` replicated.
    - ``replicated_with_grad_allreduce`` (q/k norms) stays a plain tensor, which is exact for a frozen norm.
    - ``embed_tokens: embedding_rowwise`` stays replicated when the embedding is tied to ``lm_head``: sharding it
      would re-register the embedding's parameter and leave the head holding the never-loaded original.
    - Transcoders stay replicated. Any other entry or style is refused by name.
    """
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

    tp = tp_mesh.size()
    for dim in ("num_attention_heads", "num_key_value_heads", "intermediate_size"):
        if getattr(text_cfg, dim) % tp:
            raise SystemExit(f"--tp {tp}: {dim}={getattr(text_cfg, dim)} does not divide evenly")
    styles = {"colwise": ColwiseParallel, "rowwise": RowwiseParallel}
    plan_items = text_cfg.base_model_tp_plan.items()
    for key, style in plan_items:
        if key.startswith("layers.*.") or (key == "embed_tokens" and style == "embedding_rowwise" and tied):
            continue
        raise SystemExit(f"--tp: plan entry {key!r}: {style!r} is not one this recipe places (tied={tied})")
    for layer in decoder.layers:
        plan = {}
        for key, style in plan_items:
            if not key.startswith("layers.*.") or style == "replicated_with_grad_allreduce":
                continue
            if style not in styles:
                raise SystemExit(f"--tp: plan style {style!r} for {key!r} is not one this recipe places")
            path = key[len("layers.*.") :]
            if isinstance(layer.mlp, SplicedMLP) and path.startswith("mlp."):
                path = "mlp." + path
            mod = layer.get_submodule(path)
            if isinstance(mod, LoRALinear):
                plan[f"{path}.base"] = styles[style]()
                plan[f"{path}.{'lora_B' if style == 'colwise' else 'lora_A'}"] = styles[style]()
            elif isinstance(mod, torch.nn.Linear):
                plan[path] = styles[style]()
            else:
                raise SystemExit(f"--tp: {key!r} resolves to {type(mod).__name__}, not a linear")
        # src_data_rank=None: nothing to scatter from meta parameters; the checkpoint fills each rank's shard
        parallelize_module(layer, tp_mesh, plan, src_data_rank=None)


def shard(model, decoder, args, mesh):
    """Activation checkpointing per decoder layer, then one FSDP2 unit per spliced transcoder, per decoder layer
    and the root, with the parameters (and optimizer state) offloaded to the host."""
    from torch.distributed._composable import checkpoint
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    kw = {"mesh": mesh, "offload_policy": CPUOffloadPolicy()}
    if args.fp32_islands:
        # each tensor computes in its stored dtype (bf16 base, fp32 islands); gradients reduce in fp32
        kw["mp_policy"] = MixedPrecisionPolicy(reduce_dtype=torch.float32)
    for layer in decoder.layers:  # before fully_shard, as FSDP2's composable API expects
        checkpoint(layer)
    for t in args.tc_layers:  # sharded before its layer, so the layer's all-gather does not carry it
        fully_shard(decoder.layers[t].mlp.tc, **kw)
    for layer in decoder.layers:
        fully_shard(layer, **kw)
    fully_shard(model, **kw)


def initial_values(model, args):
    """Rank 0's initial adapter values and transcoder weights, by model key."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    gen, out = torch.Generator().manual_seed(0), {}
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            a = torch.empty(m.lora_A.weight.shape, dtype=torch.float32)
            torch.nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=gen)
            out[f"{name}.lora_A.weight"] = a.to(m.lora_A.weight.dtype)
            out[f"{name}.lora_B.weight"] = torch.zeros(m.lora_B.weight.shape, dtype=m.lora_B.weight.dtype)
        elif isinstance(m, SplicedMLP):
            layer = int(name.split(".layers.")[1].split(".")[0])
            path = f"transcoder_all/layer_{layer}_{args.tc_variant}/params.safetensors"
            sd = load_file(hf_hub_download(args.tc_repo, path))
            if set(sd) != set(GemmaScopeTranscoder.NAMES):
                raise KeyError(
                    f"{args.tc_repo} layer {layer}: tensors {sorted(sd)}, expected {GemmaScopeTranscoder.NAMES}"
                )
            for k in GemmaScopeTranscoder.NAMES:
                out[f"{name}.tc.{k}"] = sd[k].to(getattr(m.tc, k).dtype)
    return out


def load(model, args, rank, tied, rename, buffers, dev):
    """Fill the sharded, offloaded model: buffers from the meta build, the base checkpoint through DCP, adapters
    and transcoders broadcast from rank 0.

    Refuses a checkpoint key the model lacks and a parameter no file supplies.
    """
    import torch.distributed.checkpoint as dcp
    from huggingface_hub import snapshot_download
    from torch.distributed.checkpoint import HuggingFaceStorageReader
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    model.to_empty(device="cpu")
    with torch.no_grad():
        for n, b in model.named_buffers():
            b.copy_(buffers[n].to(device=b.device, dtype=b.dtype))
    path = snapshot_download(args.model, allow_patterns=["*.safetensors", "*.json"])
    index = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    msd = model.state_dict()
    model_keys = {checkpoint_name(n): n for n in msd if "lora_" not in n and ".mlp.tc." not in n}
    sd = {}
    for k in index:
        mk = rename(k)
        if mk not in model_keys:
            raise KeyError(f"checkpoint key {k!r} renames to {mk!r}, which the model does not have")
        sd[k] = msd[model_keys[mk]]
    unsupplied = (
        set(model_keys.values()) - {model_keys[rename(k)] for k in index} - ({"lm_head.weight"} if tied else set())
    )
    if unsupplied:
        raise RuntimeError(
            f"{len(unsupplied)} model parameters are in no checkpoint file, e.g. {sorted(unsupplied)[:5]}"
        )
    dcp.load(sd, storage_reader=HuggingFaceStorageReader(path))
    part = initial_values(model, args) if rank == 0 else {}
    set_model_state_dict(
        model, part, options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, strict=False)
    )
    # non-persistent buffers (rotary frequencies, the embedding scale) are not FSDP-managed: they sit with the compute
    for n, b in list(model.named_buffers()):
        mod_name, _, bname = n.rpartition(".")
        model.get_submodule(mod_name)._buffers[bname] = b.to(dev)


def check_reference(model, args, ids, logits, dtype, dev, rank, res):
    """Positive control: logits against a plain ``from_pretrained`` copy on rank 0's device."""
    ok = torch.zeros(1, device=dev)
    if rank == 0:
        from transformers import AutoModelForCausalLM

        ref = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(dev).eval()
        with torch.no_grad():
            rl = ref(input_ids=ids).logits.float()
        res["ref_max_abs"] = float((logits - rl).abs().max())
        res["ref_argmax_agree"] = float((logits.argmax(-1) == rl.argmax(-1)).float().mean())
        ok[0] = float(res["ref_max_abs"] <= args.ref_atol)
        del ref, rl
        torch.cuda.empty_cache()
        log(f"reference: max |dlogit| {res['ref_max_abs']:.4g}, argmax agreement {res['ref_argmax_agree']:.4f}")
    dist.broadcast(ok, src=0)
    if not ok.item():
        raise SystemExit(f"REFERENCE FAILED: max |dlogit| above {args.ref_atol}")


def check_fvu(model, decoder, args, ids, res):
    """Gate each spliced transcoder's fraction of variance unexplained (FVU) on the reference prompt."""
    spliced = [decoder.layers[t].mlp for t in args.tc_layers]
    for m in spliced:
        m.measure = True
    with torch.no_grad():
        model(input_ids=ids)
    res["fvu"] = {}
    for t, m in zip(args.tc_layers, spliced):
        (sse, syy, n), ysum = m.fit
        res["fvu"][t] = float(sse / (syy - (ysum**2).sum() / n))
        m.measure, m.fit = False, None
    log(f"spliced FVU {res['fvu']}")
    bad = {t: v for t, v in res["fvu"].items() if not v < args.max_fvu}
    if bad:
        raise SystemExit(f"FVU GATE FAILED: layers {bad} not below {args.max_fvu}")


def target_loss(model, batch, k):
    """Cross-entropy over the last ``k`` target positions, from logits for those positions only."""
    logits = model(input_ids=batch, logits_to_keep=k + 1).logits[:, :-1].float()
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, -k:].reshape(-1))


def mem_report(mt, rank, out_dir):
    """One rank's FSDP2MemTracker peak (bytes per category and device) and module-wise report."""
    peak = {str(d): {str(k): int(v) for k, v in c.items()} for d, c in mt.get_tracker_snapshot("peak").items()}
    json.dump(peak, open(os.path.join(out_dir, f"mem_peak_rank{rank}.json"), "w"), indent=1)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mt.display_snapshot("peak", units="GiB")
        mt.display_modulewise_snapshots(depth=4, units="GiB")
    open(os.path.join(out_dir, f"mem_modulewise_rank{rank}.txt"), "w").write(buf.getvalue())


def run_phase(model, args, name, batch, dev, rank, res):
    """``--steps`` optimizer steps on the phase's trainables; the last one is profiled with ``--mem-track``."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    times = []
    for s in range(args.steps):
        tracked = args.mem_track and s == args.steps - 1 and s > 0  # after the optimizer state exists
        mt = contextlib.nullcontext()
        if tracked:
            from torch.distributed._tools.fsdp2_mem_tracker import FSDPMemTracker

            mt = FSDPMemTracker(model, opt)
        torch.cuda.synchronize()
        t = time.time()
        with mt:
            if tracked:
                mt.track_inputs((batch,))
            target_loss(model, batch, args.loss_positions).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.time() - t)
        if tracked:
            mem_report(mt, rank, args.out)
    res[name] = {
        "step_s": times,
        "peak_gpu_gib": torch.cuda.max_memory_allocated(dev) / 2**30,
        "trainable": sum(p.numel() for p in params),
        "tokens_per_s_last": batch.numel() / times[-1],
    }
    log(f"{name}: {res[name]}")
    del opt


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    ap.add_argument(
        "--batch",
        default="2x2048",
        help="BxL: the training batch on each FSDP rank (with --tp, on the whole tensor-parallel group)",
    )
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--loss-positions", type=int, default=64)
    ap.add_argument("--lora-k", type=int, default=10)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=8.0)
    ap.add_argument(
        "--fp32-islands", default="lora,tc,norm", help="comma list of lora, tc, norm stored in fp32 ('' for none)"
    )
    ap.add_argument("--tc-layers", default="", help="comma list: splice Gemma Scope 2 transcoders at these layers")
    ap.add_argument("--tc-repo", default="google/gemma-scope-2-4b-it")
    ap.add_argument("--tc-variant", default="width_16k_l0_small_affine")
    ap.add_argument("--tc-width", type=int, default=16384)
    ap.add_argument("--max-fvu", type=float, default=1.0)
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel degree on a (dp, tp) mesh")
    ap.add_argument("--reference", action="store_true", help="positive control: logits vs plain from_pretrained")
    ap.add_argument("--ref-atol", type=float, default=0.0)
    ap.add_argument("--mem-track", action="store_true", help="FSDP2MemTracker report for the last step of each phase")
    args = ap.parse_args()
    args.tc_layers = [int(x) for x in args.tc_layers.split(",") if x]
    args.fp32_islands = {x for x in args.fp32_islands.split(",") if x}
    if unknown := args.fp32_islands - {"lora", "tc", "norm"}:
        raise SystemExit(f"--fp32-islands: unknown {sorted(unknown)}; expected lora, tc, norm")
    if args.tc_layers and args.reference:
        raise SystemExit("--reference compares against the unspliced model; it cannot vouch for a spliced one")
    return args


def main():
    args = parse_args()
    dist.init_process_group()
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(dev)
    os.makedirs(args.out, exist_ok=True)
    dtype = getattr(torch, args.dtype)
    from torch.distributed.device_mesh import init_device_mesh

    tp_mesh = None
    if args.tp > 1:
        if world % args.tp:
            raise SystemExit(f"--tp {args.tp} does not divide world size {world}")
        mesh2 = init_device_mesh("cuda", (world // args.tp, args.tp), mesh_dim_names=("dp", "tp"))
        mesh, tp_mesh = mesh2["dp"], mesh2["tp"]
    else:
        mesh = init_device_mesh("cuda", (world,))
    res = {
        "args": {k: (sorted(v) if isinstance(v, set) else v) for k, v in vars(args).items()},
        "world": world,
        "torch": torch.__version__,
    }
    with HostMemWatch() as hm:
        t0 = time.time()
        cfg, model, decoder, buffers = build_meta(args, dtype)
        tied = bool(getattr(cfg.get_text_config(), "tie_word_embeddings", False))
        if tied:
            model.tie_weights()
        rename = renamer(model)
        if tp_mesh is not None:
            apply_tp(decoder, cfg.get_text_config(), tp_mesh, tied)
        shard(model, decoder, args, mesh)
        load(model, args, rank, tied, rename, buffers, dev)
        dist.barrier()
        res["load_s"] = time.time() - t0
        log(f"loaded in {res['load_s']:.1f} s")

        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        text = "The quick brown fox jumps over the lazy dog. " * 64
        ids = tok(text, return_tensors="pt").input_ids[:, :256].to(dev)
        model.eval()
        with torch.no_grad():
            logits = model(input_ids=ids).logits.float()
        if args.reference:
            check_reference(model, args, ids, logits, dtype, dev, rank, res)
        if args.tc_layers:
            check_fvu(model, decoder, args, ids, res)

        b, length = (int(x) for x in args.batch.split("x"))
        long_ids = tok(text * 40, return_tensors="pt").input_ids[0]
        if length > len(long_ids):
            raise SystemExit(f"--batch {args.batch}: the prompt has only {len(long_ids)} tokens")
        batch = long_ids[:length].repeat(b, 1).to(dev)
        model.train()
        run_phase(model, args, "lora", batch, dev, rank, res)
        if args.tc_layers:
            for t in args.tc_layers:
                tc = decoder.layers[t].mlp.tc
                tc.w_dec.requires_grad_(True)
                tc.b_dec.requires_grad_(True)
            run_phase(model, args, "latent", batch, dev, rank, res)
    res["host_mem_available_low_gib"] = hm.low
    res["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    allres = [None] * world
    dist.all_gather_object(allres, res)
    if rank == 0:
        json.dump(allres, open(os.path.join(args.out, "recipe.json"), "w"), indent=1)
        for i, r in enumerate(allres):
            print(
                f"rank {i}: lora peak {r['lora']['peak_gpu_gib']:.2f} GiB, peak RSS {r['peak_rss_gib']:.1f} GiB, "
                f"host MemAvailable low {r['host_mem_available_low_gib']:.1f} GiB",
                flush=True,
            )
        print("RECIPE PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
