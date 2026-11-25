"""One rank of the two-rank check of circuit-tracer attribution on a base model split by placement ``tp``.

Each rank builds the same tiny replacement model (``tests/core/circuit_tracer_toy.py``) twice, in a circuit-tracer
module that registers its base model (``tune_base``). One copy is placed under ``tp`` with its default plan, the one
transformers declares; the other stays whole as the reference. Rank 1's placed copy is perturbed first, so only the
broadcast can make the ranks agree. With ``--attribution-only`` the base model is not registered, and the placement's
refusal is recorded. ``--arch gemma3`` takes the toy Gemma 3 in place of the Gemma 2. With ``--meta``, the placed copy
is built on the meta device instead, and rank 0 keeps a loaded copy that the placement loads every rank from.

Readings on both copies: an attribution graph, one with ``offload='cpu'``, one with live attention, and a feature
intervention's logits with attention frozen and live. After them one rank's shard of one split weight is perturbed and
the graph taken again, so the comparison is shown to be able to fail, and the three calls that need whole layers are
made for their refusals.

Writes ``<out_dir>/rank<rank>.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from interpretune.runners.distributed import META_INIT_SOURCE


def build(tune_base: bool, replacement_model=None, source=None, arch: str = "gemma2"):
    from interpretune.adapters.circuit_tracer.adapter import BaseCircuitTracerModule, CircuitTracerAttributeMixin
    from interpretune.config import CircuitTracerConfig
    from tests.core import circuit_tracer_toy

    class _CTModule(CircuitTracerAttributeMixin, BaseCircuitTracerModule):
        """A circuit-tracer module holding only what placement and attribution read, without a session."""

    if replacement_model is None:
        replacement_model = getattr(circuit_tracer_toy, f"tiny_{arch}_replacement_model")()
    module = _CTModule.__new__(_CTModule)
    nn.Module.__init__(module)
    object.__setattr__(
        module, "it_cfg", SimpleNamespace(circuit_tracer_cfg=CircuitTracerConfig(tune_base=tune_base), model_cfg=None)
    )
    object.__setattr__(module, "_replacement_model", replacement_model)
    object.__setattr__(module, "attribution_graphs", [])
    module.model = replacement_model
    module.register_tunable_modules()
    if source is not None:
        object.__setattr__(module, META_INIT_SOURCE, source)
    return module


def build_meta(rank0: bool):
    """The placed copy built on the meta device (its base model registered as the nnsight adapter registers it),
    and the loaded copy to compare with, both with the one-dimensional thresholds of the toy's meta build."""
    from tests.core.circuit_tracer_toy import tiny_gemma2_replacement_model
    from tests.core.distributed_ct_worker import toy_meta_build, widen_scalars

    meta, source = toy_meta_build(rank0)
    placed = build(False, meta, source)
    placed._modules["hf_model"] = meta._module
    reference = tiny_gemma2_replacement_model()
    widen_scalars(reference)
    return placed, build(True, reference)


def graph(module, **kwargs):
    from tests.core.circuit_tracer_toy import TOY_PROMPT, only_token_zero_special

    with only_token_zero_special(module.replacement_model):
        return module.generate_attribution_graph(TOY_PROMPT, max_n_logits=3, batch_size=8, verbose=False, **kwargs)


def readings(module) -> dict[str, torch.Tensor]:
    from tests.core.circuit_tracer_toy import TOY_PROMPT, only_token_zero_special

    plain = graph(module)
    out = {"features": plain.active_features, "selected": plain.selected_features, "graph": plain.adjacency_matrix}
    out["offload_cpu"] = graph(module, offload="cpu").adjacency_matrix
    out["live"] = graph(module, freeze_attention=False).adjacency_matrix
    layer, position, feature = (int(i) for i in plain.active_features[plain.active_features[:, 0] == 1][0])
    with only_token_zero_special(module.replacement_model):
        for name, frozen in (("frozen", True), ("unfrozen", False)):
            for kind, edits in (("edit", [(layer, position, feature, 2.0)]), ("clean", [])):
                logits, _ = module.replacement_model.feature_intervention(TOY_PROMPT, edits, freeze_attention=frozen)
                out[f"{name}_{kind}"] = (logits.full_tensor() if hasattr(logits, "full_tensor") else logits).detach()
    return out


def refusals(module) -> dict[str, str]:
    from interpretune.adapters.circuit_tracer.backends import DEFAULT_CT_ANALYSIS_BACKEND
    from interpretune.utils import MisconfigurationException
    from tests.core.circuit_tracer_toy import TOY_PROMPT

    backend, weights = DEFAULT_CT_ANALYSIS_BACKEND, torch.zeros(16)
    calls = {
        "offload_disk": lambda: graph(module, offload="disk"),
        "head_loadings": lambda: backend.head_loading_report(module, TOY_PROMPT, torch.tensor([3])),
        "head_conductance": lambda: backend.head_conductance_report(module, TOY_PROMPT, TOY_PROMPT, weights),
    }
    record = {}
    for name, call in calls.items():
        try:
            call()
        except MisconfigurationException as err:
            record[name] = str(err)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--attribution-only", action="store_true")
    parser.add_argument("--arch", choices=("gemma2", "gemma3"), default="gemma2")
    parser.add_argument("--meta", action="store_true", help="build on the meta device and load from rank 0")
    args = parser.parse_args()

    import torch.distributed as dist

    from interpretune.runners.distributed import init_process_group, place_module, resolve_placement
    from interpretune.utils import MisconfigurationException

    run = resolve_placement("tp")
    assert run is not None
    init_process_group(run)
    if args.meta:
        placed, reference = build_meta(rank0=run.rank == 0)
    else:
        placed, reference = (build(not args.attribution_only, arch=args.arch) for _ in range(2))
    if run.rank and not args.meta:
        with torch.no_grad():
            for param in placed.parameters():
                param.add_(0.1)
    out = Path(args.out_dir, f"rank{run.rank}.json")
    try:
        place_module(placed, run)
    except MisconfigurationException as err:
        out.write_text(json.dumps({"refused": str(err)}), encoding="utf-8")
        return
    split = {n: p for n, p in placed.named_parameters() if type(p.data).__name__ == "DTensor"}
    ref, got = readings(reference), readings(placed)
    record: dict = {
        "split_params": sorted(split),
        "n_features": len(ref["features"]),
        "same_nodes": all(torch.equal(ref[k], got[k]) for k in ("features", "selected")),
        # relative: the live graph's edges are an order larger than the frozen graph's
        "diff": {
            k: float((ref[k] - got[k]).abs().max() / ref[k].abs().max()) for k in ref if ref[k].is_floating_point()
        },
        "effect": {n: float((ref[f"{n}_edit"] - ref[f"{n}_clean"]).abs().max()) for n in ("frozen", "unfrozen")},
    }
    whole = got["graph"].contiguous()
    per_rank = [torch.empty_like(whole) for _ in range(run.world_size)]
    dist.all_gather(per_rank, whole)
    record["ranks_equal"] = torch.equal(*per_rank)
    record["refusals"] = refusals(placed)
    if run.rank:
        with torch.no_grad():
            next(p for n, p in split.items() if n.endswith("layers.1.self_attn.o_proj.weight")).to_local().add_(0.5)
    control = graph(placed).adjacency_matrix
    record["control_differs"] = control.shape != ref["graph"].shape or not torch.allclose(
        control, ref["graph"], rtol=1e-2
    )
    out.write_text(json.dumps(record), encoding="utf-8")


if __name__ == "__main__":
    main()
