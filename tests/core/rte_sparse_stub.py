"""Test-side stand-ins for the sparse chain's boundary (#899): a graph hook that needs no circuit-tracer graphs,
and a probe recording what the enforcement does."""

from __future__ import annotations

import torch


class StubGraphs:
    """End hook: the enforcement from the graph split's shared template, random means and a random keep set."""

    def __call__(self, event) -> None:
        if event.depth != 0:
            return
        module = event.tuning_scheduler.model
        rows = []
        for i, batch in enumerate(module.datamodule.graph_dataloader()):
            if i == 2:
                break
            rows += [ids[mask.bool()] for ids, mask in zip(batch["input_ids"], batch["attention_mask"])]
        shortest = min(len(r) for r in rows)
        n_pre = n_suf = 0
        while n_pre < shortest and all(int(r[n_pre]) == int(rows[0][n_pre]) for r in rows):
            n_pre += 1
        while n_suf < shortest - n_pre and all(int(r[-1 - n_suf]) == int(rows[0][-1 - n_suf]) for r in rows):
            n_suf += 1
        layers = module.sparse_cfg["enforce_layers"]
        d = module.sparse_tcs[str(layers[0])].d_transcoder
        gen = torch.Generator().manual_seed(0)
        graphs = {
            "n_pre": n_pre,
            "n_suf": n_suf,
            "pre_ids": rows[0][:n_pre],
            "suf_ids": rows[0][len(rows[0]) - n_suf :],
        }
        for region, n in (("pre", n_pre), ("suf", n_suf), ("con", None)):
            shape = (d,) if n is None else (n, d)
            graphs[f"mean_{region}"] = {ly: torch.rand(shape, generator=gen) for ly in layers}
            graphs[f"keep_{region}"] = {ly: torch.rand(shape, generator=gen) > 0.5 for ly in layers}
        module.sparse_graphs = graphs


class EnforcementProbe:
    """Start hook (after the fill and payload hooks): at each phase's start, whether enforcement is on, the filled
    template, and how far enforcing moves the answer logits on one validation batch."""

    seen: list = []

    def __call__(self, event):
        module = event.tuning_scheduler.model
        batch = next(iter(module.datamodule.val_dataloader()))
        batch = {k: v.to(module.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        was = module.sparse_enforce
        with torch.no_grad():
            module.sparse_enforce = False
            free, _ = module.sparse_forward(batch)
            module.sparse_enforce = True
            enforced, _ = module.sparse_forward(batch)
        module.sparse_enforce = was
        module._sparse_saved.clear()
        template = tuple(int(v) for v in module.sparse_enforce_template)
        EnforcementProbe.seen.append((event.depth, was, template, float((enforced - free).abs().max())))
