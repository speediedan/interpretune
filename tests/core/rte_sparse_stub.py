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


class StageRecorder:
    """Core-loop callback (after ``FTSCoreCallback``): after every epoch, the phase, the trainable blocks and a
    copy of the model's parameters; and, at ``save_epoch``, the core loop's checkpoint (``resume.pt``) beside a
    copy of the checkpoint policy's directory, which a resumed run starts from."""

    seen: list = []

    def __init__(self, save_epoch: int | None = None, save_dir: str | None = None) -> None:
        self.save_epoch, self.save_dir = save_epoch, save_dir

    def on_train_epoch_end(self, ctx, metrics):
        import shutil
        from pathlib import Path

        from interpretune.tuning.core_loop import FTSCoreCallback

        fts = next(cb for cb in ctx.callbacks if isinstance(cb, FTSCoreCallback))
        model = ctx.module.model
        trainable = sorted({n.split(".")[2] for n, p in model.named_parameters() if p.requires_grad and ".h." in n})
        params = {n: p.detach().clone() for n, p in model.named_parameters()}
        StageRecorder.seen.append((ctx.epoch, fts.tuning_scheduler.curr_depth, trainable, params))
        if ctx.epoch == self.save_epoch:
            save_dir = Path(self.save_dir)
            shutil.copytree(Path("checkpoints"), save_dir / "checkpoints")
            torch.save(ctx.checkpoint_state(), save_dir / "resume.pt")


class PhaseStartParams:
    """Start hook: a copy of the model's parameters at each phase's start, after the boundary's restore."""

    seen: list = []

    def __call__(self, event):
        model = event.tuning_scheduler.model.model
        PhaseStartParams.seen.append((event.depth, {n: p.detach().clone() for n, p in model.named_parameters()}))
