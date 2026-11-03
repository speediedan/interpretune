# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""A CPU-sized sparsity chain (#698 to #700) as one phased run on the core loop (#899).

The composition is a small residual MLP classifier whose answer is the gap between two logits, with a JumpReLU
transcoder reading the input and reconstructing the output of the MLP at each regularized layer. The transcoders are
published artifacts (``instrument``); the enforcement means and keep masks are run-fitted state (``fitted``).

Phase 0 tunes the base model against the task loss plus an attribution-sparsity penalty: the $\ell_1 / \ell_2$ ratio
of the per-feature attributions $a_f (W_{dec} g)_f$, where $g$ is the gap's gradient at the MLP output, normalized by
the untuned model's penalty, plus the share of attribution the transcoders' error carries. At the boundary the best
checkpoint is restored, an end hook builds the reference graphs from it (feature means and the features that carry
the most attribution), a gate judges the phase, and a start hook fills the ``fitted`` buffers from those graphs.
Phase 1 then tunes with the simplified graph enforced: each dropped feature is held at its mean, which shifts the MLP
output by $(\mu - a) W_{dec}$ over the dropped features.

Gates and hooks are built from configuration (``FTSCoreCallback``'s ``tuning_scheduler_kwargs``), so everything here
is reachable by ``class_path``.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from finetuning_scheduler.core.transitions import GateResult

D, H, N_FEATURES = 8, 16, 12  # width, MLP hidden width, transcoder features
N_BLOCKS, REG_LAYERS = 3, (1, 2)
TEMPERATURE = 4.0


class Transcoder(nn.Module):
    """A JumpReLU transcoder from an MLP's input to its output."""

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        self.W_enc = nn.Parameter(torch.randn(D, N_FEATURES, generator=generator) / D**0.5)
        self.b_enc = nn.Parameter(torch.zeros(N_FEATURES))
        self.threshold = nn.Parameter(torch.full((N_FEATURES,), 0.05))
        self.W_dec = nn.Parameter(torch.randn(N_FEATURES, D, generator=generator) / N_FEATURES**0.5)
        self.b_dec = nn.Parameter(torch.zeros(D))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = x @ self.W_enc + self.b_enc
        return pre * (pre > self.threshold)


class ChainModel(nn.Module):
    """Residual MLP blocks and a two-logit head, with a transcoder and enforcement state per regularized layer."""

    def __init__(self, seed: int = 0, dropout: float = 0.0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.dropout = dropout
        self.blocks = nn.ModuleList(nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, D)) for _ in range(N_BLOCKS))
        self.head = nn.Linear(D, 2)
        generator = torch.Generator().manual_seed(seed + 1)
        self.tcs = nn.ModuleDict({str(layer): Transcoder(generator) for layer in REG_LAYERS})
        for layer in REG_LAYERS:
            self.register_buffer(f"enforce_mean_{layer}", torch.zeros(N_FEATURES))
            self.register_buffer(f"enforce_keep_{layer}", torch.ones(N_FEATURES, dtype=torch.bool))
        self.enforce = False

    def forward(self, x: torch.Tensor) -> SimpleNamespace:
        """The logits, and per regularized layer the MLP input, the MLP output and the transcoder features."""
        h, sites = x, {}
        for layer, block in enumerate(self.blocks):
            out = F.dropout(block(h), p=self.dropout, training=self.training)
            if layer in REG_LAYERS:
                tc = self.tcs[str(layer)]
                acts = tc.encode(h)
                if self.enforce:
                    mean, keep = getattr(self, f"enforce_mean_{layer}"), getattr(self, f"enforce_keep_{layer}")
                    out = out + torch.where(keep, torch.zeros_like(acts), mean - acts) @ tc.W_dec
                sites[layer] = SimpleNamespace(x=h, out=out, acts=acts)
            h = h + out
        return SimpleNamespace(logits=self.head(h), sites=sites)


def attributions(model: ChainModel, x: torch.Tensor, create_graph: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Per item, the transcoder features' attributions to the gap (concatenated over the regularized layers) and
    the attribution the transcoders' error carries."""
    with torch.enable_grad():
        fwd = model(x)
        gap = fwd.logits[:, 0] - fwd.logits[:, 1]
        outs = [fwd.sites[layer].out for layer in REG_LAYERS]
        grads = torch.autograd.grad(gap.sum(), outs, create_graph=create_graph)
    feats, errs = [], []
    for layer, g in zip(REG_LAYERS, grads):
        site, tc = fwd.sites[layer], model.tcs[str(layer)]
        feats.append(site.acts * (g @ tc.W_dec.T))
        error = site.out - (site.acts @ tc.W_dec + tc.b_dec)
        errs.append((error * g).sum(-1))
    return torch.cat(feats, -1), torch.stack(errs, -1).sum(-1)


def sparsity_penalty(feats: torch.Tensor) -> torch.Tensor:
    """The mean over items of the $\\ell_1 / \\ell_2$ ratio of the feature attributions."""
    return (feats.abs().sum(-1) / feats.norm(dim=-1).clamp_min(1e-8)).mean()


def error_share(feats: torch.Tensor, errs: torch.Tensor) -> torch.Tensor:
    """The mean over items of the share of attribution the transcoders' error carries."""
    err = errs.abs()
    return (err / (err + feats.abs().sum(-1)).clamp_min(1e-8)).mean()


class Normalizer:
    """The untuned penalty the sparsity term is divided by: run-fitted state, saved with checkpoints."""

    def __init__(self, value: float = 1.0) -> None:
        self.value = value

    def state_dict(self) -> dict:
        return {"value": self.value}

    def load_state_dict(self, state: dict) -> None:
        self.value = state["value"]


class ChainModule(nn.Module):
    """The core loop's surface over :class:`ChainModel`, with the phase payload's loss weights applied by
    :class:`ApplyPayload`."""

    def __init__(self, seed: int = 0, n_train: int = 4, batch_size: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.model = ChainModel(seed, dropout)
        generator = torch.Generator().manual_seed(seed + 2)
        teacher = torch.randn(D, generator=generator)
        xs = torch.randn((n_train + 2) * batch_size, D, generator=generator)
        ys = (xs @ teacher > 0).long()
        batches = [(xs[i : i + batch_size], ys[i : i + batch_size]) for i in range(0, len(xs), batch_size)]
        self.train_batches, self.val_batches, self.graph_batch = batches[:n_train], batches[n_train:-1], batches[-1]
        self.normalizer = Normalizer()
        self.reg_weight, self.err_weight = 0.0, 0.0
        # a placeholder the core loop requires; FTSCoreCallback installs the tuning scheduler's optimizer instead
        self.optimizers = [torch.optim.SGD(self.model.parameters(), lr=0.1)]
        self.lr_scheduler_configs: list = []
        self.it_cfg = SimpleNamespace(
            profiler_cfg=SimpleNamespace(which=None),
            optimizer_init={"class_path": "torch.optim.SGD", "init_args": {"lr": 0.1}},
        )
        self.current_epoch, self.global_step = 0, 0
        self.dtype, self.device = torch.float32, torch.device("cpu")
        self._logged_metrics: dict[str, list[float]] = {}
        self.events: list[tuple] = []

    def datamodule(self) -> SimpleNamespace:
        return SimpleNamespace(train_dataloader=lambda: self.train_batches, val_dataloader=lambda: self.val_batches)

    def log(self, name, value, *args, **kwargs):
        self._logged_metrics.setdefault(name, []).append(
            value.detach().item() if isinstance(value, torch.Tensor) else value
        )

    def batch_to_device(self, batch):
        return batch

    def task_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self.model(x).logits / TEMPERATURE, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.task_loss(x, y)
        if self.reg_weight or self.err_weight:
            feats, errs = attributions(self.model, x)
            loss = loss + self.reg_weight * sparsity_penalty(feats) / self.normalizer.value
            loss = loss + self.err_weight * error_share(feats, errs)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # the phase-invariant task loss: the penalty's weight and the enforcement change between phases
        self.log("val_ce", self.task_loss(*batch))


# -- hooks and gates, built from configuration ---------------------------------------------------------------------


class ApplyPayload:
    """Start hook: apply the phase's payload (``reg_weight``, ``err_weight``, ``enforce``) to the module."""

    def __call__(self, event) -> None:
        module, payload = event.tuning_scheduler.model, dict(event.phase_config or {})
        module.reg_weight = payload.get("reg_weight", 0.0)
        module.err_weight = payload.get("err_weight", 0.0)
        module.model.enforce = payload.get("enforce", False)
        module.events.append(("payload", event.depth))


class Calibrate:
    """Phase-0 start hook: fit the penalty's normalizer on the untuned model and register it as run state, so it is
    saved with every checkpoint and restored on resume."""

    def __call__(self, event) -> None:
        if event.depth != 0:
            return
        module = event.tuning_scheduler.model
        feats = torch.cat([attributions(module.model, x, create_graph=False)[0] for x, _ in module.val_batches])
        module.normalizer.value = sparsity_penalty(feats).item()
        event.tuning_scheduler.register_stateful("normalizer", module.normalizer)
        module.events.append(("calibrate", event.depth))


class BuildGraphs:
    """End hook: from the restored best weights, build the reference graphs the enforcement is fitted to: per
    regularized layer, each feature's mean activation over correctly answered items, and the ``keep`` features that
    carry the most attribution. A random subset of the graph items is drawn, which must not disturb training."""

    def __init__(self, keep: int = 4, depth: int = 0, subset: int = 12) -> None:
        self.keep, self.depth, self.subset = keep, depth, subset

    def __call__(self, event) -> None:
        if event.depth != self.depth:
            return
        module = event.tuning_scheduler.model
        x, y = module.graph_batch
        idx = torch.randperm(len(x))[: self.subset]
        x, y = x[idx], y[idx]
        with torch.no_grad():
            right = module.model(x).logits.argmax(-1) == y
        feats, _ = attributions(module.model, x, create_graph=False)
        with torch.no_grad():
            fwd = module.model(x)
        graphs = {}
        for i, layer in enumerate(REG_LAYERS):
            acts = fwd.sites[layer].acts[right]
            attr = feats[right][:, i * N_FEATURES : (i + 1) * N_FEATURES].abs().mean(0)
            keep = torch.zeros(N_FEATURES, dtype=torch.bool)
            keep[attr.topk(self.keep).indices] = True
            graphs[layer] = (acts.mean(0).detach(), keep)
        module.graphs = graphs
        module.events.append(("graphs", event.depth))


class FillEnforcement:
    """Start hook: fill the ``fitted`` enforcement buffers from the graphs the end hook built."""

    def __call__(self, event) -> None:
        module = event.tuning_scheduler.model
        if event.depth == 0 or not hasattr(module, "graphs"):
            return
        with torch.no_grad():
            for layer, (mean, keep) in module.graphs.items():
                getattr(module.model, f"enforce_mean_{layer}").copy_(mean)
                getattr(module.model, f"enforce_keep_{layer}").copy_(keep)
        module.events.append(("fill", event.depth))


class FeatureShareGate:
    """Gate: accept a phase whose transcoder features carry at least ``min_share`` of the attribution on the graph
    items, recording the share; ``veto_depths`` vetoes those phases regardless (a forced veto for tests)."""

    def __init__(self, min_share: float = 0.0, veto_depths: tuple[int, ...] = ()) -> None:
        self.min_share, self.veto_depths = min_share, tuple(veto_depths)

    def __call__(self, event) -> GateResult:
        module = event.tuning_scheduler.model
        feats, errs = attributions(module.model, module.graph_batch[0], create_graph=False)
        share = 1.0 - error_share(feats, errs).item()
        module.events.append(("gate", event.depth, event.best_depth))
        accept = share >= self.min_share and event.depth not in self.veto_depths
        return GateResult(
            accept=accept, reason=None if accept else f"feature share {share:.3f}", record={"share": share}
        )


class DrawNoise:
    """A hook that only draws random numbers: under the training streams' isolation it must change nothing."""

    def __call__(self, event) -> None:
        torch.rand(1000)


def roles(module: ChainModule) -> dict[str, str]:
    """The composition's roles: transcoders ``instrument``, enforcement buffers ``fitted``, the rest ``base``."""
    names = {n: "instrument" for n, _ in module.named_parameters() if n.startswith("model.tcs.")}
    names.update({n: "fitted" for n, _ in module.named_buffers() if n.startswith("model.enforce_")})
    return names


def chain_schedule(promote: bool = False) -> dict:
    """Phase 0 (sparsity) over the last two blocks and the head, with the penalty on; phase 1 (recovery) adds the
    first block, the enforcement on and a fresh optimizer, and with ``promote`` the last transcoder as a latent."""
    base = [f"model.blocks.{i}.{p}" for i in (1, 2) for p in ("0.weight", "0.bias", "2.weight", "2.bias")]
    phase1: dict = {
        "params": [f"model.blocks.0.{p}" for p in ("0.weight", "0.bias", "2.weight", "2.bias")],
        "phase_config": {"reg_weight": 0.3, "err_weight": 3.0, "enforce": True},
        "new_optimizer": {"optimizer_init": {"class_path": "torch.optim.SGD", "init_args": {"lr": 0.05}}},
    }
    if promote:
        phase1["promote"] = {f"model.tcs.{REG_LAYERS[-1]}.W_enc": "latent:transcoder"}
        phase1["params"] = phase1["params"] + ["role:latent:transcoder"]
    return {
        0: {
            "params": base + ["model.head.weight", "model.head.bias"],
            "phase_config": {"reg_weight": 0.3, "err_weight": 3.0, "enforce": False},
            "max_transition_epoch": 2,
        },
        1: phase1,
    }


def hook_config(veto_depths: tuple[int, ...] = ()) -> dict:
    """``tuning_scheduler_kwargs`` entries for the chain's gate and hooks, as configuration."""
    here = __name__

    def entry(name: str, **init_args) -> dict:
        return {"class_path": f"{here}.{name}", "init_args": init_args}

    return {
        "gates": [{"gate": entry("FeatureShareGate", veto_depths=list(veto_depths)), "rank_zero_only": True}],
        "phase_hooks": {
            "start": [entry("Calibrate"), entry("FillEnforcement"), entry("ApplyPayload")],
            "end": [entry("BuildGraphs", keep=4, depth=0)],
        },
    }
