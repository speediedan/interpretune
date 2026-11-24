"""Helpers for the neuron-communities replication notebook (arXiv 2611.23112 on Llama-3.1-8B).

Every measurement takes a ``model`` and its model ``backend`` rather than building either, so the notebook's
loading cell is the only place that knows how they were made: today an nnsight ``LanguageModel`` with an
``NNsightModelBackend``, and later the ``module.model`` and model backend of a task-neutral session.

A neuron is the input to an MLP's down projection (``mlp.out.hook_in``), and every edit acts at the last position
only, as in the paper. A community is addressed as explicit unit sets in a venngram address book, so "ablate every
neuron of these layers except the community and the always-on hubs" is set algebra over the book.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import statistics as st
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

RELEASE = "https://static.goodfire.ai/neuron-communities/2026-11-12-4c7e9dd"
POINT = "mlp.out.hook_in"
# the paper's medians for community 217996 on its " an" prompts (Fig 2): KL and mean |change of the top-1 probability|
PAPER_FIG2 = {
    "necessity": (4.99, 0.42),
    "sufficiency": (0.053, 0.08),
    "layer necessity": (3.15, 0.38),
}
TESTS = {"necessity": "community", "sufficiency": "suff_ablate", "layer necessity": "layer_ablate"}

Neuron = tuple[int, int]


@dataclass
class Community:
    """One released community: its members, always-on hubs and main span, with the release record it came from."""

    record: dict[str, Any]
    sha256: str
    members: list[Neuron]
    hubs: list[Neuron]
    span: list[int]

    @property
    def name(self) -> str:
        return self.record["name"]


def load_community(community_id: str, cache_dir: str | Path) -> Community:
    """Fetch a community's release record once (gzip or plain JSON) and parse its members, hubs and main span."""
    path = Path(cache_dir) / f"community_{community_id}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = urllib.request.urlopen(f"{RELEASE}/communities/{community_id}.json", timeout=60).read()
        path.write_bytes(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
    data = path.read_bytes()
    rec = json.loads(data)
    return Community(
        record=rec,
        sha256=hashlib.sha256(data).hexdigest(),
        members=sorted((n["layer"], n["idx"]) for n in rec["neurons"]),
        hubs=sorted((h["layer"], h["idx"]) for h in rec["always_on"]),
        span=list(rec["main_span"]["layers"]),
    )


def community_book(community: Community, width: int) -> dict[str, Any]:
    """The address book for the paper's three tests: the community and hubs on the main span, the whole span, and
    the two complements (sufficiency ablates ``@span - (@community | @hubs)``, layer necessity ``@span - @hubs``)."""
    members: dict[int, list[int]] = {}
    for layer, idx in community.members:
        if layer in community.span:
            members.setdefault(layer, []).append(idx)
    hubs: dict[int, list[int]] = {}
    for layer, idx in community.hubs:
        if layer in community.span:
            hubs.setdefault(layer, []).append(idx)
    return {
        "community": {"point": POINT, "layers": sorted(members), "units": members},
        "hubs": {"point": POINT, "layers": sorted(hubs), "units": hubs},
        "span": {"point": POINT, "layers": community.span, "units": {k: f"0:{width}" for k in community.span}},
        "keep": "@community | @hubs",
        "suff_ablate": "@span - @keep",
        "layer_ablate": "@span - @hubs",
    }


def zero_ablation(address: str, book: dict[str, Any], n_layers: int, width: int) -> Any:
    """Zero-replace every unit an address resolves to in ``book``, at the last position only."""
    from interpretune.analysis.venngram import unit_interventions

    return unit_interventions(
        address,
        sets=book,
        n_layers=n_layers,
        site_widths={POINT: width},
        op="replace",
        value=0.0,
        position_scope="last_token",
    )


def zero_payloads(book: dict[str, Any], n_layers: int, width: int) -> dict[str, Any]:
    """Each test's zero-ablation payload, resolved from the book by ``unit_interventions``."""
    return {test: zero_ablation(f"@{entry}", book, n_layers, width) for test, entry in TESTS.items()}


def random_payload(community: Community, n_layers: int, width: int, rng: random.Random) -> Any:
    """The control: one random neuron in each span layer, never a member or a hub, ablated the same way."""
    excluded = set(community.members) | set(community.hubs)
    units = {}
    for layer in community.span:
        unit = rng.randrange(width)
        while (layer, unit) in excluded:
            unit = rng.randrange(width)
        units[layer] = [unit]
    book = {"r": {"point": POINT, "layers": community.span, "units": units}}
    return zero_ablation("@r", book, n_layers, width)


def an_windows(tokenizer, n: int, seed: int = 0) -> list[list[int]]:
    """``n`` 64-token FineWeb-Edu windows ending at the article token " an", one per document (BOS-prepended
    documents, full windows only, as the release builds its prompts)."""
    from datasets import load_dataset

    an = tokenizer.encode(" an", add_special_tokens=False)
    if len(an) != 1:
        raise ValueError(f"' an' is not one token for this tokenizer: {an}")
    an = an[0]
    stream = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True).shuffle(seed=seed)
    out: list[list[int]] = []
    for doc in stream:
        ids = tokenizer(doc["text"], add_special_tokens=True)["input_ids"]
        hits = [i for i in range(63, len(ids)) if ids[i] == an]
        if hits:
            i = random.Random(len(out) + seed).choice(hits)
            out.append(ids[i - 63 : i + 1])
        if len(out) >= n:
            return out
    raise RuntimeError(f"the stream ended with {len(out)} windows")


def record_windows(tokenizer, community: Community) -> list[list[int]]:
    """The release's own prompts for a community, re-tokenized from their text (the release publishes text, not
    token ids), each ending at the position the release measured."""
    return [tokenizer(p["t"], add_special_tokens=False)["input_ids"][-64:] for p in community.record["prompts"]]


def kl_and_top1_change(clean: torch.Tensor, edited: torch.Tensor) -> tuple[list[float], list[float]]:
    """Per prompt at the last position: KL(clean || edited) and |change of the clean top-1 token's probability|."""
    lp0 = torch.log_softmax(clean[:, -1].float(), -1)
    lp1 = torch.log_softmax(edited[:, -1].float().to(lp0.device), -1)
    kl = (lp0.exp() * (lp0 - lp1)).sum(-1)
    top = lp0.argmax(-1, keepdim=True)
    dp = (lp0.gather(-1, top).exp() - lp1.gather(-1, top).exp()).abs()[:, 0]
    return kl.tolist(), dp.tolist()


@dataclass
class AblationResult:
    """Per-prompt KL and top-1 probability change for each test, the control and the clean top-1 probability."""

    kl: dict[str, list[float]] = field(default_factory=dict)
    dp: dict[str, list[float]] = field(default_factory=dict)
    p_clean: list[float] = field(default_factory=list)
    top1: list[int] = field(default_factory=list)

    def add(self, test: str, kl: list[float], dp: list[float]) -> None:
        self.kl.setdefault(test, []).extend(kl)
        self.dp.setdefault(test, []).extend(dp)

    def medians(self) -> dict[str, tuple[float, float]]:
        """Each test's median KL and mean |top-1 change| (the paper's two summaries)."""
        return {test: (st.median(self.kl[test]), st.mean(self.dp[test])) for test in self.kl}


def run_ablations(
    model: Any,
    backend: Any,
    windows: Sequence[Sequence[int]],
    payloads: dict[str, Any],
    random_payload_fn: Callable[[], Any] | None = None,
    *,
    batch_size: int = 4,
    draws: int = 2,
) -> AblationResult:
    """Each test's payload, and ``draws`` random controls per batch, through ``backend.fwd_w_intervention``."""
    result = AblationResult()
    for b0 in range(0, len(windows), batch_size):
        batch = {"input_ids": torch.tensor(windows[b0 : b0 + batch_size])}
        for test, payload in payloads.items():
            clean, edited = backend.fwd_w_intervention(model, batch, payload)
            result.add(test, *kl_and_top1_change(clean, edited))
            if test == "necessity":
                probs = torch.softmax(clean[:, -1].float(), -1)
                result.p_clean += probs.max(-1).values.tolist()
                result.top1 += probs.argmax(-1).tolist()
        if random_payload_fn is not None:
            for _ in range(draws):
                clean, edited = backend.fwd_w_intervention(model, batch, random_payload_fn())
                result.add("random control", *kl_and_top1_change(clean, edited))
    return result


def hand_written_zero_ablation(hf_model: Any, input_ids: torch.Tensor, units: dict[int, list[int]]) -> torch.Tensor:
    """The control the payloads are checked against: forward pre-hooks on the HuggingFace Llama's ``down_proj``
    zeroing ``units`` (layer to unit indices) at the last position, written without interpretune."""
    handles = []
    for layer, idx in units.items():

        def zero(_module, inputs, idx=idx):
            x = inputs[0].clone()
            x[:, -1, idx] = 0
            return (x, *inputs[1:])

        handles.append(hf_model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(zero))
    try:
        with torch.no_grad():
            return hf_model(input_ids.to(hf_model.device)).logits
    finally:
        for handle in handles:
            handle.remove()


def top1_log_prob(logits: torch.Tensor) -> torch.Tensor:
    """The attribution readout: log p of each prompt's own top-1 token at the last position, summed over the batch
    (each prompt's term depends only on its own activations, so one backward gives every prompt's gradients)."""
    return logits[:, -1].float().log_softmax(-1).max(-1).values.sum()


def hand_written_gradients(hf_model: Any, input_ids: torch.Tensor, n_layers: int) -> torch.Tensor:
    """The gradient control: ``top1_log_prob``'s gradient at every ``down_proj`` input, last position, by plain
    autograd through pre-hooks, ``[batch, layer, unit]``. With frozen parameters the first site becomes the leaf."""
    seen: dict[int, torch.Tensor] = {}
    handles = []
    for layer in range(n_layers):

        def grab(_module, inputs, layer=layer):
            x = inputs[0]
            if x.requires_grad:
                x.retain_grad()
            else:
                x = x.detach().requires_grad_(True)
            seen[layer] = x
            return (x, *inputs[1:])

        handles.append(hf_model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(grab))
    try:
        with torch.enable_grad():
            top1_log_prob(hf_model(input_ids.to(hf_model.device)).logits).backward()
    finally:
        for handle in handles:
            handle.remove()
    grads = []
    for layer in range(n_layers):
        grad = seen[layer].grad
        if grad is None:
            raise RuntimeError(f"no gradient reached the down_proj input of layer {layer}")
        grads.append(grad[:, -1].float().cpu())
    return torch.stack(grads, 1)


@dataclass
class Attribution:
    """Per prompt, the top-k neurons of each layer by |activation x gradient| and by the signed product."""

    top_abs: list[set[Neuron]] = field(default_factory=list)
    top_signed: list[set[Neuron]] = field(default_factory=list)
    first_batch_grads: torch.Tensor | None = None
    first_batch_ids: torch.Tensor | None = None


def attribute(
    model: Any, backend: Any, windows: Sequence[Sequence[int]], n_layers: int, *, top_k: int = 5, batch_size: int = 4
) -> Attribution:
    """The paper's per-prompt attribution at the last position, over every neuron of every layer.

    The gradients of ``top1_log_prob`` and the activations they multiply come from one ``backend.bwd_w_intervention``
    pass with no edits and ``capture_activations``.
    """
    hooks = [f"blocks.{layer}.{POINT}" for layer in range(n_layers)]
    out = Attribution()
    for b0 in range(0, len(windows), batch_size):
        batch = {"input_ids": torch.tensor(windows[b0 : b0 + batch_size])}
        grads, _, _, activations = backend.bwd_w_intervention(
            model, batch, {}, top1_log_prob, capture=hooks, capture_activations=True
        )
        acts = torch.stack([activations[h][:, -1].float().cpu() for h in hooks], 1)
        grads = torch.stack([grads[h][:, -1].float().cpu() for h in hooks], 1)
        if out.first_batch_grads is None:
            out.first_batch_grads, out.first_batch_ids = grads, batch["input_ids"]
        score = acts * grads
        for store, s in ((out.top_abs, score.abs()), (out.top_signed, score)):
            for row in s.topk(top_k, dim=-1).indices.tolist():
                store.append({(layer, unit) for layer, units in enumerate(row) for unit in units})
    return out


def present_fraction(top_sets: Sequence[set[Neuron]], neurons: set[Neuron], min_present: int = 2) -> float:
    """The fraction of prompts whose top sets hold at least ``min_present`` of ``neurons`` (the release's rule)."""
    return sum(len(neurons & s) >= min_present for s in top_sets) / len(top_sets)


def attribution_controls(top_sets: Sequence[set[Neuron]], community: Community) -> dict[str, Any]:
    """The community's presence and member count beside a hard baseline: per layer, the most frequent non-member,
    non-hub neurons on the same prompts in the community's per-layer counts.

    It is chosen in hindsight, so it bounds what attribution frequency alone gives. Also the members' and hubs' ranks by
    how many prompts each reached.
    """
    members, hubs = set(community.members), set(community.hubs)
    freq: dict[Neuron, int] = {}
    for s in top_sets:
        for n in s:
            freq[n] = freq.get(n, 0) + 1
    per_layer: dict[int, int] = {}
    for layer, _ in members:
        per_layer[layer] = per_layer.get(layer, 0) + 1
    hard: set[Neuron] = set()
    for layer, k in per_layer.items():
        pool = sorted((n for n in freq if n[0] == layer and n not in members | hubs), key=lambda n: -freq[n])
        hard |= set(pool[:k])
    order = {n: i + 1 for i, n in enumerate(sorted(freq, key=lambda n: -freq[n]))}
    missing = len(order) + 1
    return {
        "present": present_fraction(top_sets, members),
        "mean_members": st.mean(len(members & s) for s in top_sets),
        "hard_present": present_fraction(top_sets, hard),
        "hard_mean": st.mean(len(hard & s) for s in top_sets),
        "member_rank_median": st.median(order.get(n, missing) for n in members),
        "hub_ranks": sorted(order.get(n, missing) for n in hubs),
        "hub_rates": [sum(n in s for s in top_sets) / len(top_sets) for n in community.hubs],
        "distinct": len(freq),
    }
