"""Helpers for the block-sparse featurizer notebook (SASA, arXiv 2606.06333, on Mistral-7B).

Every measurement takes a ``model`` and its model ``backend`` rather than building either, so the notebook's loading
cell is the only place that knows how they were made. Edits act at one explicit position (the earlier name, or the
noun standing in for it), through ``position_scope="positions"``, as the paper's edits do.

A SASA group is block $g$ of the decoder, rows $[g r, (g + 1) r)$, so the release is a venngram ``LatentBinding`` with
``block_size`` $r$. Coordinates inside a block are only read or written through a named frame, and a chart on that
frame (a PCA plane read as a radius and an angle) is how a ring of names is edited.

The prompt templates, the nouns, and the group screen follow the authors' evaluation code
(https://github.com/arshandalili/sasa, ``eval/causal.py``, MIT license).
"""

from __future__ import annotations

import json
import math
import types
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CONCEPTS = {"month": MONTHS, "day": DAYS}
OTHER = {"month": "day", "day": "month"}
# Each template names a member, then repeats its context so that the next token is a member again.
TEMPLATES = {
    "month": [
        "The event happened in {t}. The event happened in",
        "The contract ends in {t}. The contract ends in",
        "She will return in {t}. She will return in",
        "The festival is held in {t}. The festival is held in",
        "The report is due in {t}. The report is due in",
        "The season starts in {t}. The season starts in",
        "The meeting was moved to {t}. The meeting was moved to",
        "Our lease ends in {t}. Our lease ends in",
        "The store closed in {t}. The store closed in",
        "The trial began in {t}. The trial began in",
        "The book came out in {t}. The book came out in",
        "The treaty was signed in {t}. The treaty was signed in",
        "The project was approved in {t}. The project was approved in",
        "We adopted a cat in {t}. We adopted a cat in",
        "The bridge was completed in {t}. The bridge was completed in",
        "Her contract expires in {t}. Her contract expires in",
    ],
    "day": [
        "The meeting is on {t}. The meeting is on",
        "She called me on {t}. She called me on",
        "The deadline is {t}. The deadline is",
        "They travel every {t}. They travel every",
        "The class meets on {t}. The class meets on",
        "He arrived last {t}. He arrived last",
        "Payday falls on {t}. Payday falls on",
        "The show airs every {t}. The show airs every",
        "The market opens on {t}. The market opens on",
        "We play tennis on {t}. We play tennis on",
        "The package arrived on {t}. The package arrived on",
        "Her shift starts on {t}. Her shift starts on",
        "The museum is closed on {t}. The museum is closed on",
        "Rehearsal is every {t}. Rehearsal is every",
        "The trash is collected on {t}. The trash is collected on",
        "Our next game is on {t}. Our next game is on",
    ],
}
NOUNS = [
    "lamp",
    "chair",
    "bottle",
    "hammer",
    "basket",
    "mirror",
    "pillow",
    "kettle",
    "ladder",
    "blanket",
    "candle",
    "bucket",
    "camera",
    "clock",
    "drawer",
    "envelope",
    "fork",
    "guitar",
    "helmet",
    "jacket",
    "knife",
    "napkin",
    "pencil",
    "plate",
    "radio",
    "ribbon",
    "shovel",
    "spoon",
    "towel",
    "umbrella",
    "vase",
    "wallet",
    "whistle",
]
N_PROTOTYPE_TEMPLATES = 4  # templates that estimate the steering vectors; the others evaluate them
NOUNS_PER_TEMPLATE = 8
# Calendar arithmetic (Engels et al., arXiv 2405.14860): the answer depends on where the name sits on its cycle.
ARITHMETIC = {
    "month": [
        "Let's do some calendar math. {k} months from {t} is",
        "Question: what month is {k} months after {t}? Answer:",
        "If it is {t} now, then in {k} months it will be",
    ],
    "day": [
        "Let's do some day of the week math. {k} days from {t} is",
        "Question: what day is {k} days after {t}? Answer:",
        "If today is {t}, then in {k} days it will be",
    ],
}
OFFSETS = {"Two": 2, "Three": 3, "Four": 4}
# Carrier sentences the frames and charts are fitted on, kept apart from every prompt they are tested on.
CARRIERS = {
    "month": [
        "The event happened in {t}.",
        "She will return in {t}.",
        "The report is due in {t}.",
        "The contract ends in {t}.",
        "Our lease ends in {t}.",
        "The trial began in {t}.",
    ],
    "day": [
        "The meeting is on {t}.",
        "She called me on {t}.",
        "The class meets on {t}.",
        "Payday falls on {t}.",
        "The market opens on {t}.",
        "We play tennis on {t}.",
    ],
}
# The paper's Tables 13 and 14 (Mistral-7B, layer 8), as P(C) in percent.
PAPER_REMOVAL = {
    "month": {
        "unsteered": 86.1,
        "selected group": 40.9,
        "runner-up group": 85.9,
        "other concept's group": 84.7,
        "random 8-d subspace": 86.1,
    },
    "day": {
        "unsteered": 90.0,
        "selected group": 45.2,
        "runner-up group": 88.1,
        "other concept's group": 88.9,
        "random 8-d subspace": 90.0,
    },
}
PAPER_ADDITION = {
    "month": {
        "unsteered": 5.6,
        "selected group": {1.0: 34.6, 2.0: 78.7},
        "runner-up group": {1.0: 6.1, 2.0: 7.2},
        "other concept's group": {1.0: 8.3, 2.0: 7.2},
        "random vector": {1.0: 5.6, 2.0: 5.9},
    },
    "day": {
        "unsteered": 5.0,
        "selected group": {1.0: 52.0, 2.0: 80.1},
        "runner-up group": {1.0: 5.8, 2.0: 11.3},
        "other concept's group": {1.0: 12.0, 2.0: 8.7},
        "random vector": {1.0: 5.4, 2.0: 5.9},
    },
}
LAYER_NORM_EPS = 1e-5


@dataclass
class SASARelease:
    """A released SASA model: encoder, decoder rows grouped in blocks of ``rank``, and its per-token group gate.

    It is also a latent model a model backend can attach (``latent_model_handles=[release]``), so a frame or chart
    edit of one block runs natively at ``blocks.{layer}.hook_resid_pre.hook_sae_acts_post``. Its codes are carried at
    the original activation scale (the layer-normalized codes times each token's scale), so a block's decoded value
    $a_B D_B$ is the block's contribution to the residual stream, :meth:`contribution`, and an edit of it moves the
    residual stream by exactly the edit. With the error term the attached model leaves the clean forward pass
    unchanged.
    """

    name: str
    layer: int
    w_enc: torch.Tensor
    b_enc: torch.Tensor
    w_dec: torch.Tensor
    rank: int
    k_groups: int
    threshold: float | None = None
    io_dtype: torch.dtype = torch.float32

    use_error_term = True

    @property
    def cfg(self) -> Any:
        """The hook the release reads and writes, where a model backend splices it."""
        return types.SimpleNamespace(metadata=types.SimpleNamespace(hook_name=f"blocks.{self.layer}.hook_resid_pre"))

    @property
    def W_dec(self) -> torch.Tensor:
        """The decoder rows, under the name a latent model exposes them by."""
        return self.w_dec

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Gated codes at the activation scale, ``(..., n_groups * rank)``: a group's codes where its pre-
        activation norm passes the gate, times the token's scale, else zero."""
        if self.threshold is None:
            raise ValueError("calibrate the group gate first (calibrate_threshold)")
        flat = x.reshape(-1, x.shape[-1])
        pre, scale = self.pre_activations(flat)
        codes = pre * (pre.norm(dim=-1, keepdim=True) > self.threshold) * scale[:, :, None]
        return codes.reshape(*x.shape[:-1], -1)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """The codes' reconstruction in the model's activation dtype (the layer norm's mean is left to the error
        term)."""
        return (acts.to(self.w_dec) @ self.w_dec).to(self.io_dtype)

    @property
    def n_groups(self) -> int:
        return self.w_dec.shape[0] // self.rank

    def block(self, group: int) -> torch.Tensor:
        """Group ``group``'s decoder rows, ``(rank, d_model)``."""
        return self.w_dec[group * self.rank : (group + 1) * self.rank]

    def binding(self) -> Any:
        """The release as a venngram latent binding at ``blocks.{layer}.hook_resid_pre``, one block per group."""
        from interpretune.analysis.venngram import LatentBinding

        return LatentBinding(self.name, self.layer, "hook_resid_pre", self, self.w_dec, self.rank)

    def pre_activations(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Group pre-activations ``(N, n_groups, rank)`` of the layer-normalized state, and the per-token scale."""
        h = h.to(self.w_enc)
        centered = h - h.mean(-1, keepdim=True)
        scale = centered.std(-1, keepdim=True) + LAYER_NORM_EPS
        pre = (centered / scale) @ self.w_enc + self.b_enc
        return pre.view(len(h), self.n_groups, self.rank), scale

    def group_norms(self, h: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
        """Each group's pre-activation norm, the quantity the paper's screen ranks."""
        return torch.cat(
            [self.pre_activations(h[i : i + chunk])[0].norm(dim=-1).cpu() for i in range(0, len(h), chunk)]
        )

    def contribution(self, h: torch.Tensor, group: int) -> torch.Tensor:
        """Group ``group``'s decoded contribution at the original activation scale, zero where its gate is
        closed."""
        if self.threshold is None:
            raise ValueError("calibrate the group gate first (calibrate_threshold)")
        pre, scale = self.pre_activations(h)
        codes = pre[:, group] * (pre[:, group].norm(dim=-1, keepdim=True) > self.threshold)
        return (codes @ self.block(group)) * scale


def load_sasa(
    repo_id: str,
    checkpoint: str,
    *,
    device: str,
    revision: str | None = None,
    io_dtype: torch.dtype = torch.float32,
) -> SASARelease:
    """Download one SASA checkpoint (an SAELens-format ``cfg.json`` and ``sae_weights.safetensors``).

    ``io_dtype`` is the dtype the model's activations run in, which an attached release writes back.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    cfg_path = hf_hub_download(repo_id, f"{checkpoint}/cfg.json", revision=revision)
    weights = hf_hub_download(repo_id, f"{checkpoint}/sae_weights.safetensors", revision=revision)
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    if cfg.get("normalize_activations") != "layer_norm":
        raise ValueError(f"{checkpoint}: expected layer-normalized inputs, got {cfg.get('normalize_activations')!r}")
    w = load_file(weights, device=device)
    layer = int(cfg["metadata"]["hook_name"].split(".")[1])
    return SASARelease(
        "sasa",
        layer,
        w["W_enc"].float(),
        w["b_enc"].float(),
        w["W_dec"].float(),
        cfg["group_rank"],
        cfg["k_groups"],
        io_dtype=io_dtype,
    )


def single_token_id(tokenizer, word: str) -> int:
    """The id of ``word`` as one token after a space; anything that splits is refused rather than truncated."""
    base, full = tokenizer("x")["input_ids"], tokenizer("x " + word)["input_ids"]
    if full[: len(base)] != base or len(full) != len(base) + 1:
        raise ValueError(f"{word!r} is not a single token after a space for this tokenizer")
    return full[len(base)]


def name_column(tokenizer, text: str, word: str) -> int:
    """The position of ``word``'s token where it first ends in ``text``, checked against the full tokenization."""
    prefix = text[: text.index(word) + len(word)]
    head, ids = tokenizer(prefix)["input_ids"], tokenizer(text)["input_ids"]
    if ids[: len(head)] != head:
        raise ValueError(f"{word!r} does not end on a token boundary in {text!r}")
    return len(head) - 1


def batch_of(tokenizer, texts: Sequence[str], device: str) -> dict[str, torch.Tensor]:
    """Texts that tokenize to the same length, as one batch; unequal lengths are refused (positions are
    absolute)."""
    ids = [tokenizer(t)["input_ids"] for t in texts]
    if len({len(i) for i in ids}) != 1:
        raise ValueError("a batch of positions-scoped edits needs prompts of one length")
    return {"input_ids": torch.tensor(ids, device=device)}


def capture(model, backend, batch: dict[str, torch.Tensor], site: str) -> torch.Tensor:
    """The site's activation for every position, ``(batch, seq, d_model)``, through the backend."""
    _, cache = backend.fwd_w_cache(model, batch, [site])
    return cache[site].float()


def concept_mass(logits: torch.Tensor, ids: Sequence[int]) -> torch.Tensor:
    """P(C): the next-token probability of all the concept's members together, at the final position."""
    return logits[:, -1].float().softmax(-1)[:, list(ids)].sum(-1).cpu()


def _spec(operand: torch.Tensor, op: str, column: int) -> Any:
    from interpretune.analysis.backends.interventions import InterventionSpec

    return InterventionSpec(operand[None], op=op, position_scope="positions", positions=(column,))


def random_rows(d: int, rank: int, seed: int) -> torch.Tensor:
    """Orthonormal rows of a random ``rank``-dimensional subspace."""
    gen = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(d, rank, generator=gen, dtype=torch.float64))[0].T.float()


def calibrate_threshold(
    model,
    backend,
    tokenizer,
    release: SASARelease,
    texts: Sequence[str],
    site: str,
    device: str,
    context: int = 512,
    batch_tokens: int = 4096,
) -> float:
    """The per-token gate: mean over token batches of the norm admitting ``k_groups`` groups per token on average.

    This is how the authors calibrate the released checkpoint for per-token use (their ``inference_threshold``).
    """
    states = []
    for text in texts:
        ids = tokenizer(text)["input_ids"][:context]
        if len(ids) >= 32:
            states.append(capture(model, backend, {"input_ids": torch.tensor([ids], device=device)}, site)[0, 1:])
    corpus = torch.cat(states)
    cutoffs = []
    for start in range(0, len(corpus) - batch_tokens + 1, batch_tokens):
        norms = release.pre_activations(corpus[start : start + batch_tokens])[0].norm(dim=-1)
        cutoffs.append(norms.flatten().topk(release.k_groups * batch_tokens).values[-1])
    release.threshold = float(torch.stack(cutoffs).mean())
    return release.threshold


def screen_groups(
    model, backend, tokenizer, release: SASARelease, background: torch.Tensor, site: str, device: str
) -> dict[str, dict[str, Any]]:
    """Rank groups by the AUROC of their pre-activation norm, members at the template names against background.

    The paper screens on natural Pile occurrences of the names; this uses the templates' names, which keeps the notebook
    offline apart from the background text. The top two groups per concept are the selected group and the runner-up
    control.
    """
    background_norms = release.group_norms(background)
    out = {}
    for concept, names in CONCEPTS.items():
        states = []
        for template in TEMPLATES[concept]:
            texts = [template.format(t=name) for name in names]
            column = name_column(tokenizer, texts[0], names[0])
            states.append(capture(model, backend, batch_of(tokenizer, texts, device), site)[:, column])
        positive = release.group_norms(torch.cat(states))
        n_pos, n_bg = len(positive), len(background_norms)
        ranks = torch.cat([positive, background_norms]).argsort(0).argsort(0).float() + 1
        auroc = (ranks[:n_pos].sum(0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_bg)
        group, runner_up = auroc.argsort(descending=True)[:2].tolist()
        out[concept] = {"group": group, "runner_up": runner_up, "auroc": (float(auroc[group]), float(auroc[runner_up]))}
    return out


def removal(
    model, backend, tokenizer, release: SASARelease, concept: str, screen: dict, site: str, device: str, seed: int = 0
) -> dict[str, float]:
    """The paper's Table 14 through native ``reject`` at the earlier name: P(C) in percent, averaged per template.

    ``reject`` removes the span of the group's decoder rows from the raw state, where the paper removes it from the
    layer-normalized state; the two differ by the projection of the token's mean, measured beside it.
    """
    ids = [single_token_id(tokenizer, name) for name in CONCEPTS[concept]]
    d = release.w_dec.shape[1]
    rows = {
        "selected group": release.block(screen[concept]["group"]),
        "runner-up group": release.block(screen[concept]["runner_up"]),
        "other concept's group": release.block(screen[OTHER[concept]]["group"]),
        "random 8-d subspace": random_rows(d, release.rank, seed).to(release.w_dec),
    }
    totals: dict[str, list[float]] = {"unsteered": [], **{name: [] for name in rows}}
    for template in TEMPLATES[concept]:
        texts = [template.format(t=name) for name in CONCEPTS[concept]]
        batch = batch_of(tokenizer, texts, device)
        column = name_column(tokenizer, texts[0], CONCEPTS[concept][0])
        clean = None
        for name, block in rows.items():
            clean, edited = backend.fwd_w_intervention(model, batch, {site: _spec(block, "reject", column)})
            totals[name].append(float(concept_mass(edited, ids).mean()))
        if clean is None:
            raise ValueError("removal needs at least one group to remove")
        totals["unsteered"].append(float(concept_mass(clean, ids).mean()))
    return {name: 100 * sum(values) / len(values) for name, values in totals.items()}


def steering_vectors(
    model, backend, tokenizer, release: SASARelease, screen: dict, site: str, device: str
) -> dict[str, dict[str, torch.Tensor]]:
    """Each group's mean decoded contribution at the names of the first prototype templates (the paper's $v_k$)."""
    out = {}
    for concept, names in CONCEPTS.items():
        states = []
        for template in TEMPLATES[concept][:N_PROTOTYPE_TEMPLATES]:
            texts = [template.format(t=name) for name in names]
            column = name_column(tokenizer, texts[0], names[0])
            states.append(capture(model, backend, batch_of(tokenizer, texts, device), site)[:, column])
        states = torch.cat(states)
        out[concept] = {
            key: release.contribution(states, screen[concept][key]).mean(0) for key in ("group", "runner_up")
        }
    return out


def addition(
    model,
    backend,
    tokenizer,
    release: SASARelease,
    concept: str,
    vectors: dict,
    site: str,
    device: str,
    alphas: Sequence[float] = (1.0, 2.0),
    seed: int = 100,
) -> dict[str, Any]:
    """The paper's Table 13 through native ``add`` at a noun standing in for the name: P(C) in percent.

    Every vector is scaled to the selected group's norm before $\\alpha$, so the comparison holds the edit's size fixed.
    """
    ids = [single_token_id(tokenizer, name) for name in CONCEPTS[concept]]
    nouns = [noun for noun in NOUNS if _is_single(tokenizer, noun)]
    own = vectors[concept]["group"]
    norm = own.norm()
    gen = torch.Generator().manual_seed(seed)
    random = torch.randn(own.shape[-1], generator=gen).to(own)
    controls = {
        "selected group": own,
        "runner-up group": vectors[concept]["runner_up"] / vectors[concept]["runner_up"].norm() * norm,
        "other concept's group": vectors[OTHER[concept]]["group"] / vectors[OTHER[concept]]["group"].norm() * norm,
        "random vector": random / random.norm() * norm,
    }
    totals: dict[str, Any] = {"unsteered": [], **{name: {a: [] for a in alphas} for name in controls}}
    for t in range(N_PROTOTYPE_TEMPLATES, len(TEMPLATES[concept])):
        template = TEMPLATES[concept][t]
        chosen = [nouns[(t * NOUNS_PER_TEMPLATE + q) % len(nouns)] for q in range(NOUNS_PER_TEMPLATE)]
        texts = [template.format(t=noun) for noun in chosen]
        batch = batch_of(tokenizer, texts, device)
        column = name_column(tokenizer, texts[0], chosen[0])
        clean = None
        for name, vector in controls.items():
            for alpha in alphas:
                clean, edited = backend.fwd_w_intervention(model, batch, {site: _spec(alpha * vector, "add", column)})
                totals[name][alpha].append(float(concept_mass(edited, ids).mean()))
        if clean is None:
            raise ValueError("addition needs at least one control and one alpha")
        totals["unsteered"].append(float(concept_mass(clean, ids).mean()))
    mean = lambda values: 100 * sum(values) / len(values)  # noqa: E731
    return {
        name: mean(values) if name == "unsteered" else {a: mean(v) for a, v in values.items()}
        for name, values in totals.items()
    }


def _is_single(tokenizer, word: str) -> bool:
    try:
        single_token_id(tokenizer, word)
    except ValueError:
        return False
    return True


def carrier_states(
    model, backend, tokenizer, concept: str, site: str, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Name-position states on the carrier sentences, ``(n_carriers * n_names, d)``, and each row's member
    index."""
    names = CONCEPTS[concept]
    states = []
    for carrier in CARRIERS[concept]:
        texts = [carrier.format(t=name) for name in names]
        column = name_column(tokenizer, texts[0], names[0])
        states.append(capture(model, backend, batch_of(tokenizer, texts, device), site)[:, column])
    labels = torch.arange(len(names)).repeat(len(CARRIERS[concept]))
    return torch.cat(states), labels


def named_subspace(states: torch.Tensor, labels: torch.Tensor, n_carriers: int, rank: int) -> torch.Tensor:
    """An Engels-style subspace: the top ``rank`` principal directions of the name states, each carrier sentence's
    mean removed first so that what varies is the member rather than the sentence."""
    n = int(labels.max()) + 1
    centered = states.view(n_carriers, n, -1)
    centered = (centered - centered.mean(1, keepdim=True)).reshape(len(states), -1)
    return torch.linalg.svd(centered.double(), full_matrices=False)[2][:rank].float()


def fit_ring(frame_coordinates: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    """A ``pca_plane`` chart's parameters from the members' centroids in frame coordinates, with each centroid."""
    n = int(labels.max()) + 1
    centroids = torch.stack([frame_coordinates[labels == i].mean(0) for i in range(n)]).double()
    origin = centroids.mean(0)
    _, singular, axes = torch.linalg.svd(centroids - origin, full_matrices=False)
    explained = float((singular[:2] ** 2).sum() / (singular**2).sum())
    return {"origin": origin.float(), "axes": axes[:2].float(), "centroids": centroids.float(), "explained": explained}


def ring_order(angles: Sequence[float]) -> dict[str, Any]:
    """How the members' angles run around the plane: steps in the dominant direction, winding number, largest
    gap."""
    a = torch.tensor(list(angles), dtype=torch.float64)
    steps = torch.remainder(a.roll(-1) - a + math.pi, 2 * math.pi) - math.pi
    direction = 1.0 if int((steps > 0).sum()) >= len(a) / 2 else -1.0
    oriented = steps * direction
    return {
        "in order": f"{int((oriented > 0).sum())} of {len(a)}",
        "winding": round(float(oriented.sum() / (2 * math.pi)), 2),
        "largest gap (deg)": round(math.degrees(float(oriented.max())), 1),
    }


def principal_cosines(a: torch.Tensor, b: torch.Tensor) -> list[float]:
    """Cosines of the principal angles between the row spans of ``a`` and ``b``."""
    qa = torch.linalg.qr(a.double().T)[0]
    qb = torch.linalg.qr(b.double().T)[0]
    return [round(float(c), 3) for c in torch.linalg.svdvals(qa.T @ qb)]


def edit_prompts(tokenizer, concept: str, task: str) -> list[tuple[str, int, int, int]]:
    """``(text, column, member, answer)`` for the copy templates or the arithmetic ones, the answer being the
    member the unedited model should name."""
    names = CONCEPTS[concept]
    out = []
    if task == "copy":
        for template in TEMPLATES[concept]:
            for i, name in enumerate(names):
                text = template.format(t=name)
                out.append((text, name_column(tokenizer, text, name), i, i))
    else:
        for template in ARITHMETIC[concept]:
            for word, k in OFFSETS.items():
                for i, name in enumerate(names):
                    text = template.format(k=word, t=name)
                    out.append((text, name_column(tokenizer, text, name), i, (i + k) % len(names)))
    return out


def chart_edits(
    model,
    backend,
    tokenizer,
    prompts,
    edits: dict[str, Callable[[torch.Tensor, int], torch.Tensor]],
    ids: Sequence[int],
    site: str,
    device: str,
) -> dict[str, dict[str, float]]:
    """Run each edit on each prompt and read whether the answer follows the member moved one step on.

    An edit maps the captured state at the name column and the member index to a new state, which ``replace``
    writes at that column. ``replace`` takes one operand for the whole batch, so each prompt runs alone.
    """
    tally = _ShiftTally(["none", *edits], ids)
    for text, column, member, answer in prompts:
        batch = {"input_ids": torch.tensor([tokenizer(text)["input_ids"]], device=device)}
        h = capture(model, backend, batch, site)[0, column]
        logits = {}
        clean = None
        for name, edit in edits.items():
            clean, logits[name] = backend.fwd_w_intervention(
                model, batch, {site: _spec(edit(h, member), "replace", column)}
            )
        if clean is None:
            raise ValueError("chart_edits needs at least one edit to run")
        tally.add({"none": clean, **logits}, answer)
    return tally.percentages(len(prompts))


class _ShiftTally:
    """Per edit: the probability of the expected answer and of the next member, and how often the next member is
    the top one."""

    def __init__(self, names: Sequence[str], ids: Sequence[int]) -> None:
        self.ids = list(ids)
        self.sums = {name: {"P(answer)": 0.0, "P(shifted answer)": 0.0, "shifted is top": 0.0} for name in names}

    def add(self, logits: dict[str, torch.Tensor], answer: int) -> None:
        shifted = (answer + 1) % len(self.ids)
        for name, out in logits.items():
            probs = out[0, -1].float().softmax(-1)[self.ids]
            self.sums[name]["P(answer)"] += float(probs[answer])
            self.sums[name]["P(shifted answer)"] += float(probs[shifted])
            self.sums[name]["shifted is top"] += float(int(probs.argmax()) == shifted)

    def percentages(self, n: int) -> dict[str, dict[str, float]]:
        return {name: {k: 100 * v / n for k, v in row.items()} for name, row in self.sums.items()}


def native_edits(
    model,
    backend,
    tokenizer,
    release: SASARelease,
    prompts,
    edits: dict[str, Callable[[int], tuple[str, Any]]],
    ids: Sequence[int],
    device: str,
) -> dict[str, dict[str, float]]:
    """:func:`chart_edits` for edits compiled to run natively at the release's latent sub-hook.

    An edit maps the member index to ``(hook, spec)`` from ``latent_edit_spec``; it is placed at the prompt's name
    column and runs as one op inside the forward pass, with no capture step, the release attached.
    """
    tally = _ShiftTally(["none", *edits], ids)
    for text, column, member, answer in prompts:
        batch = {"input_ids": torch.tensor([tokenizer(text)["input_ids"]], device=device)}
        logits = {}
        clean = None
        for name, edit in edits.items():
            hook, spec = edit(member)
            clean, logits[name] = backend.fwd_w_intervention(
                model, batch, {hook: at_column(spec, column)}, latent_model_handles=[release]
            )
        if clean is None:
            raise ValueError("native_edits needs at least one edit to run")
        tally.add({"none": clean, **logits}, answer)
    return tally.percentages(len(prompts))


def at_column(spec: Any, column: int) -> Any:
    """A compiled latent edit placed at one explicit position: its dose operand, one row for that position."""
    dose = torch.as_tensor(spec.intervention_tensor).reshape(1)
    return spec._replace(intervention_tensor=dose, position_scope="positions", positions=(column,))


def native_matches_functional(
    model,
    backend,
    release: SASARelease,
    catalog,
    target,
    batch: dict[str, torch.Tensor],
    column: int,
    site: str,
    hook: str,
    spec: Any,
) -> dict[str, float]:
    """Run a compiled latent edit natively, and the same edit functionally, on one batch, and compare the logits.

    The functional run captures the site state $h$ at ``column``, takes the block's contribution
    $v = a_B D_B$ from the release's own encoder, applies the catalog's edit of it (the same
    ``LatentEdit.edit_values`` the native op runs, dose included), and writes $h + (v' - v)$ back with ``replace``,
    one operand per batch row (each prompt's state differs). The two should agree to the activation dtype's rounding.
    """
    from interpretune.analysis.backends.interventions import InterventionSpec

    h = capture(model, backend, batch, site)[:, column]
    edit = spec.params["latent_edit"]
    columns = list(spec.unit_indices)
    rows = edit.decoder_rows().to(h)
    v = release.encode(h).float()[:, columns] @ rows
    dose = float(torch.as_tensor(spec.intervention_tensor))
    edited = h + dose * (edit.edit_values(v) - v)
    functional = {
        site: InterventionSpec(
            edited[:, None], op="replace", position_scope="positions", positions=(column,), operand_per_row=True
        )
    }
    clean, by_hand = backend.fwd_w_intervention(model, batch, functional)
    _, native = backend.fwd_w_intervention(
        model, batch, {hook: at_column(spec, column)}, latent_model_handles=[release]
    )
    return {
        "native vs functional": float((native[:, -1].float() - by_hand[:, -1].float()).abs().max()),
        "edit moved": float((native[:, -1].float() - clean[:, -1].float()).abs().max()),
    }


def hand_written_removal(
    hf_model: Any, input_ids: torch.Tensor, column: int, rows: torch.Tensor, layer: int, *, normalized: bool
) -> torch.Tensor:
    """The removal written by hand as a forward pre-hook on the decoder layer, for the payload control.

    With ``normalized`` it is the paper's removal: project out of the layer-normalized state, then undo the
    normalization. Without it, it is what ``reject`` should compute on the raw state.
    """
    basis = torch.linalg.qr(rows.double().T)[0]

    def hook(_module, args, kwargs):
        hidden = (args[0] if args else kwargs["hidden_states"]).clone()
        h = hidden[:, column].double()
        if normalized:
            mean = h.mean(-1, keepdim=True)
            scale = (h - mean).std(-1, keepdim=True) + LAYER_NORM_EPS
            n = (h - mean) / scale
            h = (n - (n @ basis) @ basis.T) * scale + mean
        else:
            h = h - (h @ basis) @ basis.T
        hidden[:, column] = h.to(hidden.dtype)
        return ((hidden,) + tuple(args[1:]), kwargs) if args else (args, dict(kwargs, hidden_states=hidden))

    handle = hf_model.model.layers[layer].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            return hf_model(input_ids=input_ids).logits
    finally:
        handle.remove()
