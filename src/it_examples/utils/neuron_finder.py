r"""Helpers for the neuron-finder example: its ``layer:neuron:dose`` settings as a checkpoint address book.

neuron-finder (``changjonathanc/neuron-finder``) writes each steering setting it finds as ``layer:neuron:dose``, the
dose signed and in units of :math:`m_\ell / \lVert W_u \rVert_2`: the layer's median residual norm over the norm of the
neuron's down-projection weights. :func:`settings_book` carries a list of them as a venngram checkpoint book, the units
in one set at ``point:mlp.out.hook_in`` and the doses in a treatment beside it, written in that unit through a
``statistic`` transform. :func:`dose_unit` computes the statistic for one model, so
``recipe_interventions(..., statistics={"dose_unit": dose_unit(...)})`` steers in activation units.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, NamedTuple

import torch

#: The site neuron-finder's neurons are units of: the MLP's hidden activation, the down projection's input.
SITE = "mlp.out.hook_in"


class Setting(NamedTuple):
    """One steering setting: a neuron and its signed dose."""

    layer: int
    unit: int
    dose: float

    def __str__(self) -> str:
        return f"{self.layer}:{self.unit}:{self.dose:g}"


def parse_settings(settings: Iterable[str | Setting]) -> list[Setting]:
    """Read ``layer:neuron:dose`` strings (neuron-finder's ``--neurons`` form); a :class:`Setting` passes through.

    Raises:
        ValueError: If an entry is not three fields, a layer or neuron is not a nonnegative integer, a dose is not
            finite, or one neuron appears twice (two doses for it would be two treatments).
    """
    out: list[Setting] = []
    seen: set[tuple[int, int]] = set()
    for entry in settings:
        if isinstance(entry, Setting):
            setting = entry
        else:
            fields = str(entry).strip().split(":")
            if len(fields) != 3:
                raise ValueError(f"A setting is `layer:neuron:dose`, got {entry!r}")
            try:
                setting = Setting(int(fields[0]), int(fields[1]), float(fields[2]))
            except ValueError as err:
                raise ValueError(f"A setting is `layer:neuron:dose` with integer layer and neuron: {entry!r}") from err
        if setting.layer < 0 or setting.unit < 0 or not torch.isfinite(torch.tensor(setting.dose)):
            raise ValueError(f"{entry!r}: the layer and neuron are nonnegative and the dose finite")
        if (setting.layer, setting.unit) in seen:
            raise ValueError(f"{entry!r}: neuron {setting.unit} of layer {setting.layer} is set twice")
        seen.add((setting.layer, setting.unit))
        out.append(setting)
    if not out:
        raise ValueError("No settings given")
    return out


def settings_book(
    settings: Iterable[str | Setting],
    *,
    model_id: str,
    revision: str,
    architecture: str | None = None,
    name: str = "neuron-finder",
    set_name: str = "found",
    treatment: str = "steer",
    statistic: str = "dose_unit",
    tokens: str = "all",
) -> dict[str, Any]:
    """A checkpoint address book for ``settings``: which neurons in one set, their doses in one treatment.

    A neuron index means something in one checkpoint only, so the book pins ``model_id`` at ``revision`` (in the
    unprocessed basis, the model's own). The doses stay out of the set: an address says which units, and the treatment
    says what to do to them, here ``add`` at every token (``tokens="last"`` for the final one), with each dose
    multiplied by the per-neuron ``statistic`` the caller supplies (:func:`dose_unit`).

    Returns:
        The book as a mapping, ready for ``yaml.safe_dump`` or ``interpretune.analysis.venngram.load_recipe``.
    """
    parsed = parse_settings(settings)
    units: dict[int, list[int]] = {}
    doses: dict[int, dict[int, float]] = {}
    for s in sorted(parsed):
        units.setdefault(s.layer, []).append(s.unit)
        doses.setdefault(s.layer, {})[s.unit] = s.dose
    applies_to: dict[str, Any] = {"models": [{"id": model_id, "revision": revision, "basis": "unprocessed"}]}
    if architecture is not None:
        applies_to = {"architectures": [architecture], **applies_to}
    return {
        "schema_version": 1,
        "name": name,
        "description": "Imported neuron-finder settings: " + ", ".join(str(s) for s in parsed),
        "portability": "checkpoint",
        "applies_to": applies_to,
        "sets": {set_name: {"point": SITE, "units": units}},
        "treatments": {
            treatment: {
                "model": {"id": model_id, "revision": revision},
                "stages": [
                    {
                        "site": SITE,
                        "tokens": tokens,
                        "ops": [
                            {
                                "op": "add",
                                "target": {"set": set_name},
                                "value": {"doses": doses, "transform": [{"statistic": {"name": statistic}}]},
                            }
                        ],
                    }
                ],
            }
        },
    }


def book_settings(book: Mapping[str, Any], *, set_name: str = "found", treatment: str = "steer") -> list[Setting]:
    """The settings :func:`settings_book` wrote, read back from its mapping (the export direction)."""
    (op,) = book["treatments"][treatment]["stages"][0]["ops"]
    if op["target"] != {"set": set_name}:
        raise ValueError(f"The treatment {treatment!r} does not act on the set {set_name!r}")
    return [
        Setting(int(layer), int(unit), float(dose))
        for layer, per_unit in sorted(op["value"]["doses"].items())
        for unit, dose in sorted(per_unit.items())
    ]


def dose_unit(
    down_projections: Mapping[int, tuple[torch.Tensor, int]],
    residual_medians: Mapping[int, float],
    members: Iterable[tuple[int, int]],
) -> dict[tuple[int, int], float]:
    r"""The activation size of one dose unit for each ``(layer, unit)``: :math:`m_\ell / \lVert W_u \rVert_2`.

    Args:
        down_projections: Each layer's down-projection weight and its unit axis (``unit_dim``: 0 for GPT-2's
            ``Conv1D``, 1 for an ``nn.Linear``; read it from the view's structure, ``structure().unit_dim(name, 2)``).
        residual_medians: Each layer's median residual norm :math:`m_\ell` (:func:`residual_medians`).
        members: The ``(layer, unit)`` pairs to compute it for.
    """
    from interpretune.analysis.venngram import normalized_unit_doses

    by_layer: dict[int, list[int]] = {}
    for layer, unit in members:
        by_layer.setdefault(int(layer), []).append(int(unit))
    out: dict[tuple[int, int], float] = {}
    for layer, units in sorted(by_layer.items()):
        weight, unit_dim = down_projections[layer]
        values = normalized_unit_doses(
            1.0,
            units=tuple(units),
            down_projection=weight,
            unit_dim=unit_dim,
            median_residual_norm=float(residual_medians[layer]),
        )
        out.update({(layer, u): float(v) for u, v in zip(units, values.tolist(), strict=True)})
    return out


@torch.no_grad()
def residual_medians(hidden_states: Iterable[tuple[torch.Tensor, ...]], masks: Iterable[torch.Tensor]) -> list[float]:
    r"""Neuron-finder's :math:`m_\ell`: per text, the median token norm of the residual after layer :math:`\ell`
    over its tokens from the second on, then the mean of those medians over texts.

    Args:
        hidden_states: Per batch, a model's ``output_hidden_states`` (the embeddings first, then one per layer).
        masks: Per batch, the attention mask selecting each text's real tokens.

    Returns:
        One value per layer.
    """
    sums: torch.Tensor | None = None
    count = 0
    for states, mask in zip(hidden_states, masks, strict=True):
        layers = torch.stack(states[1:]).float()  # layers, batch, tokens, width
        norms = layers.norm(dim=-1)
        for j in range(norms.shape[1]):
            real = norms[:, j][:, mask[j].bool()][:, 1:]
            med = real.median(dim=-1).values
            sums = med if sums is None else sums + med
            count += 1
    if sums is None:
        raise ValueError("No texts given")
    return (sums / count).tolist()


def activation_std(activations: torch.Tensor) -> torch.Tensor:
    """Each neuron's own dose unit: the standard deviation of its activation over a screen set.

    Args:
        activations: ``(items, layers, units)`` activations at the screened position.

    Returns:
        ``(layers, units)``, floored at ``1e-6`` so a dead neuron's unit is finite.
    """
    return activations.float().std(dim=0).clamp_min(1e-6)


def select_candidates(score: torch.Tensor, top: int) -> dict[int, list[tuple[int, int]]]:
    """The ``top`` neurons toward each target from a ``(layers, units)`` screen score.

    Returns:
        ``{+1: [...], -1: [...]}``: the ``(layer, unit)`` pairs with the largest positive and the largest negative
        scores, strongest first. A target of ``+1`` moves the behaviour's measure up, ``-1`` down.
    """
    flat = score.flatten()
    width = score.shape[1]
    pick = {t: torch.topk(t * flat, top).indices.tolist() for t in (1, -1)}
    return {t: [(i // width, i % width) for i in idx] for t, idx in pick.items()}


def edit_sign(write: torch.Tensor, layer: int, unit: int, target: int) -> int:
    """The sign of the edit that moves the measure toward ``target`` to first order.

    A neuron whose gradient (the write screen) is negative moves the measure up when its activation goes down, so the
    edit's sign is the target's times the gradient's. Taking a screen's ranking for the edit sign moves every candidate
    the same way, whatever its target.
    """
    if target not in (1, -1):
        raise ValueError(f"A target is +1 or -1, got {target!r}")
    return target * (1 if float(write[layer, unit]) >= 0 else -1)


def candidate_settings(
    write: torch.Tensor, candidates: Mapping[int, Iterable[tuple[int, int]]], dose: float
) -> list[Setting]:
    """:class:`Setting` entries for screened candidates at one dose, each signed by :func:`edit_sign`."""
    return [
        Setting(int(layer), int(unit), edit_sign(write, layer, unit, target) * float(dose))
        for target, members in candidates.items()
        for layer, unit in members
    ]


def found(signed_effects: Mapping[Any, float], control_effects: Iterable[float], q: float = 0.95) -> dict[Any, bool]:
    """Which candidates beat the controls: a signed effect toward the target above the controls' ``q`` quantile.

    Args:
        signed_effects: Each candidate's mean change of the measure, multiplied by its target.
        control_effects: The same for random neurons steered toward random targets (signed by their target).
        q: The quantile of the controls a candidate must exceed.
    """
    controls = torch.tensor(list(control_effects), dtype=torch.float64)
    if controls.numel() == 0:
        raise ValueError("No control effects given")
    threshold = float(torch.quantile(controls, q))
    return {key: float(value) > threshold for key, value in signed_effects.items()}


def answer_margin(logits: torch.Tensor, answer_ids: tuple[int, int]) -> torch.Tensor:
    """The behaviour measured: the first answer's logit minus the second's at the answer position (the last, with
    left padding), one value per item."""
    last = logits[:, -1, :].float()
    return last[:, answer_ids[0]] - last[:, answer_ids[1]]


def down_projections(module: Any, n_layers: int) -> dict[int, torch.nn.Module]:
    """Each layer's MLP down projection, found by address through the module's view, so no module path is spelled.

    Its input is the site :data:`SITE` (``point:mlp.out.hook_in``): the MLP's hidden neurons.
    """
    from interpretune.tuning.weights import resolve_weight_targets

    out: dict[int, torch.nn.Module] = {}
    for layer in range(n_layers):
        (name,) = resolve_weight_targets(module, f"component:mlp.out & layers:{layer} & leaf:weight")
        out[layer] = module.get_submodule(name.rsplit(".", 1)[0])
    return out


class Screen(NamedTuple):
    """Neuron-finder's two screens over a set of items, at the answer position.

    ``activations`` is ``(items, layers, units)``; ``write`` is the gradient of the behaviour with respect to each
    neuron, averaged over the items, ``(layers, units)``; ``labels`` holds each item's label.
    """

    activations: torch.Tensor
    write: torch.Tensor
    labels: torch.Tensor

    def read(self, positive_label: int = 0) -> torch.Tensor:
        """The read screen: each neuron's mean activation on items labelled ``positive_label`` minus the others',
        over its standard deviation, ``(layers, units)``."""
        pos = self.labels == positive_label
        acts = self.activations.float()
        return (acts[pos].mean(0) - acts[~pos].mean(0)) / activation_std(acts)


def screen(
    model: torch.nn.Module,
    projections: Mapping[int, torch.nn.Module],
    batches: Iterable[Mapping[str, torch.Tensor]],
    answer_ids: tuple[int, int],
) -> Screen:
    """Run both screens: record each neuron at the answer position, and the gradient of :func:`answer_margin`
    with respect to it, through forward pre-hooks on the down projections (the model's parameters need no gradient).
    """
    captured: dict[int, torch.Tensor] = {}

    def hook(layer: int):
        def pre(_module, args):
            x = args[0]
            if torch.is_grad_enabled():
                x.retain_grad()
            captured[layer] = x
            return None

        return pre

    handles = [proj.register_forward_pre_hook(hook(layer)) for layer, proj in projections.items()]
    trainable = [p for p in model.parameters() if p.requires_grad]
    for p in trainable:  # the gradients wanted are the neurons', not the parameters'
        p.requires_grad_(False)
    layers = sorted(projections)
    acts, labels = [], []
    write: torch.Tensor | None = None
    count = 0
    try:
        for batch in batches:
            device = next(model.parameters()).device
            ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            with torch.enable_grad():
                embeds = model.get_input_embeddings()(ids).detach().requires_grad_(True)
                logits = model(inputs_embeds=embeds, attention_mask=mask).logits
                answer_margin(logits, answer_ids).sum().backward()
            grads = torch.stack([captured[layer].grad[:, -1].float() for layer in layers], dim=1)
            write = grads.sum(0) if write is None else write + grads.sum(0)
            acts.append(torch.stack([captured[layer][:, -1].detach().float() for layer in layers], dim=1).cpu())
            labels.append(batch["labels"].view(-1).cpu())
            count += ids.shape[0]
            captured.clear()
    finally:
        for handle in handles:
            handle.remove()
        for p in trainable:
            p.requires_grad_(True)
    if write is None:
        raise ValueError("No batches given")
    return Screen(torch.cat(acts), (write / count).cpu(), torch.cat(labels))


def book_payload(book: Mapping[str, Any], width: int, unit_sizes: Mapping[tuple[int, int], float]) -> dict:
    """The backend intervention payload of a :func:`settings_book`'s treatment, its doses multiplied by
    ``unit_sizes`` (the book's ``statistic``, per ``(layer, unit)``)."""
    from interpretune.analysis.venngram import load_recipe, recipe_interventions

    (op,) = book["treatments"]["steer"]["stages"][0]["ops"]
    (transform,) = op["value"]["transform"]
    recipe = load_recipe(book).recipe
    return recipe_interventions(
        recipe, site_widths={SITE: width}, statistics={transform["statistic"]["name"]: dict(unit_sizes)}
    )


@torch.no_grad()
def steered_margins(
    backend: Any,
    model: Any,
    batches: Iterable[Mapping[str, torch.Tensor]],
    payload: Mapping[str, Any],
    answer_ids: tuple[int, int],
    input_key: str = "input_ids",
) -> tuple[torch.Tensor, torch.Tensor]:
    """The clean and the steered :func:`answer_margin` of every item, through a model backend's
    ``fwd_w_intervention``.

    ``input_key`` names the token ids for the backend's model: ``input_ids`` for an HF model (and nnsight's), ``input``
    for a TransformerLens bridge. The attention mask is passed with them, so left padding is masked on both.
    """
    clean, edited = [], []
    for batch in batches:
        inputs = {input_key: batch["input_ids"], "attention_mask": batch["attention_mask"]}
        before, after = backend.fwd_w_intervention(model, inputs, dict(payload))
        clean.append(answer_margin(before, answer_ids).cpu())
        edited.append(answer_margin(after, answer_ids).cpu())
    return torch.cat(clean), torch.cat(edited)
