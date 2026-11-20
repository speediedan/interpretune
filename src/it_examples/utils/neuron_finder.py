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
