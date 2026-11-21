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
"""The ``weight_equivalence_check`` op: a per-neuron recipe's weight form against its activation form, on a batch.

The op is driven directly with a tiny decoder whose view is a venngram ``MappedView`` over a schema-5 map (each row's
``layout`` recorded), and a backend that executes real forwards and applies the payload with the shared
``apply_intervention`` math at ``mlp.out``'s input. The same op runs on gpt2 through the runner, the real registry map
and a real backend in the model-backend conformance suite.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

pytest.importorskip("venngram", reason="the venngram extra is not installed")

from venngram.maps.schema import from_document  # noqa: E402
from venngram.maps.view import MappedView  # noqa: E402

from interpretune.analysis.backends import Capability, InterventionSupport, PositionScope  # noqa: E402
from interpretune.analysis.ops.base import AnalysisBatch  # noqa: E402
from interpretune.analysis.ops.bundled.concept import concept_ops  # noqa: E402
from interpretune.tuning.views import VIEW_ATTR  # noqa: E402
from interpretune.tuning.weights import WeightEdit, weight_form_site_widths  # noqa: E402
from tests.core.test_tuning_weights import (  # noqa: E402
    H,
    IDS,
    MAP,
    SCALE,
    SHIFT,
    TRIPLE,
    UNMAPPED_LAYOUT,
    TinyDecoder,
    _stage_doc,
    _stage_recipe,
)


class _UnitBackend:
    """Declares the intervention surface and executes it for real: each spec of the payload at ``blocks.<i>.<site>``
    is applied to the input of layer ``i``'s ``mlp.down`` (the tiny decoder's ``mlp.out``), at every position."""

    capabilities = frozenset({Capability.INTERVENTION})
    intervention_support = InterventionSupport(position_scopes=frozenset(PositionScope), every_registered_op=True)

    def __init__(self, ignore_payload: bool = False) -> None:
        self.ignore_payload = ignore_payload
        self.forwards = 0

    def fwd(self, model, batch):
        self.forwards += 1
        return model(batch["input"])

    def fwd_w_intervention(self, model, batch, interventions, latent_model_handles=None):
        from interpretune.analysis.backends import apply_intervention

        pre = self.fwd(model, batch)
        handles = []
        for pattern, specs in ({} if self.ignore_payload else interventions).items():
            layer = int(pattern.split(".")[1])

            def edit(_m, args, specs=specs):
                (x,) = args
                for spec in specs:
                    x = apply_intervention(x.clone(), spec, last_pos=x.shape[1] - 1)
                return (x,)

            handles.append(model.layers[layer].mlp.down.register_forward_pre_hook(edit))
        try:
            post = self.fwd(model, batch)
        finally:
            for handle in handles:
                handle.remove()
        return pre, post


class _Module(nn.Module):
    """An interpretune-shaped module: its model, its model backend and its attached view."""

    def __init__(self, backend: _UnitBackend, doc=MAP) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.model = TinyDecoder()
        self._model_backend = backend
        self.analysis_cfg = None
        object.__setattr__(self, VIEW_ATTR, MappedView(self, from_document(doc), mount="model"))


def _check(module: nn.Module, recipe) -> AnalysisBatch:
    return concept_ops.weight_equivalence_check_impl(module, AnalysisBatch(), {"input": IDS}, 0, recipe=recipe)


@pytest.mark.parametrize("ops", [[SCALE, SHIFT], [SHIFT, TRIPLE]], ids=["ablate-then-dose", "dose-then-scale"])
def test_a_per_neuron_recipe_is_reported_equivalent_and_the_weights_are_left_as_they_were(ops):
    module = _Module(_UnitBackend())
    base = {n: p.detach().clone() for n, p in module.named_parameters()}
    out = _check(module, _stage_recipe(ops))
    assert out.we_equivalent is True
    for name in ("we_max_abs_diff", "we_max_abs"):
        value = getattr(out, name)
        assert isinstance(value, torch.Tensor) and value.dim() == 0 and value.dtype == torch.float32, (name, value)
    assert float(out.we_max_abs) > 0.0
    assert float(out.we_max_abs_diff) <= 1e-5 + 1e-4 * float(out.we_max_abs)
    assert not any(isinstance(m, WeightEdit) for m in module.modules())
    assert all(torch.equal(p, base[n]) for n, p in module.named_parameters())


def test_an_activation_path_that_ignores_the_payload_is_reported_not_equivalent():
    """Positive control: a backend that drops the activation form leaves the weight form's change unmatched, so an op
    that always reported agreement would fail here."""
    out = _check(_Module(_UnitBackend(ignore_payload=True)), _stage_recipe([SCALE, SHIFT]))
    assert out.we_equivalent is False
    assert float(out.we_max_abs_diff) > 1e-5 + 1e-4 * float(out.we_max_abs)


def test_a_recipe_document_is_read_as_load_recipe_reads_it():
    assert _check(_Module(_UnitBackend()), _stage_doc([SCALE, SHIFT])).we_equivalent is True


@pytest.mark.parametrize(
    "tokens, site, match",
    [
        ("last", "mlp.out.hook_in", "tokens `last`"),
        ("all", "resid_post", "only unit ops at `mlp.out.hook_in`"),
    ],
)
def test_a_recipe_with_no_exact_weight_form_is_refused_with_the_reason_before_any_forward(tokens, site, match):
    backend = _UnitBackend()
    recipe = _stage_recipe([SCALE], tokens=tokens, site=site)
    with pytest.raises(ValueError, match=rf"weight_equivalence_check: recipe .* has no exact weight form.*{match}"):
        _check(_Module(backend), recipe)
    assert backend.forwards == 0


def test_a_view_whose_map_records_no_layout_is_refused_naming_the_layout():
    backend = _UnitBackend()
    with pytest.raises(ValueError, match="`layout`"):
        _check(_Module(backend, doc=UNMAPPED_LAYOUT), _stage_recipe([SCALE, SHIFT]))
    assert backend.forwards == 0


def test_a_missing_recipe_or_intervention_surface_is_refused_by_name():
    with pytest.raises(ValueError, match="requires `recipe`"):
        concept_ops.weight_equivalence_check_impl(_Module(_UnitBackend()), AnalysisBatch(), {"input": IDS}, 0)
    backend = _UnitBackend()
    backend.capabilities = frozenset()
    with pytest.raises(ValueError, match="INTERVENTION"):
        _check(_Module(backend), _stage_recipe([SCALE, SHIFT]))


def test_the_site_width_is_the_unit_axis_of_mlp_outs_weight():
    module = _Module(_UnitBackend())
    assert weight_form_site_widths(module, _stage_recipe([SCALE, SHIFT])) == {"mlp.out.hook_in": H}
