"""Attention-aware attribution through the interpretune surface, on a tiny model on CPU.

The numerics of each mode are circuit-tracer's and are tested there. What is tested here is what interpretune adds: that
a requested regime reaches the graph, that a stored or pruned graph still says which regime built it (the default reads
as frozen, so losing the record would mislabel a patterns-live graph with no error), and that the head reports keep the
sums that make them meaningful.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from interpretune.adapters.circuit_tracer import BaseCircuitTracerModule, CircuitTracerAttributeMixin
from interpretune.adapters.circuit_tracer.backends import DEFAULT_CT_ANALYSIS_BACKEND
from interpretune.adapters.circuit_tracer.config import CircuitTracerConfig
from interpretune.analysis.backends import AnalysisBackendCapability, SupportsHeadAttribution
from interpretune.analysis.ops.base import AnalysisBatch
from interpretune.analysis.ops.bundled.circuit_tracer.ct_ops import (
    compute_attribution_graph_impl,
    head_conductance_impl,
    head_loadings_impl,
)
from tests.core.circuit_tracer_toy import TOY_PROMPT, only_token_zero_special, tiny_gemma2_replacement_model

BACKEND = DEFAULT_CT_ANALYSIS_BACKEND
FROZEN = {"freeze_attention": True, "freeze_norms": True}
LIVE = {"freeze_attention": False, "freeze_norms": True}


class _CTModule(CircuitTracerAttributeMixin, BaseCircuitTracerModule):
    """A circuit-tracer module holding only what these ops read, without an interpretune session."""


def _ct_module(replacement_model, **cfg) -> _CTModule:
    module = _CTModule.__new__(_CTModule)
    nn.Module.__init__(module)
    object.__setattr__(module, "it_cfg", SimpleNamespace(circuit_tracer_cfg=CircuitTracerConfig(**cfg)))
    object.__setattr__(module, "_replacement_model", replacement_model)
    object.__setattr__(module, "_analysis_backend", BACKEND)
    object.__setattr__(module, "attribution_graphs", [])
    return module


@pytest.fixture(scope="module")
def toy():
    model = tiny_gemma2_replacement_model(n_layers=3)
    with only_token_zero_special(model):
        yield model


def _graph_batch(module, **kwargs) -> AnalysisBatch:
    defaults = {"prompt": TOY_PROMPT, "max_n_logits": 2, "desired_logit_prob": 0.5, "verbose": False}
    return compute_attribution_graph_impl(module, AnalysisBatch(), None, 0, **{**defaults, **kwargs})


class TestAttributionRegimes:
    def test_the_default_is_frozen_and_recorded(self, toy):
        batch = _graph_batch(_ct_module(toy))
        assert json.loads(batch.graph_attribution_regime_json) == FROZEN

    def test_a_regime_kwarg_changes_the_edges_and_is_recorded(self, toy):
        """Positive control included: the two regimes must disagree, or the record would describe nothing."""
        module = _ct_module(toy)
        frozen, live = _graph_batch(module), _graph_batch(module, freeze_attention=False)
        assert json.loads(live.graph_attribution_regime_json) == LIVE
        assert not torch.equal(frozen.adjacency_matrix, live.adjacency_matrix)

    def test_a_regime_from_the_config_reaches_the_graph(self, toy):
        batch = _graph_batch(_ct_module(toy, freeze_attention=False))
        assert json.loads(batch.graph_attribution_regime_json) == LIVE

    def test_a_call_kwarg_overrides_the_config(self, toy):
        batch = _graph_batch(_ct_module(toy, freeze_attention=False), freeze_attention=True)
        assert json.loads(batch.graph_attribution_regime_json) == FROZEN

    def test_the_record_survives_storage_and_pruning(self, toy):
        batch = _graph_batch(_ct_module(toy), freeze_attention=False)
        hydrated = BACKEND.hydrate_graph_from_batch(batch)
        assert hydrated.attribution_regime == LIVE
        assert BACKEND.build_pruned_graph(hydrated, 0.8, 0.98).attribution_regime == LIVE
        assert json.loads(BACKEND.decompose_graph(hydrated)["graph_attribution_regime_json"]) == LIVE

    def test_a_row_stored_before_the_field_existed_reads_frozen(self, toy):
        """Every graph that could be stored then was frozen, so this is knowledge rather than a default."""
        row = dict(BACKEND.decompose_graph(BACKEND.hydrate_graph_from_batch(_graph_batch(_ct_module(toy)))))
        row.pop("graph_attribution_regime_json")
        assert BACKEND.hydrate_graph_from_batch(row).attribution_regime == FROZEN


class TestHeadLoadings:
    @pytest.fixture(scope="class")
    def report(self, toy):
        target = torch.tensor([3])
        return head_loadings_impl(_ct_module(toy), AnalysisBatch(), None, 0, prompt=TOY_PROMPT, head_target=target)

    def test_the_backend_declares_it(self):
        assert AnalysisBackendCapability.HEAD_ATTRIBUTION in BACKEND.capabilities
        assert isinstance(BACKEND, SupportsHeadAttribution)

    def test_every_layer_is_reported_by_default(self, toy, report):
        n_layers, n_heads = toy.cfg.n_layers, toy.cfg.n_heads
        assert report.head_loading_layers.tolist() == list(range(n_layers))
        assert report.head_loading_shares.shape == (n_layers, n_heads)

    def test_the_parts_partition_the_edges_at_every_layer(self, report):
        partitioned = report.head_loading_shares.sum(1) + report.head_loading_bypass_shares
        assert partitioned == pytest.approx(torch.ones_like(partitioned), abs=1e-4)
        assert (report.head_loading_partition_errors < 1e-4).all()

    def test_a_head_carries_something(self, report):
        """A report of zeros would also partition the edges."""
        assert report.head_loading_shares.abs().max() > 1e-3

    def test_layers_can_be_narrowed(self, toy):
        batch = head_loadings_impl(
            _ct_module(toy), AnalysisBatch(), None, 0, prompt=TOY_PROMPT, head_target=torch.tensor([3]), head_layers=[1]
        )
        assert batch.head_loading_layers.tolist() == [1]
        assert batch.head_loading_shares.shape[0] == 1

    def test_the_regime_changes_the_split(self, toy, report):
        live = head_loadings_impl(
            _ct_module(toy), AnalysisBatch(), None, 0, prompt=TOY_PROMPT, head_target=torch.tensor([3]),
            freeze_attention=False,
        )  # fmt: skip
        assert not torch.equal(live.head_loading_shares, report.head_loading_shares)

    def test_a_missing_target_is_refused_by_name(self, toy):
        with pytest.raises(ValueError, match="head_target"):
            head_loadings_impl(_ct_module(toy), AnalysisBatch(), None, 0, prompt=TOY_PROMPT)

    def test_no_layers_is_refused_by_name(self, toy):
        with pytest.raises(ValueError, match="no layers"):
            head_loadings_impl(
                _ct_module(toy), AnalysisBatch(), None, 0, prompt=TOY_PROMPT, head_target=torch.tensor([3]),
                head_layers=[],
            )  # fmt: skip


class TestHeadConductance:
    SWAPPED = torch.tensor([0, 3, 9, 7, 2, 5, 3, 8])

    @pytest.fixture(scope="class")
    def weights(self, toy):
        weights = torch.zeros(toy.cfg.d_vocab)
        weights[3], weights[8] = 1.0, -1.0
        return weights

    @pytest.fixture(scope="class")
    def report(self, toy, weights):
        return head_conductance_impl(
            _ct_module(toy), AnalysisBatch(), None, 0,
            prompt=TOY_PROMPT, counterfactual_prompt=self.SWAPPED, target_weights=weights, key_positions=[2, 3],
        )  # fmt: skip

    def test_every_cut_adds_up_to_the_measured_change(self, toy, weights):
        """The completeness residual is the trapezoid rule's error alone, so it falls as $1/K^2$ with the step
        count $K$, where a cut missing a path would leave a residual that does not shrink.

        This toy's path is violent, so a fixed bound at few steps would say nothing. Measured here: $8.1$, $7.1$ and
        $-3.1$ at 32 steps, $4.4 \\times 10^{-3}$, $-6.6 \\times 10^{-2}$ and $-0.49$ at 128, and $3.6 \\times 10^{-4}$,
        $-4.3 \\times 10^{-3}$ and $-3.2 \\times 10^{-2}$ at 512.
        """
        coarse, fine = (
            head_conductance_impl(
                _ct_module(toy), AnalysisBatch(), None, 0,
                prompt=TOY_PROMPT, counterfactual_prompt=self.SWAPPED, target_weights=weights, n_steps=n_steps,
                points_per_batch=65,
            ).head_conductance_completeness_residual.abs()
            for n_steps in (128, 512)
        )  # fmt: skip
        assert (fine < 0.05).all()
        assert (fine < coarse / 8).all()

    def test_the_change_is_the_models_own(self, toy, weights, report):
        with torch.no_grad():
            measured = [
                float(toy._module(input_ids=ids[None]).logits[0, -1] @ weights) for ids in (TOY_PROMPT, self.SWAPPED)
            ]
        assert [report.head_conductance_t_from, report.head_conductance_t_to] == pytest.approx(measured, rel=1e-5)
        assert measured[0] != pytest.approx(measured[1], abs=1e-2)

    def test_pattern_and_value_parts_are_both_present(self, report):
        assert report.head_conductance_qk.abs().max() > 0
        assert report.head_conductance_ov.abs().max() > 0

    def test_the_key_split_is_part_of_each_whole(self, report):
        """Each keyed part is the share of its whole read at those key positions, so it needs the same shape."""
        assert report.head_conductance_qk_keys.shape == report.head_conductance_qk.shape
        assert report.head_conductance_ov_keys.shape == report.head_conductance_ov.shape

    def test_the_key_split_is_absent_unless_asked_for(self, toy, weights):
        batch = head_conductance_impl(
            _ct_module(toy), AnalysisBatch(), None, 0,
            prompt=TOY_PROMPT, counterfactual_prompt=self.SWAPPED, target_weights=weights, n_steps=4,
        )  # fmt: skip
        assert not hasattr(batch, "head_conductance_qk_keys")

    @pytest.mark.parametrize("missing", ["counterfactual_prompt", "target_weights"])
    def test_a_missing_input_is_refused_by_name(self, toy, weights, missing):
        given = {"counterfactual_prompt": self.SWAPPED, "target_weights": weights}
        given.pop(missing)
        with pytest.raises(ValueError, match=missing):
            head_conductance_impl(_ct_module(toy), AnalysisBatch(), None, 0, prompt=TOY_PROMPT, **given)

    def test_prompts_of_different_lengths_are_refused(self, toy, weights):
        with pytest.raises(ValueError, match="same length"):
            head_conductance_impl(
                _ct_module(toy), AnalysisBatch(), None, 0,
                prompt=TOY_PROMPT, counterfactual_prompt=self.SWAPPED[:-1], target_weights=weights,
            )  # fmt: skip


class TestTheTopLevelOps:
    """The wrappers a notebook calls: capability gating, declared defaults and the output schema all apply."""

    def test_the_regime_survives_the_graph_ops(self, toy):
        """Through the wrappers, whose schemas decide which columns reach an op and which leave it."""
        import interpretune as it
        import interpretune.analysis  # registers the top-level op wrappers

        module = _ct_module(toy)
        graph = it.compute_attribution_graph(
            module, None, None, 0, prompt=TOY_PROMPT, max_n_logits=2, desired_logit_prob=0.5, verbose=False,
            freeze_attention=False,
        )  # fmt: skip
        assert json.loads(graph.graph_attribution_regime_json) == LIVE
        pruned = it.graph_prune(module, AnalysisBatch(**dict(graph)), None, 0, node_threshold=0.8)
        assert json.loads(pruned.graph_attribution_regime_json) == LIVE

    def test_head_loadings(self, toy):
        import interpretune as it
        import interpretune.analysis  # registers the top-level op wrappers

        result = it.head_loadings(_ct_module(toy), None, None, 0, prompt=TOY_PROMPT, head_target=torch.tensor([3]))
        assert result.head_loading_shares.shape == (toy.cfg.n_layers, toy.cfg.n_heads)

    def test_head_conductance(self, toy):
        import interpretune as it
        import interpretune.analysis  # registers the top-level op wrappers

        weights = torch.zeros(toy.cfg.d_vocab)
        weights[3], weights[8] = 1.0, -1.0
        result = it.head_conductance(
            _ct_module(toy), None, None, 0,
            prompt=TOY_PROMPT, counterfactual_prompt=TestHeadConductance.SWAPPED, target_weights=weights, n_steps=4,
        )  # fmt: skip
        assert result.head_conductance_qk.shape == (toy.cfg.n_layers, toy.cfg.n_heads)

    def test_a_backend_declaring_it_without_the_reports_is_refused_by_name(self, toy):
        """The op's own check, for a backend the capability gate cannot catch because it declares the
        capability."""
        declared_only = SimpleNamespace(
            capabilities=frozenset({AnalysisBackendCapability.HEAD_ATTRIBUTION}), supports=lambda capability: True
        )
        module = _ct_module(toy)
        object.__setattr__(module, "_analysis_backend", declared_only)
        with pytest.raises(ValueError, match="provides no head attribution reports"):
            head_loadings_impl(module, AnalysisBatch(), None, 0, prompt=TOY_PROMPT, head_target=torch.tensor([3]))

    def test_a_backend_without_the_capability_is_refused_by_name(self, toy):
        """The capability is what a backend without these reports is refused on, before anything runs."""
        import interpretune as it
        import interpretune.analysis  # registers the top-level op wrappers

        class _NoHeads(type(BACKEND)):
            @property
            def capabilities(self):
                return frozenset({AnalysisBackendCapability.ATTRIBUTION_GRAPH})

        module = _ct_module(toy)
        object.__setattr__(module, "_analysis_backend", _NoHeads())
        with pytest.raises(Exception, match="head_attribution"):
            it.head_loadings(module, None, None, 0, prompt=TOY_PROMPT, head_target=torch.tensor([3]))
