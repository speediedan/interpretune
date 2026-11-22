"""Attention-aware attribution through the interpretune surface, on a tiny model on CPU.

The numerics of each mode are circuit-tracer's and are tested there. What is tested here is what interpretune adds: that
a requested regime reaches the graph, that a stored or pruned graph still says which regime built it (the default reads
as frozen, so losing the record would mislabel a patterns-live graph with no error), and that the head reports keep the
sums that make them meaningful.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from interpretune.adapters.circuit_tracer import BaseCircuitTracerModule, CircuitTracerAttributeMixin
from interpretune.adapters.circuit_tracer.backends import DEFAULT_CT_ANALYSIS_BACKEND, _head_source_report
from interpretune.adapters.circuit_tracer.config import CircuitTracerConfig
from interpretune.analysis.backends import AttributionGraphSupport, HeadAttributionSupport, SupportsHeadAttribution
from interpretune.analysis.ops.base import AnalysisBatch
from interpretune.analysis.ops.bundled.circuit_tracer.ct_ops import (
    compute_attribution_graph_impl,
    head_conductance_impl,
    head_loadings_impl,
)
from tests.core.circuit_tracer_toy import (
    TOY_PROMPT,
    BaseModelBackendStandIn,
    only_token_zero_special,
    tiny_gemma2_replacement_model,
)

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
    object.__setattr__(module, "_model_backend", BaseModelBackendStandIn())
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
        assert isinstance(BACKEND.technique_support["head_attribution"], HeadAttributionSupport)
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

    def test_raw_sources_partition_the_same_total_as_the_heads(self, toy, report):
        n_features = len(report.head_loading_feature_nodes)
        assert report.head_loading_feature_nodes.shape == (n_features, 3)
        assert report.head_loading_feature_edges.shape == (n_features,)
        assert report.head_loading_error_edges.shape == (toy.cfg.n_layers, len(TOY_PROMPT))
        assert report.head_loading_embedding_edges.shape == (len(TOY_PROMPT),)
        source_total = sum(
            values.double().sum()
            for values in (
                report.head_loading_feature_edges,
                report.head_loading_error_edges,
                report.head_loading_embedding_edges,
            )
        )
        assert float(source_total) == pytest.approx(report.head_loading_total, rel=1e-5, abs=1e-5)
        assert report.head_loading_feature_edges.abs().max() > 0
        assert report.head_loading_error_edges.abs().max() > 0
        assert report.head_loading_embedding_edges.abs().max() > 0
        assert torch.allclose(report.head_loading_head_edges / report.head_loading_total, report.head_loading_shares)
        assert torch.allclose(
            report.head_loading_bypass_edges / report.head_loading_total, report.head_loading_bypass_shares
        )
        assert json.loads(report.head_loading_regime_json) == FROZEN

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
        assert json.loads(live.head_loading_regime_json) == LIVE

    def test_a_missing_target_is_refused_by_name(self, toy):
        with pytest.raises(ValueError, match="head_target"):
            head_loadings_impl(_ct_module(toy), AnalysisBatch(), None, 0, prompt=TOY_PROMPT)

    def test_no_layers_is_refused_by_name(self, toy):
        with pytest.raises(ValueError, match="no layers"):
            head_loadings_impl(
                _ct_module(toy), AnalysisBatch(), None, 0, prompt=TOY_PROMPT, head_target=torch.tensor([3]),
                head_layers=[],
            )  # fmt: skip


class TestRawSourceIdentity:
    @pytest.fixture
    def parts(self):
        from circuit_tracer.attribution.head_loadings_nnsight import HeadLoadings

        edges = torch.arange(1.0, 8.0)
        reference = HeadLoadings(
            layer=0,
            per_head=torch.stack([edges / 4, edges / 4]),
            bypass=edges / 2,
            total=edges,
            active_features=torch.tensor([[0, 1, 2]]),
            n_layers=2,
            n_pos=2,
        )
        return [reference, replace(reference, layer=1)]

    def test_source_coordinates_keep_the_upstream_row_order(self, parts):
        result = _head_source_report(parts)
        assert result["head_loading_feature_nodes"].tolist() == [[0, 1, 2]]
        assert result["head_loading_feature_edges"].tolist() == [1.0]
        assert result["head_loading_error_edges"].tolist() == [[2.0, 3.0], [4.0, 5.0]]
        assert result["head_loading_embedding_edges"].tolist() == [6.0, 7.0]

    @pytest.mark.parametrize(
        "change",
        [
            {"active_features": torch.tensor([[0, 1, 3]])},
            {"total": torch.arange(2.0, 9.0)},
            {"total": torch.ones(6)},
            {"n_pos": 3},
        ],
    )
    def test_a_cut_attributing_other_sources_is_refused(self, parts, change):
        parts[1] = replace(parts[1], **change)
        with pytest.raises(ValueError, match="layer 1 attributes different source"):
            _head_source_report(parts)

    def test_an_incomplete_source_vector_is_refused(self, parts):
        parts[0] = replace(parts[0], total=torch.ones(6))
        with pytest.raises(ValueError, match="incomplete source-node layout"):
            _head_source_report(parts)

    def test_nonfinite_source_edges_are_refused(self, parts):
        parts = [replace(part, total=torch.full((7,), float("inf"))) for part in parts]
        with pytest.raises(ValueError, match="nonfinite source edges"):
            _head_source_report(parts)


class TestMeasuredBasisIdentity:
    @pytest.fixture
    def basis_toy(self):
        # Structural mutations below must not leave the shared model's Envoy children pointing at replaced modules.
        return tiny_gemma2_replacement_model(n_layers=2)

    @pytest.mark.parametrize("field", ["decoder", "threshold", "bandwidth", "zero-positions", "hook"])
    def test_actual_basis_changes_cannot_keep_the_same_identity(self, basis_toy, monkeypatch, field):
        from it_examples.utils.sparse_attribution_collect import transcoder_basis_id

        before = transcoder_basis_id(basis_toy)
        basis = basis_toy.transcoders._module
        tc = basis[0]
        if field in ("decoder", "threshold"):
            tensor = tc.W_dec if field == "decoder" else tc.activation_function.threshold
            saved = tensor.detach().clone()
            try:
                with torch.no_grad():
                    tensor.add_(0.25)
                assert transcoder_basis_id(basis_toy) != before
            finally:
                with torch.no_grad():
                    tensor.copy_(saved)
        else:
            obj, attribute, value = {
                "bandwidth": (tc.activation_function, "bandwidth", 0.25),
                "zero-positions": (basis_toy, "zero_positions", slice(0, 2)),
                "hook": (basis, "feature_input_hook", "another-hook"),
            }[field]
            monkeypatch.setattr(obj, attribute, value)
            assert transcoder_basis_id(basis_toy) != before

    def test_an_unrecorded_activation_kind_is_refused(self, basis_toy, monkeypatch):
        from it_examples.utils.sparse_attribution_collect import transcoder_basis_id

        monkeypatch.setattr(basis_toy.transcoders._module[0], "activation_function", torch.nn.ReLU())
        with pytest.raises(ValueError, match="supported JumpReLU"):
            transcoder_basis_id(basis_toy)


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
        assert result.head_loading_feature_nodes.shape[1] == 3
        assert result.head_loading_error_edges.shape == (toy.cfg.n_layers, len(TOY_PROMPT))
        assert result.head_loading_head_edges.shape == result.head_loading_shares.shape
        assert json.loads(result.head_loading_regime_json) == FROZEN

    def test_a_logit_gap_report_keeps_the_decision_separate_from_source_edges(self, toy):
        import interpretune as it
        import interpretune.analysis
        from circuit_tracer.attribution.targets import CustomTarget

        from it_examples.utils.sparse_attribution_collect import head_loading_row
        from it_examples.utils.sparse_attribution_report import checkpoint_attribution_report

        positive, negative = 3, 8
        with torch.no_grad():
            logits = toy._module(input_ids=TOY_PROMPT[None]).logits[0, -1]
            decision = float(logits[positive] - logits[negative])
        unembed = toy._module.get_output_embeddings().weight.detach()
        target = CustomTarget("logit_gap", 1.0, unembed[positive] - unembed[negative])
        raw = it.head_loadings(_ct_module(toy), None, None, 0, prompt=TOY_PROMPT, head_target=target)
        row = head_loading_row(
            raw, row_id="validation-1", checkpoint_sha256="a" * 64, basis_id="toy-transcoders",
            decision=decision, answer_position=len(TOY_PROMPT) - 1,
        )  # fmt: skip
        summary = checkpoint_attribution_report(
            [row], checkpoint_sha256="a" * 64, basis_id="toy-transcoders", model_id="tiny-gemma2",
            model_revision="random-test-model", target_token_ids=(positive, negative),
        )  # fmt: skip
        assert row.decision == decision
        assert row.bias is None
        assert summary["attribution_regime"] == FROZEN
        assert summary["source_shares"]["bias"] is None
        assert summary["source_shares"]["unassigned"] == pytest.approx(1 - row.edge_total / decision, abs=1e-5)
        assert row.edge_total != pytest.approx(decision, abs=1e-2)
        assert torch.tensor(summary["head_shares"]) == pytest.approx(raw.head_loading_head_edges / decision, abs=1e-5)
        assert row.feature_nodes == [tuple(node) for node in raw.head_loading_feature_nodes.tolist()]

    @pytest.mark.parametrize("regime", [{}, {"freeze_attention": True, "freeze_norms": "false"}])
    def test_a_row_with_incomplete_or_ambiguous_regime_is_refused(self, regime):
        from it_examples.utils.sparse_attribution_collect import head_loading_row

        with pytest.raises(ValueError, match="regime must"):
            head_loading_row(
                {"head_loading_regime_json": json.dumps(regime)},
                row_id="validation-1", checkpoint_sha256="a" * 64, basis_id="toy-transcoders",
                decision=1.0, answer_position=1,
            )  # fmt: skip

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
        """A backend that declares the technique without the report methods is refused by the record's protocol,
        naming it, before the op reaches a missing method."""
        declared_only = SimpleNamespace(
            capabilities=frozenset(),
            supports=lambda capability: False,
            technique_support={"head_attribution": HeadAttributionSupport()},
        )
        module = _ct_module(toy)
        object.__setattr__(module, "_analysis_backend", declared_only)
        with pytest.raises(ValueError, match="does not implement SupportsHeadAttribution"):
            head_loadings_impl(module, AnalysisBatch(), None, 0, prompt=TOY_PROMPT, head_target=torch.tensor([3]))

    def test_a_backend_without_the_technique_is_refused_by_name(self, toy):
        """The technique is what a backend without these reports is refused on, before anything runs."""
        import interpretune as it
        import interpretune.analysis  # registers the top-level op wrappers

        class _NoHeads(type(BACKEND)):
            @property
            def technique_support(self):
                return {"attribution_graph": AttributionGraphSupport()}

        module = _ct_module(toy)
        object.__setattr__(module, "_analysis_backend", _NoHeads())
        with pytest.raises(Exception, match="head_attribution"):
            it.head_loadings(module, None, None, 0, prompt=TOY_PROMPT, head_target=torch.tensor([3]))
