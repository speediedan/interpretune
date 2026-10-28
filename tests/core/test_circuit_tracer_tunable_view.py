"""The circuit-tracer composition's tunable view (#787): transcoders as promotable instruments, the base model
refused by name unless ``tune_base`` asks for it (#813), attribution unaffected by what a schedule thawed, and low-
rank corrections that let lazily loaded transcoder weights be tuned."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("finetuning_scheduler.tunable_view", reason="finetuning-scheduler predates the tunable view")
pytest.importorskip("circuit_tracer", reason="circuit-tracer is not installed")

from circuit_tracer.transcoder import SingleLayerTranscoder, TranscoderSet  # noqa: E402
from circuit_tracer.transcoder.activation_functions import JumpReLU  # noqa: E402
from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder  # noqa: E402
from finetuning_scheduler.testing import TunableViewConformance  # noqa: E402
from finetuning_scheduler.tunable_view import Unresolvable, as_view  # noqa: E402

from interpretune.adapters.circuit_tracer.adapter import BaseCircuitTracerModule, CircuitTracerAttributeMixin  # noqa: E402
from interpretune.adapters.circuit_tracer.config import CircuitTracerConfig  # noqa: E402
from interpretune.adapters.circuit_tracer.tunable_view import (  # noqa: E402
    BASE_MODEL_REASON,
    BASE_UNREACHABLE_REASON,
    DISK_OFFLOAD_REASON,
    PROMOTE_BASE_REASON,
    CircuitTracerView,
    CrossLayerTranscoderWithDeltas,
    SingleLayerTranscoderWithDeltas,
    attach_lazy_deltas,
    attribution_frozen,
    lazy_weights,
)
from interpretune.tuning.views import VIEW_ATTR  # noqa: E402
from tests.core.circuit_tracer_toy import TOY_PROMPT, only_token_zero_special, tiny_gemma2_replacement_model  # noqa: E402

D_MODEL, D_TC = 8, 32


class _CTModule(CircuitTracerAttributeMixin, BaseCircuitTracerModule):
    """A circuit-tracer module holding only what its view reads, without an interpretune session."""


def _ct_module(replacement_model, hf_model: nn.Module | None = None, **cfg) -> _CTModule:
    module = _CTModule.__new__(_CTModule)
    nn.Module.__init__(module)
    object.__setattr__(module, "it_cfg", SimpleNamespace(circuit_tracer_cfg=CircuitTracerConfig(**cfg)))
    object.__setattr__(module, "_replacement_model", replacement_model)
    if hf_model is not None:
        module._modules["hf_model"] = hf_model
    module.register_tunable_modules()
    return module


def _toy_module(**cfg) -> _CTModule:
    rm = tiny_gemma2_replacement_model()
    return _ct_module(rm, hf_model=rm._model, **cfg)


def _uniform_(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.uniform_(p, -1, 1)
    return module


def _eager_and_lazy_slt(path: str) -> tuple[SingleLayerTranscoder, SingleLayerTranscoder]:
    """A random eager transcoder, and the same transcoder saved to ``path`` and loaded with both weights lazy."""
    eager = _uniform_(SingleLayerTranscoder(D_MODEL, D_TC, JumpReLU(torch.tensor(0.0), 0.1), 0, dtype=torch.float32))
    eager.to_safetensors(path)
    lazy = SingleLayerTranscoder(
        D_MODEL,
        D_TC,
        JumpReLU(torch.tensor(0.0), 0.1),
        0,
        transcoder_path=path,
        lazy_encoder=True,
        lazy_decoder=True,
        dtype=torch.float32,
    )
    lazy.load_state_dict({k: v for k, v in eager.state_dict().items() if k not in ("W_enc", "W_dec")}, strict=False)
    return eager, lazy


def _eager_and_lazy_clt(path: str, n_layers: int = 3) -> tuple[CrossLayerTranscoder, CrossLayerTranscoder]:
    eager = _uniform_(
        CrossLayerTranscoder(n_layers, D_TC, D_MODEL, lazy_decoder=False, lazy_encoder=False, dtype=torch.float32)
    )
    os.makedirs(path)
    eager.to_safetensors(path)
    lazy = CrossLayerTranscoder(
        n_layers, D_TC, D_MODEL, lazy_decoder=True, lazy_encoder=True, dtype=torch.float32, clt_path=path
    )
    lazy.load_state_dict(
        {k: v for k, v in eager.state_dict().items() if not k.startswith(("W_enc", "W_dec"))}, strict=False
    )
    return eager, lazy


def _promoted_view(module: nn.Module, layer: int = 2):
    """``module``'s view with one transcoder layer promoted to ``latent:transcoder``, as #709's adaptation
    would."""
    view = as_view(module.build_tunable_view())
    view.promote([n for n in view.roles() if f".transcoders.{layer}." in n], "latent:transcoder")
    return view


def _randomize_corrections(module: nn.Module) -> None:
    """Move every correction off zero (its initialization), so a read that skipped it would be detected."""
    with torch.no_grad():
        for name, p in module.named_parameters():
            if name.endswith(".B"):
                p.normal_()


class TestTranscoderView:
    def test_transcoders_are_instruments_and_the_base_model_is_refused(self):
        module = _toy_module()
        view = as_view(module.build_tunable_view())
        assert isinstance(view.view, CircuitTracerView) and view.support().trainable
        transcoder_names = [n for n, _ in view.logical_named_parameters()]
        assert transcoder_names and all(".transcoders." in n for n in transcoder_names)
        assert set(view.roles().values()) == {"instrument"} and set(view.roles()) == set(transcoder_names)
        base = next(n for n, _ in module.named_parameters() if n not in set(transcoder_names))
        with pytest.raises(Unresolvable, match="base-model parameters"):
            view.to_physical([base])
        assert view.to_physical(transcoder_names[:1]) == {transcoder_names[0]: (transcoder_names[0],)}

    def test_promotion_reaches_transcoders_only(self):
        view = as_view(_toy_module().build_tunable_view())
        name = next(n for n, _ in view.logical_named_parameters() if n.endswith("W_dec"))
        view.promote([name], "latent:transcoder")
        assert view.roles()[name] == "latent:transcoder"
        base = next(n for n, _ in view.root.named_parameters() if n not in view.roles())
        with pytest.raises(Unresolvable, match="base-model parameters"):
            view.promote([base], "latent:transcoder")

    def test_depth_groups_are_whole_layers_from_the_last(self):
        view = as_view(_toy_module().build_tunable_view())
        groups = view.depth_groups()
        assert len(groups) == 3
        assert all(f".transcoders.{layer}." in n for layer, g in zip((2, 1, 0), groups) for n in g)

    def test_transcoders_are_registered_once(self):
        """With no HF module registered (circuit-tracer builds its model from its own config), the transcoders are
        registered as `transcoders`; with one, they are already reachable through it and are not registered
        again."""
        rm = tiny_gemma2_replacement_model()
        bare = _ct_module(rm)
        assert bare._modules["transcoders"] is rm.transcoders._module  # nnsight wraps the transcoders in an envoy
        with_hf = _ct_module(rm, hf_model=rm._model)
        assert "transcoders" not in with_hf._modules
        keys = list(with_hf.state_dict())
        assert len(keys) == len(set(keys)) and any(k.startswith("hf_model.transcoders.") for k in keys)

    def test_nothing_trains_until_promoted(self):
        """Every transcoder tensor is `instrument`, so the default schedule is empty until a promotion."""
        from finetuning_scheduler.core.compile import view_default_schedule

        module = _toy_module()
        assert view_default_schedule(module.build_tunable_view()) == {}
        assert list(view_default_schedule(_promoted_view(module)).values()) == [
            {"params": [n for n, _ in module.named_parameters() if ".transcoders.2." in n]}
        ]

    def test_conformance(self):
        """Finetuning-scheduler's cases on the view as #709 uses it, with one layer promoted."""
        report = TunableViewConformance(lambda: _promoted_view(_toy_module())).run()
        assert report.ok, report.failures
        assert {"step_changes_only_thawed", "checkpoint_round_trip", "frozen_roles_refuse", "promotion"} <= set(
            report.ran
        )

    def test_disk_offload_refuses_by_name(self):
        support = as_view(_toy_module(offload="disk").build_tunable_view()).support()
        assert not support.trainable and support.reason == DISK_OFFLOAD_REASON

    def test_an_unknown_transcoder_layout_is_refused_by_name(self):
        with pytest.raises(TypeError, match="not a transcoder layout this view knows"):
            CircuitTracerView(nn.Linear(2, 2), nn.Linear(2, 2))

    def test_base_reason_names_the_opt_in(self):
        assert "tune_base=True" in BASE_MODEL_REASON


def _base_names(module: nn.Module) -> list[str]:
    return [n for n, _ in module.named_parameters() if ".transcoders." not in n]


class TestBaseTuning:
    """``tune_base``: the base model's parameters as ``base``, beside transcoders that stay ``instrument``."""

    def test_base_parameters_resolve_as_base(self):
        module = _toy_module(tune_base=True)
        view = as_view(module.build_tunable_view())
        base = _base_names(module)
        assert base and set(base) <= {n for n, _ in view.logical_named_parameters()}
        assert set(view.roles().values()) == {"instrument"} and not set(base) & set(view.roles())
        assert view.to_physical(base[:2]) == {n: (n,) for n in base[:2]}
        assert "base" in view.support().declared_roles

    def test_base_parameters_are_not_promoted(self):
        view = as_view(_toy_module(tune_base=True).build_tunable_view())
        with pytest.raises(Unresolvable) as excinfo:
            view.promote(_base_names(view.root)[:1], "latent:transcoder")
        assert PROMOTE_BASE_REASON in str(excinfo.value)

    def test_default_schedule_thaws_base_pairs_from_the_last(self):
        """The base model in pairs from its last parameter; the transcoder groups follow, empty until a
        promotion."""
        from finetuning_scheduler.core.compile import view_default_schedule

        module = _toy_module(tune_base=True)
        base = _base_names(module)
        phases = [phase["params"] for phase in view_default_schedule(module.build_tunable_view()).values()]
        assert phases[0] == base[::-1][:2]
        assert sorted(n for phase in phases for n in phase) == sorted(base)

    def test_conformance(self):
        """Finetuning-scheduler's cases with the base model tunable and one transcoder layer promoted."""
        report = TunableViewConformance(lambda: _promoted_view(_toy_module(tune_base=True))).run()
        assert report.ok, report.failures
        assert {"step_changes_only_thawed", "checkpoint_round_trip", "frozen_roles_refuse"} <= set(report.ran)

    def test_an_unreachable_base_is_refused_by_name(self):
        """With no HF module registered, the nnsight backend's base model is not this module's: the opt-in would
        tune nothing of it, so the composition is refused rather than tuning its transcoders alone (#859)."""
        support = as_view(_ct_module(tiny_gemma2_replacement_model(), tune_base=True).build_tunable_view()).support()
        assert not support.trainable and support.reason == BASE_UNREACHABLE_REASON


class TestDefaultOptimizer:
    """The default optimizer over a composition with no HF module registered (the circuit-tracer notebooks'), whose
    parameters are all transcoder ``instrument`` tensors until a promotion (#859)."""

    @staticmethod
    def _bare_module_with_view(layer: int | None = None) -> _CTModule:
        module = _ct_module(tiny_gemma2_replacement_model())
        module.it_cfg.optimizer_init = {"class_path": "torch.optim.AdamW", "init_args": {"lr": 1e-3}}
        module.it_cfg.lr_scheduler_init = {
            "class_path": "torch.optim.lr_scheduler.StepLR",
            "init_args": {"step_size": 1},
        }
        module.it_cfg.pl_lrs_cfg = {}
        view = as_view(module.build_tunable_view())
        if layer is not None:
            view.promote([n for n in view.roles() if f".transcoders.{layer}." in n], "latent:transcoder")
        object.__setattr__(module, VIEW_ATTR, view)
        return module

    def test_nothing_to_optimize_configures_no_optimizer(self):
        module = self._bare_module_with_view()
        assert all(".transcoders." in n for n, _ in module.named_parameters())
        with pytest.warns(UserWarning, match="nothing to optimize"):
            assert module.configure_optimizers() is None

    def test_promoted_transcoders_are_optimized(self):
        module = self._bare_module_with_view(layer=2)
        optimizers, schedulers = module.configure_optimizers()
        promoted = {id(p) for n, p in module.named_parameters() if ".transcoders.2." in n}
        assert promoted and {id(p) for g in optimizers[0].param_groups for p in g["params"]} == promoted
        assert len(schedulers) == 1


class TestAttributionAfterThawing:
    """Circuit-tracer freezes the replacement model at construction and attributes with ``backward()``, so what a
    schedule thawed must not reach attribution."""

    @staticmethod
    def _thawed(embedding: bool):
        model = tiny_gemma2_replacement_model()
        tc = {id(p) for p in model.transcoders.parameters()}
        thawed = [p for n, p in model.named_parameters() if id(p) not in tc and (embedding or "embed" not in n)]
        for p in thawed:
            p.requires_grad_(True)
        return model, thawed

    @staticmethod
    def _attribute(model):
        from circuit_tracer import attribute

        with only_token_zero_special(model):
            return attribute(TOY_PROMPT, model, max_n_logits=3, batch_size=8, verbose=False)

    def test_unguarded_attribution_is_changed_by_thawed_base_parameters(self):
        """The positive control: without the guard a thawed embedding fails attribution, and other thawed base
        parameters accumulate its gradients."""
        model, _ = self._thawed(embedding=True)
        with pytest.raises(RuntimeError, match="leaf variables"):
            self._attribute(model)
        model, thawed = self._thawed(embedding=False)
        self._attribute(model)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in thawed)

    def test_guarded_attribution_leaves_thawed_parameters_untouched(self):
        model, thawed = self._thawed(embedding=True)
        with attribution_frozen(model):
            graph = self._attribute(model)
        assert graph.adjacency_matrix.numel() > 0
        assert all(p.grad is None for p in thawed) and all(p.requires_grad for p in thawed)

    def test_the_module_attributes_under_the_guard(self):
        module = _toy_module(tune_base=True)
        module.attribution_graphs = []
        for p in module.parameters():
            p.requires_grad_(True)
        with only_token_zero_special(module.replacement_model), pytest.warns(UserWarning, match="consider re-fitting"):
            module.generate_attribution_graph(TOY_PROMPT, max_n_logits=3, batch_size=8, verbose=False)
        assert all(p.grad is None and p.requires_grad for p in module.parameters())

    def test_the_refit_suggestion_needs_a_thawed_base(self, recwarn):
        """Thawed transcoders alone (the latent path) do not move what the transcoders reconstruct."""
        module = _toy_module(tune_base=True)
        module.attribution_graphs = []
        for n, p in module.named_parameters():
            p.requires_grad_(".transcoders." in n)
        with only_token_zero_special(module.replacement_model):
            module.generate_attribution_graph(TOY_PROMPT, max_n_logits=3, batch_size=8, verbose=False)
        assert not [w for w in recwarn if "consider re-fitting" in str(w.message)]


class TestLazyCorrections:
    def test_single_layer_reads_go_through_the_correction(self, tmp_path):
        eager, lazy = _eager_and_lazy_slt(str(tmp_path / "layer_0.safetensors"))
        transcoders = TranscoderSet({0: lazy}, feature_input_hook="hook_resid_mid", feature_output_hook="hook_mlp_out")
        assert lazy_weights(transcoders) == ["transcoders.0.W_enc", "transcoders.0.W_dec"]
        attach_lazy_deltas(transcoders, rank=2)
        assert type(lazy) is SingleLayerTranscoderWithDeltas
        x = torch.randn(5, D_MODEL)
        torch.testing.assert_close(lazy(x), eager(x))  # zero at initialization
        _randomize_corrections(lazy)
        d_dec = lazy.lazy_deltas["W_dec"].A @ lazy.lazy_deltas["W_dec"].B
        d_enc = lazy.lazy_deltas["W_enc"].A @ lazy.lazy_deltas["W_enc"].B
        torch.testing.assert_close(lazy.W_dec, eager.W_dec + d_dec)
        torch.testing.assert_close(lazy.W_enc, eager.W_enc + d_enc)
        rows = torch.tensor([3, 7, 7, 30])
        torch.testing.assert_close(lazy._get_decoder_vectors(rows), (eager.W_dec + d_dec)[rows])
        # the sparse decode attribution uses reads only the active features' corrected rows
        with torch.no_grad():
            sparse, _ = lazy.encode_sparse(x, zero_positions=slice(0, 0))
            reconstruction, _ = lazy.decode_sparse(sparse, x)
            expected = lazy.encode(x) @ (eager.W_dec + d_dec) + eager.b_dec
        torch.testing.assert_close(reconstruction, expected, rtol=1e-5, atol=1e-5)
        lazy(x).sum().backward()
        assert all(p.grad is not None for p in lazy.lazy_deltas.parameters())

    def test_cross_layer_reads_go_through_the_correction(self, tmp_path):
        eager, lazy = _eager_and_lazy_clt(str(tmp_path / "clt"))
        attach_lazy_deltas(lazy, rank=2)
        assert type(lazy) is CrossLayerTranscoderWithDeltas
        x = torch.randn(3, 4, D_MODEL)
        torch.testing.assert_close(lazy(x), eager(x))
        _randomize_corrections(lazy)
        d_enc = torch.stack([d.A @ d.B for d in lazy.lazy_deltas["W_enc"]])
        torch.testing.assert_close(lazy._get_encoder_weights(), eager.W_enc + d_enc)
        torch.testing.assert_close(lazy._get_encoder_weights(2), (eager.W_enc + d_enc)[2])
        rows = torch.tensor([0, 5, 31])
        for layer in range(3):
            delta = lazy.lazy_deltas["W_dec"][layer]
            d_dec = (delta.A @ delta.B).reshape(eager.W_dec[layer].shape)
            torch.testing.assert_close(lazy._get_decoder_vectors(layer, rows), (eager.W_dec[layer] + d_dec)[rows])

    def test_attaching_twice_keeps_the_first_corrections(self, tmp_path):
        _, lazy = _eager_and_lazy_slt(str(tmp_path / "layer_0.safetensors"))
        transcoders = TranscoderSet({0: lazy}, feature_input_hook="hook_resid_mid", feature_output_hook="hook_mlp_out")
        attach_lazy_deltas(transcoders, rank=2)
        first = lazy.lazy_deltas["W_dec"]
        attach_lazy_deltas(transcoders, rank=4)
        assert lazy.lazy_deltas["W_dec"] is first

    def test_a_transcoder_subclass_is_refused_by_name(self, tmp_path):
        class Custom(SingleLayerTranscoder):
            pass

        _, lazy = _eager_and_lazy_slt(str(tmp_path / "layer_0.safetensors"))
        lazy.__class__ = Custom
        transcoders = TranscoderSet({0: lazy}, feature_input_hook="hook_resid_mid", feature_output_hook="hook_mlp_out")
        with pytest.raises(TypeError, match="subclasses SingleLayerTranscoder"):
            attach_lazy_deltas(transcoders, rank=2)

    def test_the_composition_exposes_corrections_through_its_view(self, tmp_path):
        _, lazy = _eager_and_lazy_slt(str(tmp_path / "layer_0.safetensors"))
        transcoders = TranscoderSet({0: lazy}, feature_input_hook="hook_resid_mid", feature_output_hook="hook_mlp_out")
        module = _ct_module(SimpleNamespace(transcoders=transcoders), lazy_delta_rank=2)
        view = as_view(module.build_tunable_view())
        names = {n for n, _ in view.logical_named_parameters()}
        assert {
            "transcoders.transcoders.0.lazy_deltas.W_dec.A",
            "transcoders.transcoders.0.lazy_deltas.W_dec.B",
        } <= names
        report = TunableViewConformance(lambda: _promoted_view(module, layer=0)).run()
        assert report.ok, report.failures

    def test_without_a_rank_lazy_weights_are_left_out_with_a_warning(self, tmp_path):
        _, lazy = _eager_and_lazy_slt(str(tmp_path / "layer_0.safetensors"))
        transcoders = TranscoderSet({0: lazy}, feature_input_hook="hook_resid_mid", feature_output_hook="hook_mlp_out")
        module = _ct_module(SimpleNamespace(transcoders=transcoders))
        with pytest.warns(UserWarning, match="lazy_delta_rank"):
            view = as_view(module.build_tunable_view())
        assert type(lazy) is SingleLayerTranscoder
        assert not any("W_dec" in n or "W_enc" in n for n, _ in view.logical_named_parameters())


class TestTranscoderFit:
    @staticmethod
    def _set_and_sample(n_layers: int = 3, n_pos: int = 40):
        torch.manual_seed(7)
        transcoders = TranscoderSet(
            {
                i: _uniform_(
                    SingleLayerTranscoder(D_MODEL, D_TC, JumpReLU(torch.tensor(0.5), 0.1), i, dtype=torch.float32)
                )
                for i in range(n_layers)
            },
            feature_input_hook="hook_resid_mid",
            feature_output_hook="hook_mlp_out",
        )
        x = torch.randn(n_layers, n_pos, D_MODEL)
        y = torch.randn(n_layers, n_pos, D_MODEL)
        return transcoders, x, y

    def test_matches_a_direct_computation_and_is_chunk_invariant(self):
        from interpretune.adapters.circuit_tracer.drift import TranscoderFitMeter

        transcoders, x, y = self._set_and_sample()
        whole = TranscoderFitMeter(transcoders, dense_threshold=0.3)
        whole.update(x, y)
        chunked = TranscoderFitMeter(transcoders, dense_threshold=0.3)
        for xs, ys in zip(x.split(13, dim=1), y.split(13, dim=1)):
            chunked.update(xs, ys)
        with torch.no_grad():
            acts = transcoders.encode(x)
            recon = transcoders.decode(acts, x)
        for fit, other in zip(whole.compute(), chunked.compute()):
            layer = fit.layer
            fvu = ((y[layer] - recon[layer]) ** 2).sum() / ((y[layer] - y[layer].mean(0)) ** 2).sum()
            frequency = (acts[layer] != 0).float().mean(0)
            assert fit.fvu == pytest.approx(float(fvu), rel=1e-5) and other.fvu == pytest.approx(fit.fvu, rel=1e-6)
            assert fit.l0 == pytest.approx(float((acts[layer] != 0).sum(-1).float().mean()))
            assert fit.dead == int((frequency == 0).sum()) == other.dead
            assert fit.dense == int((frequency > 0.3).sum()) == other.dense
            assert fit.positions == other.positions == x.shape[1]

    def test_a_perfect_reconstruction_has_no_unexplained_variance(self):
        from interpretune.adapters.circuit_tracer.drift import TranscoderFitMeter

        transcoders, x, _ = self._set_and_sample()
        with torch.no_grad():
            y = transcoders.decode(transcoders.encode(x), x)
        meter = TranscoderFitMeter(transcoders)
        meter.update(x, y)
        assert all(fit.fvu == pytest.approx(0.0, abs=1e-10) for fit in meter.compute())

    def test_refuses_an_empty_meter_and_mismatched_shapes(self):
        from interpretune.adapters.circuit_tracer.drift import TranscoderFitMeter

        transcoders, x, y = self._set_and_sample()
        meter = TranscoderFitMeter(transcoders)
        with pytest.raises(ValueError, match="No positions were added"):
            meter.compute()
        with pytest.raises(ValueError, match="share one"):
            meter.update(x, y[:, :5])
