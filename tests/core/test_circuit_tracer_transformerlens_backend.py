"""Circuit-tracer's TransformerLens backend through interpretune, on a ``TransformerBridge``.

TransformerLens 4 removed the ``HookedTransformer`` circuit-tracer's TransformerLens replacement model subclassed, and
the backend was refused until circuit-tracer moved it onto a ``TransformerBridge``. These tests run it through
interpretune's own attribution op on a tiny Gemma 2 that shares its weights and transcoders with the nnsight backend,
so agreement is evidence about the TransformerLens path rather than about either model alone. CPU only.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from interpretune.adapters.circuit_tracer import BaseCircuitTracerModule
from interpretune.adapters.circuit_tracer.adapter import CircuitTracerTLModuleMixin
from interpretune.adapters.circuit_tracer.backends import DEFAULT_CT_ANALYSIS_BACKEND
from interpretune.adapters.circuit_tracer.config import CircuitTracerConfig
from interpretune.adapters.transformer_lens.backends import _build_available_hook_map
from interpretune.adapters.transformer_lens.config import ITLensBridgeConfig
from interpretune.analysis.backends.interventions import expand_intervention_patterns
from interpretune.utils import MisconfigurationException
from tests.core.circuit_tracer_toy import (
    TOY_PROMPT,
    only_token_zero_special,
    tiny_gemma2_backend_twins,
    tiny_gemma2_tl_twin,
)
from tests.core.test_circuit_tracer_attention_attribution import _ct_module, _graph_batch


@pytest.fixture(scope="module")
def twins():
    nnsight, _ = tiny_gemma2_backend_twins()
    transformerlens = tiny_gemma2_tl_twin(nnsight)
    with only_token_zero_special(nnsight), only_token_zero_special(transformerlens):
        yield nnsight, transformerlens


class TestTheTransformerLensBackend:
    def test_it_is_circuit_tracers_bridge_backend(self, twins):
        from transformer_lens.model_bridge import TransformerBridge

        _, transformerlens = twins
        assert transformerlens.backend == "transformerlens"
        assert isinstance(transformerlens, TransformerBridge)

    def test_the_twins_share_their_weights(self, twins):
        nnsight, transformerlens = twins
        with torch.no_grad():
            expected = nnsight._module(TOY_PROMPT.unsqueeze(0)).logits
            logits = transformerlens(TOY_PROMPT.unsqueeze(0))
        torch.testing.assert_close(logits, expected)

    def test_the_graph_is_the_nnsight_backends(self, twins):
        batches = [_graph_batch(_ct_module(model)) for model in twins]
        assert torch.equal(batches[1].active_features, batches[0].active_features)
        assert batches[0].adjacency_matrix.abs().max() > 1e-3, "an empty graph would match trivially"
        torch.testing.assert_close(batches[1].adjacency_matrix, batches[0].adjacency_matrix, atol=1e-4, rtol=1e-3)

    def test_a_stored_graph_reads_back_once_the_bridge_config_is_completed(self, twins):
        """A bridge leaves ``tokenizer_name`` and ``original_architecture`` unset, so a graph built on it cannot
        rebuild its config from storage until the composition fills them."""
        _, transformerlens = twins
        cfg = transformerlens.cfg
        assert cfg.tokenizer_name is None and cfg.original_architecture is None, "the control needs the bridge's own"
        with pytest.raises(KeyError):
            DEFAULT_CT_ANALYSIS_BACKEND.hydrate_graph_from_batch(_graph_batch(_ct_module(transformerlens)))
        BaseCircuitTracerModule._complete_bridge_cfg(transformerlens)
        try:
            graph = DEFAULT_CT_ANALYSIS_BACKEND.hydrate_graph_from_batch(_graph_batch(_ct_module(transformerlens)))
            assert graph.cfg.tokenizer_name == transformerlens.tokenizer.name_or_path
            assert graph.cfg.original_architecture == cfg.architecture == "Gemma2ForCausalLM"
        finally:
            cfg.tokenizer_name = cfg.original_architecture = None

    def test_the_unembedding_input_resolves_by_its_bridge_name(self, twins):
        """``unembed.hook_in`` needed a legacy alias on the HookedTransformer model; the bridge has it by name."""
        _, transformerlens = twins
        expanded = expand_intervention_patterns(["unembed.hook_in"], _build_available_hook_map(transformerlens))
        assert expanded == {"unembed.hook_in": ["unembed.hook_in"]}

    def test_concept_work_projects_into_the_vocabulary(self, twins):
        _, transformerlens = twins
        weight = DEFAULT_CT_ANALYSIS_BACKEND.get_embedding_weight(SimpleNamespace(model=transformerlens))
        assert weight.shape == (transformerlens.cfg.d_vocab, transformerlens.cfg.d_model)
        assert torch.equal(weight, transformerlens.W_U.T)


class TestTheCompositionRefuses:
    def test_compatibility_mode(self):
        """Weight processing changes what the transcoders read, so the composition refuses it before building."""

        class _Base:
            def _convert_hf_to_bridge(self):
                raise AssertionError("the bridge was built despite the refusal")

        class _Module(CircuitTracerTLModuleMixin, _Base):
            pass

        module = _Module.__new__(_Module)
        object.__setattr__(module, "it_cfg", SimpleNamespace(tl_cfg=ITLensBridgeConfig(enable_compatibility_mode=True)))
        with pytest.raises(MisconfigurationException, match="enable_compatibility_mode: false"):
            module._convert_hf_to_bridge()

    def test_a_bridge_without_a_tokenizer_name(self):
        bridge = SimpleNamespace(
            cfg=SimpleNamespace(tokenizer_name=None, original_architecture=None, architecture="Gemma2ForCausalLM"),
            tokenizer=None,
        )
        with pytest.raises(MisconfigurationException, match=r"\['tokenizer_name'\]"):
            BaseCircuitTracerModule._complete_bridge_cfg(bridge)

    def test_a_bridge_for_another_backend(self):
        with pytest.raises(MisconfigurationException, match="only the 'transformerlens' backend"):
            BaseCircuitTracerModule._replacement_model_from_bridge(
                CircuitTracerConfig(backend="nnsight"), object(), False
            )


def test_the_replacement_model_is_made_from_a_completed_bridge_config(monkeypatch):
    """The composition's own path fills the config before circuit-tracer copies it into its model."""
    import circuit_tracer.utils.hf_utils as hf_utils
    from circuit_tracer.replacement_model.replacement_model_transformerlens import TransformerLensReplacementModel

    monkeypatch.setattr(hf_utils, "load_transcoder_from_hub", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(TransformerLensReplacementModel, "from_bridge", classmethod(lambda cls, bridge, _: bridge))
    bridge = SimpleNamespace(
        cfg=SimpleNamespace(
            device="cpu", tokenizer_name=None, original_architecture=None, architecture="Gemma2ForCausalLM"
        ),
        tokenizer=SimpleNamespace(name_or_path="google/gemma-2-2b"),
    )
    made = BaseCircuitTracerModule._replacement_model_from_bridge(
        CircuitTracerConfig(backend="transformerlens"), bridge, False
    )
    assert (made.cfg.tokenizer_name, made.cfg.original_architecture) == ("google/gemma-2-2b", "Gemma2ForCausalLM")
