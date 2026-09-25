"""Core as the first consumer of the conformance suite: the bundled compositions that attach a model backend.

This is where the contract is proven before any hub repository sees it. Each class is one target; the
suite selects cases from the live declarations, so a bundled backend that stops declaring something turns
its own row red here rather than only in a downstream repository.
"""

from __future__ import annotations

import pytest

from interpretune.analysis.backends import ModelBackendCapability
from interpretune.testing.conformance import ConformanceTarget, ModelBackendConformance, OpCollectionConformance
from interpretune.testing.conformance.gates import conformance_case
from interpretune.testing.conformance.inputs import CAPTURE_LAYER, ConformanceInputs
from interpretune.utils.import_utils import package_available
from tests.runif import RunIf

pytest_plugins = ["interpretune.testing.conformance.plugin"]


class TestBridgeConformance(ModelBackendConformance):
    """TransformerBridge over gpt2: executes the HF forward in place, so the value cases apply."""

    target = ConformanceTarget(
        composition=("core", "sae_lens"),
        forward_family="hf_native",
        datamodule_flavour="bridge",
        # a declared field, so the composition-survival case fires on an in-tree target rather than only downstream
        module_cfg_extras={"experiment_tag": "conformance-bridge"},
    )


@pytest.mark.skipif(not package_available("nnsight"), reason="nnsight is not installed")
class TestNNsightConformance(ModelBackendConformance):
    target = ConformanceTarget(
        composition=("core", "nnsight", "sae_lens"), forward_family="hf_native", datamodule_flavour="nnsight"
    )


class TestConceptCollectionOnBridge(OpCollectionConformance):
    """The bundled `concept` family, validated as a collection against the bridge target."""

    target = ConformanceTarget(
        composition=("core", "sae_lens"), forward_family="hf_native", datamodule_flavour="bridge"
    )
    collection = "concept"


GEMMA3_ATTRIBUTION_PROMPT = "<bos><start_of_turn>user\nThe National Digital Analytics Group (ND"


def _gemma3_circuit_tracer(inputs):
    """The seed's gemma-3-1b-it circuit-tracer config (nnsight backend), with the intervention settings the
    analysis-backend cases assume: value from the top feature's activation, sign-aware scale, every layer
    constrained, no activation function on the intervened value."""

    def _settings(_dm_cfg, it_cfg):
        ct = it_cfg.circuit_tracer_cfg
        ct.analysis_target_tokens = None
        ct.target_token_ids = None
        ct.max_feature_nodes = None
        ct.offload = None
        ct.intervention_value_source = "top_feature_activation_values"
        ct.intervention_scale_factor = inputs.attribution_scale_factor
        ct.intervention_max_influence_norm_scale = False
        ct.intervention_sign_aware_scale = True
        ct.intervention_apply_activation_function = False
        ct.intervention_freeze_attention = None
        ct.intervention_sparse = False

    return inputs.session_cfg(("core", "nnsight", "circuit_tracer"), flavour="circuit_tracer", prepare=_settings)


@RunIf(min_cuda_gpus=1)
@pytest.mark.usefixtures("unpatched_gemma_eager_attention")
class TestCircuitTracerConformance(ModelBackendConformance):
    """circuit-tracer over the nnsight backend on gemma-3-1b-it: the analysis-backend gates' first consumer.

    Marked at the class: its cases are inherited, so the class is the only place the mark can go, and the phase
    selector reads class-level marks for exactly this reason. The unpatched-attention fixture is class-scoped for
    the same reason: circuit-tracer resolves its attention locations through nnsight's source tracing, which a
    TransformerLens bridge built earlier in the session breaks for every gemma model in the process (see the
    fixture). The model backend cases run too, on a model the suite carries no latent model for, so the latent
    cases skip with that reason.
    """

    target = ConformanceTarget(
        composition=("core", "nnsight", "circuit_tracer"),
        session_cfg_factory=_gemma3_circuit_tracer,
        forward_family="hf_native",
        datamodule_flavour="circuit_tracer",
    )
    inputs = ConformanceInputs(
        model_id="google/gemma-3-1b-it",
        device_type="cuda",
        attribution_prompt=GEMMA3_ATTRIBUTION_PROMPT,
    )


def _compatibility_mode_bridge(inputs):
    """The bridge seed's session config with HookedTransformer-equivalent weight processing turned on.

    This target used to select the weight-converting ``HookedTransformer`` via ``tl_cfg.use_bridge=False``.
    TransformerLens 4.0 removes that class, and ``enable_compatibility_mode()`` is what reproduces its default
    processing (LayerNorm folding, ``center_writing_weights``, ``center_unembed``), which upstream verifies
    against frozen ``HookedTransformer`` reference activations. So the distinction this target exists to cover
    survives the removal: raw bridge weights versus HookedTransformer-equivalent ones.
    """

    def _compat(_dm_cfg, it_cfg):
        it_cfg.tl_cfg.enable_compatibility_mode = True

    return inputs.session_cfg(("core", "sae_lens"), flavour="bridge", prepare=_compat)


class TestCompatibilityModeConformance(ModelBackendConformance):
    """A bridge carrying HookedTransformer-equivalent weight processing, rather than raw HF weights.

    Processed weights change the numbers a forward produces, so the `hf_native` reference cases do not apply
    and skip by family; the causal and structural cases run, and capture names are the same vocabulary
    spellings the raw bridge takes.
    """

    target = ConformanceTarget(
        composition=("core", "sae_lens"),
        session_cfg_factory=_compatibility_mode_bridge,
        forward_family="weight_converted",
        datamodule_flavour="bridge",
    )
    # The legacy grammar spells the sublayer outputs `hook_mlp_out` / `hook_attn_out`, and the vocabulary does not
    # alias those to `mlp.hook_out` / `attn.hook_out` on purpose: on a sandwich-norm architecture they are different
    # tensors (the post-norm output versus the sublayer's own), so an alias would be right on gpt2 and wrong on
    # Gemma. Until the resolver decides that per architecture, a HookedTransformer cannot capture those two points
    # by their component spelling; the capture case refuses by name, and this target captures what it can spell.
    # This is the capturable-subset gap the suite will let a target declare (#450).
    inputs = ConformanceInputs(
        capture_points=(
            f"blocks.{CAPTURE_LAYER}.hook_in",
            f"blocks.{CAPTURE_LAYER}.hook_out",
            f"blocks.{CAPTURE_LAYER}.ln2.hook_out",
            "unembed.hook_in",
        )
    )

    # TransformerLens 4.0 returns non-finite gradients through a compatibility-mode bridge on a masked (padded)
    # batch, at the pad positions and at real positions of the padded row, while the forward stays finite and
    # 3.5.1 is finite throughout. The inherited cases are overridden rather than marked, because a mark on the
    # shared method would reach every target. Strict, so the pass that follows an upstream fix fails loudly and
    # these overrides get removed. Reported upstream as TransformerLensOrg/TransformerLens#1809.
    _TL4_MASKED_GRADIENTS = (
        "TransformerLens 4.0: non-finite gradients through a compatibility-mode bridge on a padded, masked batch "
        "(TransformerLensOrg/TransformerLens#1809)"
    )

    @pytest.mark.xfail(strict=True, reason=_TL4_MASKED_GRADIENTS)
    @conformance_case(capability=ModelBackendCapability.GRADIENTS)
    def test_gradient_op_stores_the_declared_schema(self, suite):
        super().test_gradient_op_stores_the_declared_schema(suite)

    @pytest.mark.xfail(strict=True, reason=_TL4_MASKED_GRADIENTS)
    @conformance_case(capability=ModelBackendCapability.GRADIENTS)
    def test_gradient_predicts_a_small_perturbation_to_first_order(self, suite):
        super().test_gradient_predicts_a_small_perturbation_to_first_order(suite)

    def test_the_family_label_is_true_of_the_model(self, suite):
        """Positive control on the label: this target's weights really are processed, not raw HF weights.

        The control used to assert the model was NOT a bridge, because `weight_converted` meant the
        weight-converting `HookedTransformer` and "is a bridge" was therefore proof the label was a lie.
        TransformerLens 4.0 removes that class, so every model is a bridge and that assertion can no longer
        fail for the right reason -- it would fail always, which is not a control.

        What the label still means is unchanged: processed weights, so the `hf_native` reference numbers do
        not apply. `enable_compatibility_mode()` sets `compatibility_mode` on the bridge and its components,
        so that flag is what makes the claim checkable now. Without this, a target could silently run on raw
        weights while skipping the reference cases that would have caught it.
        """
        from transformer_lens.model_bridge import TransformerBridge

        model = suite.module.model
        assert isinstance(model, TransformerBridge), (
            f"the weight_converted target built a {type(model).__name__}, which is not a bridge at all"
        )
        assert getattr(model, "compatibility_mode", False), (
            "the weight_converted target ran on RAW bridge weights: compatibility_mode is not set, so its "
            "family cases skipped the hf_native references while asserting nothing about processed weights"
        )
