"""Full-set collection uses the real Hub sparse forward and refuses reconstructions that lose its behavior."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from it_examples.utils.rte_sparse_attribution import RteSparseAttributionCollector
from it_examples.utils.sparse_attribution_checkpoint import verify_attribution_checkpoint
from it_examples.utils.sparse_attribution_report import checkpoint_attribution_report
from tests.core.circuit_tracer_toy import tiny_gemma2_replacement_model, only_token_zero_special
from tests.core.test_circuit_tracer_attention_attribution import _ct_module

pytest.importorskip("finetuning_scheduler.core.tuning_scheduler")


@pytest.fixture
def sparse_module(request):
    from transformers import Gemma2ForCausalLM
    from tests.rte_component import rte_entrypoint_module

    initial = tiny_gemma2_replacement_model(n_layers=2)

    class Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if getattr(request, "param", "gemma2") == "gemma3":
                from transformers import Gemma3ForCausalLM, Gemma3TextConfig

                self.model = Gemma3ForCausalLM(
                    Gemma3TextConfig(
                        architectures=["Gemma3ForCausalLM"],
                        vocab_size=16,
                        hidden_size=8,
                        intermediate_size=16,
                        num_hidden_layers=2,
                        num_attention_heads=2,
                        num_key_value_heads=2,
                        head_dim=4,
                        layer_types=["full_attention", "full_attention"],
                        attn_implementation="eager",
                    )
                )
                out_site = "model.layers.{layer}.post_feedforward_layernorm"
            else:
                self.model = Gemma2ForCausalLM(initial._module.config)
                selected = initial._module.state_dict()
                self.model.load_state_dict({name: selected[name] for name in self.model.state_dict()})
                out_site = "model.layers.{layer}.mlp"
            self.it_cfg = SimpleNamespace(
                entailment_mapping_indices=torch.tensor([3, 8]),
                sparse_chain={
                    "reg_layers": [0, 1],
                    "enforce_layers": [0, 1],
                    "transcoders": {"random": {"d_transcoder": 32, "seed": 4, "threshold": 0.0}},
                    "sites": {"x": "model.layers.{layer}.pre_feedforward_layernorm", "out": out_site},
                },
            )
            self.datamodule = SimpleNamespace(
                tokenizer=initial.tokenizer,
                dataset={
                    "validation": [
                        {"idx": 17, "input_ids": [0, 0, 3, 4, 8], "attention_mask": [0, 1, 1, 1, 1]},
                        {"idx": 29, "input_ids": [0, 3, 2, 5, 3, 8], "attention_mask": [1, 1, 1, 1, 1, 1]},
                        {"idx": 41, "input_ids": [0, 9, 4, 8], "attention_mask": [1, 1, 1, 1]},
                    ]
                },
            )

        def register_tunable_modules(self):
            pass

        def forward(self, **batch):
            return self.model(**batch)

    class Sparse(rte_entrypoint_module().RTEBoolqSparseMixin, Base):
        pass

    module = Sparse()
    module.register_tunable_modules()
    return module


class RebuiltModel:
    """A CPU measurement factory copying selected weights into a separate real replacement model.

    It deliberately supplies no RTE enforcement hooks, so an enforced source must fail the logit control.
    """

    def __init__(self, defect=None):
        self.defect = defect

    @contextmanager
    def __call__(self, module, checkpoint):
        replacement = tiny_gemma2_replacement_model(n_layers=2, seed=12)
        # The measurement's basis is an independent artifact; only the selected HF weights are copied into it.
        state = replacement._module.state_dict()
        state.update(module.model.state_dict())
        replacement._module.load_state_dict(state)
        if self.defect == "weights":
            with torch.no_grad():
                replacement._module.model.norm.weight.add_(0.5)
        elif self.defect == "same-gap-other-logits":

            def shifted(_module, _inputs, output):
                output.logits[..., [3, 8]] += 1.0
                return output

            replacement._module.register_forward_hook(shifted)
        elif self.defect == "softcap":
            replacement._module.config.final_logit_softcapping = 30.0
        elif self.defect == "changed-enforcement-mode":
            module.sparse_enforce = not module.sparse_enforce
        with only_token_zero_special(replacement):
            yield _ct_module(replacement)


def _proof(tmp_path, module):
    path = tmp_path / "selected.pt"
    torch.save({"version": 1, "include": "all", "sharded": False, "model": module.state_dict()}, path)
    return verify_attribution_checkpoint(path, module.state_dict())


def _collector(factory=None):
    return RteSparseAttributionCollector(factory or RebuiltModel(), "tiny-gemma2", "test-revision")


def test_full_validation_collection_checks_actual_weights_logits_and_source_partitions(sparse_module, tmp_path):
    module = sparse_module
    module.train()
    module.model.model.layers[0].eval()
    modes = [child.training for child in module.modules()]
    retained = {"retained": torch.tensor(7)}
    mask = torch.tensor([[1, 1]])
    module._sparse_saved, module._sparse_mask = retained, mask
    proof = _proof(tmp_path, module)
    measurements = _collector()(module, proof)
    assert measurements.expected_row_ids == ["17", "29", "41"]
    assert [row.row_id for row in measurements.rows] == ["17", "29", "41"]
    assert [len(row.embedding_edges) for row in measurements.rows] == [4, 6, 4]
    assert measurements.checks["max_answer_logit_abs_error"] < 1e-5
    assert len(measurements.checks["validation_inputs_sha256"]) == 64
    assert measurements.basis_id.startswith("sha256:")
    report = checkpoint_attribution_report(
        measurements.rows, checkpoint_sha256=proof.sha256, basis_id=measurements.basis_id,
        model_id=measurements.model_id, model_revision=measurements.model_revision,
        target_token_ids=measurements.target_token_ids,
    )  # fmt: skip
    assert report["items"] == 3 and report["complete_head_layers"]
    assert module._sparse_saved is retained and module._sparse_mask is mask
    assert [child.training for child in module.modules()] == modes
    assert verify_attribution_checkpoint(proof.path, module.state_dict()) == proof


@pytest.mark.parametrize(
    "defect,match",
    [
        ("weights", "HF weights do not match"),
        ("same-gap-other-logits", "answer logits.*do not match"),
        ("softcap", "final_logit_softcapping"),
        ("changed-enforcement-mode", "changed the selected sparse enforcement mode"),
    ],
)
def test_reconstruction_defects_are_refused_before_accepting_measurements(sparse_module, tmp_path, defect, match):
    with pytest.raises(ValueError, match=match):
        _collector(RebuiltModel(defect))(sparse_module, _proof(tmp_path, sparse_module))


def test_exporting_bare_weights_does_not_reproduce_an_enforced_selected_run(sparse_module, tmp_path):
    module = sparse_module
    for layer in module.sparse_cfg["enforce_layers"]:
        getattr(module, f"sparse_enforce_keep_con_{layer}").fill_(False)
        getattr(module, f"sparse_enforce_mean_con_{layer}").fill_(0.75)
    module.sparse_enforce = True
    with pytest.raises(ValueError, match="answer logits.*preserve fitted enforcement"):
        _collector()(module, _proof(tmp_path, module))
    assert module.sparse_enforce is True


def _production_factory(module):
    from copy import deepcopy
    from circuit_tracer.transcoder import TranscoderSet
    from it_examples.utils.rte_sparse_measurement import RteSparseMeasurementFactory

    basis = TranscoderSet(
        {int(layer): deepcopy(tc) for layer, tc in module.sparse_tcs.items()},
        feature_input_hook=("mlp.hook_in" if module.model.config.model_type == "gemma3_text" else "hook_resid_mid"),
        feature_output_hook="hook_mlp_out",
    )
    return RteSparseMeasurementFactory(basis)


@pytest.mark.parametrize("enforced", [False, True])
@pytest.mark.parametrize("sparse_module", ["gemma2", "gemma3"], indirect=True)
def test_production_factory_preserves_actual_enforcement_and_public_head_partitions(
    sparse_module, tmp_path, enforced, monkeypatch
):
    module = sparse_module
    for layer in module.sparse_cfg["enforce_layers"]:
        getattr(module, f"sparse_enforce_keep_con_{layer}").fill_(False)
        getattr(module, f"sparse_enforce_mean_con_{layer}").fill_(0.75)
    module.sparse_enforce = enforced
    retained, mask = {"retained": torch.tensor(7)}, torch.tensor([[1, 1]])
    module._sparse_saved, module._sparse_mask = retained, mask
    module.train()
    module.model.model.layers[0].eval()
    modes = [child.training for child in module.modules()]
    first = next(module.model.parameters())
    first.grad = torch.ones_like(first)
    params = [(p, p.requires_grad, p.grad) for p in module.parameters()]
    checked_batches = []
    check_template = module.check_template

    def check(ids, mask):
        checked_batches.append(tuple(ids.shape))
        check_template(ids, mask)

    monkeypatch.setattr(module, "check_template", check)
    proof = _proof(tmp_path, module)
    rng = torch.random.get_rng_state()
    with only_token_zero_special(SimpleNamespace(tokenizer=module.datamodule.tokenizer)):
        measurements = _collector(_production_factory(module))(module, proof)
    assert len(measurements.rows) == 3
    assert measurements.checks["max_answer_logit_abs_error"] < 1e-5
    report = checkpoint_attribution_report(
        measurements.rows, checkpoint_sha256=proof.sha256, basis_id=measurements.basis_id,
        model_id=measurements.model_id, model_revision=measurements.model_revision,
        target_token_ids=measurements.target_token_ids,
    )  # fmt: skip
    assert report["items"] == 3 and report["complete_head_layers"]
    assert module.sparse_enforce is enforced
    if enforced:
        assert any(batch > 1 for batch, _length in checked_batches)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert module._sparse_saved is retained and module._sparse_mask is mask
    assert [child.training for child in module.modules()] == modes
    assert all(p.requires_grad is required and p.grad is grad for p, required, grad in params)
    assert verify_attribution_checkpoint(proof.path, module.state_dict()) == proof


def test_production_factory_rejects_other_enforcement_basis(sparse_module, tmp_path):
    factory = _production_factory(sparse_module)
    with torch.no_grad():
        factory.transcoder_set[0].W_dec.add_(1.0)
    with pytest.raises(ValueError, match="differs from the fitted enforcement transcoder"):
        with factory(sparse_module, _proof(tmp_path, sparse_module)):
            pytest.fail("accepted the other basis")


@pytest.mark.parametrize("defect", ["precision", "architecture"])
def test_unsupported_models_are_refused_before_loading_a_basis(sparse_module, tmp_path, monkeypatch, defect):
    from circuit_tracer.utils import hf_utils
    from it_examples.utils.rte_sparse_measurement import RteSparseMeasurementFactory

    def unexpected_load(*args, **kwargs):
        pytest.fail("loaded a basis before checking the selected model")

    monkeypatch.setattr(hf_utils, "load_transcoder_from_hub", unexpected_load)
    if defect == "precision":
        sparse_module.model.double()
    else:
        sparse_module.model.config.architectures = ["GPT2LMHeadModel"]
    with pytest.raises(ValueError, match="float32 selected model|Unsupported model architecture: GPT2LMHeadModel"):
        with RteSparseMeasurementFactory("must-not-load")(sparse_module, _proof(tmp_path, sparse_module)):
            pytest.fail("accepted an unsupported selected model")


def test_a_partial_measurement_basis_is_refused(sparse_module, tmp_path):
    factory = _production_factory(sparse_module)
    del factory.transcoder_set.transcoders[-1]
    with pytest.raises(ValueError, match="basis for every model layer"):
        with factory(sparse_module, _proof(tmp_path, sparse_module)):
            pytest.fail("accepted the incomplete basis")


def test_production_factory_restores_caches_and_removes_owned_hooks_after_failure(sparse_module, tmp_path):
    factory = _production_factory(sparse_module)
    proof = _proof(tmp_path, sparse_module)
    retained, mask = {"retained": torch.tensor(7)}, torch.tensor([[1, 1]])
    sparse_module._sparse_saved, sparse_module._sparse_mask = retained, mask
    clone = None
    with pytest.raises(RuntimeError, match="measurement failure"):
        with factory(sparse_module, proof) as subject:
            clone = subject.replacement_model._module
            sparse_module._sparse_saved["temporary"] = torch.tensor(3)
            raise RuntimeError("measurement failure")
    assert sparse_module._sparse_saved is retained and sparse_module._sparse_mask is mask
    assert not clone._forward_pre_hooks
    assert all(not child._forward_hooks for child in clone.modules())
    assert verify_attribution_checkpoint(proof.path, sparse_module.state_dict()) == proof


def test_a_checkpoint_hash_cannot_certify_other_live_weights(sparse_module, tmp_path):
    proof = _proof(tmp_path, sparse_module)
    with torch.no_grad():
        sparse_module.model.model.norm.weight.add_(0.5)
    with pytest.raises(ValueError, match="RTE module tensors do not match"):
        _collector()(sparse_module, proof)


@pytest.mark.parametrize("defect", ["duplicate-idx", "internal-padding", "float-token", "empty-mask"])
def test_invalid_validation_inputs_are_refused(sparse_module, tmp_path, defect):
    items = sparse_module.datamodule.dataset["validation"]
    if defect == "duplicate-idx":
        items[1]["idx"] = items[0]["idx"]
    elif defect == "internal-padding":
        items[0]["attention_mask"] = [0, 1, 0, 1, 1]
    elif defect == "float-token":
        items[0]["input_ids"] = [float(token) for token in items[0]["input_ids"]]
    else:
        items[0]["attention_mask"] = [0] * 5
    with pytest.raises(ValueError, match="validation"):
        _collector()(sparse_module, _proof(tmp_path, sparse_module))
