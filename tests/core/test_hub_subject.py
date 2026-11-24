"""The ``subject`` of an AnalysisStore: what its columns were computed on, read from the module at write time."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from interpretune.hub.subject import StoreSubject, StoreSubjectError, subject_from_module


class _HFConfig:
    """Stands in for a transformers config: the attributes the subject reads, and ``to_dict``."""

    def __init__(self, *, name_or_path="google/gemma-2-2b", architectures=("Gemma2ForCausalLM",), commit="abc123"):
        self._name_or_path = name_or_path
        self.architectures = list(architectures)
        self._commit_hash = commit

    def to_dict(self):
        return {}


def _module(*, it_cfg=None, config=None, tokenizer_id="google/gemma-2-2b"):
    it_cfg = it_cfg if it_cfg is not None else SimpleNamespace(model_name_or_path="google/gemma-2-2b")
    datamodule = SimpleNamespace(tokenizer=SimpleNamespace(name_or_path=tokenizer_id)) if tokenizer_id else None
    return SimpleNamespace(it_cfg=it_cfg, model=SimpleNamespace(config=config or _HFConfig()), datamodule=datamodule)


class TestStoreSubject:
    def test_the_envelope_form_round_trips(self):
        subject = StoreSubject(
            model_id="google/gemma-2-2b",
            model_revision="abc123",
            architecture="Gemma2ForCausalLM",
            tokenizer_id="google/gemma-2-2b",
            component_map={"architecture": "Gemma2ForCausalLM", "source": "gemma2.yaml", "schema_version": 1},
            dictionaries=({"kind": "sae_lens", "release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_16k"},),
        )
        assert StoreSubject.from_dict(subject.to_dict()) == subject

    def test_every_field_is_present_so_an_unknown_reads_as_none(self):
        record = StoreSubject(model_id="gpt2").to_dict()
        assert record == {
            "model": {"id": "gpt2", "revision": None, "n_layers": None},
            "architecture": None,
            "tokenizer": {"id": None},
            "component_map": None,
            "dictionaries": [],
        }

    @pytest.mark.parametrize("model_id", ["", None])
    def test_a_subject_needs_a_model_id(self, model_id):
        with pytest.raises(StoreSubjectError, match="needs a model id"):
            StoreSubject(model_id=model_id)  # type: ignore[arg-type]

    @pytest.mark.parametrize("record", [None, {}, {"model": {}}, {"model": "gpt2"}])
    def test_a_record_without_a_model_is_refused_by_name(self, record):
        with pytest.raises(StoreSubjectError, match="must name the model as `model.id`"):
            StoreSubject.from_dict(record, source="me/store")


class TestSubjectFromModule:
    def test_reads_model_revision_architecture_tokenizer_and_the_component_map(self):
        subject = subject_from_module(_module())
        assert subject.model_id == "google/gemma-2-2b"
        assert subject.model_revision == "abc123"
        assert subject.architecture == "Gemma2ForCausalLM"
        assert subject.tokenizer_id == "google/gemma-2-2b"
        assert subject.component_map is not None
        assert subject.component_map["architecture"] == "Gemma2ForCausalLM"

    def test_an_architecture_with_no_registered_map_states_none_rather_than_a_near_match(self):
        subject = subject_from_module(_module(config=_HFConfig(architectures=("NoSuchArchForCausalLM",))))
        assert subject.architecture == "NoSuchArchForCausalLM" and subject.component_map is None

    def test_an_adapter_model_name_takes_precedence_and_dictionaries_are_listed(self):
        it_cfg = SimpleNamespace(
            model_name_or_path="local/checkpoint",
            circuit_tracer_cfg=SimpleNamespace(model_name="google/gemma-2-2b", transcoder_set="gemma"),
            sae_cfgs=[SimpleNamespace(release="gemma-scope-2b-pt-res", sae_id="layer_12/width_16k")],
        )
        subject = subject_from_module(_module(it_cfg=it_cfg))
        assert subject.model_id == "google/gemma-2-2b"
        assert subject.dictionaries == (
            {"kind": "sae_lens", "release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_16k"},
            {"kind": "circuit_tracer", "transcoder_set": "gemma"},
        )

    def test_what_cannot_be_read_is_recorded_as_unknown(self):
        module = SimpleNamespace(it_cfg=SimpleNamespace(model_name_or_path="gpt2"), model=None, datamodule=None)
        assert subject_from_module(module) == StoreSubject(model_id="gpt2")

    def test_stand_in_objects_are_not_taken_for_identifiers(self):
        """A mock answers every attribute truthily: none of that may reach a published record."""
        module = MagicMock()
        module.it_cfg.model_name_or_path = "gpt2"
        module.it_cfg.circuit_tracer_cfg.model_name = None
        module.it_cfg.nnsight_cfg.model_name = None
        subject = subject_from_module(module)
        assert subject.model_id == "gpt2"
        record = subject.to_dict()
        assert record["model"]["revision"] is None and record["architecture"] is None
        assert record["tokenizer"]["id"] is None and record["dictionaries"] == []

    def test_a_module_stating_no_model_id_is_refused_by_name(self):
        module = SimpleNamespace(it_cfg=SimpleNamespace(model_name_or_path=None), model=None, datamodule=None)
        with pytest.raises(StoreSubjectError, match="states no model id"):
            subject_from_module(module)


class TestWriteTimeStamp:
    """``AnalysisCfg.apply`` stamps the store it writes, as it stamps op provenance."""

    def test_the_store_is_stamped_with_the_modules_subject(self):
        from interpretune.config.analysis import AnalysisCfg

        store = SimpleNamespace(subject=None)
        AnalysisCfg._stamp_subject(store, _module())
        assert store.subject == subject_from_module(_module())

    def test_a_module_with_no_model_id_leaves_the_store_without_one_and_says_so(self, recwarn):
        from interpretune.config.analysis import AnalysisCfg

        store = SimpleNamespace(subject=None)
        module = SimpleNamespace(it_cfg=SimpleNamespace(model_name_or_path=None), model=None, datamodule=None)
        AnalysisCfg._stamp_subject(store, module)
        assert store.subject is None
        assert any("subject was not recorded" in str(w.message) for w in recwarn.list)
