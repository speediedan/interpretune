"""The ``subject`` of an AnalysisStore: what its columns were computed on, read from the module at write time."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from interpretune.hub.subject import StoreSubject, StoreSubjectError, loaded_revision, subject_from_module


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


def _cached_repo(cache, repo="org/model", **refs):
    """A Hub cache holding ``repo`` with ``config.json`` at each commit, and each ref pointing at its commit."""
    root = cache / ("models--" + repo.replace("/", "--"))
    for ref, commit in refs.items():
        (root / "snapshots" / commit).mkdir(parents=True, exist_ok=True)
        (root / "snapshots" / commit / "config.json").write_text("{}", encoding="utf-8")
        (root / "refs").mkdir(exist_ok=True)
        (root / "refs" / ref).write_text(commit, encoding="utf-8")
    return cache


class TestLoadedRevision:
    """Transformers 5.18 and later record the commit a model was loaded from nowhere on it (#1243)."""

    MAIN, TAGGED = "a" * 40, "b" * 40

    def _loaded(self, tmp_path, name="org/model", **kwargs):
        module = _module(config=_HFConfig(name_or_path=name, commit=None))
        cache = _cached_repo(tmp_path, main=self.MAIN, v1=self.TAGGED)
        return loaded_revision(module, cache_dir=cache, **kwargs)

    def test_the_default_branch_and_a_named_revision(self, tmp_path):
        assert self._loaded(tmp_path) == self.MAIN
        assert self._loaded(tmp_path, revision="v1") == self.TAGGED
        assert self._loaded(tmp_path, revision=self.TAGGED) == self.TAGGED

    @pytest.mark.parametrize("name", ["", "org/not-cached", "not a repository/at/all"])
    def test_what_no_hub_load_left_in_the_cache_is_unknown(self, tmp_path, name):
        assert self._loaded(tmp_path, name=name) is None

    def test_a_local_directory_has_no_revision(self, tmp_path, monkeypatch):
        (tmp_path / "org" / "model").mkdir(parents=True)  # shadows the cached repository of the same name
        monkeypatch.chdir(tmp_path)
        assert self._loaded(tmp_path) is None

    def test_a_revision_the_config_states_is_kept(self, tmp_path):
        assert loaded_revision(_module(config=_HFConfig(commit="abc123")), cache_dir=tmp_path) == "abc123"

    def test_a_module_records_it_when_its_model_is_built(self):
        """What the store was computed on stays what was loaded, wherever the cache's branch moves afterwards."""
        module = _module(config=_HFConfig(commit=None))
        module._it_state = SimpleNamespace(_model_revision=self.MAIN)
        assert subject_from_module(module).model_revision == self.MAIN

    def test_a_loaded_model_states_its_commit(self):
        """A real load, so a change in what transformers records makes this fail."""
        from huggingface_hub import try_to_load_from_cache
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained("gpt2")
        commit = loaded_revision(_module(config=config))
        assert commit is not None and commit in str(try_to_load_from_cache("gpt2", "config.json"))

    @pytest.mark.parametrize("session", ["ns_gpt2", "sl_br_gpt2", "core_cust"])
    def test_a_composition_records_the_commit_of_the_model_it_loads(self, request, session):
        from copy import deepcopy

        from interpretune.hub.subject import model_identity
        from interpretune.session import ITSession

        module = ITSession(deepcopy(request.getfixturevalue(f"get_it_session_cfg__{session}"))).module
        _, commit = model_identity(module)
        if session == "core_cust":  # built from a configuration, so there is no commit to state
            assert commit is None
        else:
            assert commit is not None and len(commit) == 40
