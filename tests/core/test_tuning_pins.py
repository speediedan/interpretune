"""Address books pinned to the models they were written against (#875)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from interpretune.analysis.venngram import LatentBinding, LatentCatalog, LatentFrame, export_book, load_recipe
from interpretune.tuning.interventions import OpIntervention, attach_configured_interventions, attach_intervention
from interpretune.tuning.latents import LatentSite, attach_latent_model, detach_latent_model, latent_model_releases
from interpretune.tuning.pins import book_pins, check_book_pins, loaded_pins, pinned_book
from interpretune.tuning.sites import book_sets
from interpretune.utils import MisconfigurationException
from tests.core.test_tuning_sites import BOOK, D, Core, TinySAE
from venngram import AddressBook, PinMismatchError

REVISION = "0" * 40
SAE = {"sae_lens": "gpt2-small-res-jb", "id": "blocks.0.hook_resid_pre"}
BSF = {"hub": "org/bsf-l17", "revision": "9ab1f0e"}
SITE = LatentSite("point:hook_resid_post & layers:0")


def _root(*, model_id="gpt2", revision=REVISION, release=SAE) -> Core:
    """A core composition stating its base model, with one latent model attached from ``release``."""
    root = Core()
    root.it_cfg = SimpleNamespace(model_name_or_path=model_id)
    root.model.config._commit_hash = revision
    attach_latent_model(root, "sae", TinySAE(), SITE, release=release)
    return root


def _catalog(*versions: str) -> LatentCatalog:
    """A catalog declaring the frame ``weekday`` of the latent model ``bsf``'s first block in each version."""
    decoder = torch.eye(4, dtype=torch.float64)[:2]
    bsf = LatentBinding("bsf", 0, "hook_resid_post", nn.Identity(), decoder, 2)
    (target,) = LatentCatalog([bsf]).resolve("features:bsf & units:[0]", n_layers=1).targets
    return LatentCatalog([bsf], frames=[LatentFrame("weekday", v, target, decoder) for v in versions])


def _pinned(root, *args, **kwargs) -> AddressBook:
    return AddressBook.from_mapping(pinned_book(BOOK, root, *args, **kwargs))


class TestReading:
    def test_a_module_reports_its_base_model_and_its_latent_models_sources(self):
        root = _root()
        attach_latent_model(root, "unsourced", TinySAE(), SITE)
        assert loaded_pins(root) == {
            "base_model": {"id": "gpt2", "revision": REVISION},
            "latent_models": {
                "sae": {"release": SAE, "frames": {}},
                "unsourced": {"release": {}, "frames": {}},
            },
        }

    def test_a_detached_latent_model_is_not_reported(self):
        root = _root()
        detach_latent_model(root, "sae")
        assert latent_model_releases(root) == {} and loaded_pins(root)["latent_models"] == {}

    def test_a_catalogs_latent_models_and_frames_are_reported(self):
        loaded = loaded_pins(_root(), catalog=_catalog("3"), releases={"bsf": BSF})
        assert loaded["latent_models"]["bsf"] == {"release": BSF, "frames": {"weekday": "3"}}

    def test_a_frame_in_two_versions_is_refused(self):
        with pytest.raises(MisconfigurationException, match=r"frame 'weekday' in versions \['2', '3'\]"):
            loaded_pins(_root(), catalog=_catalog("3", "2"))

    def test_a_release_contradicting_the_modules_is_refused(self):
        with pytest.raises(MisconfigurationException, match="latent model 'sae' was loaded from"):
            loaded_pins(_root(), releases={"sae": BSF})
        assert loaded_pins(_root(release=None), releases={"sae": BSF})["latent_models"]["sae"]["release"] == BSF


class TestWriting:
    def test_a_pinned_book_carries_what_it_was_written_against(self):
        document = pinned_book(BOOK, _root())
        assert document["requires"] == {
            "base_model": {"id": "gpt2", "revision": REVISION},
            "latent_models": {"sae": {"release": SAE}},
        }
        assert document["schema_version"] == 8 and BOOK["schema_version"] == 1  # the source document is not edited
        assert AddressBook.from_mapping(yaml.safe_load(yaml.safe_dump(document))).pins() == document["requires"]

    def test_the_latent_models_a_book_names_can_be_chosen(self):
        root = _root()
        attach_latent_model(root, "other", TinySAE(), SITE)  # states no release, and the book does not name it
        assert set(book_pins(root, ["sae"])["latent_models"]) == {"sae"}
        assert book_pins(root, []) == {"base_model": {"id": "gpt2", "revision": REVISION}}

    def test_frames_are_pinned_with_their_latent_model(self):
        pins = book_pins(_root(), ["bsf"], catalog=_catalog("3"), releases={"bsf": BSF})
        assert pins["latent_models"] == {"bsf": {"release": BSF, "frames": {"weekday": "3"}}}

    def test_other_requirements_and_newer_schemas_are_kept(self):
        document = {**BOOK, "schema_version": 11, "requires": {"naming_scheme": "physical"}}
        pinned = pinned_book(document, _root())
        assert pinned["schema_version"] == 11 and pinned["requires"]["naming_scheme"] == "physical"

    @pytest.mark.parametrize(
        ("kwargs", "names", "match"),
        [
            ({"revision": None}, None, "does not state its base model's revision"),
            ({"model_id": None, "revision": None}, None, "does not state its base model's id or revision"),
            ({"release": None}, None, r"latent models \['sae'\] state no release"),
            ({}, ["absent"], r"latent models \['absent'\] are not loaded \(loaded: \['sae'\]\)"),
        ],
    )
    def test_what_cannot_be_identified_is_not_pinned(self, kwargs, names, match):
        root = _root(**kwargs)
        root.model.config._name_or_path = ""  # what a model built from a configuration states
        with pytest.raises(MisconfigurationException, match=match):
            pinned_book(BOOK, root, names)

    def test_a_book_already_pinned_is_not_pinned_again(self):
        root = _root()
        with pytest.raises(MisconfigurationException, match="already pins its models"):
            pinned_book(pinned_book(BOOK, root), root)

    def test_an_exported_book_is_pinned(self):
        root, add = _root(), {"blocks.1.hook_resid_post": {"intervention_tensor": torch.ones(()), "op": "add"}}
        kwargs = {"name": "exported", "revision": REVISION, "pins": book_pins(root)}
        document = export_book(add, model_id="gpt2", **kwargs)
        assert document["schema_version"] == 8 and document["requires"] == book_pins(root)
        assert load_recipe(document, module=root).recipe.name == "exported"
        with pytest.raises(Exception, match="base_model|base model"):
            export_book(add, model_id="another", **kwargs)
        assert "requires" not in export_book(add, name="exported", model_id="gpt2", revision=REVISION)


DIFFERENCES = [
    ({"model_id": "other/gpt2"}, ("requires.base_model",)),
    ({"revision": "1" * 40}, ("requires.base_model",)),
    ({"revision": None}, ("requires.base_model",)),
    ({"release": {**SAE, "id": "blocks.1.hook_resid_pre"}}, ("requires.latent_models.sae.release.id",)),
    (
        {"release": None},
        ("requires.latent_models.sae.release.sae_lens", "requires.latent_models.sae.release.id"),
    ),
]


class TestChecking:
    def test_the_same_models_are_accepted(self):
        book = _pinned(_root())
        check_book_pins(book, _root())
        assert book_sets(book, _root()) == book.sets()

    @pytest.mark.parametrize(("loaded", "entries"), DIFFERENCES)
    def test_each_difference_is_refused(self, loaded, entries):
        with pytest.raises(PinMismatchError) as err:
            check_book_pins(_pinned(_root()), _root(**loaded))
        assert set(err.value.entries) == set(entries)

    def test_a_latent_model_not_attached_is_refused(self):
        root = _root()
        detach_latent_model(root, "sae")
        with pytest.raises(PinMismatchError, match="which is not loaded"):
            check_book_pins(_pinned(_root()), root)

    @pytest.mark.parametrize("version", [None, "2"])
    def test_another_frame_version_is_refused(self, version):
        book = _pinned(_root(), ["bsf"], catalog=_catalog("3"), releases={"bsf": BSF})
        check_book_pins(book, _root(), catalog=_catalog("3"), releases={"bsf": BSF})
        with pytest.raises(PinMismatchError) as err:
            check_book_pins(book, _root(), catalog=version and _catalog(version), releases={"bsf": BSF})
        assert err.value.entries == ("requires.latent_models.bsf.frames.weekday",)

    def test_an_imported_books_pins_bind(self, tmp_path):
        (tmp_path / "pinned.yaml").write_text(yaml.safe_dump(pinned_book(BOOK, _root())), encoding="utf-8")
        outer = {"schema_version": 1, "name": "outer", "imports": {"p": "pinned.yaml"}, "sets": {"s": "@p.resid1"}}
        (tmp_path / "outer.yaml").write_text(yaml.safe_dump(outer), encoding="utf-8")
        assert book_sets(tmp_path / "outer.yaml", _root())
        with pytest.raises(PinMismatchError):
            book_sets(tmp_path / "outer.yaml", _root(release=BSF))

    def test_a_book_that_pins_nothing_needs_no_module(self):
        assert set(book_sets(BOOK)) == {"resid1", "mlp1_in"}

    def test_a_pinned_book_with_no_module_to_check_is_refused(self):
        with pytest.raises(ValueError, match="there is no module to check them against"):
            book_sets(pinned_book(BOOK, _root()))


class TestWhereBooksAreRead:
    """Every place interpretune reads a book checks its pins against the module it will be used on."""

    def test_a_configured_interventions_book(self, tmp_path):
        path = tmp_path / "book.yaml"
        path.write_text(yaml.safe_dump(pinned_book(BOOK, _root())), encoding="utf-8")
        cfg = {"name": "steer", "site": "@resid1", "book": str(path), "shape": D}
        attach_configured_interventions(_root(), [cfg])
        with pytest.raises(PinMismatchError):
            attach_configured_interventions(_root(release=BSF), [cfg])

    def test_the_sets_an_intervention_is_sited_by(self):
        root = _root()
        attach_intervention(root, "s", OpIntervention("add", D), "@resid1", sets=book_sets(_pinned(root), root))
        assert "s" in root.interventions

    @pytest.mark.parametrize("form", ["mapping", "path", "reference"])
    def test_a_treatment(self, tmp_path, form):
        add = {"blocks.1.hook_resid_post": {"intervention_tensor": torch.ones(()), "op": "add"}}
        document = export_book(add, name="t", model_id="gpt2", revision=REVISION, pins=book_pins(_root()))
        path = tmp_path / "book.yaml"
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        if form == "reference":
            from venngram import store

            store.publish("local", str(tmp_path), tmp_path, documents=[("address_book", "book.yaml")])
        source = {"mapping": document, "path": path, "reference": f"local:{tmp_path}:book.yaml"}[form]
        assert load_recipe(source, module=_root()).recipe.name == "t"
        with pytest.raises(PinMismatchError):
            load_recipe(source, module=_root(release=BSF))
        with pytest.raises(ValueError, match="there is no module to check them against"):
            load_recipe(source)


class TestSchedules:
    """Finetuning-scheduler reads a schedule's books and asks the module's view to check the ones that pin
    models."""

    PINS = {"base_model": {"id": "tiny", "revision": REVISION}}

    @pytest.fixture(autouse=True)
    def _scheduler(self):
        pytest.importorskip("finetuning_scheduler.core.tuning_scheduler")

    @staticmethod
    def _module(attach=True, revision=REVISION):
        from interpretune.tuning.views import attach_tunable_view, identity_view
        from tests.core import test_core_loop_training as core_loop

        module = core_loop.TinyModule()
        module.it_cfg.model_name_or_path = "tiny"
        module._it_state = SimpleNamespace(_model_revision=revision)
        if attach:
            module.build_tunable_view = lambda: identity_view(module)
            attach_tunable_view(module)
        return module

    def _train(self, module, tmp_path, book, address="@bias"):
        from functools import partial

        from interpretune.tuning.core_loop import FTSCoreCallback
        from tests.core import test_core_loop_training as core_loop

        (tmp_path / "book.yaml").write_text(yaml.safe_dump(book), encoding="utf-8")
        schedule = {"addresses": str(tmp_path / "book.yaml"), 0: {"params": [address]}}
        (tmp_path / "schedule.yaml").write_text(yaml.safe_dump(schedule), encoding="utf-8")
        fts = FTSCoreCallback(
            str(tmp_path / "schedule.yaml"), tuning_scheduler_kwargs={"optimizer": partial(torch.optim.SGD, lr=0.1)}
        )
        before = {n: p.detach().clone() for n, p in module.model.named_parameters()}
        try:
            core_loop._train(module, max_epochs=1, callbacks=[fts])
        finally:
            self.trained = sorted(n for n, p in module.model.named_parameters() if not torch.equal(p, before[n]))

    @property
    def _book(self):
        return {"schema_version": 8, "name": "tiny", "requires": self.PINS, "sets": {"bias": "leaf:bias"}}

    def test_a_book_pinned_to_the_loaded_models_runs(self, tmp_path):
        self._train(self._module(), tmp_path, self._book)
        assert self.trained == ["bias"]

    def test_a_book_pinned_to_other_models_is_refused(self, tmp_path):
        from finetuning_scheduler.core.schedule import FTSConfigurationError

        with pytest.raises(FTSConfigurationError, match="do not apply through this view") as err:
            self._train(self._module(revision="1" * 40), tmp_path, self._book)
        assert isinstance(err.value.__cause__, PinMismatchError)
        assert err.value.__cause__.entries == ("requires.base_model",)
        assert self.trained == []

    def test_an_imported_books_pins_bind(self, tmp_path):
        from finetuning_scheduler.core.schedule import FTSConfigurationError

        (tmp_path / "pinned.yaml").write_text(yaml.safe_dump(self._book), encoding="utf-8")
        outer = {"schema_version": 1, "name": "outer", "imports": {"p": "pinned.yaml"}, "sets": {"s": "@p.bias"}}
        self._train(self._module(), tmp_path, outer, "@s")
        assert self.trained == ["bias"]
        with pytest.raises(FTSConfigurationError, match="do not apply through this view"):
            self._train(self._module(revision="1" * 40), tmp_path, outer, "@s")

    def test_the_latent_models_of_the_module_are_checked(self):
        """The view answers for the module it was attached to, as it is when the schedule compiles."""
        from interpretune.tuning.views import attach_tunable_view, default_view

        root = _root()
        root.build_tunable_view = lambda: default_view(root)
        book = _pinned(root)
        attach_tunable_view(root).check_book(book)
        detach_latent_model(root, "sae")
        with pytest.raises(PinMismatchError, match="which is not loaded"):
            root.tunable_view.check_book(book)

    def test_a_view_that_checks_nothing_still_refuses_a_pinned_book(self, tmp_path):
        from finetuning_scheduler.core.schedule import FTSConfigurationError

        with pytest.raises(FTSConfigurationError, match="defines no `check_book`"):
            self._train(self._module(attach=False), tmp_path, self._book)
        assert self.trained == []

    def test_a_views_own_check_is_kept(self):
        from interpretune.tuning.views import check_books_of

        class Own:
            def check_book(self, book):
                return "own"

        class Slotted:
            __slots__ = ()

        own, slotted = Own(), Slotted()
        check_books_of(own, _root())
        check_books_of(slotted, _root())
        assert own.check_book(None) == "own" and not hasattr(slotted, "check_book")
