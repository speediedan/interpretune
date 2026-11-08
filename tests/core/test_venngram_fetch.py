"""Fetching recipes by reference (#887): local paths, inline documents and store references, pinned."""

import pytest
import yaml

pytest.importorskip("venngram", reason="the venngram extra is not installed")

from interpretune.analysis.venngram import load_recipe
from interpretune.utils import MisconfigurationException

RECIPE = {
    "schema_version": 2,
    "kind": "recipe",
    "name": "dose-the-units",
    "model": {"id": "m", "revision": "r1"},
    "book": {
        "schema_version": 1,
        "name": "units",
        "portability": "checkpoint",
        "applies_to": {"models": [{"id": "m", "revision": "r1"}]},
        "sets": {"chess": {"point": "mlp.out.hook_in", "layers": [15], "units": {15: [7, 4041]}}},
    },
    "stages": [
        {
            "site": "mlp.hook_out",
            "ops": [{"op": "add", "target": {"set": "chess"}, "value": {"dose": 0.8}}],
        }
    ],
}


def test_a_mapping_and_a_recipe_pass_through():
    from venngram.recipe import Recipe

    fetched = load_recipe(RECIPE)
    assert isinstance(fetched.recipe, Recipe) and fetched.reference is None and fetched.source == "<inline>"
    assert load_recipe(fetched.recipe).recipe is fetched.recipe


def test_a_local_path_reads_the_file(tmp_path):
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(RECIPE))
    fetched = load_recipe(path)
    assert fetched.recipe.name == "dose-the-units" and fetched.reference is None and fetched.source == str(path)


def test_a_store_reference_fetches_and_pins(tmp_path):
    from venngram import store

    root = tmp_path / "published"
    root.mkdir()
    (root / "recipe.yaml").write_text(yaml.safe_dump(RECIPE))
    store.publish("local", str(root), root, documents=[("recipe", "recipe.yaml")])

    fetched = load_recipe(f"local:{root}:recipe.yaml")
    assert fetched.recipe.name == "dose-the-units"
    assert fetched.reference is not None and "#" in fetched.reference  # pinned by content
    assert store.pin(fetched.reference).__str__() == fetched.reference  # already exact


def test_a_non_recipe_document_and_a_bare_number_refuse(tmp_path):
    from venngram import store

    root = tmp_path / "published"
    root.mkdir()
    (root / "book.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "b",
                "portability": "checkpoint",
                "applies_to": {"models": [{"id": "m", "revision": "r1"}]},
                "sets": {"any": "component:mlp.out & layers:0"},
            }
        )
    )
    store.publish("local", str(root), root, documents=[("address_book", "book.yaml")])
    with pytest.raises(MisconfigurationException, match="not a recipe"):
        load_recipe(f"local:{root}:book.yaml")
    with pytest.raises(MisconfigurationException, match="must be a document reference"):
        load_recipe(42)


class TestBookTreatments:
    BOOK = {
        "schema_version": 1,
        "name": "units",
        "portability": "checkpoint",
        "applies_to": {"models": [{"id": "m", "revision": "r1"}]},
        "sets": {"chess": {"point": "mlp.hook_out", "layers": [15], "units": {15: [7]}}},
    }
    BODY = {
        "model": {"id": "m", "revision": "r1"},
        "stages": [{"site": "mlp.hook_out", "ops": [{"op": "add", "target": {"set": "chess"}, "value": {"dose": 1}}]}],
    }

    def _book(self, *names):
        return {**self.BOOK, "treatments": {name: self.BODY for name in names}}

    def test_a_book_with_one_treatment_reads_it_from_every_source(self, tmp_path):
        from venngram import store

        book = self._book("dose")
        assert load_recipe(book).recipe.name == "dose"
        root = tmp_path / "published"
        root.mkdir()
        (root / "book.yaml").write_text(yaml.safe_dump(book))
        assert load_recipe(root / "book.yaml").recipe.name == "dose"
        store.publish("local", str(root), root, documents=[("address_book", "book.yaml")])
        fetched = load_recipe(f"local:{root}:book.yaml")
        assert fetched.recipe.name == "dose" and "#" in fetched.reference

    def test_a_book_with_several_treatments_needs_a_name(self):
        book = self._book("first", "second")
        assert load_recipe(book, treatment="second").recipe.name == "second"
        with pytest.raises(MisconfigurationException, match=r"name one of the book's treatments \['first', 'second'\]"):
            load_recipe(book)
        with pytest.raises(MisconfigurationException, match="not 'third'"):
            load_recipe(book, treatment="third")
