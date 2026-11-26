"""Publisher trust and analysis-store discovery (``interpretune.hub.discovery``).

The envelopes served here are the frozen, writer-produced fixtures the reader matrix uses, so a listing is checked
against what a real push wrote rather than against a hand-built envelope. Only the schema number is changed for the too-
new case, which no writer can produce yet.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from huggingface_hub.errors import EntryNotFoundError

from interpretune.hub.artifacts import ARTIFACT_SCHEMA_VERSION, SUBJECT_REQUIRED_FROM_SCHEMA
from interpretune.hub.cards import generate_artifact_card
from interpretune.hub.discovery import (
    ANALYSIS_STORE_TAG,
    DEFAULT_TRUSTED_PUBLISHERS,
    IT_TRUSTED_PUBLISHERS_ENV_VAR,
    StoreSchemaSkippedWarning,
    UntrustedPublisherSpecError,
    find_analysis_stores,
    trusted_publishers,
)

FIXTURES = Path(__file__).parent / "fixtures"
GEMMA = "google/gemma-2-2b"


def _fixture(version: int) -> dict:
    return json.loads((FIXTURES / f"it_artifact_schema{version}.json").read_text("utf-8"))


@pytest.fixture(autouse=True)
def _default_trust(monkeypatch) -> None:
    monkeypatch.delenv(IT_TRUSTED_PUBLISHERS_ENV_VAR, raising=False)


class TestTrustedPublishers:
    def test_the_default_is_the_project_namespace(self) -> None:
        assert trusted_publishers() == DEFAULT_TRUSTED_PUBLISHERS == ("speediedan",)

    def test_the_environment_variable_is_read_at_call_time(self, monkeypatch) -> None:
        monkeypatch.setenv(IT_TRUSTED_PUBLISHERS_ENV_VAR, " acme , speediedan,acme ")
        assert trusted_publishers() == ("acme", "speediedan")

    def test_the_keyword_overrides_the_environment(self, monkeypatch) -> None:
        monkeypatch.setenv(IT_TRUSTED_PUBLISHERS_ENV_VAR, "acme")
        assert trusted_publishers(["other-org"]) == ("other-org",)
        assert trusted_publishers("solo") == ("solo",)

    @pytest.mark.parametrize("bad", ["speediedan/ct-concept-steering-demo", "", "-lead", "has space"])
    def test_anything_but_a_namespace_is_refused_by_name(self, bad) -> None:
        with pytest.raises(UntrustedPublisherSpecError, match="not a Hub namespace") as excinfo:
            trusted_publishers([bad])
        assert repr(bad) in str(excinfo.value)

    def test_an_empty_list_is_refused_rather_than_trusting_nobody_silently(self, monkeypatch) -> None:
        with pytest.raises(UntrustedPublisherSpecError, match="names no publishers"):
            trusted_publishers([])
        monkeypatch.setenv(IT_TRUSTED_PUBLISHERS_ENV_VAR, "acme,")
        with pytest.raises(UntrustedPublisherSpecError, match=r"\$IT_TRUSTED_PUBLISHERS names ''"):
            trusted_publishers()


def test_the_discovery_tag_is_the_one_every_pushed_store_carries() -> None:
    card = generate_artifact_card(_fixture(2), "ns/store")
    assert ANALYSIS_STORE_TAG in card.data.tags


def _hub(stores: dict[str, list[tuple[str, str, object]]]):
    """Serve ``{publisher: [(repo_id, sha, envelope or exception)]}`` in place of the Hub listing and downloads."""
    listed: list[tuple[str, str]] = []
    reads: list[tuple[str, str]] = []
    served = {repo: (sha, body) for rows in stores.values() for repo, sha, body in rows}

    def list_datasets(self, *, author=None, filter=None, token=None, **_):
        listed.append((author, filter))
        return [SimpleNamespace(id=repo, sha=sha, private=True) for repo, sha, _ in stores.get(author, [])]

    def read_envelope(repo_id, revision, token, cache_dir):
        reads.append((repo_id, revision))
        sha, body = served[repo_id]
        assert revision == sha
        if isinstance(body, Exception):
            raise body
        return json.loads(json.dumps(body))

    return (
        patch("huggingface_hub.HfApi.list_datasets", list_datasets),
        patch("interpretune.hub.discovery._read_envelope", read_envelope),
        listed,
        reads,
    )


class TestFindAnalysisStores:
    def test_rows_describe_each_store_at_the_commit_listed(self) -> None:
        hub, read, listed, reads = _hub(
            {"speediedan": [("speediedan/v2", "a" * 40, _fixture(2)), ("speediedan/v1", "b" * 40, _fixture(1))]}
        )
        with hub, read:
            rows = {r.repo_id: r for r in find_analysis_stores()}
        assert listed == [("speediedan", ANALYSIS_STORE_TAG)]
        assert reads == [("speediedan/v2", "a" * 40), ("speediedan/v1", "b" * 40)]
        v2, v1 = rows["speediedan/v2"], rows["speediedan/v1"]
        assert (v2.publisher, v2.revision, v2.schema, v2.private) == ("speediedan", "a" * 40, 2, True)
        assert v2.store_id == _fixture(2)["identity"]["store_id"]
        assert v2.subject is not None and v2.subject.model_id == GEMMA
        assert v2.num_rows == 2 and v2.columns == tuple(_fixture(2)["artifacts"]["columns"])
        assert (v1.schema, v1.subject) == (1, None)

    def test_a_model_filter_matches_the_subject_and_drops_stores_without_one(self) -> None:
        other = _fixture(2)
        other["subject"]["model"]["id"] = "gpt2"
        hub, read, _, _ = _hub(
            {
                "speediedan": [
                    ("speediedan/gemma", "a" * 40, _fixture(2)),
                    ("speediedan/gpt2", "c" * 40, other),
                    ("speediedan/v1", "b" * 40, _fixture(1)),
                ]
            }
        )
        with hub, read:
            assert [r.repo_id for r in find_analysis_stores(GEMMA)] == ["speediedan/gemma"]
            assert [r.repo_id for r in find_analysis_stores("gpt2")] == ["speediedan/gpt2"]

    def test_only_trusted_publishers_are_listed(self, monkeypatch) -> None:
        stores = {"speediedan": [("speediedan/a", "a" * 40, _fixture(2))], "acme": [("acme/b", "b" * 40, _fixture(2))]}
        hub, read, listed, _ = _hub(stores)
        with hub, read:
            assert [r.repo_id for r in find_analysis_stores()] == ["speediedan/a"]
            assert [r.repo_id for r in find_analysis_stores(publishers="acme")] == ["acme/b"]
            monkeypatch.setenv(IT_TRUSTED_PUBLISHERS_ENV_VAR, "acme,speediedan")
            assert [r.repo_id for r in find_analysis_stores()] == ["acme/b", "speediedan/a"]
        assert [author for author, _ in listed] == ["speediedan", "acme", "acme", "speediedan"]

    def test_an_unreadable_store_is_left_out_by_name_without_hiding_the_rest(self) -> None:
        newer = _fixture(2)
        newer["schema"] = 99
        hub, read, _, _ = _hub(
            {
                "speediedan": [
                    ("speediedan/newer", "n" * 40, newer),
                    ("speediedan/no-envelope", "m" * 40, EntryNotFoundError("it_artifact.json not found")),
                    ("speediedan/good", "a" * 40, _fixture(2)),
                ]
            }
        )
        with hub, read, pytest.warns(StoreSchemaSkippedWarning) as caught:
            rows = find_analysis_stores()
        assert [r.repo_id for r in rows] == ["speediedan/good"]
        messages = [str(w.message) for w in caught]
        assert any("'speediedan/newer'" in m and "newer Interpretune" in m for m in messages)
        assert any("'speediedan/no-envelope'" in m and "could not be read" in m for m in messages)

    def test_a_failure_to_reach_the_hub_raises_rather_than_shortening_the_listing(self) -> None:
        hub, read, _, _ = _hub({"speediedan": [("speediedan/a", "a" * 40, ConnectionError("hub unreachable"))]})
        with hub, read, pytest.raises(ConnectionError, match="hub unreachable"):
            find_analysis_stores()

    @pytest.mark.hf_live
    def test_the_published_demo_store_is_listed_when_the_token_can_read_it(self) -> None:
        """The project's one published store is private, so this needs a token that can read it."""
        import os

        token = os.environ.get("IT_HF_TOKEN") or os.environ.get("HF_TOKEN")
        if not token:
            pytest.skip("IT_HF_TOKEN or HF_TOKEN required to read the private demo store")
        rows = {r.repo_id: r for r in find_analysis_stores(GEMMA, token=token)}
        demo = rows["speediedan/ct-concept-steering-demo"]
        # the store is re-uploaded by whatever code renders its notebook, so its schema moves within the readable window
        assert SUBJECT_REQUIRED_FROM_SCHEMA <= demo.schema <= ARTIFACT_SCHEMA_VERSION
        assert demo.subject is not None and demo.subject.model_id == GEMMA
        assert len(demo.revision) == 40
