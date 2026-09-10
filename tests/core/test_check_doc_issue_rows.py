"""The documentation issue-state checker, in its three states, with the GitHub call replaced by a chosen mapping.

The checker keeps `docs/jlens_paper_alignment.md` honest from the weekly link-check workflow, online by construction.
These tests pin what it does with each answer GitHub can give, offline, so an edit that breaks the closed-issue
detection fails here rather than leaving the page silently unchecked while the workflow keeps passing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_doc_issue_rows.py"


@pytest.fixture(scope="module")
def checker():
    spec = importlib.util.spec_from_file_location("check_doc_issue_rows", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _page(tmp_path: Path, text: str) -> Path:
    page = tmp_path / "page.md"
    page.write_text(text, encoding="utf-8")
    return page


class TestThreeStates:
    def test_a_page_citing_only_open_issues_passes(self, checker, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(checker, "issue_state", lambda repo, number, token: "open")
        page = _page(tmp_path, "| gap | Not implemented. Open: #423. |\n| gap | Open: #539 and #540. |\n")
        assert checker.main(["--repo", "owner/name", str(page)]) == 0
        out = capsys.readouterr().out
        assert "#423: open" in out and "all 3 cited issues are open" in out

    def test_a_closed_citation_fails_by_name_with_its_line(self, checker, monkeypatch, tmp_path, capsys):
        states = {423: "open", 521: "closed"}
        monkeypatch.setattr(checker, "issue_state", lambda repo, number, token: states[number])
        page = _page(tmp_path, "row one names #423\nrow two names #521 as open\n")
        assert checker.main(["--repo", "owner/name", str(page)]) == 1
        err = capsys.readouterr().err
        assert "#521 is closed" in err and f"{page}:2" in err
        assert "#423" not in err, "an open citation is not reported as a failure"

    def test_a_page_with_no_citations_does_not_pass(self, checker, monkeypatch, tmp_path, capsys):
        """The vacuity control: an empty check must not read as a passing one."""
        monkeypatch.setattr(checker, "issue_state", lambda repo, number, token: pytest.fail("no issue to look up"))
        page = _page(tmp_path, "nothing cited here\n")
        assert checker.main(["--repo", "owner/name", str(page)]) == 2
        assert "no issue citations found" in capsys.readouterr().err

    def test_a_path_fragment_is_not_a_citation(self, checker, monkeypatch, tmp_path):
        """`refs/pull/546/merge` and `sha#12` shapes must not be read as issue numbers."""
        seen: list[int] = []

        def fake(repo, number, token):
            seen.append(number)
            return "open"

        monkeypatch.setattr(checker, "issue_state", fake)
        page = _page(tmp_path, "see refs/pull/546/merge and commit abc#12 and issue #77\n")
        assert checker.main(["--repo", "owner/name", str(page)]) == 0
        assert seen == [77]
