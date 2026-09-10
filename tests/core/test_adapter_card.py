"""The generated card for an `adapters` component says what a model card never has to: this runs code.

A model card describes weights, which are data. An adapter is code that executes in the caller's process and composes
into the MRO of the module their session runs. Capability is largely inferable from a manifest; **exposure is not**,
which is why the trust block is the one this card most exists for and the one these tests guard hardest.

Generated at publish, from the validated manifest, so no publish path can produce a card-less or hand-drifted adapter
repo — the same property the other kinds' cards already have.
"""

from __future__ import annotations

import pytest

from interpretune.hub.cards import ComponentCardError, generate_component_card

REPO = "speediedan/it-interp-engine-adapter"


def _manifest(compositions=None, declares=("interp_engine",), entrypoint="adapter.py"):
    return {
        "it_schema_version": 1,
        "kinds": ["adapters"],
        "adapters": {
            "declares": list(declares),
            "entrypoint": entrypoint,
            "compositions": compositions if compositions is not None else [],
        },
    }


TWO_COMPOSITIONS = [
    {"component": "module", "adapters": ["core", "interp_engine"]},
    {
        "component": "module",
        "adapters": ["core", "interp_engine", "circuit_tracer"],
        "requires": {"pip": ["circuit-tracer"]},
    },
]


class TestTheCardStatesExposure:
    """The block that distinguishes an adapter card from a model card."""

    def test_it_says_the_component_executes_code_in_the_caller_process(self):
        body = str(generate_component_card(_manifest(TWO_COMPOSITIONS), REPO))
        assert "executes code in your process" in body
        assert "IT_TRUST_REMOTE_CODE" in body, "the card must name the gate, not merely allude to a gate"
        assert "composes into the MRO" in body, (
            "an adapter is the only kind that becomes part of the object the session runs; a card that "
            "omits this understates the exposure in exactly the way a model card is entitled to"
        )

    def test_it_names_the_entrypoint_that_would_run(self):
        body = str(generate_component_card(_manifest(TWO_COMPOSITIONS, entrypoint="custom_entry.py"), REPO))
        assert "custom_entry.py" in body, "a reader deciding whether to trust this needs the file named"

    def test_it_points_at_the_inspect_before_executing_path(self):
        body = str(generate_component_card(_manifest(TWO_COMPOSITIONS), REPO))
        assert f'interpretune.hub.pull("{REPO}")' in body, (
            "stating the exposure without the way to inspect it first leaves a reader with a warning and no action"
        )


class TestConditionalityIsVisibleBeforePulling:
    def test_a_conditional_composition_names_what_it_requires(self):
        body = str(generate_component_card(_manifest(TWO_COMPOSITIONS), REPO))
        assert "requires `circuit-tracer`" in body

    def test_an_unconditional_composition_reads_as_always(self):
        """Negative control: 'requires X' must be caused by the requires block, not printed for every row."""
        body = str(generate_component_card(_manifest(TWO_COMPOSITIONS), REPO))
        rows = [line for line in body.splitlines() if line.startswith("| `core` + `interp_engine` |")]
        assert rows and "always" in rows[0], (
            "an unconditional composition rendered as conditional would make the distinction meaningless"
        )

    def test_the_skip_is_described_as_reported_not_absent(self):
        body = str(generate_component_card(_manifest(TWO_COMPOSITIONS), REPO))
        assert "skipped and reported" in body, (
            '"unavailable here" and "does not exist" are the two states #431 exists to keep distinct; the '
            "card is where a reader first meets that distinction"
        )


class TestProvenanceIsStatedRatherThanImplied:
    def test_the_card_says_these_are_declarations(self):
        """The publisher cannot execute the entrypoint, so the card must not imply it verified anything.

        Reconciliation against what actually registers happens in `load_hub_adapter`, at load, behind the
        trust gate. A card claiming more than the publisher could know would be the exact overstatement
        this block exists to prevent.
        """
        body = str(generate_component_card(_manifest(TWO_COMPOSITIONS), REPO))
        assert "DECLARATIONS" in body
        assert "load_hub_adapter" in body, "a reader should be told where the real reconciliation happens"


class TestCoherenceIsCheckedAtPublish:
    def test_a_composition_naming_an_unknown_adapter_is_refused(self):
        """The subset the publisher CAN check without executing: a composition nothing could ever register."""
        bad = _manifest([{"component": "module", "adapters": ["core", "interp_engine", "not_a_real_adapter"]}])
        with pytest.raises(ComponentCardError, match="not_a_real_adapter"):
            generate_component_card(bad, REPO)

    def test_a_declared_adapter_is_accepted_even_though_it_is_not_bundled(self):
        """Positive control: the check must accept the component's OWN adapter, or it refuses every card.

        Without this, a check that only accepted bundled `Adapter` members would reject exactly the
        components this kind exists for — and it would look like strictness rather than a bug.
        """
        ok = _manifest([{"component": "module", "adapters": ["core", "interp_engine"]}])
        assert "## Adapters" in str(generate_component_card(ok, REPO))


class TestOtherKindsAreUnaffected:
    def test_a_module_component_gets_no_adapters_section(self):
        """The branch must be gated on the kind, not run for every component."""
        module_manifest = {
            "it_schema_version": 1,
            "kinds": ["module"],
            "module": {"configs": {"demo": "demo.yaml"}},
        }
        body = str(generate_component_card(module_manifest, "speediedan/rte"))
        assert "## Adapters" not in body
        assert "## Configurations" in body, "the existing module rendering must still work"


class TestTheCardNamesWhatItCannotReport:
    """An absent section reads as an absent limit, which is the stronger claim and the false one.

    The adapter card renders the validated manifest. Capabilities and hook refusals live in the code, and
    the publisher never executes the entrypoint, so they are structurally unreachable. That is the right
    trade -- rendering them would mean either executing hub-resident code at publish time, which the trust
    gate exists to prevent, or publishing an undeclared claim unchallenged.

    But a reader cannot distinguish "not reported" from "none exist" unless the card says which. This pins
    that it says so, because the sentence is exactly the kind a later tidy-up deletes as boilerplate.
    """

    def test_the_card_says_capabilities_and_refusals_are_not_derivable(self):
        card = str(generate_component_card(_manifest(TWO_COMPOSITIONS), REPO))
        assert "What this card cannot tell you" in card
        assert "refuses" in card or "refusal" in card
        assert "no limits" in card, (
            "the card must state that an absent section is not a claim of no limits; without that "
            "sentence the omission reads as the stronger claim"
        )

    def test_it_points_somewhere_actionable(self):
        """Naming a gap without a next step just relocates the reader's problem."""
        card = str(generate_component_card(_manifest(TWO_COMPOSITIONS), REPO))
        assert "documentation" in card or "registered backend" in card


class TestMeasuredCapabilitiesAreRenderedOnlyWhenTrustworthy:
    """The measured block renders from a published conformance report, and never by the report's mere presence."""

    HEAD = "a" * 40

    def _report(self, *, head=None, exit_status=0, fmt="interpretune.conformance.report/1"):
        return {
            "format": fmt,
            "provenance": {
                "interpretune_version": "0.1.0",
                "git_head": self.HEAD if head is None else head,
                "measured_at": "2026-09-10T15:00:00+00:00",
                "exit_status": exit_status,
            },
            "targets": {
                "TestX": {
                    "composition": ["core", "interp_engine"],
                    "model_id": "gpt2",
                    "model_capabilities": ["activation_intervention"],
                    "analysis_capabilities": [],
                    "intervention": {"modes": ["add"], "position_scopes": ["last_token"]},
                    "capture": {
                        "capturable": ["hook_in", "hook_out"],
                        "uncapturable": {"ln2.hook_scale": "the vocabulary has no spelling this engine can address"},
                        "n_layers": 12,
                        "architecture": "GPT2LMHeadModel",
                    },
                }
            },
            "ran": ["TestX::a"],
            "skipped_undeclared": ["TestX::b"],
            "skipped_other": [],
            "failed": [],
        }

    def _card(self, tmp_path, report, source_revision):
        import json

        from interpretune.hub.cards import CONFORMANCE_REPORT_FILE

        if report is not None:
            (tmp_path / CONFORMANCE_REPORT_FILE).write_text(json.dumps(report), encoding="utf-8")
        return str(
            generate_component_card(_manifest(TWO_COMPOSITIONS), REPO, tree=tmp_path, source_revision=source_revision)
        )

    def test_no_report_keeps_the_cannot_tell_sentence(self, tmp_path):
        body = self._card(tmp_path, None, self.HEAD)
        assert "What this card cannot tell you" in body and "Measured capabilities" not in body

    def test_a_matching_green_report_renders_the_measured_block(self, tmp_path):
        body = self._card(tmp_path, self._report(), self.HEAD)
        assert "### Measured capabilities" in body and "What this card cannot tell you" not in body
        assert "component revision `aaaaaaaaaaaa`" in body and "exit status 0" in body
        assert "modes `add`" in body and "2 of 3 base points on `GPT2LMHeadModel`" in body
        assert "cannot capture `ln2.hook_scale`: the vocabulary has no spelling" in body
        assert "**vocabulary gap**" in body and "**capability gap**" in body
        assert "1 ran, 1 skipped because the surface is undeclared" in body

    def test_a_stale_report_is_named_and_not_shown(self, tmp_path):
        body = self._card(tmp_path, self._report(head="b" * 40), self.HEAD)
        assert "Measured capabilities" not in body and "What this card cannot tell you" in body
        assert "exists for component revision `bbbbbbbbbbbb`, not the one published (`aaaaaaaaaaaa`)" in body

    def test_a_red_run_is_named_and_not_shown(self, tmp_path):
        body = self._card(tmp_path, self._report(exit_status=1), self.HEAD)
        assert "Measured capabilities" not in body and "did not pass, so it is not shown" in body

    def test_an_unknown_publishing_revision_treats_the_report_as_absent(self, tmp_path):
        body = self._card(tmp_path, self._report(), None)
        assert "Measured capabilities" not in body and "could not be determined, so it is not shown" in body

    def test_an_unknown_format_is_not_read(self, tmp_path):
        body = self._card(tmp_path, self._report(fmt="something/else"), self.HEAD)
        assert "Measured capabilities" not in body and "its format is not one this card reads" in body


class TestTheRevisionKeyIsStableAcrossUnrelatedCommits:
    """The positive control for the renderer: a report keyed to the component's directory revision still renders after
    an unrelated commit lands elsewhere in the repository, because both sides compute the last commit touching the
    component directory with the same helper. Two repository heads would match only on the exact measured commit."""

    @staticmethod
    def _git(repo, *args):
        import subprocess

        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    def _repo(self, tmp_path):
        repo = tmp_path / "repo"
        (repo / "component").mkdir(parents=True)
        self._git(repo, "init", "-q")
        self._git(repo, "config", "user.email", "t@example.invalid")
        self._git(repo, "config", "user.name", "t")
        (repo / "component" / "adapter.py").write_text("x = 1\n")
        self._git(repo, "add", "-A")
        self._git(repo, "commit", "-q", "-m", "component")
        return repo

    def test_publish_and_suite_compute_the_same_key_and_it_survives_an_unrelated_commit(self, tmp_path, monkeypatch):
        import json

        from interpretune.hub.cards import CONFORMANCE_REPORT_FILE
        from interpretune.hub.publish import source_revision_of
        from interpretune.hub.revisions import directory_revision, repo_head

        repo = self._repo(tmp_path)
        component = repo / "component"
        measured = directory_revision(component)
        assert measured is not None and measured == self._git(repo, "rev-parse", "HEAD")
        # an unrelated commit elsewhere in the repository
        (repo / "README.md").write_text("unrelated\n")
        self._git(repo, "add", "-A")
        self._git(repo, "commit", "-q", "-m", "unrelated")
        assert repo_head(repo) != measured, "the repository head moved"
        assert source_revision_of(component) == measured, "the component's own revision did not"
        report = {
            "format": "interpretune.conformance.report/1",
            "provenance": {
                "interpretune_version": "0.1.0",
                "git_head": measured,
                "component_revision": measured,
                "measured_at": "2026-09-10T15:00:00+00:00",
                "exit_status": 0,
            },
            "targets": {},
            "ran": [],
            "skipped_undeclared": [],
            "skipped_other": [],
            "failed": [],
        }
        staged = tmp_path / "staged"
        staged.mkdir()
        (staged / CONFORMANCE_REPORT_FILE).write_text(json.dumps(report))
        body = str(
            generate_component_card(
                _manifest(TWO_COMPOSITIONS), REPO, tree=staged, source_revision=source_revision_of(component)
            )
        )
        assert "### Measured capabilities" in body, "the block must still render after the unrelated commit"

    def test_a_report_with_only_a_repository_head_matches_only_that_commit(self, tmp_path):
        import json

        from interpretune.hub.cards import CONFORMANCE_REPORT_FILE
        from interpretune.hub.revisions import repo_head

        repo = self._repo(tmp_path)
        head = repo_head(repo)
        report = {
            "format": "interpretune.conformance.report/1",
            "provenance": {"git_head": head, "exit_status": 0, "interpretune_version": "x", "measured_at": "t"},
            "targets": {},
            "ran": [],
            "skipped_undeclared": [],
            "skipped_other": [],
            "failed": [],
        }
        staged = tmp_path / "staged"
        staged.mkdir()
        (staged / CONFORMANCE_REPORT_FILE).write_text(json.dumps(report))
        assert "### Measured capabilities" in str(
            generate_component_card(_manifest(TWO_COMPOSITIONS), REPO, tree=staged, source_revision=head)
        )
        assert "not the one published" in str(
            generate_component_card(_manifest(TWO_COMPOSITIONS), REPO, tree=staged, source_revision="f" * 40)
        )
