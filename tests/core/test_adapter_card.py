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
