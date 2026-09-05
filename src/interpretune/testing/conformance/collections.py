"""The op-collection conformance cases: does a collection's every op declare what it needs, and does the target
honour or refuse each by name?

A repository subclasses ``OpCollectionConformance``, sets ``target`` (the composition under test, exactly as for
``ModelBackendConformance``) and ``collection``: a bundled op family name (``"concept"``) or a hub repo id
(``"org/repo"``, pulled or staged by the target's ``load``). Cases are read off the dispatcher's definitions, so a
collection is validated through the same objects a session executes.
"""

from __future__ import annotations

from typing import Any, ClassVar, cast

import pytest

from interpretune.analysis.backends import BackendCapability
from interpretune.analysis.ops.base import AnalysisOp

from .inputs import ConformanceInputs, ConformanceTarget
from .oracles import expect_refusal
from .session import ConformanceSession, build_conformance_session

BUNDLED_PREFIX = "interpretune.analysis.ops.bundled."


def belongs_to_collection(op_def: Any, collection: str, *, family_members: set[str] | None = None) -> bool:
    """Whether a canonical ``OpDef`` is part of ``collection``.

    A hub collection is identified by its declared ``collection_name`` or by ``hub:<user.repo>`` provenance. A
    bundled family is identified by the implementation path of a leaf op, and a composite belongs when every
    member does (``family_members`` carries the leaves already admitted, so composites are decided second).
    """
    if "/" in collection:
        namespaced = collection.replace("/", ".")
        return op_def.collection_name == collection or op_def.source == f"hub:{namespaced}"
    if op_def.source != "bundled":
        return False
    if op_def.composition:
        members = family_members or set()
        return bool(members) and all(name.split(".")[-1] in members for name in op_def.composition)
    return str(op_def.implementation).startswith(f"{BUNDLED_PREFIX}{collection}.")


def ops_in_collection(collection: str) -> dict[str, Any]:
    """Canonical ``{name: OpDef}`` for every op of a bundled family or a hub collection, composites last."""
    from interpretune.analysis.ops.dispatcher import DISPATCHER

    definitions = {name: d for name, d in DISPATCHER._op_definitions.items() if d.name == name}
    leaves = {n: d for n, d in definitions.items() if not d.composition and belongs_to_collection(d, collection)}
    members = {n.split(".")[-1] for n in leaves}
    composites = {
        n: d
        for n, d in definitions.items()
        if d.composition and belongs_to_collection(d, collection, family_members=members)
    }
    return {**leaves, **composites}


class OpCollectionConformance:
    """Subclass, set ``target`` and ``collection``, and pytest does the rest."""

    target: ClassVar[ConformanceTarget]
    collection: ClassVar[str]
    inputs: ClassVar[ConformanceInputs | None] = None

    @pytest.fixture(scope="class")
    def suite(self, request) -> ConformanceSession:
        """One composed session per target class, exactly as for the model-backend cases."""
        cls = request.cls
        return build_conformance_session(cls.target, cls.inputs or ConformanceInputs())

    @pytest.fixture(scope="class")
    def collection_ops(self, request, suite) -> dict[str, Any]:
        """The collection's canonical definitions, after the target's ``load`` has run (via ``suite``)."""
        ops = ops_in_collection(request.cls.collection)
        assert ops, f"collection {request.cls.collection!r} declares no ops the dispatcher can see"
        return ops

    def test_every_declared_requirement_is_a_known_axis(self, collection_ops):
        """Each op's declared capabilities, modes and scopes are members of the vocabulary.

        Instantiation normalizes every axis and raises naming the stray value, so instantiating is the check.
        """
        from interpretune.analysis.ops.dispatcher import DISPATCHER

        for name in collection_ops:
            DISPATCHER.get_op(name)

    def test_requirements_are_satisfied_or_refused_by_name(self, suite, collection_ops):
        """For each op: the target's live declarations satisfy its requirements, or validation refuses naming
        the missing capability, mode or scope. Nothing executes either way."""
        from interpretune.analysis.ops.dispatcher import DISPATCHER

        caps = suite.capabilities
        record = caps.intervention
        for name in collection_ops:
            op = cast(AnalysisOp, DISPATCHER.get_op(name))
            missing: list[str] = [c.value for c in op.required_capabilities if not caps.supports(c)]
            if op.requires_intervention_axes:
                if record is None:
                    missing.append(BackendCapability.INTERVENTION.value)
                else:
                    declared_modes = {m.value for m in record.modes}
                    declared_scopes = {s.value for s in record.position_scopes}
                    missing += [m.value for m in op.required_intervention_modes if m.value not in declared_modes]
                    missing += [s.value for s in op.required_position_scopes if s.value not in declared_scopes]
            if not missing:
                op._validate_capabilities(suite.module)  # the runner's pre-execution check, no execution
                continue
            with expect_refusal(ValueError, match=missing[0]):
                op._validate_capabilities(suite.module)

    def test_each_op_runs_on_its_declared_sample(self, suite, collection_ops):
        """An op declaring ``conformance.run_inputs`` runs through the runner and yields its output columns."""
        from interpretune import AnalysisCfg

        sampled = {
            n: d.conformance for n, d in collection_ops.items() if d.conformance and "run_inputs" in d.conformance
        }
        if not sampled:
            pytest.skip(
                f"no op in {self.collection!r} declares a `conformance.run_inputs` sample: {sorted(collection_ops)}"
            )
        for name, sample in sampled.items():
            store = suite.run(AnalysisCfg(target_op=name, run_inputs=dict(sample["run_inputs"])))
            expected = {
                col for col, cfg in collection_ops[name].output_schema.items() if getattr(cfg, "required", True)
            }
            present = set(store.dataset.column_names)
            assert expected <= present, f"{name}: output columns {sorted(expected - present)} missing from the store"


__all__ = ["OpCollectionConformance", "belongs_to_collection", "ops_in_collection"]
