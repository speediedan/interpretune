"""The op-collection conformance cases: does a collection's every op declare what it needs, and does the target
honour or refuse each by name?

A repository subclasses ``OpCollectionConformance``, sets ``target`` (the composition under test, exactly as for
``ModelBackendConformance``) and ``collection``: a bundled op family name (``"concept"``), a hub repo id
(``"org/repo"``, pulled by the target's ``load``), or the declared name of a collection the target's ``load`` stages
from a local op path (see :func:`stage_local_collection`). Cases are read off the dispatcher's definitions, so a
collection is validated through the same objects a session executes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, cast

import pytest

from interpretune.analysis.backends import ModelBackendCapability
from interpretune.analysis.ops.base import AnalysisOp

from .inputs import ConformanceInputs, ConformanceTarget
from .oracles import expect_refusal
from .session import build_conformance_session

BUNDLED_PREFIX = "interpretune.analysis.ops.bundled."


def belongs_to_collection(op_def: Any, collection: str, *, family_members: set[str] | None = None) -> bool:
    """Whether a canonical ``OpDef`` is part of ``collection``.

    A hub collection is identified by its declared ``collection_name`` or by ``hub:<user.repo>`` provenance. A
    locally staged collection (loaded from an op path, ``source == "local"``) is identified by the name its
    ``collection:`` header declares, which is what lets a collection repository validate its own working tree
    before it is published. A bundled family is identified by the implementation path of a leaf op. In both of the
    last two, a composite belongs when every member does (``family_members`` carries the leaves already admitted,
    so composites are decided second).
    """
    if "/" in collection:
        namespaced = collection.replace("/", ".")
        return op_def.collection_name == collection or op_def.source == f"hub:{namespaced}"
    if op_def.source not in ("bundled", "local"):
        return False
    if op_def.source == "local" and op_def.collection_name == collection:
        return True
    if op_def.composition:
        members = family_members or set()
        return bool(members) and all(name.split(".")[-1] in members for name in op_def.composition)
    return op_def.source == "bundled" and str(op_def.implementation).startswith(f"{BUNDLED_PREFIX}{collection}.")


def stage_local_collection(path: Any) -> Callable[[], None]:
    """A ``ConformanceTarget.load`` hook that loads the op collection at ``path`` (a directory of op YAMLs).

    For a collection repository validating its own working tree before publishing: set ``collection`` to the name
    its ``collection:`` header declares, and the cases read the staged ops through the dispatcher exactly as a
    session would.
    """

    def _load() -> None:
        from interpretune.analysis.ops.dispatcher import DISPATCHER

        DISPATCHER.add_op_path(path)

    return _load


def ops_in_collection(collection: str) -> dict[str, Any]:
    """Canonical ``{name: OpDef}`` for every op of a bundled family, a hub collection or a locally staged one,
    composites last."""
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
    #: Per-op values merged over each declared ``conformance.run_inputs`` sample, for inputs only the test
    #: environment can supply (a fixture path, a locally generated artifact). The declaration still says which
    #: inputs the sample needs; an override may only fill or replace keys, never add a sample for an undeclared op.
    run_input_overrides: ClassVar[dict[str, dict[str, Any]]] = {}

    @pytest.fixture(scope="class")
    def suite(self, request):
        """One composed session per target class, exactly as for the model-backend cases; cleaned up with it."""
        cls = request.cls
        inputs = cls.inputs or ConformanceInputs()
        yield build_conformance_session(cls.target, inputs)
        inputs.cleanup()

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
                    missing.append(ModelBackendCapability.ACTIVATION_INTERVENTION.value)
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
        undeclared = sorted(set(self.run_input_overrides) - set(sampled))
        assert not undeclared, (
            f"run_input_overrides name {undeclared}, which declare no `conformance.run_inputs` sample: an override "
            "fills a declared sample and cannot stand in for one"
        )
        if not sampled:
            pytest.skip(
                f"no op in {self.collection!r} declares a `conformance.run_inputs` sample: {sorted(collection_ops)}"
            )
        for name, sample in sampled.items():
            run_inputs = {**sample["run_inputs"], **self.run_input_overrides.get(name, {})}
            store = suite.run(AnalysisCfg(target_op=name, run_inputs=run_inputs))
            # An intermediate-only column is consumed inside the op's composition and never persisted, by definition.
            expected = {
                col
                for col, cfg in collection_ops[name].output_schema.items()
                if getattr(cfg, "required", True) and not getattr(cfg, "intermediate_only", False)
            }
            present = set(store.dataset.column_names)
            assert expected <= present, f"{name}: output columns {sorted(expected - present)} missing from the store"


__all__ = ["OpCollectionConformance", "belongs_to_collection", "ops_in_collection", "stage_local_collection"]
