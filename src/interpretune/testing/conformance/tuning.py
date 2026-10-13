"""The tuning conformance cases: finetuning-scheduler's ``TunableView`` contract, run against a composition's view.

Tracked in #760.

A repository subclasses ``TuningConformance`` and sets ``target``, as it does for ``ModelBackendConformance``::

    class TestMyAdapterTuning(TuningConformance):
        target = ConformanceTarget(composition=("core", "my_adapter"), session_cfg_factory=build_session_cfg)

finetuning-scheduler owns the cases (``finetuning_scheduler.testing``); this module selects them from what the
composition's view declares. A view declaring ``trainable=True`` runs every case that applies to it, one declaring
``trainable=False`` (or a composition declaring nothing, which is refused) runs the refusal case, and a run in which no
case applies fails. Each case builds a fresh session, because cases change parameters and ``requires_grad``.

When the installed finetuning-scheduler predates views, every case skips with that reason.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any, ClassVar

import pytest

from interpretune.tuning.views import VIEW_ATTR, fts_views_available

from .inputs import ConformanceInputs, ConformanceTarget


def _case_names() -> list[str]:
    if not fts_views_available():
        return ["unavailable"]
    from finetuning_scheduler.testing import TunableViewConformance

    return [case.name for case in TunableViewConformance(lambda: None).cases()]


def tunable_view_of(module: Any) -> Any:
    """The view attached to an interpretune module.

    Raises:
        AssertionError: If none is attached, which should not happen when finetuning-scheduler provides views.
    """
    view = getattr(module, VIEW_ATTR, None)
    assert view is not None, f"{type(module).__name__} has no `{VIEW_ATTR}` attached after model init."
    return view


class TuningConformance:
    """Subclass, set ``target``, and pytest runs finetuning-scheduler's view cases against the composition."""

    target: ClassVar[ConformanceTarget]
    inputs: ClassVar[ConformanceInputs | None] = None

    @pytest.fixture(scope="class")
    def make_view(self, request) -> Any:
        """A factory returning the view of a freshly composed session, one per call."""
        if not fts_views_available():
            pytest.skip("the installed finetuning-scheduler predates TunableView support")
        from interpretune import ITSession

        cls = request.cls
        inputs = cls.inputs or ConformanceInputs()
        if cls.target.load is not None:
            cls.target.load()
        session_cfg = cls.target.build_session_cfg(inputs)

        def factory() -> Any:
            return tunable_view_of(ITSession(deepcopy(session_cfg)).module)

        yield factory
        inputs.cleanup()

    @pytest.fixture(scope="class")
    def probe(self, make_view: Callable[[], Any]) -> Any:
        """One bound view, for case selection."""
        from finetuning_scheduler.core.views import as_view

        return as_view(make_view())

    def test_some_case_applies(self, make_view, probe) -> None:
        """The vacuity guard: a target for which no case applies has checked nothing."""
        from finetuning_scheduler.testing import TunableViewConformance

        support = probe.support()
        applicable = [c.name for c in TunableViewConformance(make_view).cases() if c.applies(support, probe)]
        assert applicable, f"No tuning conformance case applies to this target's view ({support})."

    @pytest.mark.parametrize("case_name", _case_names())
    def test_tuning_case(self, make_view, probe, case_name: str) -> None:
        """One finetuning-scheduler case, skipped (with the view's support record) when it does not apply."""
        from finetuning_scheduler.testing import TunableViewConformance

        suite = TunableViewConformance(make_view)
        case = next(c for c in suite.cases() if c.name == case_name)
        support = probe.support()
        if not case.applies(support, probe):
            pytest.skip(f"does not apply to this view: {support}")
        report = suite.run(select=[case_name])
        assert not report.failures, report.failures[case_name]
