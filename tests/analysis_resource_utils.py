"""Shared helpers for memory-aware analysis fixtures in tests.

Preferred pattern for expensive analysis-session tests:

1. Choose fixture scope dynamically: ``class`` on higher-RAM runners and
   ``function`` on lower-RAM runners.
2. Deep-copy only the values a test class needs.
3. Reuse those copied values across methods via the public ``extract_values()``
   helper on ``AnalysisExtractionMixin``.

Tradeoff summary:
- Dynamic fixture scope is preferred over ad hoc transient session builders
  because it keeps lifecycle management inside pytest's fixture machinery and
  avoids duplicating analysis-session construction logic.
- Low-RAM runners trade some setup time for prompt teardown.
- High-RAM runners retain class-scoped reuse for throughput.
"""

from __future__ import annotations

import gc
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, ClassVar, Literal

from tests.runif import get_runner_ram_gb

ANALYSIS_LOW_RAM_GB = 16
FixtureScope = Literal["function", "class", "module", "session"]


def analysis_fixture_scope(
    *,
    min_ram_gb: int = ANALYSIS_LOW_RAM_GB,
    low_ram_scope: FixtureScope = "function",
    high_ram_scope: FixtureScope = "class",
) -> FixtureScope:
    """Return a pytest fixture scope based on available system RAM."""

    return low_ram_scope if get_runner_ram_gb() < min_ram_gb else high_ram_scope


@contextmanager
def conditional_clean_cpu(fixture: Any, min_ram_gb: int = ANALYSIS_LOW_RAM_GB):
    """Clear large fixture payloads after extraction on low-RAM runners."""

    try:
        yield fixture
    finally:
        if get_runner_ram_gb() < min_ram_gb:
            for attr in ("result", "runner", "run_config", "it_session"):
                if hasattr(fixture, attr):
                    setattr(fixture, attr, None)
            gc.collect()


def extract_fixture_result(request: Any, fixture_key: str, min_ram_gb: int = ANALYSIS_LOW_RAM_GB) -> Any:
    """Deep-copy a fixture result and release heavyweight fixture state when possible."""

    fixture = request.getfixturevalue(fixture_key)
    with conditional_clean_cpu(fixture, min_ram_gb=min_ram_gb) as active_fixture:
        return deepcopy(active_fixture.result)


def extract_fixture_data(
    request: Any,
    fixture_key: str,
    extractor: Any,
    min_ram_gb: int = ANALYSIS_LOW_RAM_GB,
) -> Any:
    """Extract arbitrary data from a fixture and clear heavyweight state on low-RAM runners."""

    fixture = request.getfixturevalue(fixture_key)
    with conditional_clean_cpu(fixture, min_ram_gb=min_ram_gb) as active_fixture:
        extracted = extractor(active_fixture)
    return deepcopy(extracted)


class AnalysisExtractionMixin:
    """Class-level cache helper for memory-heavy analysis tests.

    Subclasses implement ``build_extracted_values()`` and test methods call the
    public ``extract_values()`` helper to populate and reuse the copied payloads.
    """

    _extracted: ClassVar[Any | None] = None

    def build_extracted_values(self, request: Any) -> Any:
        raise NotImplementedError

    def extract_values(self, request: Any) -> Any:
        cls = type(self)
        extracted = getattr(cls, "_extracted", None)
        if extracted is None:
            extracted = self.build_extracted_values(request)
            cls._extracted = extracted
        return extracted

    @classmethod
    def clear_extracted_values(cls) -> None:
        cls._extracted = None
        gc.collect()
