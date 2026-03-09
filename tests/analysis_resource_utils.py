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

import torch

from interpretune.analysis.core import LatentAnalysisDict

from tests.runif import get_runner_ram_gb

ANALYSIS_LOW_RAM_GB = 16
FixtureScope = Literal["function", "class", "module", "session"]


class ExtractedAnalysisStore:
    """Minimal ``AnalysisStore`` view backed by a selected subset of fields."""

    def __init__(self, **fields: Any) -> None:
        self._fields = fields

    def __deepcopy__(self, memo: dict[int, Any]) -> ExtractedAnalysisStore:
        copied = type(self)(**deepcopy(self._fields, memo))
        memo[id(self)] = copied
        return copied

    def __getattr__(self, name: str) -> Any:
        if name in self._fields:
            return self._fields[name]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def by_latent_model(self, field_name: str, stack_latents: bool = True) -> LatentAnalysisDict:
        """Match ``AnalysisStore.by_latent_model`` for selected nested fields."""

        values = getattr(self, field_name)
        assert values, f"No values found for field {field_name}"
        if not isinstance(values[0], dict):
            raise TypeError(
                f"Values for field {field_name} must be dictionaries to be transformed into a LatentAnalysisDict"
            )

        result = LatentAnalysisDict()
        sae_names = values[0].keys()
        for sae in sae_names:
            if isinstance(values[0][sae], dict) and stack_latents:
                batch_tensors = []
                for batch in values:
                    latent_tensors = [tensor for tensor in batch[sae].values()]
                    batch_tensors.append(torch.stack(latent_tensors) if latent_tensors else None)
                result[sae] = batch_tensors  # type: ignore[assignment]
            else:
                result[sae] = [  # type: ignore[assignment]
                    None if isinstance(batch[sae], list) and not batch[sae] else batch[sae] for batch in values
                ]
        return result


def analysis_fixture_scope(
    *,
    min_ram_gb: int = ANALYSIS_LOW_RAM_GB,
    low_ram_scope: FixtureScope = "function",
    high_ram_scope: FixtureScope = "class",
) -> FixtureScope:
    """Return a pytest fixture scope based on available system RAM."""

    return low_ram_scope if get_runner_ram_gb() <= min_ram_gb else high_ram_scope


@contextmanager
def conditional_clean_cpu(fixture: Any, min_ram_gb: int = ANALYSIS_LOW_RAM_GB):
    """Clear large fixture payloads after extraction on low-RAM runners."""

    try:
        yield fixture
    finally:
        if get_runner_ram_gb() <= min_ram_gb:
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


def extract_analysis_store_fields(
    request: Any,
    fixture_key: str,
    field_names: list[str] | tuple[str, ...],
    min_ram_gb: int = ANALYSIS_LOW_RAM_GB,
) -> ExtractedAnalysisStore:
    """Copy selected ``fixture.result`` fields into a lightweight analysis-store view."""

    selected_fields = tuple(field_names)
    return extract_fixture_data(
        request,
        fixture_key,
        lambda fixture: ExtractedAnalysisStore(
            **{field_name: getattr(fixture.result, field_name) for field_name in selected_fields}
        ),
        min_ram_gb=min_ram_gb,
    )


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
