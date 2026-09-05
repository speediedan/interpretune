"""Oracles shared by every conformance case: tolerances with their reasons, and the causal discriminators.

Nothing here knows about a backend. That is the point: an oracle that consulted the thing under test could
agree with it for the wrong reason.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.testing import assert_close

#: Value tolerance for a backend that EXECUTES the HuggingFace forward (family ``hf_native``). Tight, because a
#: difference must come from the observation or edit mechanism, which is what is under test, and not from
#: arithmetic drift.
CONVERGENCE_RTOL = 1e-4
CONVERGENCE_ATOL = 1e-4

#: "Changed" needs a stated threshold or the position set goes noisy from nondeterminism at ~1e-9 and the
#: pressure becomes to loosen the assertion until it passes. Far above numerical noise, far below any real
#: intervention's effect.
CHANGED_ATOL = 1e-3

#: Scale for the suite's steering vector: large enough that every steered position moves by more than
#: ``CHANGED_ATOL`` downstream, small enough to stay in the model's numerical range.
STEER_SCALE = 8.0


def changed_positions(baseline: torch.Tensor, intervened: torch.Tensor, *, atol: float = CHANGED_ATOL) -> set[int]:
    """Positions whose downstream tensor moved by more than ``atol``.

    Both tensors are ``[pos, ...]`` or ``[batch, pos, ...]``; the trailing axes are reduced with a max, and
    a batch axis is reduced with a max as well, so the returned set is the union over batch rows. Asserting
    the SET rather than a count is deliberate: a bug steering position 0 under a last-token spec gives a
    count of 1 and passes, and a whole-prompt intervention inert at one position also gives 1 while
    operating on the wrong scope.
    """
    if baseline.shape != intervened.shape:
        raise ValueError(f"shape mismatch: baseline {tuple(baseline.shape)} vs intervened {tuple(intervened.shape)}")
    delta = (intervened.to(torch.float32) - baseline.to(torch.float32)).abs()
    if delta.ndim >= 3:
        delta = delta.amax(dim=0)  # batch -> [pos, ...]
    while delta.ndim > 1:
        delta = delta.amax(dim=-1)
    return {int(i) for i in torch.nonzero(delta > atol).flatten().tolist()}


def expected_positions(scope: str, seq_len: int) -> set[int]:
    """The position set each scope must produce, by definition of the scope."""
    if scope == "last_token":
        return {seq_len - 1}
    if scope == "all_positions":
        return set(range(seq_len))
    raise ValueError(f"unknown position scope {scope!r}")


def assert_converges(got: torch.Tensor, ref: torch.Tensor, *, what: str) -> None:
    """Value convergence on the HF forward, with the tolerance stated once."""
    assert_close(
        got.to(ref.dtype),
        ref,
        rtol=CONVERGENCE_RTOL,
        atol=CONVERGENCE_ATOL,
        msg=f"{what} diverged from the HuggingFace forward (rtol={CONVERGENCE_RTOL}, atol={CONVERGENCE_ATOL})",
    )


def assert_non_degenerate(tensor: torch.Tensor, *, what: str) -> None:
    """A reference or a capture that is all zeros, NaN, or constant would make every comparison vacuous."""
    assert torch.isfinite(tensor).all(), f"{what} contains non-finite values"
    assert tensor.abs().max() > 0, f"{what} is identically zero"
    assert tensor.float().std() > 0, f"{what} is constant"


def steering_vector(reference_activation: torch.Tensor) -> torch.Tensor:
    """A unit direction with real effect for THIS model: the final position of a captured activation."""
    vec = reference_activation.reshape(-1, reference_activation.shape[-1])[-1]
    return (vec / vec.norm()).detach().clone()


def all_equal_sets(sets: Iterable[set[int]]) -> bool:
    """Whether every batch row reported the same position set (a per-row discriminator's sanity check)."""
    sets = list(sets)
    return all(s == sets[0] for s in sets)


class expect_refusal:
    """``pytest.raises`` for a refusal that may surface WRAPPED.

    A backend's refusal (``NotImplementedError`` naming the axis) raised inside the analysis runner reaches the
    caller as ``datasets``' ``DatasetGenerationError`` with the refusal as ``__cause__``. A case that matched only
    the outer exception would fail every correctly refusing backend, so this walks the cause and context chain and
    accepts the first exception of the expected type whose message matches. Usable as a context manager without
    pytest, so the package stays import-safe.
    """

    def __init__(self, exc_type: type[BaseException], match: str) -> None:
        self.exc_type = exc_type
        self.match = match
        self.found: BaseException | None = None

    def __enter__(self) -> "expect_refusal":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        import re

        if exc is None:
            raise AssertionError(
                f"expected a refusal ({self.exc_type.__name__} matching {self.match!r}); nothing was raised"
            )
        seen: set[int] = set()
        cur: BaseException | None = exc
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            if isinstance(cur, self.exc_type) and re.search(self.match, str(cur)):
                self.found = cur
                return True
            cur = cur.__cause__ or cur.__context__
        raise AssertionError(
            f"expected a refusal ({self.exc_type.__name__} matching {self.match!r}) in the exception chain; got "
            f"{type(exc).__name__}: {str(exc)[:200]}"
        ) from exc
