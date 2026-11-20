"""Linear spaces: named, versioned linear maps from a site's activations to the space its features are expressed in.

A latent model trained on $J_\\ell h$ rather than on $h$ (``space:jlens``) has decoder rows in that space, not in the
site's. :class:`LinearSpace` carries the map, reads site activations into the space, and writes an edit made there back
to the site. The write uses the map's declared inverse, or its pseudoinverse, and is exact: the map composed with its
inverse is the identity on the space, so the edited space coordinates are exactly the ones requested, and the part of
the activation the map does not see (its null space) is left as it was. A map that cannot be written exactly, one whose
rank is below its output dimension, is refused rather than written in least squares.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from interpretune.utils import MisconfigurationException


def _matrix(value: torch.Tensor, where: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or min(value.shape) < 1
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise MisconfigurationException(f"{where} needs a nonempty, finite floating-point matrix")
    return value.detach().clone()


@dataclass(frozen=True, eq=False)
class LinearSpace:
    """A linear map from site activations, width ``width``, to a space of dimension ``dim``.

    Args:
        name: The space's name, as a binding's ``space`` and an address's ``space:`` qualifier spell it.
        version: The map's exact version, recorded for provenance.
        matrix: The map, shape ``(dim, width)``: space coordinates are ``h @ matrix.T``.
        inverse: The write-back map, shape ``(width, dim)``, or ``None`` for the pseudoinverse.
        atol: Tolerance on ``matrix @ inverse`` equalling the identity, checked in double precision.
    """

    name: str
    version: str
    matrix: torch.Tensor
    inverse: torch.Tensor | None = None
    atol: float = 1e-6

    def __post_init__(self) -> None:
        for field in ("name", "version"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise MisconfigurationException(f"linear space needs a nonempty {field}, got {value!r}")
        if type(self.atol) not in (int, float) or not 0 < self.atol < 1:
            raise MisconfigurationException(f"linear space {self.name!r} needs atol strictly between zero and one")
        matrix = _matrix(self.matrix, f"linear space {self.name!r}")
        exact = matrix.to(torch.float64)
        if self.inverse is None:
            inverse = torch.linalg.pinv(exact).to(matrix.dtype)
        else:
            inverse = _matrix(self.inverse, f"inverse of linear space {self.name!r}")
            if inverse.shape != (matrix.shape[1], matrix.shape[0]):
                raise MisconfigurationException(
                    f"linear space {self.name!r}: inverse shape {tuple(inverse.shape)} is not"
                    f" {(matrix.shape[1], matrix.shape[0])}"
                )
        error = float((exact @ inverse.to(torch.float64) - torch.eye(len(matrix), dtype=torch.float64)).abs().max())
        if error > self.atol:
            rank = int(torch.linalg.matrix_rank(exact))
            raise MisconfigurationException(
                f"linear space {self.name!r} cannot be written exactly: its map has rank {rank} for {len(matrix)} space"
                f" dimensions, and map times inverse differs from the identity by {error:g}"
            )
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "inverse", inverse)

    @property
    def dim(self) -> int:
        """The space's dimension."""
        return self.matrix.shape[0]

    @property
    def width(self) -> int:
        """The site activation width the map reads."""
        return self.matrix.shape[1]

    def _like(self, tensor: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=h.device, dtype=h.dtype)

    def _check(self, h: torch.Tensor, width: int, what: str) -> None:
        if (
            not isinstance(h, torch.Tensor)
            or h.ndim < 1
            or h.shape[-1] != width
            or not h.is_floating_point()
            or not bool(torch.isfinite(h).all())
        ):
            raise MisconfigurationException(
                f"linear space {self.name!r} needs finite floating-point {what} with final width {width}"
            )

    def read(self, h: torch.Tensor) -> torch.Tensor:
        """Site activations ``(..., width)`` to space coordinates ``(..., dim)``."""
        self._check(h, self.width, "site activations")
        return h @ self._like(self.matrix, h).T

    def write(self, h: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        """Site activations whose space coordinates are ``coordinates``, changed only along the inverse's range."""
        current = self.read(h)
        if (
            not isinstance(coordinates, torch.Tensor)
            or coordinates.shape != current.shape
            or coordinates.dtype != h.dtype
            or coordinates.device != h.device
            or not bool(torch.isfinite(coordinates).all())
        ):
            raise MisconfigurationException(
                f"linear space {self.name!r} needs finite space coordinates with shape {tuple(current.shape)}, dtype"
                f" {h.dtype} and device {h.device}"
            )
        assert self.inverse is not None  # set in __post_init__
        return h + (coordinates - current) @ self._like(self.inverse, h).T
