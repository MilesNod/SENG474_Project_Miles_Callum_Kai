"""A single collocated fluid state, independent of plotting or ML libraries."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class FluidState:
    """Own float64 arrays of equal shape (H, W), with rows increasing in y.

    Arrays remain mutable for callers preparing initial conditions. Construction
    copies all components so states do not accidentally share storage.
    """

    u: NDArray[np.float64]
    v: NDArray[np.float64]
    p: NDArray[np.float64]

    def __post_init__(self) -> None:
        components = [np.asarray(field) for field in (self.u, self.v, self.p)]
        shape = components[0].shape
        if len(shape) != 2 or 0 in shape:
            raise ValueError("State components must be nonempty 2D arrays.")
        for field in components:
            if field.shape != shape:
                raise ValueError("u, v, and p must have the same shape.")
            if not (
                np.issubdtype(field.dtype, np.integer)
                or np.issubdtype(field.dtype, np.floating)
            ):
                raise ValueError("State components must contain real numbers.")
            if not np.isfinite(field).all():
                raise ValueError("State components must contain finite values.")
        self.u, self.v, self.p = (
            np.array(field, dtype=np.float64, copy=True) for field in components
        )

    def to_array(self) -> NDArray[np.float64]:
        """Return a new (3, H, W) array with channels ordered [u, v, p]."""
        return np.stack((self.u, self.v, self.p), axis=0)

    @classmethod
    def from_array(cls, array: NDArray[np.float64]) -> "FluidState":
        """Copy a (3, H, W) array with channels [u, v, p] into a state."""
        array = np.asarray(array)
        if array.ndim != 3 or array.shape[0] != 3:
            raise ValueError("Expected an array of shape (3, H, W).")
        return cls(u=array[0], v=array[1], p=array[2])
