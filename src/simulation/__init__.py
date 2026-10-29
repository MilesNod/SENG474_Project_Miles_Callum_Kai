"""NumPy-based, collocated lid-driven-cavity simulation."""

from .solver import NavierStokesSolver, SolverConfig
from .state import FluidState

__all__ = ["FluidState", "NavierStokesSolver", "SolverConfig"]
