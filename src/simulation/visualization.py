"""Optional pressure and streamline plotting; never imported by the solver."""

from typing import TYPE_CHECKING

import numpy as np

from .state import FluidState

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def visualize(state: FluidState, domain_size: float = 1.0) -> tuple["Figure", "Axes"]:
    """Create the baseline pressure/streamline plot without calling plt.show()."""
    from matplotlib import pyplot as plt

    height, width = state.u.shape
    x = np.linspace(0.0, domain_size, width)
    y = np.linspace(0.0, domain_size, height)
    grid_x, grid_y = np.meshgrid(x, y)
    figure, axes = plt.subplots()
    pressure = axes.contourf(grid_x, grid_y, state.p)
    figure.colorbar(pressure, ax=axes, label="Pressure")
    axes.streamplot(grid_x, grid_y, state.u, state.v, color="black")
    axes.set(xlabel="x", ylabel="y", title="Lid-driven cavity", aspect="equal")
    return figure, axes
