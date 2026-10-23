from simulation import Fluid

from matplotlib import pyplot as plt
import numpy as np

def plot_fluid(fluid: Fluid, show_grid=True):
    cs = fluid.cell_size
    W, H = fluid.width, fluid.height
    div = fluid.compute_divergence()

    fig, ax = plt.subplots(figsize=(9, 9 * H / W))

    vmax = float(np.abs(div).max()) or 1.0
    im = ax.imshow(
        div,
        origin="upper",                      # row 0 → top of plot
        extent=(0, W * cs, 0, H * cs),
        cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="divergence", fraction=0.046, pad=0.04)

    max_v = max(np.abs(fluid.vx).max(), np.abs(fluid.vy).max(), 1e-12)
    scale = max_v / (0.8 * cs)

    # vx on vertical edges: row i at y = H*cs - (i+0.5)*cs
    xs = np.arange(W + 1) * cs
    ys = H * cs - (np.arange(H) + 0.5) * cs
    X, Y = np.meshgrid(xs, ys)
    ax.quiver(X, Y, fluid.vx, np.zeros_like(fluid.vx),
              color="black", angles="xy", scale_units="xy", scale=scale,
              width=0.003, pivot="mid")

    # vy on horizontal edges: row i at y = H*cs - i*cs
    xs = (np.arange(W) + 0.5) * cs
    ys = H * cs - np.arange(H + 1) * cs
    X, Y = np.meshgrid(xs, ys)
    ax.quiver(X, Y, np.zeros_like(fluid.vy), fluid.vy,
              color="black", angles="xy", scale_units="xy", scale=scale,
              width=0.003, pivot="mid")

    if show_grid:
        for x in np.arange(W + 1) * cs:
            ax.axvline(x, color="gray", lw=0.3, alpha=0.4)
        for y in np.arange(H + 1) * cs:
            ax.axhline(y, color="gray", lw=0.3, alpha=0.4)

    ax.set_aspect("equal")
    ax.set_xlim(0, W * cs)
    ax.set_ylim(0, H * cs)
    ax.set_xlabel("x")
    ax.set_ylabel("y (up)")
    ax.set_title("Edge velocities (arrows) & cell divergence (color)")
    plt.tight_layout()
    plt.show()
    return fig, ax