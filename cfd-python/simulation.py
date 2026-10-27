import numpy as np

class Fluid:
    def __init__(self, width: int, height: int, cell_size: float):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.density = 1

        self.rng = np.random.default_rng(seed=42)
        self.vx = self.rng.random(size=(height, width + 1)) * 2 - 1
        self.vy = self.rng.random(size=(height + 1, width)) * 2 - 1

        self.vx[:, 0] = 0
        self.vx[:, -1] = 0
        self.vy[0, :] = 0
        self.vy[-1, :] = 0

    def compute_divergence(self, pressure=None):
        left = self.vx[:, :-1]
        right = self.vx[:, 1:]
        top = self.vy[:-1, :]
        bottom = self.vy[1:, :]

        if(pressure != None):
            pass

        return (right - left) / self.cell_size + (top - bottom) / self.cell_size

    def solve_pressure(self, n: int, dt: float = 1):
        pressure = np.zeros((self.height, self.width))
        dx = self.cell_size
        dy = self.cell_size
        coeff = dx**2*dy**2*self.density/dt
        denom = 2*(dx**2 + dy**2)
        div = self.compute_divergence()

        for i in range(n):
            left = np.pad(pressure[:, :-1], ((0, 0), (1, 0)), mode="edge")
            right = np.pad(pressure[:, 1:], ((0, 0), (0, 1)), mode="edge")
            top = np.pad(pressure[:-1, :], ((1, 0), (0, 0)), mode="edge")
            bottom = np.pad(pressure[1:, :], ((0, 1), (0, 0)), mode="edge")

            pressure = (dy**2*(top + bottom) + dx**2*(right + left) - coeff * div) / denom
            pressure[0, 0] = 0.0

        return pressure

    def apply_pressure(self, pressure: np.ndarray, dt: float = 1):
        dx = self.cell_size
        dy = self.cell_size
        left = np.pad(pressure, ((0, 0), (1, 0)), mode="edge")
        right = np.pad(pressure, ((0, 0), (0, 1)), mode="edge")
        top = np.pad(pressure, ((1, 0), (0, 0)), mode="edge")
        bottom = np.pad(pressure, ((0, 1), (0, 0)), mode="edge")

        dpx = right - left
        self.vx = self.vx - dt/self.density * dpx/dx
        dpy = top - bottom
        self.vy = self.vy - dt/self.density * dpy/dy