from simulation import Fluid
from visualization import plot_fluid

fluid = Fluid(10, 10, 1)

plot_fluid(fluid) # before jacobi solver

pressure = fluid.solve_pressure(1000)
fluid.apply_pressure(pressure)

plot_fluid(fluid) # after jacobi solver