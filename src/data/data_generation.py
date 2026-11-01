'''
This file will be the script ran to generate the dataset

Do note the set values(class vars) in simulation/solver/ if those are changed, make sure that we use the right data-set for that pls
Or I will just make that an input idk

'''
import csv
import numpy as np
from dataclasses import asdict

from src.simulation import FluidState, NavierStokesSolver, SolverConfig

def generate_data_set():
    # Create conflig
    datasetConfig = SolverConfig(
        # Change these as needed or make it so that each of these is a field in generate_data_set such that we can do it from cmd

        # These are the defualts set in the solver:

        # n_points = 41
        # domain_size = 1.0
        # dt = 0.001
        # viscosity = 0.1
        # density = 1.0
        # pressure_iterations = 50
        # top_wall_velocity = 1.0
        # stability_safety_factor = 0.5
    )


    # Create solver
    solver = NavierStokesSolver(datasetConfig)

    # Get Initial State
    initial_state = solver.initial_state()

    # Record initial states
    # Combine snaps into array, and save arr + config
    n_steps = 100 # Change this later or make this another input field
    trajectory: list[FluidState] = solver.run(initial_state, n_steps)

    # convert every state in traj
    snaps = np.stack (
        [state.to_array() for state in trajectory],
        axis = 0,
    )
    # output - (time, channels, grid/n_points, grid/n_points)


    np.savez_compressed(
        "datasetForTestingDefault.npz",
        states=snaps,
        **asdict(datasetConfig), # convert datasetConfig to dict
    )

    with np.load("datasetForTestingDefault.npz") as dataset:
        print(dataset["states"].shape)
        print(dataset["dt"].item())

def open_dataset():
    with np.load("datasetForTestingDefault.npz") as dataset:
            print(dataset.files)
            states = dataset["states"]
            print(states.shape)
            print(states[10,0])

if __name__ == "__main__":
    # generate_data_set()
    open_dataset()