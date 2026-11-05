import torch
import numpy as np
from torch.utils.data import Dataset
from dataclasses import replace
import os
import sys

# this is the code to import from other folders, there's definetly a better way than this
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))

if root_dir not in sys.path:sys.path.insert(0, root_dir)

from src.simulation import SolverConfig, NavierStokesSolver, FluidState

# I think we can ignore this function if we want, leaving it here just incase
def make_configs(n, seed):
    """
    n: number of configs that you want
    seed: random seed
    ex. make_configs(50, 3)
    returns a list of 50 SolverConfig() objects that have different viscosities and speeds
    makes different random configs based on what seed we use for random
    changes lid speed and viscosity

    side note: do we want speed and viscosity to be different? how will we give this to the model?
    """
    rng = np.random.default_rng(seed)
    base = SolverConfig()
    configs = []
    
    while len(configs) < n:
        speed = rng.uniform(0.3, 2.0) * rng.choice([-1, 1])   # no 0 because 0 is at rest
        cfg = replace(base, top_wall_velocity=speed, viscosity=base.viscosity * rng.uniform(0.5, 1.0))
        try:
            NavierStokesSolver(cfg)     
        except ValueError:
            continue
        configs.append(cfg)
    return configs 

def make_random_initial_state(solver, rng, amplitude):
    """
    makes a totally random initial state, might be violent, idk how fluid sims really work
    """
    H, W = solver.initial_state().u.shape
    u = rng.normal(size=(H, W)) * amplitude
    v = rng.normal(size=(H, W)) * amplitude
    for f in (u, v):               
        f[0, :] = f[-1, :] = 0
        f[:, 0] = f[:, -1] = 0
    return FluidState(u=u, v=v, p=np.zeros((H, W)))

class FluidDataset(Dataset):
    def __init__(self, config=None, num_steps=250, noise = 0.0, seed = 3, num_runs = 100, amplitude = 0.1):
        """
        config: the parameters of the system: will be default as discussed but is some legacy code now
        num_steps: number of steps the solver runs
        noise: amount of noise that you want, 0 is no noise
        *** NOISE IS ONLY ADDED WHEN getitems IS CALLED TO GET NOISE FOR THAT SPECIFIC STATE ***

        seed: for random seeds
        num_runs: the amount of seperate runs that we want to do for the fluid sim
        amplitude: the amount you want it to be randomized, I set it at 0.1 so the results won't be too crazy

        self.trajs stores all the data

        ex.
        bazinga = FluidDataset()
        bazinga.trajs[3, 50]
        returns the 50th state for the 3rd run of the 100 make configs things
        """
        # for base case incase you dont put anything
        # base case is just the default system incase we don't even want to use configs
        if config is None:
            config = SolverConfig()
        solver = NavierStokesSolver(config)
        rng = np.random.default_rng(seed)


        trajs = []
        # make a fluid sim for every run that we want
        for run in range(num_runs-1):
            s0 = make_random_initial_state(solver, rng, amplitude)
            states = solver.run(s0, num_steps)
            trajs.append(np.stack([state.to_array().reshape(3, 41, 41) for state in states]))

        self.trajs = np.stack(trajs).astype(np.float32)
        self.noise = noise
        self.num_steps = num_steps

        # standard deviation for each of the three fields, adds noise based on how the values of u, v, and p
        # used AI for this as I didn't know how to add noise effectivly
        ch = self.trajs.reshape(-1, 3, 41 * 41)      # (N*(T+1), 3, 1681)
        std_per_channel = ch.std(axis=(0, 2))                       # shape (3,)
        self.channel_std = torch.from_numpy(
        np.repeat(std_per_channel, 41 * 41).astype(np.float32)  # shape (5043,)
        )

    def __len__(self):
        """
        Get the total number of samples in the dataset.
        """

        return len(self.trajs) * self.num_steps
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        takes a number from 0 to len
        returns one sample from the dataset based on an index.
        Returns (state, next_state)
        """

        # divmod translates it too which run and step it is
        i, t = divmod(index, self.num_steps) # i is which run, t is which step
        state = torch.from_numpy(self.trajs[i, t])
        next_state = torch.from_numpy(self.trajs[i, t + 1])

        if self.noise > 0:                       # noise for training only
            state = state + self.noise * self.channel_std * torch.randn_like(state)
        return state, next_state # return two states based on index and noise

    def save(self, path):
        """
        saves the data

        ex:
        bazinga = FluidDataset(num_runs=100, seed=3)
        bazinga.save("data/train.pt")
        """
        torch.save({
            "trajs": torch.from_numpy(self.trajs),
            "num_steps": self.num_steps,
            "noise": self.noise,
            "channel_std": self.channel_std,
        }, path)

    @classmethod
    def load(cls, path, noise=None):
        """
        loads the data into trajs and all other relevant variables

        ex:
        train = FluidDataset.load("data/train.pt")
        val = FluidDataset.load("data/val.pt", noise=0.0)
        """
        data = torch.load(path)
        ds = cls.__new__(cls)                      # skip __init__
        ds.trajs = data["trajs"].numpy()
        ds.num_steps = data["num_steps"]
        ds.noise = data["noise"] if noise is None else noise
        ds.channel_std = data["channel_std"]
        return ds