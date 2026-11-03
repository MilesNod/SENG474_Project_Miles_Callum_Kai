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

from src.simulation import SolverConfig, NavierStokesSolver

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

class FluidDataset(Dataset):
    def __init__(self, configs=None, num_steps=250, noise = 0.0):
        """
        num_steps: number of steps the solver runs
        noise: amount of noise that you want, 0 is no noise
        self.trajs stores all the data

        ex.
        bazinga = FluidDataset(make_configs(100, 3))
        bazinga.trajs[3, 50]
        returns the 50th state for the 3rd run of the 100 make configs things
        """
        # for base case incase you dont put anything
        # base case is just the default system incase we don't even want to use configs
        if configs is None:
            configs = [SolverConfig()]
        
        trajs = []
        # for every config put the dataset in trajs
        for config in configs:
            solver = NavierStokesSolver(config)
            states = solver.run(solver.initial_state(), num_steps)
            trajs.append(np.stack([state.to_array().reshape(-1) for state in states]))

        self.trajs = np.stack(trajs).astype(np.float32)
        self.noise = noise
        self.num_steps = num_steps
        self.params = np.array([[c.top_wall_velocity, c.viscosity] for c in configs], dtype=np.float32)

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

        # concatenates the parameters that we're working with to the state (top_wall_velocity and viscosity)
        state = torch.cat([state, torch.from_numpy(self.params[i])]) 
        return state, next_state # return two states based on index and noise

# doing some testing, if I forget to take this out, this is AI
# def main():
#     ds = FluidDataset(make_configs(3, 0), num_steps=50)

#     # 1. Is anything nonzero at all?
#     print("nonzero fraction:", np.count_nonzero(ds.trajs) / ds.trajs.size)

#     # 2. Does each run develop over time? (frame 0 is zero by design)
#     for r in range(len(ds.trajs)):
#         for t in (0, 1, 2, 10, 50):
#             ch = ds.trajs[r, t].reshape(3, 41, 41)
#             print(f"run {r} t={t}",
#                 "max|u|:", round(float(np.abs(ch[0]).max()), 4),
#                 "max|v|:", round(float(np.abs(ch[1]).max()), 4),
#                 "max|p|:", round(float(np.abs(ch[2]).max()), 4))

#     # 3. Do different runs actually differ? (different lid speed and viscosity)
#     print("lid speeds:", ds.params[:, 0])
#     print("u top row, t=50:", [float(ds.trajs[r, 50].reshape(3, 41, 41)[0, -1, 0]) for r in range(3)])
#     # should match each run's top_wall_velocity

#     # 4. Are the training pairs real? (state != next_state)
#     x, y = ds[10]
#     print("pair differs:", not torch.equal(x[:5043], y))
#     print("diff magnitude:", (y - x[:5043]).abs().max().item())

#     # 5. Is it finite?
#     print("finite:", np.isfinite(ds.trajs).all())

# if __name__ == "__main__":
#     main()