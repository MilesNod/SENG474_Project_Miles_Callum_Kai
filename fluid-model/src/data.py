import torch
from torch.utils.data import Dataset

class FluidDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        """
        Get the total number of samples in the dataset.
        """

        return 1000

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generates/returns one sample from the dataset based on an index.
        Returns (state, next_state)
        """

        state = torch.rand(11 * 10 * 2)
        next_state = state

        return state, next_state # return a random 10x10 flat fluid state that is unchanging