import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

class FluidModel(nn.Module):
    def __init__(self, width: int, height: int):
        super().__init__()

        input_size = (width + 1) * height + width * (height + 1)
        hidden_size = round(input_size * 1)

        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        state_dict = torch.load(path, weights_only=True)
        self.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.stack(x)
        return x