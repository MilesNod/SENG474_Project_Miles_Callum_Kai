import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

class FluidModel(nn.Module):
    def __init__(self, input_shape: torch.Size, hidden_factor: float = 1):
        super().__init__()

        input_size = input_shape.numel()
        hidden_size = round(input_size * hidden_factor)

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

    def save(self, path: str | Path):
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), checkpoint_path)

    def load(self, path: str):
        state_dict = torch.load(path, weights_only=True)
        self.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.stack(x)
        return x
