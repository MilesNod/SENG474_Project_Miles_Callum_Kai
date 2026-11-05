import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

class AddSpatialCoordinates2d(nn.Module):
    def forward(self, x):
        batch_size, _, height, width = x.size()
        x_coords = torch.linspace(-1, 1, width, device=x.device).view(1, 1, 1, width).expand(batch_size, 1, height, width)
        y_coords = torch.linspace(-1, 1, height, device=x.device).view(1, 1, height, 1).expand(batch_size, 1, height, width)
        return torch.cat([x, x_coords, y_coords], dim=1)

class FluidModel(nn.Module):
    def __init__(self, channels=3, hidden_dim=32):
        super().__init__()

        self.coord_injector = AddSpatialCoordinates2d()

        self.stack = nn.Sequential(
            nn.Conv2d(channels + 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.loss_func = nn.L1Loss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def save(self, path: str | Path):
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), checkpoint_path)

    def load(self, path: str):
        state_dict = torch.load(path, weights_only=True)
        self.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 3, 41, 41)
        x = self.coord_injector(x)
        x = self.stack(x)
        return x
