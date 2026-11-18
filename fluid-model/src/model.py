"""Signed, normalized state predictors with coordinates and physical-parameter inputs."""

# These classes describe prediction only. experiment.py handles fitting and scoring.
# state: (batch, 3, H, W), channels [u, v, p]. H/W are the grid dimensions.
# parameters: (batch, 2), values [viscosity, lid speed] for each sample.
# Both learned models predict a change, then add it to the current state.

from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class StatePredictor(nn.Module):
    def __init__(self, mean=None, std=None):
        # Shared setup for CNN and ridge. Buffers move with the model to the GPU
        # and get saved in checkpoints, but the optimizer does not learn these stats.
        super().__init__()
        self.register_buffer("field_mean", torch.as_tensor(np.zeros(3) if mean is None else mean,
                                                          dtype=torch.float32).reshape(1, 3, 1, 1))
        self.register_buffer("field_std", torch.as_tensor(np.ones(3) if std is None else std,
                                                         dtype=torch.float32).reshape(1, 3, 1, 1))

    def features(self, state, parameters):
        # Called inside either model's forward pass to build its 7 input channels.
        if state.ndim != 4 or state.shape[1] != 3:
            raise ValueError("States must have shape (batch, 3, height, width).")
        if parameters.shape != (len(state), 2):
            raise ValueError("Supply viscosity/lid-speed parameters of shape (batch, 2).")
        batch, _, height, width = state.shape
        # A convolution sees local patterns; coordinates also tell it where they are.
        # expand repeats the coordinate/parameter grids across the batch and grid area.
        x = torch.linspace(-1, 1, width, device=state.device, dtype=state.dtype).view(1, 1, 1, width)
        y = torch.linspace(-1, 1, height, device=state.device, dtype=state.dtype).view(1, 1, height, 1)
        physical = parameters / parameters.new_tensor([0.1, 1.0])
        # Put physical parameters on a useful scale using fixed reference values.
        # These constants do not depend on validation or test data.
        return torch.cat(((state - self.field_mean) / self.field_std,
                          x.expand(batch, 1, height, width), y.expand(batch, 1, height, width),
                          physical[:, :, None, None].expand(batch, 2, height, width)), dim=1)

    def save(self, path, metadata=None):
        # Used for fitted ridge checkpoints. CNN epoch saving is handled by experiment.py
        # because resuming it also needs Adam, shuffle, and random-generator state.
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format_version": 2, "architecture": self.architecture,
                    "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
                    "metadata": metadata or {}}, path)


class FluidModel(StatePredictor):
    def __init__(self, channels=3, hidden_dim=32, mean=None, std=None):
        # train_cnn makes a fresh instance per fold; hidden_dim is the width we tune.
        if channels != 3 or hidden_dim < 1:
            raise ValueError("Use three state channels and a positive hidden width.")
        super().__init__(mean, std)
        self.architecture = {"kind": "cnn", "hidden_dim": hidden_dim}
        self.stack = nn.Sequential(nn.Conv2d(7, hidden_dim, 3, padding=1), nn.ReLU(),
                                   nn.Conv2d(hidden_dim, 3, 3, padding=1))
        # No final ReLU: velocities and pressure need to be able to go negative.
        # Start at persistence. The signed linear output learns a normalized delta.
        nn.init.zeros_(self.stack[-1].weight)
        nn.init.zeros_(self.stack[-1].bias)

    def forward(self, state, parameters):
        # Called by model(state, parameters) during training, scoring, and each rollout step.
        # Convert the predicted normalized change back to field units, then advance the state.
        return state + self.stack(self.features(state, parameters)) * self.field_std


class LocalRidge(StatePredictor):
    """A shared 5x5 linear stencil with the CNN's inputs and receptive-field extent."""
    def __init__(self, mean=None, std=None):
        # fit_ridge fills these weights from a linear solve; we do not train them with Adam.
        super().__init__(mean, std)
        self.architecture = {"kind": "ridge"}
        self.linear = nn.Conv2d(7, 3, 5, padding=2)

    def forward(self, state, parameters):
        # Same current-state + predicted-change idea as the CNN, but with one linear stencil.
        return state + self.linear(self.features(state, parameters)) * self.field_std

    def design(self, state, parameters):
        # Only used while fitting ridge: unfold turns every 5x5 neighbourhood into a row.
        # Each row has 7 * 25 = 175 inputs; the fit code adds an intercept as input 176.
        features = F.unfold(self.features(state, parameters), kernel_size=5, padding=2)
        return features.transpose(1, 2).reshape(-1, 7 * 25)


def load_predictor(path, device="cpu"):
    # Used for evaluation/resume. Read the saved architecture to recreate the right model,
    # then load its weights and its own training normalizer before moving it to the device.
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    if checkpoint.get("format_version") != 2:
        raise ValueError("Legacy/incompatible checkpoint. Retrain with the current train.py; old weights are preserved.")
    architecture = checkpoint["architecture"]
    if architecture["kind"] == "cnn":
        model = FluidModel(hidden_dim=architecture["hidden_dim"])
    elif architecture["kind"] == "ridge":
        model = LocalRidge()
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint["metadata"]
