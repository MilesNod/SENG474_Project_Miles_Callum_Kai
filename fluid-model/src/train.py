from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import FluidModel
from data import FluidDataset, make_configs

print("loading training dataset... ", end="", flush=True)
dataset = FluidDataset(num_runs=100, num_steps=50)
data_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)
print(len(dataset), "samples loaded")

print("loading training dataset... ", end="", flush=True)
val_dataset = FluidDataset(num_runs=20, num_steps=50)
print(len(val_dataset), "samples loaded")

print("loading model...")
checkpoint_path = Path(__file__).resolve().parents[1] / "weights" / "newest_model.pt"
model = FluidModel()

print("training...")
best_validation = 10e10
for epoch in range(50):
    progress = tqdm(total=len(data_loader), desc=f"epoch {epoch}")

    total_loss = 0
    for batch_index, (state, next_state) in enumerate(data_loader):
        model.optimizer.zero_grad()

        predicted_state = model(state)
        loss: torch.Tensor = model.loss_func(predicted_state, next_state)
        total_loss += loss.item()
        progress.set_postfix({ "avg_loss": total_loss / (batch_index + 1) })
        loss.backward()

        model.optimizer.step()
        progress.update(1)

    model.save("weights/newest_model.pt")