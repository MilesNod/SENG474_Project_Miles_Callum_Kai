import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import FluidModel
from data import FluidDataset

dataset = FluidDataset()
data_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

model = FluidModel(10, 10)

for epoch in range(50):
    total_loss = 0

    progress = tqdm(data_loader, desc=f"epoch {epoch}")
    for batch_index, (state, next_state) in enumerate(progress):
        model.optimizer.zero_grad()

        predicted_state = model(state)
        loss: torch.Tensor = model.loss_func(predicted_state, next_state)
        total_loss += loss.item()
        progress.set_postfix(avg_loss=total_loss / (batch_index + 1))
        loss.backward()

        model.optimizer.step()

    model.save("weights/newest_model.pt")