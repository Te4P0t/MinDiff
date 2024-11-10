"""
Minimal Example for Iterative alpha Belnding: a Minimalist Deterministic Diffusion Model
https://arxiv.org/abs/2305.03486
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from schedulefree.adamw_schedulefree import AdamWScheduleFree
from torchvision.transforms import transforms as trns
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm, trange


class Gen(nn.Module):
    def __init__(
        self, input_dim=28 * 28 * 1, hidden_dim=1024, classes=None, class_embed_dim=128
    ):
        super(Gen, self).__init__()
        if classes is not None:
            self.class_embed = nn.Embedding(classes, class_embed_dim)
        self.model = nn.Sequential(
            nn.Linear(input_dim + 1 + bool(classes) * class_embed_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, input_dim),
        )
        nn.init.constant_(self.model[-1].weight, 0)
        nn.init.constant_(self.model[-1].bias, 0)

    def forward(self, x, t, cond=None):
        x = torch.cat([x, t], dim=-1)
        if cond is not None:
            x = torch.cat([x, self.class_embed(cond)], dim=-1)
        return self.model(x)


def train(epoch, model, optimizer, loss, dataloader, train_with_cond=False):
    model.train()
    ema_loss = 0
    with tqdm(dataloader, desc=f"Epoch {epoch}", smoothing=0.01) as pbar:
        for i, (x, cond) in enumerate(dataloader):
            x = x.view(-1, 28 * 28 * 1).to(DEVICE)
            cond = cond.to(DEVICE).long()

            ## alpha Blending Diffusion
            ## x0 ~ Image, x1 ~ N, t ~ [0, 1], xt = x0 + (x1 - x0) * t
            ## Model(xt, t) = x0 - x1
            ## xt-dt = xt + Model(xt, t) * dt
            t = torch.rand(x.shape[0], 1).to(DEVICE).float()
            eps = torch.randn_like(x)
            xt = x * (1 - t) + eps * t
            target = x - eps

            optimizer.zero_grad()
            y = model(xt, t, cond if train_with_cond else None)
            l = loss(y, target)
            l.backward()
            optimizer.step()

            ema_decay = min(0.99, i / 100)
            ema_loss = ema_decay * ema_loss + (1 - ema_decay) * l.item()

            pbar.update(1)
            pbar.set_postfix({"loss": ema_loss})
    torch.save(model.state_dict(), "gen.pth")


def test(epoch, model, gen_with_cond=False):
    model.eval()
    IMAGE_COUNT = 16 * 16
    with torch.no_grad():
        pred_x = torch.randn(IMAGE_COUNT, 28 * 28 * 1).to(DEVICE)
        cond = torch.arange(IMAGE_COUNT).long().to(DEVICE) % 10
        t = torch.ones(IMAGE_COUNT, 1).to(DEVICE)
        STEPS = 100
        for i in range(STEPS):
            pred = model(pred_x, t, cond if gen_with_cond else None)
            pred_x = pred_x + pred * 1 / STEPS
            t = t - 1 / STEPS
    # save image as single grid
    pred_x = pred_x.reshape(16, 16, 28, 28).permute(0, 2, 1, 3) * 0.5 + 0.5
    pred_x = pred_x.reshape(16 * 28, 16 * 28).cpu().numpy()
    pred_x = (pred_x * 255).clip(0, 255).astype(np.uint8)
    pred_x = Image.fromarray(pred_x)
    pred_x.save(f"./result/gen-{epoch}.png")


if __name__ == "__main__":
    os.makedirs("./result", exist_ok=True)
    DEVICE = "mps"
    CLASSES = 10
    EPOCHS = 100

    model = Gen(
        input_dim=28 * 28 * 1,
        hidden_dim=2048,
        classes=CLASSES,
        class_embed_dim=1024 - 785,
    ).to(DEVICE)
    print(sum(p.numel() for p in model.parameters()) / 1e6)
    optimizer = AdamWScheduleFree(
        model.parameters(), 1e-3, (0.9, 0.98), weight_decay=0.01, warmup_steps=100
    )
    loss = nn.MSELoss()
    transform = trns.Compose([trns.ToTensor(), trns.Normalize((0.5,), (0.5,))])
    dataset = MNIST("./data", download=True, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    for i in trange(EPOCHS):
        test(i, model, gen_with_cond=bool(CLASSES))
        train(i, model, optimizer, loss, dataloader, train_with_cond=bool(CLASSES))
    test(i + 1, model, gen_with_cond=bool(CLASSES))
