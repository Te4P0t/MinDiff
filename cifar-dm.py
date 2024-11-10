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
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from PIL import Image
from tqdm import tqdm, trange

from cifar_vae import Encoder, Decoder


class Gen(nn.Module):
    def __init__(
        self, input_dim=16 * 16 * 4, hidden_dim=1024, classes=None, class_embed_dim=128
    ):
        super(Gen, self).__init__()
        if classes is not None:
            self.class_embed = nn.Embedding(classes, class_embed_dim)
        self.input_proj = nn.Linear(
            input_dim + 1 + bool(classes) * class_embed_dim, hidden_dim
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 3),
            nn.Mish(),
            nn.Linear(hidden_dim * 3, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 3),
            nn.Mish(),
            nn.Linear(hidden_dim * 3, hidden_dim),
        )
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        nn.init.constant_(self.output_proj.weight, 0)
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(self, x, t, cond=None):
        x = torch.cat([x, t], dim=-1)
        if cond is not None:
            x = torch.cat([x, self.class_embed(cond)], dim=-1)
        x = self.input_proj(x)
        x = self.norm1(x + self.ffn1(x))
        x = self.norm2(x + self.ffn2(x))
        return self.output_proj(x)


def train(epoch, model, encoder, optimizer, loss, dataloader, train_with_cond=False):
    model.train()
    ema_loss = 0
    with tqdm(dataloader, desc=f"Epoch {epoch}", smoothing=0.01) as pbar:
        for i, (x, cond) in enumerate(dataloader):
            x = encoder(x.to(DEVICE)).view(x.shape[0], -1)
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


def test(epoch, model, decoder, gen_with_cond=False):
    model.eval()
    IMAGE_COUNT = 16 * 16
    with torch.no_grad():
        pred_x = torch.randn(IMAGE_COUNT, 16 * 16 * 4).to(DEVICE)
        cond = torch.arange(IMAGE_COUNT).long().to(DEVICE) % 10
        t = torch.ones(IMAGE_COUNT, 1).to(DEVICE)
        STEPS = 100
        for i in range(STEPS):
            pred = model(pred_x, t, cond if gen_with_cond else None)
            pred_x = pred_x + pred * 1 / STEPS
            t = t - 1 / STEPS
    pred_x = decoder(pred_x.reshape(IMAGE_COUNT, 4, 16, 16))
    # save image as single grid
    pred_x = pred_x.reshape(16, 16, 3, 32, 32).permute(0, 3, 1, 4, 2) * 0.5 + 0.5
    pred_x = pred_x.reshape(16 * 32, 16 * 32, 3).cpu().numpy()
    pred_x = (pred_x * 255).clip(0, 255).astype(np.uint8)
    pred_x = Image.fromarray(pred_x)
    pred_x.save(f"./result/gen-{epoch}.png")


if __name__ == "__main__":
    os.makedirs("./result", exist_ok=True)
    DEVICE = "mps"
    CLASSES = 10
    EPOCHS = 100

    encoder = Encoder().requires_grad_(False).eval().to(DEVICE)
    decoder = Decoder().requires_grad_(False).eval().to(DEVICE)
    encoder.load_state_dict(torch.load("enc.pth", weights_only=True))
    decoder.load_state_dict(torch.load("dec.pth", weights_only=True))

    model = Gen(
        input_dim=16 * 16 * 4,
        hidden_dim=1536,
        classes=CLASSES,
        class_embed_dim=128,
    ).to(DEVICE)
    print(sum(p.numel() for p in model.parameters()) / 1e6)
    optimizer = AdamWScheduleFree(
        model.parameters(), 5e-4, (0.9, 0.98), weight_decay=0.01, warmup_steps=100
    )
    loss = nn.MSELoss()
    transform = trns.Compose(
        [
            trns.RandomHorizontalFlip(),
            trns.ToTensor(),
            trns.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = [
        CIFAR10("./data", download=True, transform=transform)
        for _ in range(1)
    ]
    dataset = ConcatDataset(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    for i in trange(EPOCHS):
        test(i, model, decoder, gen_with_cond=bool(CLASSES))
        train(
            i,
            model,
            encoder,
            optimizer,
            loss,
            dataloader,
            train_with_cond=bool(CLASSES),
        )
    test(i, model, decoder, gen_with_cond=bool(CLASSES))
