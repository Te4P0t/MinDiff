"""
Minimal Example for Iterative alpha Belnding: a Minimalist Deterministic Diffusion Model
https://arxiv.org/abs/2305.03486
"""

import os
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from schedulefree.adamw_schedulefree import AdamWScheduleFree
from torchvision.transforms import transforms as trns
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
from PIL import Image
from tqdm import tqdm, trange


class DownSample(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, 2, 2)


class UpSample(nn.Module):
    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode="nearest")


def standardize(x: torch.Tensor, dim=-1):
    return (x - x.mean(dim=dim, keepdim=True)) / x.std(dim=dim, keepdim=True)


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim=3,
        hidden_dim=(32, 64),
        latent_dim=4,
    ):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            *(
                [nn.Sequential(nn.Conv2d(input_dim, hidden_dim[0], 3, 1, 1), nn.Mish())]
                + [
                    nn.Sequential(
                        nn.GroupNorm(8, hid1),
                        nn.Conv2d(hid1, hid2, 3, 1, 1),
                        nn.Mish(),
                        DownSample(),
                        nn.GroupNorm(8, hid2),
                        nn.Conv2d(hid2, hid2, 3, 1, 1),
                        nn.Mish(),
                    )
                    for hid1, hid2 in zip(hidden_dim[:-1], hidden_dim[1:])
                ]
                + [
                    nn.Sequential(
                        nn.GroupNorm(8, hidden_dim[-1]),
                        nn.Conv2d(hidden_dim[-1], latent_dim, 3, 1, 1),
                    )
                ]
            )
        )

    def forward(self, x):
        latent = self.model(x)
        return standardize(latent, dim=[-1, -2, -3])


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim=3,
        hidden_dim=(32, 64),
        latent_dim=4,
    ):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            *(
                [
                    nn.Sequential(
                        nn.Conv2d(latent_dim, hidden_dim[-1], 3, 1, 1), nn.Mish()
                    )
                ]
                + [
                    nn.Sequential(
                        nn.GroupNorm(8, hid2),
                        nn.Conv2d(hid2, hid2, 3, 1, 1),
                        nn.Mish(),
                        UpSample(),
                        nn.GroupNorm(8, hid2),
                        nn.Conv2d(hid2, hid1, 3, 1, 1),
                        nn.Mish(),
                    )
                    for hid1, hid2 in reversed(
                        list(zip(hidden_dim[:-1], hidden_dim[1:]))
                    )
                ]
                + [
                    nn.Sequential(
                        nn.GroupNorm(8, hidden_dim[0]),
                        nn.Conv2d(hidden_dim[0], input_dim, 3, 1, 1),
                    )
                ]
            )
        )

    def forward(self, x):
        return self.model(x)


def train(epoch, encoder, decoder, optimizer, lr_sch, loss, dataloader):
    encoder.train()
    decoder.train()
    ema_loss = 0
    with tqdm(dataloader, desc=f"Epoch {epoch}", smoothing=0.01) as pbar:
        for i, (x, cond) in enumerate(dataloader):
            x = x.to(DEVICE)
            optimizer.zero_grad()
            latent = encoder(x)
            y = decoder(latent)
            l = loss(y, x) + (
                1 - ssim(y * 0.5 + 0.5, x * 0.5 + 0.5, win_size=7, data_range=1.0)
            )
            l.backward()
            optimizer.step()
            lr_sch.step()

            ema_decay = min(0.99, i / 100)
            ema_loss = ema_decay * ema_loss + (1 - ema_decay) * l.item()

            pbar.update(1)
            pbar.set_postfix({"loss": ema_loss})
    torch.save(encoder.state_dict(), "enc.pth")
    torch.save(decoder.state_dict(), "dec.pth")


def test(epoch, encoder, decoder, loss, dataloader):
    encoder.eval()
    decoder.eval()
    IMAGE_COUNT = 16
    input_images = None
    output_images = None
    with torch.no_grad():
        for i, (x, cond) in enumerate(dataloader):
            x = x.to(DEVICE)
            latent = encoder(x)
            y = decoder(latent)
            if input_images is None:
                input_images = x
                output_images = y
            else:
                input_images = torch.cat([input_images, x], dim=0)
                output_images = torch.cat([output_images, y], dim=0)
            if input_images.shape[0] >= IMAGE_COUNT**2:
                break
    input_images = input_images[: IMAGE_COUNT**2].cpu()
    output_images = output_images[: IMAGE_COUNT**2].cpu()
    input_images = (
        input_images.reshape(IMAGE_COUNT, IMAGE_COUNT, 3, 32, 32)
        .permute(0, 3, 1, 4, 2)
        .reshape(IMAGE_COUNT * 32, IMAGE_COUNT * 32, 3)
    ) * 0.5 + 0.5
    output_images = (
        output_images.reshape(IMAGE_COUNT, IMAGE_COUNT, 3, 32, 32)
        .permute(0, 3, 1, 4, 2)
        .reshape(IMAGE_COUNT * 32, IMAGE_COUNT * 32, 3)
    ) * 0.5 + 0.5
    input_images = Image.fromarray(
        (input_images.numpy() * 255).clip(0, 255).astype(np.uint8)
    )
    output_images = Image.fromarray(
        (output_images.numpy() * 255).clip(0, 255).astype(np.uint8)
    )
    input_images.save(f"./ae-result/{epoch}-input.png")
    output_images.save(f"./ae-result/{epoch}-output.png")


if __name__ == "__main__":
    os.makedirs("./ae-result", exist_ok=True)
    DEVICE = "cuda"
    EPOCHS = 1000
    transform = trns.Compose(
        [
            trns.RandomHorizontalFlip(),
            trns.ToTensor(),
            trns.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = CIFAR10("./data", download=True, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    test_dataset = CIFAR10("./data", train=False, transform=transform)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
    )

    encoder = Encoder(hidden_dim=(32, 64, 128), latent_dim=8).to(DEVICE)
    decoder = Decoder(hidden_dim=(32, 64, 128), latent_dim=8).to(DEVICE)
    print(sum(p.numel() for p in encoder.parameters()) / 1e6)
    print(sum(p.numel() for p in decoder.parameters()) / 1e6)
    optimizer = AdamWScheduleFree(
        chain(encoder.parameters(), decoder.parameters()),
        5e-3,
        (0.9, 0.995),
        weight_decay=0.01,
        warmup_steps=100,
    )
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, EPOCHS * len(dataloader), 5e-5
    )
    loss = nn.MSELoss()

    for i in trange(EPOCHS):
        test(i, encoder, decoder, loss, test_dataloader)
        optimizer.train()
        train(i, encoder, decoder, optimizer, lr_scheduler, loss, dataloader)
        optimizer.eval()
    test(i, encoder, decoder, loss, test_dataloader)
