import os
import argparse
import ffmpeg
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from schedulefree.adamw_schedulefree import AdamWScheduleFree
from torchvision.transforms import transforms as trns
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm, trange

from cifar_vae import Encoder, Decoder


class ConvAdaNorm(nn.Module):
    def __init__(self, dim):
        super(ConvAdaNorm, self).__init__()
        self.inp = nn.Sequential(
            nn.GroupNorm(8, dim),
            nn.Conv2d(dim, dim, 3, 1, 1),
        )
        self.adaln = nn.Linear(dim, dim * 2)
        self.out = nn.Sequential(
            nn.Mish(),
            nn.GroupNorm(8, dim),
            nn.Conv2d(dim, dim, 3, 1, 1),
        )
        nn.init.constant_(self.out[-1].weight, 0)
        nn.init.constant_(self.out[-1].bias, 0)

    def forward(self, x, t):
        scale, shift = self.adaln(t).chunk(2, dim=-1)
        scale = scale[..., None, None]
        shift = shift[..., None, None]
        return x + self.out(self.inp(x) * scale + shift)


class Gen(nn.Module):
    def __init__(
        self,
        input_dim=3,
        hidden_dim=1024,
        blocks=3,
        classes=None,
        class_embed_dim=128,
    ):
        super(Gen, self).__init__()
        if classes is not None:
            self.class_embed = nn.Sequential(
                nn.Embedding(classes, class_embed_dim),
                nn.LayerNorm(class_embed_dim),
            )
        self.time_embed = nn.Sequential(
            nn.Linear(1 + bool(classes) * class_embed_dim, hidden_dim),
            nn.Mish(),
        )

        self.input_proj = nn.Conv2d(input_dim, hidden_dim, 3, 1, 1)
        self.model = nn.ModuleList([ConvAdaNorm(hidden_dim) for _ in range(blocks)])
        self.output_proj = nn.Sequential(
            nn.Mish(),
            nn.GroupNorm(8, hidden_dim),
            nn.Conv2d(hidden_dim, input_dim, 3, 1, 1),
        )
        nn.init.constant_(self.output_proj[-1].weight, 0)
        nn.init.constant_(self.output_proj[-1].bias, 0)

    def forward(self, x, t, cond=None):
        if cond is not None:
            t = torch.cat([t.reshape(-1, 1), self.class_embed(cond)], dim=-1)
        t = self.time_embed(t)
        x = self.input_proj(x)
        for layer in self.model:
            x = layer(x, t)
        return self.output_proj(x)


def train(
    epoch, model, encoder, optimizer, lr_sch, loss, dataloader, train_with_cond=False
):
    model.train()
    ema_loss = 0
    with tqdm(dataloader, desc=f"Epoch {epoch}", smoothing=0.01) as pbar:
        for i, (x, cond) in enumerate(dataloader):
            if encoder is not None:
                x = encoder(x.to(DEVICE))
            else:
                x = x.to(DEVICE)
            cond = cond.to(DEVICE).long()

            ## Flow Matching parameterization
            ## x0 ~ Image, x1 ~ N, t ~ [0, 1], xt = x0 + (x1 - x0) * t
            ## Model(xt, t) = x0 - x1
            ## xt-dt = xt + Model(xt, t) * dt
            t = torch.rand(x.shape[0], *[1] * (x.dim() - 1)).to(DEVICE).float()
            eps = torch.randn_like(x)
            xt = x * (1 - t) + eps * t
            target = x - eps

            optimizer.zero_grad()
            y = model(xt, t, cond if train_with_cond else None)
            # eps_pred = xt - (1 - t) * y
            # x0_pred = xt + t * y

            l = loss(y, target)# + loss(eps_pred, eps) + loss(x0_pred, x) + loss(xt, xt)
            l.backward()
            optimizer.step()
            lr_sch.step()

            ema_decay = min(0.99, i / 100)
            ema_loss = ema_decay * ema_loss + (1 - ema_decay) * l.item()

            pbar.update(1)
            pbar.set_postfix({"loss": ema_loss})
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"cifar-gen-{epoch}.pth")


def test(epoch, model, decoder, gen_with_cond=False):
    rng_state = torch.get_rng_state()
    torch.manual_seed(0)
    model.eval()
    IMAGE_COUNT = 16 * 16
    with torch.no_grad():
        if decoder is not None:
            pred_x = torch.randn(IMAGE_COUNT, 8, 8, 8).to(DEVICE)
        else:
            pred_x = torch.randn(IMAGE_COUNT, 3, 32, 32).to(DEVICE)
        cond = torch.arange(IMAGE_COUNT).long().to(DEVICE) % 10
        t = torch.ones(IMAGE_COUNT, 1).to(DEVICE)
        STEPS = 100
        for i in range(STEPS):
            pred = model(pred_x, t, cond if gen_with_cond else None)
            pred_x = pred_x + pred * (1 / STEPS)
            t = t - (1 / STEPS)
    if decoder is not None:
        pred_x = decoder(pred_x.reshape(IMAGE_COUNT, 8, 8, 8))
    else:
        pred_x = pred_x.reshape(IMAGE_COUNT, 3, 32, 32)
    # save image as single grid
    pred_x = pred_x.reshape(16, 16, 3, 32, 32).permute(0, 3, 1, 4, 2) * 0.5 + 0.5
    pred_x = pred_x.reshape(16 * 32, 16 * 32, 3).cpu().numpy()
    pred_x = (pred_x * 255).clip(0, 255).astype(np.uint8)
    pred_x = Image.fromarray(pred_x)
    pred_x.save(f"./cifar-result/gen-{epoch}.png")
    torch.set_rng_state(rng_state)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="./ckpts/cifar-gen")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    DEVICE = "cuda"
    CLASSES = 10
    EPOCHS = args.epochs
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
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True,
    )

    encoder = decoder = None
    encoder = Encoder(hidden_dim=(32, 64, 128), latent_dim=8)
    decoder = Decoder(hidden_dim=(32, 64, 128), latent_dim=8)
    encoder = encoder.requires_grad_(False).eval().to(DEVICE)
    decoder = decoder.requires_grad_(False).eval().to(DEVICE)
    encoder.load_state_dict(torch.load("enc.pth", weights_only=True))
    decoder.load_state_dict(torch.load("dec.pth", weights_only=True))

    model = Gen(
        input_dim=8, hidden_dim=256, classes=CLASSES, class_embed_dim=256, blocks=8
    ).to(DEVICE)
    print(sum(p.numel() for p in model.parameters()) / 1e6)
    optimizer = AdamWScheduleFree(
        model.parameters(), 1e-3, (0.9, 0.995), weight_decay=0.01, warmup_steps=1000
    )
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, EPOCHS * len(dataloader), 1e-4
    )
    loss = nn.MSELoss()

    frame_idx = 0
    for i in trange(EPOCHS):
        if i % 10 == 0:
            test(frame_idx, model, decoder, gen_with_cond=bool(CLASSES))
            frame_idx += 1
        optimizer.train()
        train(
            i,
            model,
            encoder,
            optimizer,
            lr_scheduler,
            loss,
            dataloader,
            train_with_cond=bool(CLASSES),
        )
        optimizer.eval()
    test(i, model, decoder, gen_with_cond=bool(CLASSES))

    stream = ffmpeg.input(
        "./cifar-result/gen-%d.png", pattern_type="sequence", framerate=60
    )
    stream = ffmpeg.output(stream, "cifar-gen.mp4", crf=20, pix_fmt="yuv420p")
    ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
