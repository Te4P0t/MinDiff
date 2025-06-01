import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from accelerate.utils import set_seed

from cifar_vae import Decoder
from cifar_dm import Gen

@torch.inference_mode()
def generate(model, decoder, num_samples, output_dir, batch_size=100):
    model.eval()
    decoder.eval()
    for i in tqdm(range(num_samples // batch_size)):
        x = torch.randn(batch_size, 8, 8, 8).to(DEVICE)
        cond = torch.arange(i * batch_size, (i + 1) * batch_size).long().to(DEVICE) % CLASSES
        t = torch.ones(batch_size, 1).to(DEVICE)
        STEPS = 100

        for j in range(STEPS):
            pred = model(x, t, cond)
            x = x + pred * (1 / STEPS)
            t = t - (1 / STEPS)

        x = decoder(x.reshape(batch_size, 8, 8, 8))
        x = x.reshape(batch_size, 3, 32, 32).permute(0, 2, 3, 1) * 0.5 + 0.5
        x = x.cpu().numpy() * 255
        x = x.clip(0, 255).astype(np.uint8)
        for j in range(batch_size):
            Image.fromarray(x[j]).save(f"{output_dir}/sample_{i*batch_size+j}.png")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gen-path", type=str, default="cifar-gen-4900.pth")
    parser.add_argument("--decoder-path", type=str, default="dec.pth")
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--output-dir", type=str, default="cifar-result")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    DEVICE = "cuda"
    CLASSES = 10
    
    decoder = Decoder(hidden_dim=(32, 64, 128), latent_dim=8)
    decoder = decoder.requires_grad_(False).eval().to(DEVICE)
    decoder.load_state_dict(torch.load(args.decoder_path, weights_only=True))

    model = Gen(
        input_dim=8, hidden_dim=256, classes=CLASSES, class_embed_dim=256, blocks=8
    )
    model = model.requires_grad_(False).eval().to(DEVICE)
    model.load_state_dict(torch.load(args.gen_path, weights_only=True))

    generate(model, decoder, args.num_samples, args.output_dir)