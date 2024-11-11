import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image

from cifar_vae import Encoder, Decoder


if __name__ == "__main__":
    DEVICE = "cuda"
    encoder = decoder = None
    encoder = Encoder(hidden_dim=(32, 64, 128), latent_dim=8)
    decoder = Decoder(hidden_dim=(32, 64, 128), latent_dim=8)
    encoder = encoder.requires_grad_(False).eval().to(DEVICE)
    decoder = decoder.requires_grad_(False).eval().to(DEVICE)
    encoder.load_state_dict(torch.load("enc.pth", weights_only=True))
    decoder.load_state_dict(torch.load("dec.pth", weights_only=True))

    test_image = Image.open("image/test/test-ae-inp.png")
    test_input = to_tensor(test_image).unsqueeze(0).to(DEVICE) * 2 - 1
    latent = encoder(test_input)
    output = decoder(latent) * 0.5 + 0.5
    output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output = (output * 255).clip(0, 255).astype(np.uint8)
    output = Image.fromarray(output)
    output.save("image/test/test-ae-out.png")