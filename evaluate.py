import os
import argparse
import torch
import json
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from torchvision.datasets import CIFAR10
from torchvision import transforms
from tqdm import tqdm

def load_image(image_path):
    with open(image_path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
    return img

class GenDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        self.image_paths.sort()
        self.images = [load_image(image_path) for image_path in self.image_paths]
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-path", type=str, default="cifar-result/gen-4900.png")
    parser.add_argument("--cifar-path", type=str, default="data")
    parser.add_argument("--save-path", type=str, default="results.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gen_dataset = GenDataset(args.gen_path, transform=transforms.ToTensor())
    cifar_dataset = CIFAR10(root=args.cifar_path, train=True, download=True, transform=transforms.ToTensor())
    assert len(gen_dataset) == len(cifar_dataset), "Number of generated and real images must be the same"
    gen_dataloader = DataLoader(gen_dataset, batch_size=64, shuffle=False, num_workers=4)
    cifar_dataloader = DataLoader(cifar_dataset, batch_size=64, shuffle=False, num_workers=4)
    fid = FrechetInceptionDistance(normalize=True).to("cuda").set_dtype(torch.float32)
    for gen_images, cifar_images in tqdm(zip(gen_dataloader, cifar_dataloader), total=len(gen_dataloader)):
        fid.update(gen_images.to("cuda"), real=False)
        fid.update(cifar_images[0].to("cuda"), real=True)
    fid_result = fid.compute().item()
    print(f"FID: {fid_result:.2f}")
    results = {
        "fid": fid_result
    }
    save_path = Path(args.save_path)
    with open(save_path, "w") as f:
        json.dump(results, f)