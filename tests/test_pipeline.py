import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Ensure src/ is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_loader import NpySpectrogramDataset, PngSpectrogramDataset, CropSpectrogramPlot
from model import GravitySpyCNN


def get_dataset(dataset_type, split, root):
    transform = transforms.Compose([
        CropSpectrogramPlot(crop_box=(70, 50, 730, 570)) if dataset_type == "png" else lambda x: x,
        transforms.Resize((128, 128)),
        transforms.ToTensor() if dataset_type == "png" else lambda x: x, 
    ])

    if dataset_type == "png":
        dataset = PngSpectrogramDataset(f"{root}/png/{split}", transform=transform)
    elif dataset_type == "npy":
        dataset = NpySpectrogramDataset(f"{root}/npy/{split}", transform=transform)
    else:
        raise ValueError("Unsupported dataset type")
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, choices=["png", "npy"], required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--root", type=str, default="./data")
    args = parser.parse_args()

    dataset = get_dataset(args.dataset_type, args.split, args.root)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = GravitySpyCNN(input_shape=(1, 128, 128), num_classes=7)
    model.eval()

    for inputs, labels in dataloader:
        print(f"Input shape: {inputs.shape}")  # [B, 1, 128, 128]
        outputs = model(inputs)
        print(f"Output shape: {outputs.shape}")  # [B, 7]
        break


if __name__ == "__main__":
    main()
