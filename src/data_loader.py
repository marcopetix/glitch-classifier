import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from gwpy.timeseries import TimeSeries
from scipy.signal import spectrogram

class NpySpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            self.class_to_idx[class_name] = idx
            for fname in os.listdir(class_dir):
                if fname.endswith(".npy"):
                    self.samples.append((os.path.join(class_dir, fname), idx))
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        spec = np.load(path).astype(np.float32)
        spec = torch.tensor(spec).unsqueeze(0)  # Add channel dim
        if self.transform:
            spec = self.transform(spec)
        return spec, label
    

class PngSpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            self.class_to_idx[class_name] = idx
            for fname in os.listdir(class_dir):
                if fname.endswith(".png"):
                    self.samples.append((os.path.join(class_dir, fname), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        return image, label