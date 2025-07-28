import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from .transforms import get_png_transform

class PngSpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform if transform else get_png_transform()

        self.samples = []
        self.class_to_idx = {
            class_name: idx
            for idx, class_name in enumerate(sorted(os.listdir(root_dir)))
            if os.path.isdir(os.path.join(root_dir, class_name))
        }

        for class_name, class_idx in self.class_to_idx.items():
            class_folder = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_folder):
                if fname.endswith(".png"):
                    self.samples.append((os.path.join(class_folder, fname), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("L")  # Grayscale
        if self.transform:
            image = self.transform(image)
        return image, label

def get_dataloader(dataset_type, split, root, batch_size):
    if dataset_type == "png":
        dataset = PngSpectrogramDataset(os.path.join(root, "png", split))
    else:
        raise ValueError("Unsupported dataset type")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=4,
        pin_memory=True,
    )
