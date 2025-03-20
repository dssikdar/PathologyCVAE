import os
from torch.utils.data import Dataset
from PIL import Image

class BreastCancerDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Ensure 3-channel RGB

        if self.transform:
            image = self.transform(image)

        return image, 0  # Dummy label (not used in VAE)
