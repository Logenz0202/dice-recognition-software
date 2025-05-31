import os
from PIL import Image
from torch.utils.data import Dataset

"""
This script defines a custom dataset class 
for loading images of dice faces from a specified directory.
"""

class DiceDataset(Dataset):
    def __init__(self, root_dir, dice_type=None, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        self.dice_type = dice_type.lower() if dice_type else None

        for fname in os.listdir(root_dir):
            if not fname.lower().endswith(".jpg"):
                continue
            if self.dice_type and not fname.lower().startswith(self.dice_type + "_"):
                continue

            parts = fname.lower().split("_")
            try:
                # [type]_[face]_set[1-5]_[001-004]_aug[1-10].jpg
                label = int(parts[1])
                self.images.append(os.path.join(root_dir, fname))
                self.labels.append(label)
            except Exception as e:
                print(f"Error parsing {fname}: {e}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx] - 1  # 0-based labels
        if self.transform:
            img = self.transform(img)
        return img, label
