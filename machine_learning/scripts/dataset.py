import os
from PIL import Image
from torch.utils.data import Dataset

class DiceDataset(Dataset):
    DICE_TYPE_TO_IDX = {
        "d4": 0,
        "d6": 1,
        "d8": 2,
        "d10": 3,
        "d12": 4,
        "d20": 5
    }

    def __init__(self, root_dir, dice_type=None, transform=None, label_type="value"):
        self.images = []
        self.labels = []
        self.transform = transform
        self.label_type = label_type.lower()
        self.dice_type_filter = dice_type.lower() if dice_type and dice_type != "type" else None

        for fname in os.listdir(root_dir):
            if not fname.lower().endswith(".jpg"):
                continue

            parts = fname.lower().split("_")
            if len(parts) < 4:
                print(f"Skipping malformed filename: {fname}")
                continue

            dice_type_from_file = parts[0]
            if self.dice_type_filter and dice_type_from_file != self.dice_type_filter:
                continue

            try:
                if self.label_type == "value":
                    face_value = int(parts[1])
                    label = face_value - 1  # zero-based
                elif self.label_type == "type":
                    label = self.DICE_TYPE_TO_IDX[dice_type_from_file]
                else:
                    raise ValueError(f"Invalid label_type: {self.label_type}")

                self.images.append(os.path.join(root_dir, fname))
                self.labels.append(label)

            except Exception as e:
                print(f"Error parsing {fname}: {e}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
