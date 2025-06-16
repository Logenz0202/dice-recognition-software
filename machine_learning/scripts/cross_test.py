import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from dataset import DiceDataset

class DiceCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

dice_types = ["d4", "d6", "d8", "d10", "d12", "d20", "type"]
dice_classes = {"d4": 4, "d6": 6, "d8": 8, "d10": 10, "d12": 12, "d20": 20, "type": 6}
model_dir = "../models_push"
results = np.zeros((len(dice_types), len(dice_types)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def load_model(model_path, num_classes):
    model = DiceCNN(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def evaluate_model(model, loader, device):
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0

for j, dice_type in enumerate(dice_types):
    model_path = os.path.join(model_dir, f"{dice_type}_classifier.pth" if dice_type != "type" else "type_classifier.pth")
    if not os.path.exists(model_path):
        continue
    num_classes = dice_classes[dice_type]
    model = load_model(model_path, num_classes).to(device)
    for k, test_type in enumerate(dice_types):
        label_type = "type" if test_type == "type" else "value"
        dataset = DiceDataset("../dataset/augmented", dice_type=test_type, transform=transform, label_type=label_type)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
        acc = evaluate_model(model, loader, device)
        results[j, k] = acc

plt.figure(figsize=(12, 8))
yticklabels = [f"{t}" for t in dice_types]
sns.heatmap(results, annot=True, fmt=".2f", xticklabels=dice_types, yticklabels=yticklabels)
plt.xlabel("Zbiór testowy")
plt.ylabel("Model")
plt.title("Cross-test dokładności modeli (models_push)")
plt.tight_layout()
plt.show()
