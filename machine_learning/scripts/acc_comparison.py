import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from machine_learning.scripts.dataset import DiceDataset

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

def load_model(model_path, num_classes):
    model = DiceCNN(num_classes)
    model.load_state_dict(torch.load(model_path))
    return model

def evaluate_model(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    correct = total = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = correct / total
    cm = confusion_matrix(y_true, y_pred)
    return accuracy, cm

def is_valid_dice_type(dice_type):
    pattern = r'^d(4|6|8|10|12|20)$'
    return bool(re.match(pattern, dice_type.lower())) or dice_type.lower() == "type"

def main():
    dice_classes = {
        'd4': 4, 'd6': 6, 'd8': 8, 'd10': 10, 'd12': 12, 'd20': 20, 'type': 6
    }

    while True:
        dice_type = input("Enter dice type (d4/d6/d8/d10/d12/d20) or 'type': ").lower()
        if is_valid_dice_type(dice_type):
            break
        print("Invalid dice type.")

    num_classes = dice_classes[dice_type]

    # Setup paths
    if dice_type == "type":
        model1_path = os.path.join('../models', 'type_classifier.pth')
        model2_path = os.path.join('../models_push', 'type_classifier.pth')
    else:
        model1_path = os.path.join('../models', f'{dice_type}_classifier.pth')
        model2_path = os.path.join('../models_push', f'{dice_type}_classifier.pth')

    if not (os.path.exists(model1_path) and os.path.exists(model2_path)):
        print(f'Error: Could not find both model files for {dice_type}')
        return

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    print("Loading models...")
    model1 = load_model(model1_path, num_classes).to(device)
    model2 = load_model(model2_path, num_classes).to(device)

    # Prepare dataset
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    print("Loading dataset...")
    if dice_type == "type":
        dataset = DiceDataset("../dataset/augmented", dice_type=dice_type, transform=transform, label_type="type")
    else:
        dataset = DiceDataset("../dataset/augmented", dice_type=dice_type, transform=transform)

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Evaluate both models
    print("Evaluating models...")
    acc1, cm1 = evaluate_model(model1, test_loader, device)
    acc2, cm2 = evaluate_model(model2, test_loader, device)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Create labels for the confusion matrices
    if dice_type == "type":
        labels = ["d4", "d6", "d8", "d10", "d12", "d20"]
    else:
        labels = [str(i + 1) for i in range(num_classes)]

    # Plot confusion matrices
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=labels, yticklabels=labels)
    ax1.set_title(f'Tested model (Accuracy: {acc1:.2f})')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')

    sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=labels, yticklabels=labels)
    ax2.set_title(f'Pushed model (Accuracy: {acc2:.2f})')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')

    plt.suptitle(f'Confusion Matrices Comparison for {dice_type.upper()} Models')
    plt.tight_layout()
    plt.show()

    # Print accuracy comparison
    print(f"\nAccuracy comparison for {dice_type}:")
    print(f"Model 1 (models/): {acc1:.4f}")
    print(f"Model 2 (models_push/): {acc2:.4f}")
    print(f"Difference: {abs(acc1 - acc2):.4f}")

if __name__ == '__main__':
    main()
