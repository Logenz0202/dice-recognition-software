import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import sys

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

def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def get_prediction_with_confidence(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)
        return predicted.item(), confidence.item()

def load_model(model_path, num_classes):
    try:
        model = DiceCNN(num_classes)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {str(e)}")
        return None

def process_image(image_path, confidence_threshold=0.7):
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")

        # First, classify dice type
        dice_type_model = load_model("../models_push/type_classifier.pth", num_classes=6)
        if dice_type_model is None:
            return {"error": "Could not load dice type classifier model"}

        input_tensor = transform_image(img)
        dice_types = ['d4', 'd6', 'd8', 'd10', 'd12', 'd20']
        type_pred, type_conf = get_prediction_with_confidence(dice_type_model, input_tensor)
        predicted_type = dice_types[type_pred]

        if type_conf < confidence_threshold:
            return {
                "warning": f"Low confidence ({type_conf:.2f}) in dice type prediction",
                "dice_type": predicted_type,
                "confidence": type_conf
            }

        # Then, recognize face value
        dice_classes = {'d4': 4, 'd6': 6, 'd8': 8, 'd10': 10, 'd12': 12, 'd20': 20}
        face_model = load_model(f"../models_push/{predicted_type}_classifier.pth",
                                dice_classes[predicted_type])

        if face_model is None:
            return {"error": f"Could not load face recognition model for {predicted_type}"}

        face_pred, face_conf = get_prediction_with_confidence(face_model, input_tensor)
        predicted_face = face_pred + 1  # Classes are 0-based, faces are 1-based

        result = {
            "dice_type": predicted_type,
            "face_value": predicted_face,
            "confidence": {
                "type_confidence": float(type_conf),
                "face_confidence": float(face_conf)
            }
        }

        # Add a warning if confidence is low
        if face_conf < confidence_threshold:
            result["warning"] = f"Low confidence ({face_conf:.2f}) in face value prediction"

        return result

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)

    result = process_image(sys.argv[1])
    print(json.dumps(result, indent=2))
