import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import re
from io import BytesIO

app = FastAPI()

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

# Helpers
def is_valid_dice_type(dice_type):
    return re.match(r'^d(4|6|8|10|12|20)$', dice_type.lower())

def load_model(model_dir, dice_type, num_classes):
    model_path = os.path.join(model_dir, f"{dice_type}_classifier.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    model = DiceCNN(num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Endpoint
@app.post("/predict/")
async def predict(
    image: UploadFile = File(...),
    dice_type: str = Form(...),
    model_source: str = Form(...)  # "models" or "models_push"
):
    dice_classes = {'d4': 4, 'd6': 6, 'd8': 8, 'd10': 10, 'd12': 12, 'd20': 20}
    
    if not is_valid_dice_type(dice_type):
        raise HTTPException(status_code=400, detail="Invalid dice type.")

    if model_source not in ["models", "models_push"]:
        raise HTTPException(status_code=400, detail="Invalid model source.")

    try:
        image_bytes = await image.read()
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        input_tensor = transform_image(img)

        model = load_model(model_source, dice_type, dice_classes[dice_type])
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item() + 1  # Classes are 1-based

        return JSONResponse(content={"predicted_class": predicted_class})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))