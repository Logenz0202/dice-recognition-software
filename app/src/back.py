import sys
import os
sys.path.append(os.path.abspath('../../machine_learning/scripts'))
from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
from train_d6 import DiceCNN


app = Flask(__name__)

# Load model
model = DiceCNN()
model.load_state_dict(torch.load("../models/d6_classifier.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    img = Image.open(file.stream).convert("RGB")
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        pred = output.argmax(1).item() + 1  # 1-based label
    return jsonify({'prediction': pred})

if __name__ == '__main__':
    app.run(port=5000)