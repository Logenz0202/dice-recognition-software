import sys
import os
import json

if len(sys.argv) < 2:
    print("No image path provided")
    sys.exit(1)

image_path = sys.argv[1]
filename = os.path.basename(image_path)

result = {
    "face_value": 15,
    "dice_type": "d20"
}
print(json.dumps(result))