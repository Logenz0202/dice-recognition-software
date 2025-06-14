import os
import json
from main import process_image

test_folder = "../dataset/batch_test"

valid_extensions = [".jpg", ".jpeg", ".png"]

results = {}

for fname in os.listdir(test_folder):
    if not any(fname.lower().endswith(ext) for ext in valid_extensions):
        continue

    img_path = os.path.join(test_folder, fname)
    print(f"Processing {fname}...")
    result = process_image(img_path)
    results[fname] = result

with open("batch_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Finished. Results saved to batch_results.json.")
