import cv2
import os

INPUT_DIR = "../dataset/raw"
OUTPUT_DIR = "../dataset/resized"
TARGET_SIZE = (256, 256)
QUALITY = 85

dice_structure = {
    "d4": 4,
    "d6": 6,
    "d8": 8,
    "d10": 10,
    "d12": 12,
    "d20": 20
}
sets_count = 5
images_per_face = 4

# Sort files for consistent processing
all_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

if len(all_files) != sets_count * sum(dice_structure.values()) * images_per_face:
    raise ValueError("Number of images doesn't match expected count (1200)")

os.makedirs(OUTPUT_DIR, exist_ok=True)

index = 0
for set_id in range(1, sets_count + 1):
    for dice_type, face_count in dice_structure.items():
        for face_value in range(1, face_count + 1):
            for img_id in range(1, images_per_face + 1):
                filename = all_files[index]
                img_path = os.path.join(INPUT_DIR, filename)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Skipping unreadable image: {filename}")
                    index += 1
                    continue

                resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

                new_name = f"{dice_type}_{face_value}_set{set_id}_{img_id:03}.jpg"
                output_path = os.path.join(OUTPUT_DIR, new_name)
                cv2.imwrite(output_path, resized, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])

                index += 1
