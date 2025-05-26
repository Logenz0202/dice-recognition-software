import cv2
import os
import albumentations as A

# SETTINGS
INPUT_DIR = "../dataset/resized"
OUTPUT_DIR = "../dataset/augmented"
AUGS_PER_IMAGE = 10

# Create output directory if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Augmentation pipeline
transform = A.Compose([
    A.Rotate(limit=25, p=0.7),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.MotionBlur(blur_limit=3, p=0.3),
    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.3),
    A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
])

# Process all images
for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    path = os.path.join(INPUT_DIR, filename)
    image = cv2.imread(path)
    if image is None:
        continue

    name, ext = os.path.splitext(filename)

    for i in range(AUGS_PER_IMAGE):
        augmented = transform(image=image)['image']
        new_filename = f"{name}_aug{i+1}{ext}"
        cv2.imwrite(os.path.join(OUTPUT_DIR, new_filename), augmented)
