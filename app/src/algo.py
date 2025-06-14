import sys
import os

if len(sys.argv) < 2:
    print("No image path provided")
    sys.exit(1)

image_path = sys.argv[1]
filename = os.path.basename(image_path)

print(filename)
print(filename)