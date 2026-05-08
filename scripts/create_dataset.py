import sys
import os

# 🔥 Fix import path (same as model script)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import zipfile
from torchvision import datasets

# -------------------------
# CONFIG
# -------------------------
OUTPUT_FOLDER = "mnist_images"
ZIP_NAME = "mnist_dataset.zip"

# -------------------------
# LOAD MNIST
# -------------------------
mnist = datasets.MNIST('./data', train=False, download=True)

# -------------------------
# CREATE IMAGE FOLDER
# -------------------------
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("Saving images...")

for i, (img, label) in enumerate(mnist):

    label_folder = os.path.join(OUTPUT_FOLDER, str(label))
    os.makedirs(label_folder, exist_ok=True)

    img_path = os.path.join(label_folder, f"{i}.png")
    img.save(img_path)

print("✅ Images saved")

# -------------------------
# CREATE ZIP FILE
# -------------------------
print("Creating ZIP...")

with zipfile.ZipFile(ZIP_NAME, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(OUTPUT_FOLDER):
        for file in files:
            full_path = os.path.join(root, file)

            # 🔥 Important: preserve relative structure
            arcname = os.path.relpath(full_path, OUTPUT_FOLDER)

            zipf.write(full_path, arcname)

print(f"✅ Dataset ZIP created: {ZIP_NAME}")