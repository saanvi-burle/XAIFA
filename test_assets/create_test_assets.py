from pathlib import Path
import csv
import zipfile

from PIL import Image, ImageDraw
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATASET_DIR = ROOT / "datasets"
LABEL_DIR = ROOT / "labels"


class TinyRgbClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc(self.pool(x).flatten(1))


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, 3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(26 * 26 * 16, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def make_dirs():
    for path in [MODEL_DIR, DATASET_DIR, LABEL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def create_models():
    tiny = TinyRgbClassifier().eval()
    scripted = torch.jit.trace(tiny, torch.zeros(1, 3, 224, 224))
    scripted.save(str(MODEL_DIR / "tiny_rgb_classifier.pt"))

    torch.save(SimpleCNN(num_classes=10).state_dict(), MODEL_DIR / "simple_cnn_mnist_like.pth")


def image(path, color, text, size=(64, 64)):
    img = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(img)
    draw.text((8, 24), text, fill=(255, 255, 255))
    img.save(path)


def create_folder_dataset():
    root = DATASET_DIR / "folder_dataset"
    for class_name, color in [("cat", (220, 45, 45)), ("dog", (45, 80, 220))]:
        (root / class_name).mkdir(parents=True, exist_ok=True)
        for idx in range(1, 4):
            image(root / class_name / f"{class_name}_{idx}.png", color, class_name[:3])

    zip_path = DATASET_DIR / "folder_dataset_cat_dog.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in root.rglob("*.png"):
            zf.write(path, path.relative_to(root).as_posix())


def create_csv_dataset():
    root = DATASET_DIR / "csv_dataset"
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, color in [("cat", (220, 45, 45)), ("dog", (45, 80, 220))]:
        for idx in range(1, 3):
            name = f"{label}_{idx}.png"
            image(image_dir / name, color, label[:3])
            rows.append({"image_path": f"images/{name}", "true_label": label})

    csv_path = root / "labels.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "true_label"])
        writer.writeheader()
        writer.writerows(rows)

    zip_path = DATASET_DIR / "csv_dataset_cat_dog.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in root.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(root).as_posix())


def create_labels():
    (LABEL_DIR / "cat_dog_labels.txt").write_text("0: cat\n1: dog\n", encoding="utf-8")
    (LABEL_DIR / "mnist_labels.txt").write_text(
        "\n".join(f"{idx}: {idx}" for idx in range(10)) + "\n",
        encoding="utf-8",
    )


def main():
    make_dirs()
    create_models()
    create_folder_dataset()
    create_csv_dataset()
    create_labels()
    print(f"Created test assets in {ROOT}")


if __name__ == "__main__":
    main()
