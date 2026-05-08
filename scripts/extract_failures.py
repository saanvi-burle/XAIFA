import sys
from pathlib import Path

# ADD PROJECT ROOT
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.cnn_model import SimpleCNN

# =========================
# DEVICE
# =========================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =========================
# LOAD MODEL
# =========================
model = SimpleCNN().to(device)

model.load_state_dict(
    torch.load(
        ROOT_DIR / "models" / "mnist_model.pth"
    )
)

model.eval()

# =========================
# DATASET
# =========================
transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    ROOT_DIR / "data",
    train=False,
    download=True,
    transform=transform
)

loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)

# =========================
# EXTRACT FAILURES
# =========================
failures = []

with torch.no_grad():

    for img, lbl in loader:

        img = img.to(device)

        output = model(img)

        pred = output.argmax(dim=1)

        if pred.item() != lbl.item():

            failures.append(
                (
                    img.cpu(),
                    lbl.item(),
                    pred.item()
                )
            )

# =========================
# SAVE FAILURES
# =========================
data_dir = ROOT_DIR / "data"

data_dir.mkdir(exist_ok=True)

torch.save(
    failures,
    data_dir / "failures.pt"
)

print(f"\nSAVED {len(failures)} FAILURES")

print(
    f" FILE: {data_dir / 'failures.pt'}"
)