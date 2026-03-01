# ============================================
# collect_failures.py
# Purpose:
# Collect misclassified MNIST samples
# (First XAIFA step)
# ============================================

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

from models.cnn_model import SimpleCNN


# ---------- DEVICE ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- LOAD TEST DATA ----------
transform = transforms.ToTensor()

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=False,
    transform=transform
)

test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


# ---------- LOAD TRAINED MODEL ----------
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("models/mnist_model.pth"))
model.eval()


# ---------- FAILURE COLLECTION ----------
failed_images = []
failed_true_labels = []
failed_pred_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        # find wrong predictions
        mismatch = preds != labels

        failed_images.append(images[mismatch].cpu())
        failed_true_labels.append(labels[mismatch].cpu())
        failed_pred_labels.append(preds[mismatch].cpu())


# ---------- COMBINE ALL FAILURES ----------
failed_images = torch.cat(failed_images)
failed_true_labels = torch.cat(failed_true_labels)
failed_pred_labels = torch.cat(failed_pred_labels)

print("Total failures collected:", len(failed_images))


# ---------- SAVE FAILURE DATA ----------
np.save("outputs/failed_images.npy", failed_images.numpy())
np.save("outputs/failed_true.npy", failed_true_labels.numpy())
np.save("outputs/failed_pred.npy", failed_pred_labels.numpy())

print("Failure dataset saved in outputs/")