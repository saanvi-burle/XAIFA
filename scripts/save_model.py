import sys
import os

# 🔥 FIX PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.cnn_model import SimpleCNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()

train = datasets.MNIST('./data', train=True, download=True, transform=transform)
loader = DataLoader(train, batch_size=64, shuffle=True)

model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    for img, lbl in loader:
        img, lbl = img.to(device), lbl.to(device)

        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, lbl)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} done")

# 🔥 SAVE MODEL (root folder)
torch.save(model.state_dict(), "mnist_model.pth")

print("✅ Model saved at mnist_model.pth")