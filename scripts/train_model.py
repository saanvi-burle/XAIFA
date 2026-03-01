# ============================================
# train_model.py
# Purpose:
# Train a simple CNN on MNIST dataset
# and save the trained model.
# ============================================

# ---------- IMPORTS ----------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# import CNN model from models folder
from models.cnn_model import SimpleCNN


# ---------- DEVICE SETUP ----------
# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---------- DATA TRANSFORM ----------
# Convert images to tensor format
transform = transforms.ToTensor()


# ---------- LOAD DATASET ----------
# Training dataset
train_data = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Test dataset
test_data = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)


# ---------- DATA LOADERS ----------
# DataLoader feeds data in batches
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


# ---------- MODEL INITIALIZATION ----------
model = SimpleCNN().to(device)


# ---------- LOSS FUNCTION & OPTIMIZER ----------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ---------- TRAINING LOOP ----------
epochs = 5

for epoch in range(epochs):

    model.train()   # set model to training mode
    running_loss = 0.0

    for images, labels in train_loader:

        # move data to device (GPU/CPU)
        images = images.to(device)
        labels = labels.to(device)

        # clear previous gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(images)

        # calculate loss
        loss = criterion(outputs, labels)

        # backpropagation
        loss.backward()

        # update weights
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss:.4f}")


# ---------- SAVE TRAINED MODEL ----------
torch.save(model.state_dict(), "models/mnist_model.pth")
print("Model saved successfully -> models/mnist_model.pth")