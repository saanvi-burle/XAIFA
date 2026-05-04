import torch
from torchvision import datasets, transforms
from models.cnn_model import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN().to(device)
model.load_state_dict(torch.load("models/mnist_model.pth"))
model.eval()

data = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())

failures = []

for i in range(len(data)):
    img, lbl = data[i]
    inp = img.unsqueeze(0).to(device)

    pred = model(inp).argmax().item()

    if pred != lbl:
        failures.append((img, lbl, pred))

torch.save(failures, "data/failures.pt")
print("Failures:", len(failures))