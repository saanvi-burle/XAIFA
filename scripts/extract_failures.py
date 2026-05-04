import torch
from torchvision import datasets, transforms
from models.cnn_model import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN().to(device)
model.load_state_dict(torch.load("models/mnist_model.pth"))
model.eval()

dataset = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())

failures = []

for i in range(len(dataset)):
    image, label = dataset[i]
    image = image.unsqueeze(0).to(device)

    pred = model(image).argmax().item()

    if pred != label:
        failures.append((image.cpu(), label, pred))

torch.save(failures, "data/failures.pt")
print("Failures saved:", len(failures))