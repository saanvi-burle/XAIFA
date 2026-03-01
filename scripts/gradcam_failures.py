# ============================================
# gradcam_failures.py
# Purpose:
# Generate Grad-CAM explanation for failures
# ============================================

import torch
import numpy as np
import matplotlib.pyplot as plt

from models.cnn_model import SimpleCNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- LOAD MODEL ----------
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("models/mnist_model.pth"))
model.eval()


# ---------- LOAD FAILURE DATA ----------
failed_images = np.load("outputs/failed_images.npy")
failed_true = np.load("outputs/failed_true.npy")
failed_pred = np.load("outputs/failed_pred.npy")


# pick one failure example
img = torch.tensor(failed_images[0]).unsqueeze(0).to(device)
img.requires_grad = True


# ---------- GRAD-CAM HOOK ----------
activations = []
gradients = []

def forward_hook(module, inp, out):
    activations.append(out)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])


hook_f = model.conv.register_forward_hook(forward_hook)
hook_b = model.conv.register_backward_hook(backward_hook)


# ---------- FORWARD ----------
output = model(img)
pred_class = output.argmax(dim=1)

model.zero_grad()
output[0, pred_class].backward()


# ---------- BUILD CAM ----------
grad = gradients[0][0]
act = activations[0][0]

weights = grad.mean(dim=(1, 2))

cam = torch.zeros(act.shape[1:], device=device)

for i, w in enumerate(weights):
    cam += w * act[i]

cam = torch.relu(cam)
cam = cam.cpu().detach().numpy()

# normalize
cam = (cam - cam.min()) / (cam.max() - cam.min())


# ---------- DISPLAY ----------
plt.figure(figsize=(6,3))

plt.subplot(1,2,1)
plt.imshow(img.cpu().detach().numpy()[0][0], cmap="gray")
plt.title(f"True:{failed_true[0]} Pred:{failed_pred[0]}")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(cam, cmap="jet")
plt.title("Grad-CAM")
plt.axis("off")

plt.tight_layout()
plt.show()


# remove hooks
hook_f.remove()
hook_b.remove()