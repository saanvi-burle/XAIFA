# ============================================
# extract_xai_features.py
# Purpose:
# Convert Grad-CAM outputs into feature vectors
# ============================================

import torch
import numpy as np
from models.cnn_model import SimpleCNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- LOAD MODEL ----------
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("models/mnist_model.pth"))
model.eval()


# ---------- LOAD FAILURES ----------
failed_images = np.load("outputs/failed_images.npy")


# ---------- HOOKS ----------
activations = []
gradients = []

def forward_hook(module, inp, out):
    activations.append(out)

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

hook_f = model.conv.register_forward_hook(forward_hook)
hook_b = model.conv.register_backward_hook(backward_hook)


# ---------- FEATURE STORAGE ----------
feature_list = []


# ---------- PROCESS FAILURES ----------
for idx in range(len(failed_images)):

    img = torch.tensor(failed_images[idx]).unsqueeze(0).to(device)
    img.requires_grad = True

    activations.clear()
    gradients.clear()

    # forward
    output = model(img)
    pred_class = output.argmax(dim=1)

    model.zero_grad()
    output[0, pred_class].backward()

    # build CAM
    grad = gradients[0][0]
    act = activations[0][0]

    weights = grad.mean(dim=(1,2))

    cam = torch.zeros(act.shape[1:], device=device)

    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = torch.relu(cam)
    cam = cam.cpu().detach().numpy()

    # normalize
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # ---------- EXTRACT FEATURES ----------
    mean_val = cam.mean()
    max_val = cam.max()
    std_val = cam.std()
    active_ratio = np.sum(cam > 0.5) / cam.size

    feature_vector = [
        mean_val,
        max_val,
        std_val,
        active_ratio
    ]

    feature_list.append(feature_vector)

    if idx % 100 == 0:
        print(f"Processed {idx} failures")


# ---------- SAVE FEATURES ----------
features = np.array(feature_list)

np.save("outputs/xai_features.npy", features)

print("XAI feature extraction completed.")
print("Feature shape:", features.shape)


# remove hooks
hook_f.remove()
hook_b.remove()