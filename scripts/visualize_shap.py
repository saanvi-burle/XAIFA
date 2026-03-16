# ============================================
# visualize_shap.py
# Show SHAP explanation images
# ============================================

import torch
import numpy as np
import shap
import matplotlib.pyplot as plt

from models.cnn_model import SimpleCNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- LOAD MODEL ----------
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("models/mnist_model.pth"))
model.eval()


# ---------- LOAD FAILURES ----------
failed_images = np.load("outputs/failed_images.npy")
inputs = torch.tensor(failed_images[:5]).float().to(device)


# ---------- SHAP ----------
explainer = shap.DeepExplainer(model, inputs[:2])
shap_values = explainer.shap_values(inputs)

if isinstance(shap_values, list):
    shap_values = shap_values[0]

# ---------- VISUALIZE ----------
for i in range(5):

    plt.figure(figsize=(6,3))

    # original image
    plt.subplot(1,2,1)
    plt.imshow(inputs[i][0].cpu(), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # -------- FIXED SHAP MAP --------
    # pick predicted class explanation
    output = model(inputs[i:i+1])
    pred_class = output.argmax(dim=1).item()

    # shap shape = (N,1,28,28,10)
    shap_map = shap_values[i, 0, :, :, pred_class]

    plt.subplot(1,2,2)
    plt.imshow(shap_map, cmap="seismic")
    plt.title(f"SHAP (class {pred_class})")
    plt.axis("off")

    plt.tight_layout()
    plt.show()