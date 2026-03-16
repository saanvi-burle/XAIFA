# ============================================
# extract_shap_features.py  (FIXED)
# ============================================

import torch
import numpy as np
import shap

from models.cnn_model import SimpleCNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- LOAD MODEL ----------
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("models/mnist_model.pth"))
model.eval()


# ---------- LOAD FAILURES ----------
failed_images = np.load("outputs/failed_images.npy")

# small subset (SHAP is heavy)
subset = failed_images[:20]

inputs = torch.tensor(subset).float().to(device)


# ---------- SHAP EXPLAINER ----------
background = inputs[:5]

explainer = shap.DeepExplainer(model, background)

shap_values = explainer.shap_values(inputs)

# IMPORTANT FIX
# SHAP may return list OR array depending on version
if isinstance(shap_values, list):
    shap_values = shap_values[0]

print("SHAP shape:", np.array(shap_values).shape)


# ---------- FEATURE EXTRACTION ----------
feature_list = []

for i in range(len(inputs)):

    sv = np.abs(shap_values[i])

    mean_val = sv.mean()
    max_val = sv.max()
    std_val = sv.std()

    feature_list.append([mean_val, max_val, std_val])


features = np.array(feature_list)

np.save("outputs/shap_features.npy", features)

print("SHAP features saved:", features.shape)