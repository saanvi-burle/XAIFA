# ============================================
# extract_lime_features.py (FIXED)
# ============================================

import torch
import numpy as np
from lime import lime_image

from models.cnn_model import SimpleCNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- LOAD MODEL ----------
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("models/mnist_model.pth"))
model.eval()


# ---------- LOAD FAILURES ----------
failed_images = np.load("outputs/failed_images.npy")

explainer = lime_image.LimeImageExplainer()


# ---------- PREDICTION FUNCTION ----------
def predict_fn(images):

    # LIME gives RGB → convert back to grayscale
    imgs = np.mean(images, axis=-1, keepdims=True)

    imgs = torch.tensor(imgs).float().permute(0,3,1,2).to(device)

    outputs = model(imgs)

    return outputs.detach().cpu().numpy()


feature_list = []


# ---------- PROCESS FEW IMAGES ----------
for i in range(20):   # keep small (LIME slow)

    img = failed_images[i][0]   # (28,28)

    # convert grayscale → RGB
    img_rgb = np.stack([img, img, img], axis=-1)

    explanation = explainer.explain_instance(
        img_rgb,
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=100
    )

    weights = list(explanation.local_exp.values())[0]
    vals = np.array([abs(w[1]) for w in weights])

    mean_val = vals.mean()
    max_val = vals.max()
    std_val = vals.std()

    feature_list.append([mean_val, max_val, std_val])

    print("Processed", i)


features = np.array(feature_list)

np.save("outputs/lime_features.npy", features)

print("LIME features saved:", features.shape)