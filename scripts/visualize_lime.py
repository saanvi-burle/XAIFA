# ============================================
# visualize_lime.py
# Show LIME explanations
# ============================================

import torch
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries

from models.cnn_model import SimpleCNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- LOAD MODEL ----------
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("models/mnist_model.pth"))
model.eval()


# ---------- LOAD FAILURES ----------
failed_images = np.load("outputs/failed_images.npy")

explainer = lime_image.LimeImageExplainer()


def predict_fn(images):

    imgs = np.mean(images, axis=-1, keepdims=True)
    imgs = torch.tensor(imgs).float().permute(0,3,1,2).to(device)

    outputs = model(imgs)
    return outputs.detach().cpu().numpy()


# ---------- VISUALIZE ONE SAMPLE ----------
img = failed_images[0][0]
img_rgb = np.stack([img,img,img], axis=-1)

explanation = explainer.explain_instance(
    img_rgb,
    predict_fn,
    top_labels=1,
    hide_color=0,
    num_samples=100
)

temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],
    positive_only=True,
    hide_rest=False
)

plt.figure(figsize=(6,3))

plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(mark_boundaries(temp, mask))
plt.title("LIME Explanation")
plt.axis("off")

plt.tight_layout()
plt.show()