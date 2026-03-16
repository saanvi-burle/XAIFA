# ============================================
# compare_all_xai.py
# Show GradCAM + SHAP + LIME together
# ============================================

import torch
import numpy as np
import matplotlib.pyplot as plt
import shap

from lime import lime_image
from skimage.segmentation import mark_boundaries

from models.cnn_model import SimpleCNN


# ---------- DEVICE ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- LOAD MODEL ----------
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("models/mnist_model.pth"))
model.eval()


# ---------- LOAD FAILURES ----------
failed_images = np.load("outputs/failed_images.npy")
failed_true = np.load("outputs/failed_true.npy")
failed_pred = np.load("outputs/failed_pred.npy")

# choose one failure
img = failed_images[0]
img_tensor = torch.tensor(img).unsqueeze(0).float().to(device)
img_tensor.requires_grad = True


# =====================================================
# 1️⃣ GRADCAM
# =====================================================
acts = []
grads = []

def f_hook(m, i, o):
    acts.append(o)

def b_hook(m, gi, go):
    grads.append(go[0])

hf = model.conv.register_forward_hook(f_hook)
hb = model.conv.register_full_backward_hook(b_hook)

out = model(img_tensor)
pred_class = out.argmax(dim=1)

model.zero_grad()
out[0, pred_class].backward()

grad = grads[0][0]
act = acts[0][0]

weights = grad.mean(dim=(1,2))

cam = torch.zeros(act.shape[1:], device=device)
for i, w in enumerate(weights):
    cam += w * act[i]

cam = torch.relu(cam).cpu().detach().numpy()
cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

hf.remove()
hb.remove()


# =====================================================
# 2️⃣ SHAP
# =====================================================
background = img_tensor
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(img_tensor)

if isinstance(shap_values, list):
    shap_values = shap_values[0]

shap_map = shap_values[0, 0, :, :, pred_class.item()]


# =====================================================
# 3️⃣ LIME
# =====================================================
explainer_lime = lime_image.LimeImageExplainer()

def predict_fn(images):
    imgs = np.mean(images, axis=-1, keepdims=True)
    imgs = torch.tensor(imgs).float().permute(0,3,1,2).to(device)
    outputs = model(imgs)
    return outputs.detach().cpu().numpy()

img_gray = img[0]
img_rgb = np.stack([img_gray]*3, axis=-1)

explanation = explainer_lime.explain_instance(
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

lime_vis = mark_boundaries(temp, mask)


# =====================================================
# 4️⃣ DISPLAY ALL TOGETHER
# =====================================================

plt.figure(figsize=(12,3))

plt.subplot(1,4,1)
plt.imshow(img_gray, cmap="gray")
plt.title(f"Original\nT:{failed_true[0]} P:{failed_pred[0]}")
plt.axis("off")

plt.subplot(1,4,2)
plt.imshow(cam, cmap="jet")
plt.title("Grad-CAM")
plt.axis("off")

plt.subplot(1,4,3)

# =====================================================
# SHAP  (FIXED VERSION)
# =====================================================

# use multiple background samples
background = torch.tensor(
    failed_images[:10]
).float().to(device)
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(img_tensor)

# convert list if needed
if isinstance(shap_values, list):
    shap_values = shap_values[0]

# shap shape = (1,1,28,28,10)

# ⭐ IMPORTANT FIX:
# combine absolute contribution across classes
# aggregate importance for visualization
shap_map = np.abs(shap_values[0,0]).mean(axis=-1)

# normalize manually
shap_map = (shap_map - shap_map.min()) / (shap_map.max() - shap_map.min() + 1e-8)
plt.imshow(shap_map, cmap="hot")
plt.title("SHAP")
plt.axis("off")

plt.subplot(1,4,4)
plt.imshow(lime_vis)
plt.title("LIME")
plt.axis("off")

plt.tight_layout()
plt.show()