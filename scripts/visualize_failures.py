# ============================================
# visualize_failures.py
# Purpose:
# Visualize misclassified MNIST samples
# ============================================

import numpy as np
import matplotlib.pyplot as plt

# ---------- LOAD FAILURE DATA ----------
failed_images = np.load("outputs/failed_images.npy")
failed_true = np.load("outputs/failed_true.npy")
failed_pred = np.load("outputs/failed_pred.npy")

print("Total failures loaded:", len(failed_images))


# ---------- VISUALIZE ----------
num_show = 9   # show first 9 failures

plt.figure(figsize=(8, 8))

for i in range(num_show):

    plt.subplot(3, 3, i + 1)

    # MNIST image shape: (1, 28, 28)
    img = failed_images[i][0]

    plt.imshow(img, cmap="gray")

    plt.title(f"T:{failed_true[i]}  P:{failed_pred[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()