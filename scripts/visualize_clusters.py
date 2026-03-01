# ============================================
# visualize_clusters.py
# Purpose:
# Visualize failure samples grouped by cluster
# ============================================

import numpy as np
import matplotlib.pyplot as plt


# ---------- LOAD DATA ----------
failed_images = np.load("outputs/failed_images.npy")
failed_true = np.load("outputs/failed_true.npy")
failed_pred = np.load("outputs/failed_pred.npy")
clusters = np.load("outputs/failure_clusters.npy")


print("Total failures:", len(failed_images))


# ---------- VISUALIZE EACH CLUSTER ----------
num_clusters = len(np.unique(clusters))

for c in range(num_clusters):

    print(f"\nShowing samples from Cluster {c}")

    idx = np.where(clusters == c)[0]

    # show up to 9 samples per cluster
    show_n = min(9, len(idx))

    plt.figure(figsize=(8, 8))

    for i in range(show_n):

        plt.subplot(3, 3, i + 1)

        img = failed_images[idx[i]][0]

        plt.imshow(img, cmap="gray")
        plt.title(f"T:{failed_true[idx[i]]} P:{failed_pred[idx[i]]}")
        plt.axis("off")

    plt.suptitle(f"Failure Cluster {c}")
    plt.tight_layout()
    plt.show()