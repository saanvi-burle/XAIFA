# ============================================
# VISUALIZATION FOR XAIFA PROJECT
# ============================================

from pathlib import Path
import numpy as np
import matplotlib

# IMPORTANT
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import cv2

# ============================================
# GLOBAL RESULTS DIRECTORY
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

print(f"[DEBUG] Saving results to: {RESULTS_DIR}")

# ============================================
# 1. FAILURE DATASET (Actual vs Predicted)
# ============================================

def show_failures_grid(failures, n=8):

    fig, axs = plt.subplots(2, n//2, figsize=(10,4))

    for i in range(n):

        img, lbl, pred = failures[i]

        axs[i//(n//2), i%(n//2)].imshow(
            img.squeeze(),
            cmap='gray'
        )

        axs[i//(n//2), i%(n//2)].set_title(
            f"T:{lbl} P:{pred}"
        )

        axs[i//(n//2), i%(n//2)].axis('off')

    plt.tight_layout()

    save_path = RESULTS_DIR / "failure_grid.png"

    plt.savefig(save_path)

    print(f"[DEBUG] Saved: {save_path}")

    plt.close()


# ============================================
# 2. ALL METHODS
# ============================================

def show_all_methods_full(image, g, s, l, cmb):

    outputs = {
        "GradCAM": g,
        "SHAP": s,
        "LIME": l,
        "G+S": cmb.g_s(g, s),
        "G+L": cmb.g_l(g, l),
        "S+L": cmb.s_l(s, l),
        "ALL": cmb.all_three(g, s, l)
    }

    fig, axs = plt.subplots(2, 4, figsize=(12,6))
    axs = axs.flatten()

    axs[0].imshow(image.squeeze(), cmap='gray')
    axs[0].set_title("Original")

    for i, (name, exp) in enumerate(outputs.items()):

        axs[i+1].imshow(exp, cmap='jet')
        axs[i+1].set_title(name)

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()

    save_path = RESULTS_DIR / "all_7_methods.png"

    plt.savefig(save_path)

    print(f"[DEBUG] Saved: {save_path}")

    plt.close()


# ============================================
# 3. OVERLAY
# ============================================

def overlay(image, exp, title="overlay"):

    image = image.squeeze().cpu().numpy()

    heatmap = (exp * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    image_rgb = np.stack([image]*3, axis=-1) * 255

    overlay_img = 0.6 * heatmap + 0.4 * image_rgb

    overlay_img = overlay_img.astype(np.uint8)

    plt.imshow(overlay_img)

    plt.title(title)

    plt.axis('off')

    save_path = RESULTS_DIR / f"{title}.png"

    plt.savefig(save_path)

    print(f"[DEBUG] Saved: {save_path}")

    plt.close()


# ============================================
# 4. PCA
# ============================================

def plot_pca(features, title):

    from sklearn.decomposition import PCA

    features = np.array(features)

    pca = PCA(n_components=2)

    reduced = pca.fit_transform(features)

    plt.figure()

    plt.scatter(
        reduced[:,0],
        reduced[:,1],
        s=10
    )

    plt.title(f"PCA - {title}")

    save_path = RESULTS_DIR / f"{title}_pca.png"

    plt.savefig(save_path)

    print(f"[DEBUG] Saved: {save_path}")

    plt.close()


# ============================================
# 5. BEST VS ALL
# ============================================

def show_best_vs_all(image, g, s, l, cmb):

    image = image.squeeze().cpu().numpy()

    best = cmb.g_s(g, s)

    all_map = cmb.all_three(g, s, l)

    fig, axs = plt.subplots(1, 3, figsize=(9,3))

    axs[0].imshow(image, cmap='gray')
    axs[0].set_title("Original")

    axs[1].imshow(best, cmap='jet')
    axs[1].set_title("Best (G+S)")

    axs[2].imshow(all_map, cmap='jet')
    axs[2].set_title("All (+LIME)")

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()

    save_path = RESULTS_DIR / "best_vs_all.png"

    plt.savefig(save_path)

    print(f"[DEBUG] Saved: {save_path}")

    plt.close()