# ============================================
# multi_xai_explanations.py
# FINAL XAIFA INTELLIGENT EXPLANATIONS
# ============================================

import numpy as np


# ---------- LOAD DATA ----------
features = np.load("outputs/combined_xai_features.npy")
clusters = np.load("outputs/multi_xai_clusters.npy")

global_mean = features.mean(axis=0)

print("\n=========== XAIFA MULTI-XAI INTELLIGENT EXPLANATIONS ===========\n")


# ---------- FEATURE GROUP INDEX ----------
# GradCAM: [0:4]
# SHAP:    [4:7]
# LIME:    [7:10]


for c in np.unique(clusters):

    idx = np.where(clusters == c)[0]
    cluster_data = features[idx]
    mean_vals = cluster_data.mean(axis=0)

    print(f"\n--- Cluster {c} ({len(idx)} samples) ---")

    # ---- GradCAM interpretation ----
    grad_mean = mean_vals[0]
    grad_spread = mean_vals[3]

    # ---- SHAP interpretation ----
    shap_strength = mean_vals[4]

    # ---- LIME interpretation ----
    lime_stability = mean_vals[9]

    print("Detected Multi-XAI Pattern:")

    # GradCAM
    if grad_spread > global_mean[3]:
        print("• GradCAM: broad attention regions")
    else:
        print("• GradCAM: narrow focused attention")

    # SHAP
    if shap_strength > global_mean[4]:
        print("• SHAP: strong feature attribution dominance")
    else:
        print("• SHAP: weak distributed feature influence")

    # LIME
    if lime_stability > global_mean[9]:
        print("• LIME: unstable local perturbation sensitivity")
    else:
        print("• LIME: stable local decision behavior")

    # ---------- INTELLIGENT SUMMARY ----------
    print("\nAuto Explanation:")

    if grad_spread < global_mean[3] and shap_strength > global_mean[4]:
        print("→ Model relies heavily on a small region.")
        print("→ Possible overfitting to specific strokes or shapes.")

    if grad_spread > global_mean[3]:
        print("→ Model attention is scattered.")
        print("→ Input ambiguity or noisy structure likely.")

    if lime_stability > global_mean[9]:
        print("→ Small input changes strongly affect prediction.")
        print("→ Decision boundary is fragile.")

    if shap_strength < global_mean[4]:
        print("→ No dominant features detected.")
        print("→ Model may be uncertain or under-trained.")

print("\n================================================================")