# ============================================
# recommend_improvements.py
# Purpose:
# Generate improvement suggestions
# from failure cluster statistics
# ============================================

import numpy as np


# ---------- LOAD DATA ----------
features = np.load("outputs/xai_features.npy")
clusters = np.load("outputs/failure_clusters.npy")

feature_names = [
    "Mean activation",
    "Max activation",
    "Activation spread",
    "Active attention ratio"
]

global_mean = features.mean(axis=0)

print("\n========== XAIFA IMPROVEMENT RECOMMENDATIONS ==========\n")


# ---------- ANALYZE EACH CLUSTER ----------
for c in np.unique(clusters):

    idx = np.where(clusters == c)[0]
    cluster_features = features[idx]
    cluster_mean = cluster_features.mean(axis=0)

    print(f"\n--- Cluster {c} ({len(idx)} samples) ---")

    mean_act = cluster_mean[0]
    spread = cluster_mean[2]
    active_ratio = cluster_mean[3]

    print("\nDetected pattern:")

    # pattern reasoning
    if active_ratio > global_mean[3]:
        print("• Attention spread is HIGH")
    else:
        print("• Attention spread is LOW")

    if spread > global_mean[2]:
        print("• Attention pattern is unstable")
    else:
        print("• Attention pattern is stable")

    if mean_act < global_mean[0]:
        print("• Model confidence appears weak")
    else:
        print("• Model confidence is strong")

    # ---------- RECOMMENDATIONS ----------
    print("\nSuggested Improvements:")

    if active_ratio > global_mean[3]:
        print("→ Apply data augmentation (rotation, shift, noise).")
        print("→ Increase training variety to improve focus.")

    if spread > global_mean[2]:
        print("→ Add regularization or dropout.")
        print("→ Improve feature consistency using normalization.")

    if mean_act < global_mean[0]:
        print("→ Increase training samples for difficult cases.")
        print("→ Consider deeper model architecture.")

    if mean_act >= global_mean[0] and spread <= global_mean[2]:
        print("→ Model is confident but wrong.")
        print("→ Check for ambiguous classes or labeling issues.")

print("\n=======================================================")