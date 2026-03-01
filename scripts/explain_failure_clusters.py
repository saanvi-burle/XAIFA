# ============================================
# explain_failure_clusters.py
# Purpose:
# Generate automatic explanations for
# discovered failure clusters
# ============================================

import numpy as np


# ---------- LOAD DATA ----------
features = np.load("outputs/xai_features.npy")
clusters = np.load("outputs/failure_clusters.npy")

feature_names = [
    "Mean activation",
    "Max activation",
    "Activation spread (std)",
    "Active attention ratio"
]


# ---------- GLOBAL STATS ----------
global_mean = features.mean(axis=0)

print("\n========== XAIFA AUTOMATIC FAILURE EXPLANATIONS ==========\n")


# ---------- EXPLAIN EACH CLUSTER ----------
unique_clusters = np.unique(clusters)

for c in unique_clusters:

    idx = np.where(clusters == c)[0]
    cluster_features = features[idx]

    cluster_mean = cluster_features.mean(axis=0)

    print(f"\n--- Cluster {c} ---")
    print(f"Samples: {len(idx)}")

    # compare against global average
    for i, name in enumerate(feature_names):

        diff = cluster_mean[i] - global_mean[i]

        if diff > 0.05:
            trend = "HIGHER than average"
        elif diff < -0.05:
            trend = "LOWER than average"
        else:
            trend = "NEAR average"

        print(f"{name}: {trend}")

    # ---------- SIMPLE AUTO INTERPRETATION ----------
    mean_act = cluster_mean[0]
    spread = cluster_mean[2]
    active_ratio = cluster_mean[3]

    print("\nAuto interpretation:")

    if active_ratio > global_mean[3]:
        print("• Model attention is spread across larger regions.")
    else:
        print("• Model attention is concentrated in smaller regions.")

    if spread > global_mean[2]:
        print("• Attention pattern is unstable / scattered.")
    else:
        print("• Attention pattern is consistent and focused.")

    if mean_act < global_mean[0]:
        print("• Overall activation strength is weak (possible uncertainty).")
    else:
        print("• Model shows strong confidence focus.")

print("\n==========================================================")