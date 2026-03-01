# ============================================
# xaifa_summary.py
# FINAL XAIFA SUMMARY REPORT
# ============================================

import numpy as np

# ---------- LOAD DATA ----------
features = np.load("outputs/xai_features.npy")
clusters = np.load("outputs/failure_clusters.npy")

global_mean = features.mean(axis=0)

print("\n================ XAIFA FINAL SUMMARY ================\n")

print(f"Total failures analyzed: {len(features)}")
print(f"Total discovered failure patterns: {len(np.unique(clusters))}")

print("\n----------------------------------------------------")

for c in np.unique(clusters):

    idx = np.where(clusters == c)[0]
    cluster_features = features[idx]
    cluster_mean = cluster_features.mean(axis=0)

    print(f"\nCluster {c}")
    print(f"Samples: {len(idx)}")

    mean_act = cluster_mean[0]
    spread = cluster_mean[2]
    active_ratio = cluster_mean[3]

    # AUTO EXPLANATION
    if active_ratio > global_mean[3]:
        focus = "broad attention"
    else:
        focus = "narrow attention"

    if spread > global_mean[2]:
        stability = "unstable focus"
    else:
        stability = "stable focus"

    if mean_act > global_mean[0]:
        confidence = "strong confidence"
    else:
        confidence = "weak confidence"

    print(f"Pattern: {focus}, {stability}, {confidence}")

    # AUTO RECOMMENDATION
    print("Recommended Action:")

    if active_ratio > global_mean[3]:
        print(" - Add data augmentation")
    if spread > global_mean[2]:
        print(" - Apply regularization/dropout")
    if mean_act < global_mean[0]:
        print(" - Improve model capacity / data")
    if mean_act > global_mean[0] and spread <= global_mean[2]:
        print(" - Check ambiguous classes")

print("\n====================================================")
print("XAIFA analysis completed successfully.")