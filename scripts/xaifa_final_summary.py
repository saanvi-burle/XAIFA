# ============================================
# xaifa_final_summary.py
# FINAL XAIFA SUMMARY + RECOMMENDATIONS
# ============================================

import numpy as np

features = np.load("outputs/combined_xai_features.npy")
clusters = np.load("outputs/multi_xai_clusters.npy")

global_mean = features.mean(axis=0)

print("\n=========== XAIFA FINAL SUMMARY ===========\n")

print(f"Total failures analyzed : {len(features)}")
print(f"Failure patterns found  : {len(np.unique(clusters))}")


for c in np.unique(clusters):

    idx = np.where(clusters == c)[0]
    data = features[idx]
    mean_vals = data.mean(axis=0)

    print("\n---------------------------------------")
    print(f"Cluster {c}  |  Samples: {len(idx)}")

    grad_focus = "NARROW" if mean_vals[3] < global_mean[3] else "BROAD"
    shap_strength = "STRONG" if mean_vals[4] > global_mean[4] else "WEAK"
    lime_stability = "UNSTABLE" if mean_vals[9] > global_mean[9] else "STABLE"

    print("\nDetected Pattern:")
    print(f"• Attention focus : {grad_focus}")
    print(f"• Feature dominance (SHAP): {shap_strength}")
    print(f"• Local behaviour (LIME): {lime_stability}")

    print("\nAuto Summary:")

    if grad_focus == "NARROW" and shap_strength == "STRONG":
        print("→ Model over-relies on small regions.")
        print("→ Possible overfitting to specific strokes.")

    if grad_focus == "BROAD":
        print("→ Attention scattered across image.")
        print("→ Input ambiguity or noise likely.")

    if lime_stability == "UNSTABLE":
        print("→ Small input changes strongly affect prediction.")
        print("→ Decision boundary is fragile.")

    print("\nRecommended Actions:")

    if grad_focus == "BROAD":
        print("✔ Apply augmentation (rotation/noise).")

    if lime_stability == "UNSTABLE":
        print("✔ Add regularization / dropout.")

    if shap_strength == "WEAK":
        print("✔ Improve feature learning (deeper model / more data).")

print("\n==========================================")
print("XAIFA analysis complete.")