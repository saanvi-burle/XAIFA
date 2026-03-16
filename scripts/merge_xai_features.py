import numpy as np

gradcam = np.load("outputs/xai_features.npy")
shap = np.load("outputs/shap_features.npy")
lime = np.load("outputs/lime_features.npy")

# match minimum length
n = min(len(gradcam), len(shap), len(lime))

combined = np.concatenate(
    [gradcam[:n], shap[:n], lime[:n]],
    axis=1
)

np.save("outputs/combined_xai_features.npy", combined)

print("Combined feature shape:", combined.shape)