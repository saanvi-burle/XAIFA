# ============================================
# cluster_multi_xai.py
# Purpose:
# Cluster failures using combined XAI features
# ============================================

import numpy as np
from sklearn.cluster import KMeans


# ---------- LOAD DATA ----------
features = np.load("outputs/combined_xai_features.npy")

print("Feature shape:", features.shape)


# ---------- CLUSTER ----------
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(features)

np.save("outputs/multi_xai_clusters.npy", clusters)

print("Clustering complete.")
print("Cluster counts:", np.bincount(clusters))