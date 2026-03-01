# ============================================
# cluster_failures.py
# Purpose:
# Cluster XAI feature vectors to discover
# failure patterns
# ============================================

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# ---------- LOAD XAI FEATURES ----------
features = np.load("outputs/xai_features.npy")

print("Feature shape:", features.shape)


# ---------- APPLY K-MEANS ----------
k = 3   # number of failure groups (can change later)

kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(features)


# ---------- SAVE CLUSTER LABELS ----------
np.save("outputs/failure_clusters.npy", clusters)

print("Clustering complete.")
print("Cluster counts:", np.bincount(clusters))


# ---------- SIMPLE VISUALIZATION ----------
plt.scatter(features[:,0], features[:,1], c=clusters, cmap="viridis")
plt.xlabel("Mean Activation")
plt.ylabel("Max Activation")
plt.title("Failure Clusters (XAIFA)")
plt.show()