import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

def clustering(features):
    features = np.array(features)

    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(features)

    return silhouette_score(features, labels), davies_bouldin_score(features, labels)