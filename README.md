
# XAIFA Complete Workflow

## 1. Model Training

Train the image classification model.

```
train_model
```

Output

* trained CNN model

---

# 2. Failure Detection

Run the trained model and collect incorrect predictions.

```
collect_failures
```

Output

* failed_images.npy
* failed_true.npy
* failed_pred.npy

---

# 3. Failure Visualization (optional)

Inspect the misclassified samples.

```
visualize_failures
```

---

# 4. Generate GradCAM Explanations

Compute GradCAM attention maps for failure cases.

```
gradcam_failures
```

---

# 5. Extract GradCAM Features

Convert GradCAM heatmaps into structured numerical features.

```
extract_xai_features
```

Output

* xai_features.npy

---

# 6. Failure Pattern Discovery (GradCAM)

Cluster failures based on GradCAM features.

```
cluster_failures
```

Output

* failure_clusters.npy

---

# 7. Visualize Failure Clusters

Display the clustered failure samples.

```
visualize_clusters
```

---

# 8. Explain Failure Clusters

Generate explanations for each discovered cluster.

```
explain_failure_clusters
```

---

# 9. Generate Improvement Recommendations

Suggest model improvements based on cluster patterns.

```
recommend_improvements
```

---

# 10. Extract SHAP Features

Compute SHAP explanations for failure samples.

```
extract_shap_features
```

Output

* shap_features.npy

---

# 11. Extract LIME Features

Compute LIME explanations.

```
extract_lime_features
```

Output

* lime_features.npy

---

# 12. Merge XAI Features

Combine GradCAM, SHAP, and LIME features.

```
merge_xai_features
```

Output

* combined_xai_features.npy

---

# 13. Multi-XAI Failure Clustering

Cluster failures using combined XAI features.

```
cluster_multi_xai
```

Output

* multi_xai_clusters.npy

---

# 14. Multi-XAI Explanation Generation

Explain patterns using GradCAM + SHAP + LIME together.

```
multi_xai_explanations
```

---

# 15. System Summary

Generate overall XAIFA system analysis.

```
xaifa_summary
```

---

# 16. Detailed Final Summary

More structured explanation of failure patterns.

```
xaifa_final_summary
```

---

# 17. SHAP Visualization (optional)

Visualize SHAP attribution maps.

```
visualize_shap
```

---

# 18. LIME Visualization (optional)

Visualize LIME explanation regions.

```
visualize_lime
```

---

# 19. XAI Comparison Visualization

Compare all explainability methods.

```
compare_all_xai
```

Displays:

```
Original | GradCAM | SHAP | LIME
```

---

# 20. Final XAIFA Intelligent Report

Complete system output.

```
xaifa_intelligent_report
```

Output includes:

* visual explanation
* failure analysis
* automated recommendations

---