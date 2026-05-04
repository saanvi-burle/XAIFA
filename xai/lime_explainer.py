import numpy as np
import cv2
from lime import lime_image


class LimeExplainer:

    def __init__(self, predict_fn):
        self.explainer = lime_image.LimeImageExplainer()
        self.predict_fn = predict_fn

    def generate(self, image):

        image = (image - image.min()) / (image.max() + 1e-8)

        # 🔥 GRID SEGMENTATION (FIXES ZOOM ISSUE)
        def segment_fn(img):
            h, w, _ = img.shape
            segments = np.zeros((h, w), dtype=int)

            grid_size = 4   # 4x4 blocks → 16 segments
            seg_id = 0

            for i in range(0, h, grid_size):
                for j in range(0, w, grid_size):
                    segments[i:i+grid_size, j:j+grid_size] = seg_id
                    seg_id += 1

            return segments

        explanation = self.explainer.explain_instance(
            image,
            self.predict_fn,
            top_labels=1,
            hide_color=0.5,
            num_samples=100,
            segmentation_fn=segment_fn
        )

        segments = explanation.segments
        weights = dict(explanation.local_exp[explanation.top_labels[0]])

        heatmap = np.zeros_like(segments, dtype=float)

        for seg_val in np.unique(segments):
            weight = weights.get(seg_val, 0.0)
            heatmap[segments == seg_val] = abs(weight)

        # normalize
        if heatmap.max() != heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        else:
            heatmap = np.zeros_like(heatmap)

        # smooth
        heatmap = cv2.GaussianBlur(heatmap, (5,5), 0)

        return heatmap