from lime import lime_image
import numpy as np

class LimeExplainer:

    def __init__(self, predict_fn):
        self.explainer = lime_image.LimeImageExplainer()
        self.predict_fn = predict_fn

    def generate(self, image_np):

        explanation = self.explainer.explain_instance(
            image_np,
            self.predict_fn,
            top_labels=1,
            num_samples=500
        )

        _, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=10
        )

        mask = mask.astype(float)

        return (mask - mask.min())/(mask.max()+1e-8)