import numpy as np
from lime import lime_image

class LimeExplainer:

    def __init__(self, predict_fn):
        self.explainer = lime_image.LimeImageExplainer()
        self.predict_fn = predict_fn

    def generate(self, image):

        explanation = self.explainer.explain_instance(
            image,
            self.predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=100
        )

        # extract weights only
        weights = list(explanation.local_exp.values())[0]
        vals = np.array([abs(w[1]) for w in weights])

        score = vals.mean()

        # create uniform map scaled by importance
        heatmap = np.ones((28,28)) * score

        return heatmap