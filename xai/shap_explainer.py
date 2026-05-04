import shap
import numpy as np

class ShapExplainer:

    def __init__(self, model, background):
        self.explainer = shap.DeepExplainer(model, background)

    def generate(self, image):
        shap_values = self.explainer.shap_values(image)
        shap_map = np.abs(shap_values[0]).mean(axis=1)[0]

        return (shap_map - shap_map.min())/(shap_map.max()+1e-8)