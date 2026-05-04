import shap
import torch
import numpy as np

class ShapExplainer:

    def __init__(self, model, background):
        self.model = model
        self.explainer = shap.GradientExplainer(model, background)

    def generate(self, image):

        # Get SHAP values
        shap_values = self.explainer.shap_values(image)

        # ---------- HANDLE OUTPUT FORMAT ----------
        if isinstance(shap_values, list):
            sv = shap_values[0]  # take first class (consistent baseline)
        else:
            sv = shap_values

        # Convert to numpy safely
        if torch.is_tensor(sv):
            sv = sv.detach().cpu().numpy()

        # Remove batch/channel dims
        sv = np.squeeze(sv)

        # ---------- CRITICAL FIX ----------
        # Case 1: multiple class maps → (28,28,10)
        if sv.ndim == 3:
            shap_map = np.mean(np.abs(sv), axis=2)

        # Case 2: already correct → (28,28)
        elif sv.ndim == 2:
            shap_map = np.abs(sv)

        else:
            raise ValueError(f"Unexpected SHAP shape: {sv.shape}")

        # Normalize
        shap_map = (shap_map - shap_map.min()) / (shap_map.max() + 1e-8)

        return shap_map