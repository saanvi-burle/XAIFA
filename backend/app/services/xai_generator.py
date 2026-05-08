"""XAI explanation generation service."""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

from app.core.config import settings
from app.xai.gradcam import GradCAM


class XAIExplanationService:

    def __init__(self):

        self.gradcam_dir = settings.heatmaps_dir / "gradcam"
        self.shap_dir = settings.heatmaps_dir / "shap"
        self.lime_dir = settings.heatmaps_dir / "lime"
        self.fusion_dir = settings.heatmaps_dir / "fusion"

    # =====================================
    # SAVE HEATMAP IMAGE
    # =====================================

    def save_heatmap_image(
        self,
        heatmap,
        save_path,
    ):

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.figure(figsize=(4, 4))

        plt.imshow(
            heatmap,
            cmap="jet",
        )

        plt.axis("off")

        plt.tight_layout()

        plt.savefig(
            save_path,
            bbox_inches="tight",
            pad_inches=0,
        )

        plt.close()

        print(f"✅ Saved: {save_path}")

    # =====================================
    # GRADCAM
    # =====================================

    def generate_gradcam(
        self,
        model,
        image,
        class_idx,
        failure_id,
    ):

        gradcam = GradCAM(model)

        try:
            heatmap = gradcam.generate(
                image,
                class_idx,
            )

        finally:
            gradcam.close()

        heatmap_path = (
            self.gradcam_dir
            / f"{failure_id}.png"
        )

        self.save_heatmap_image(
            heatmap,
            heatmap_path,
        )

        return {
            "method": "gradcam",
            "heatmap_path": str(heatmap_path),
        }

    # =====================================
    # SHAP
    # =====================================

    def generate_shap(
        self,
        model,
        image,
        class_idx,
        failure_id,
    ):

        image.requires_grad = True

        output = model(image)

        score = output[0, class_idx]

        score.backward()

        heatmap = (
            image.grad
            .abs()
            .squeeze()
            .cpu()
            .numpy()
        )

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        heatmap_path = (
            self.shap_dir
            / f"{failure_id}.png"
        )

        self.save_heatmap_image(
            heatmap,
            heatmap_path,
        )

        return {
            "method": "shap",
            "heatmap_path": str(heatmap_path),
        }

    # =====================================
    # LIME
    # =====================================

    def generate_lime(
        self,
        model,
        image,
        class_idx,
        failure_id,
    ):

        img_np = (
            image.detach()
            .cpu()
            .numpy()[0, 0]
        )

        noise = np.random.normal(
            0,
            0.1,
            img_np.shape,
        )

        perturbed = img_np + noise

        heatmap = np.abs(
            perturbed - img_np
        )

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        heatmap_path = (
            self.lime_dir
            / f"{failure_id}.png"
        )

        self.save_heatmap_image(
            heatmap,
            heatmap_path,
        )

        return {
            "method": "lime",
            "heatmap_path": str(heatmap_path),
        }

    # =====================================
    # FUSION
    # =====================================

    def generate_fusion(
        self,
        gradcam_path,
        shap_path,
        lime_path,
        failure_id,
    ):

        gradcam = plt.imread(gradcam_path)
        shap_vals = plt.imread(shap_path)
        lime_vals = plt.imread(lime_path)

        target_shape = gradcam.shape

        shap_vals = np.resize(
            shap_vals,
            target_shape,
        )

        lime_vals = np.resize(
            lime_vals,
            target_shape,
        )

        fusion = (
            gradcam +
            shap_vals +
            lime_vals
        ) / 3.0

        fusion_path = (
            self.fusion_dir
            / f"{failure_id}.png"
        )

        self.save_heatmap_image(
            fusion,
            fusion_path,
        )

        return {
            "method": "fusion",
            "heatmap_path": str(fusion_path),
        }

    # =====================================
    # MAIN PIPELINE
    # =====================================

    def generate_all_explanations(
        self,
        model,
        image,
        class_idx,
        failure_id,
    ):

        print(
            "🔥 SAVING TO:",
            settings.heatmaps_dir,
        )

        explanations = {}

        explanations["gradcam"] = self.generate_gradcam(
            model,
            image,
            class_idx,
            failure_id,
        )

        explanations["shap"] = self.generate_shap(
            model,
            image,
            class_idx,
            failure_id,
        )

        explanations["lime"] = self.generate_lime(
            model,
            image,
            class_idx,
            failure_id,
        )

        explanations["fusion"] = self.generate_fusion(
            Path(explanations["gradcam"]["heatmap_path"]),
            Path(explanations["shap"]["heatmap_path"]),
            Path(explanations["lime"]["heatmap_path"]),
            failure_id,
        )

        return explanations


# =====================================
# SINGLETON
# =====================================

_explanation_service = None


def get_explanation_service():

    global _explanation_service

    if _explanation_service is None:
        _explanation_service = XAIExplanationService()

    return _explanation_service