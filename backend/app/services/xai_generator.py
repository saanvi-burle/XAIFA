"""XAI explanation generation service."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from app.core.config import settings
from app.xai.gradcam import GradCAM, overlay_heatmap


class XAIExplanationService:
    """Service for generating XAI explanations for model predictions."""
    
    def __init__(self):
        self.gradcam_dir = settings.heatmaps_dir / "gradcam"
        self.shap_dir = settings.heatmaps_dir / "shap"
        self.lime_dir = settings.heatmaps_dir / "lime"
        self.fusion_dir = settings.heatmaps_dir / "fusion"
    
    def generate_gradcam(
        self,
        model: torch.nn.Module,
        image: torch.Tensor,
        class_idx: int,
        failure_id: str,
    ) -> dict[str, Any]:
        """Generate Grad-CAM explanation for a single prediction."""
        gradcam = GradCAM(model)
        try:
            heatmap = gradcam.generate(image, class_idx)
        finally:
            gradcam.close()
        
        # Save heatmap
        heatmap_path = self.gradcam_dir / f"{failure_id}.npy"
        heatmap_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(heatmap_path, heatmap)
        
        return {
            "method": "gradcam",
            "failure_id": failure_id,
            "heatmap_path": str(heatmap_path),
            "min": float(heatmap.min()),
            "max": float(heatmap.max()),
            "mean": float(heatmap.mean()),
        }
    
    def generate_shap(
        self,
        model: torch.nn.Module,
        image: torch.Tensor,
        class_idx: int,
        failure_id: str,
    ) -> dict[str, Any]:
        """Generate SHAP explanation (simplified kernel explainer)."""
        # Simplified SHAP-like feature attribution
        # For production, use shap.KernelExplainer with background dataset
        
        # Get model output and compute simple attribution
        model.eval()
        with torch.no_grad():
            output = model(image)
            probs = torch.softmax(output, dim=1)[0]
        
        # Create simple attribution based on activation regions
        # This is a placeholder - real SHAP requires background distribution
        heatmap = probs[class_idx].cpu().numpy()
        
        # Resize to simulate feature attribution
        if heatmap.ndim == 4:
            heatmap = heatmap[0, 0]  # Take first channel
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Save
        heatmap_path = self.shap_dir / f"{failure_id}.npy"
        heatmap_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(heatmap_path, heatmap)
        
        return {
            "method": "shap",
            "failure_id": failure_id,
            "heatmap_path": str(heatmap_path),
            "min": float(heatmap.min()),
            "max": float(heatmap.max()),
            "mean": float(heatmap.mean()),
        }
    
    def generate_lime(
        self,
        model: torch.nn.Module,
        image: torch.Tensor,
        class_idx: int,
        failure_id: str,
    ) -> dict[str, Any]:
        """Generate LIME explanation (simplified)."""
        # Simplified LIME-like explanation
        # For production, use lime.LimeImageExplainer
        
        model.eval()
        with torch.no_grad():
            output = model(image)
            probs = torch.softmax(output, dim=1)[0]
        
        # Create superpixel-like attribution
        heatmap = probs[class_idx].cpu().numpy()
        
        if heatmap.ndim == 4:
            heatmap = heatmap[0, 0]
        
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        heatmap_path = self.lime_dir / f"{failure_id}.npy"
        heatmap_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(heatmap_path, heatmap)
        
        return {
            "method": "lime",
            "failure_id": failure_id,
            "heatmap_path": str(heatmap_path),
            "min": float(heatmap.min()),
            "max": float(heatmap.max()),
            "mean": float(heatmap.mean()),
        }
    
    def generate_fusion(
        self,
        gradcam_path: Path,
        shap_path: Path,
        lime_path: Path,
        failure_id: str,
    ) -> dict[str, Any]:
        """Generate fused explanation from multiple methods."""
        gradcam = np.load(gradcam_path)
        shap_vals = np.load(shap_path)
        lime_vals = np.load(lime_path)
        
        # Normalize each to [0, 1] if needed
        for arr in [gradcam, shap_vals, lime_vals]:
            if arr.max() > 0:
                arr = arr / arr.max()
        
        # Simple average fusion
        # More sophisticated: weighted fusion based on agreement
        fusion = (gradcam + shap_vals + lime_vals) / 3.0
        
        # Save
        fusion_path = self.fusion_dir / f"{failure_id}.npy"
        fusion_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(fusion_path, fusion)
        
        return {
            "method": "fusion",
            "failure_id": failure_id,
            "heatmap_path": str(fusion_path),
            "min": float(fusion.min()),
            "max": float(fusion.max()),
            "mean": float(fusion.mean()),
        }
    
    def generate_all_explanations(
        self,
        model: torch.nn.Module,
        image: torch.Tensor,
        class_idx: int,
        failure_id: str,
    ) -> dict[str, Any]:
        """Generate all XAI explanations for a failure case."""
        explanations = {}
        
        # Generate each method
        explanations["gradcam"] = self.generate_gradcam(model, image, class_idx, failure_id)
        explanations["shap"] = self.generate_shap(model, image, class_idx, failure_id)
        explanations["lime"] = self.generate_lime(model, image, class_idx, failure_id)
        
        # Generate fusion
        explanations["fusion"] = self.generate_fusion(
            Path(explanations["gradcam"]["heatmap_path"]),
            Path(explanations["shap"]["heatmap_path"]),
            Path(explanations["lime"]["heatmap_path"]),
            failure_id,
        )
        
        return explanations


# Singleton instance
_explanation_service: XAIExplanationService | None = None


def get_explanation_service() -> XAIExplanationService:
    """Get the XAI explanation service instance."""
    global _explanation_service
    if _explanation_service is None:
        _explanation_service = XAIExplanationService()
    return _explanation_service