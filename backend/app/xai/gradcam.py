"""Grad-CAM implementation for XAIFA."""

import torch
import numpy as np
from PIL import Image
from typing import Protocol


class FeatureExtractor(Protocol):
    """Protocol for models that can provide feature maps for Grad-CAM."""
    
    def get_conv_layers(self) -> torch.nn.Module:
        """Return the convolutional layers for hook."""
        ...


class GradCAM:
    """Gradient-weighted Class Activation Mapping."""
    
    def __init__(self, model: torch.nn.Module, target_layer: str | None = None):
        self.model = model
        self.target_layer = target_layer
        self.activations: list[torch.Tensor] = []
        self.gradients: list[torch.Tensor] = []
        self._register_hooks()
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target layer."""
        # Find the last convolutional layer if not specified
        target = None
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target = module
        
        if target is None:
            raise ValueError("No convolutional layer found in model")
        
        self.target = target
        self.forward_handle = target.register_forward_hook(self._forward_hook)
        self.backward_handle = target.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module: torch.nn.Module, input: tuple, output: torch.Tensor) -> None:
        self.activations.append(output.detach())
    
    def _backward_hook(self, module: torch.nn.Module, grad_input: tuple, grad_output: tuple) -> None:
        self.gradients.append(grad_output[0].detach())
    
    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the input.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            class_idx: Target class index. If None, uses predicted class.
            
        Returns:
            Heatmap array (H, W) normalized to [0, 1]
        """
        self.model.eval()
        self.activations.clear()
        self.gradients.clear()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Determine target class
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()
        
        # Compute weights
        activations = self.activations[0]  # (1, K, H, W)
        gradients = self.gradients[0]      # (1, K, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, K, 1, 1)
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        
        # ReLU and normalize
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def generate_for_batch(self, input_tensors: torch.Tensor, class_indices: list[int] | None = None) -> list[np.ndarray]:
        """Generate Grad-CAM for a batch of inputs."""
        results = []
        n = input_tensors.shape[0]
        targets = class_indices if class_indices else [None] * n
        
        for i in range(n):
            heatmap = self.generate(input_tensors[i:i+1], targets[i] if targets[i] is not None else None)
            results.append(heatmap)
        
        return results
    
    def close(self) -> None:
        """Remove hooks."""
        self.forward_handle.remove()
        self.backward_handle.remove()


def apply_gradcam_to_image(
    model: torch.nn.Module,
    image: Image.Image | np.ndarray | torch.Tensor,
    class_idx: int | None = None,
    target_layer: str | None = None,
) -> np.ndarray:
    """
    Apply Grad-CAM to an image and return heatmap.
    
    Args:
        model: PyTorch model
        image: Input image (PIL Image, numpy array, or tensor)
        class_idx: Target class index
        target_layer: Name of target convolutional layer
        
    Returns:
        Heatmap as normalized numpy array
    """
    # Convert to tensor if needed
    if isinstance(image, Image.Image):
        image = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
    elif isinstance(image, np.ndarray):
        if image.ndim == 3:
            image = torch.from_numpy(image).permute(2, 0, 1)
        else:
            image = torch.from_numpy(image)
    
    # Add batch dimension
    if image.ndim == 3:
        image = image.unsqueeze(0)
    
    # Generate CAM
    gradcam = GradCAM(model, target_layer)
    try:
        heatmap = gradcam.generate(image, class_idx)
    finally:
        gradcam.close()
    
    return heatmap


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.
    
    Args:
        image: Original image (H, W, C)
        heatmap: Heatmap (H, W) normalized to [0, 1]
        alpha: Transparency of heatmap
        colormap: Matplotlib colormap name
        
    Returns:
        overlayed image
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    
    # Resize heatmap to match image
    if heatmap.shape != image.shape[:2]:
        from PIL import Image as PILImage
        heatmap_pil = PILImage.fromarray((heatmap * 255).astype(np.uint8), mode="L")
        heatmap_pil = heatmap_pil.resize(image.shape[:2][::-1], PILImage.BILINEAR)
        heatmap = np.array(heatmap_pil) / 255.0
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(heatmap)[:, :, :3]  # Remove alpha
    
    # Blend with original
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
        colored = (colored * 255).astype(np.uint8)
    
    result = (1 - alpha) * image + alpha * colored
    return result.astype(np.uint8)