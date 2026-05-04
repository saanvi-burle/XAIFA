import torch
import numpy as np
import shap

def generate_shap_heatmap(model: torch.nn.Module, image: torch.Tensor, class_idx: int) -> np.ndarray:
    """
    Generate SHAP heatmap using GradientExplainer.
    
    Args:
        model: PyTorch model
        image: Input image tensor (1, C, H, W)
        class_idx: Target class index
        
    Returns:
        Heatmap array (H, W) normalized to [0, 1]
    """
    model.eval()
    
    # We need a background dataset for GradientExplainer. 
    # For a single image explanation without a real dataset, 
    # using a zero tensor is a common baseline approach.
    background = torch.zeros_like(image)
    
    try:
        explainer = shap.GradientExplainer(model, background)
        # shap_values gives attribution for each class
        shap_values, _ = explainer.shap_values(image)
    except Exception as e:
        # Fallback to DeepExplainer if GradientExplainer fails (e.g. for some architectures)
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(image)
    
    # Extract the target class SHAP values
    if isinstance(shap_values, list):
        target_shap = shap_values[class_idx]
    else:
        # Depending on shap version and model output, it might be a tensor/array directly
        if shap_values.shape[0] == 1 and len(shap_values.shape) == 5:
            # Shape might be (1, N_classes, C, H, W)
            target_shap = shap_values[0, class_idx]
        elif len(shap_values.shape) == 4 and shap_values.shape[0] > 1:
            # If shap_values is (N_classes, C, H, W)
            target_shap = shap_values[class_idx]
        else:
            target_shap = shap_values
            
    # Ensure it is a numpy array
    if isinstance(target_shap, torch.Tensor):
        target_shap = target_shap.detach().cpu().numpy()
        
    # Remove batch dimension if present
    if target_shap.ndim == 4 and target_shap.shape[0] == 1:
        target_shap = target_shap[0]
        
    # Aggregate across channels (C, H, W) -> (H, W)
    # Using absolute mean is standard for visualizing SHAP feature importance
    heatmap = np.abs(target_shap).mean(axis=0)
    
    # Normalize to [0, 1]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
        
    return heatmap
