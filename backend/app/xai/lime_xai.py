import torch
import numpy as np
from lime import lime_image

def generate_lime_heatmap(model: torch.nn.Module, image: torch.Tensor, class_idx: int) -> np.ndarray:
    """
    Generate LIME heatmap for the given input.
    
    Args:
        model: PyTorch model
        image: Input image tensor (1, C, H, W)
        class_idx: Target class index
        
    Returns:
        Heatmap array (H, W) normalized to [0, 1]
    """
    model.eval()
    
    # LIME expects images in (H, W, C) format as numpy arrays
    img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    is_grayscale = img_np.shape[2] == 1
    
    # If image is grayscale (H, W, 1), repeat to 3 channels since LIME ImageExplainer prefers RGB
    if is_grayscale:
        img_np = np.repeat(img_np, 3, axis=2)
        
    explainer = lime_image.LimeImageExplainer()
    device = image.device
    
    # Define a prediction function for LIME
    def batch_predict(images_np):
        # images_np is (N, H, W, C)
        batch_tensors = []
        for img in images_np:
            # Revert to grayscale if the original model expects 1 channel
            if is_grayscale:
                img = img[:, :, :1]
            t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
            batch_tensors.append(t)
            
        batch_tensor = torch.stack(batch_tensors).to(device)
        
        with torch.no_grad():
            outputs = model(batch_tensor)
            probs = torch.softmax(outputs, dim=1)
            
        return probs.cpu().numpy()
        
    # Generate explanation
    # Note: num_samples=250 for speed in web demo. Increase to 1000+ for production quality.
    explanation = explainer.explain_instance(
        img_np.astype('double'), 
        batch_predict, 
        top_labels=None, 
        labels=[class_idx], 
        hide_color=0, 
        num_samples=250
    )
    
    # Construct a heatmap from the superpixels and their weights
    try:
        dict_heatmap = dict(explanation.local_exp[class_idx])
    except KeyError:
        # Fallback if class_idx not found (unlikely if passed explicitly)
        dict_heatmap = dict(explanation.local_exp[list(explanation.local_exp.keys())[0]])
    
    # Initialize heatmap
    heatmap = np.zeros(explanation.segments.shape)
    for k, v in dict_heatmap.items():
        # LIME weights can be negative (evidence against class). 
        # We highlight features with positive evidence for the class.
        if v > 0:
            heatmap[explanation.segments == k] = v
            
    # Normalize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
        
    return heatmap
