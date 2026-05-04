import numpy as np
import torch

def fidelity(model, image, exp, pred):

    exp = exp / (np.max(exp)+1e-8)
    threshold = np.percentile(exp, 80)

    mask = exp > threshold

    perturbed = image.clone()
    perturbed[:, :, mask] = 0

    with torch.no_grad():
        orig = model(image)[0, pred].item()
        pert = model(perturbed)[0, pred].item()

    return orig - pert