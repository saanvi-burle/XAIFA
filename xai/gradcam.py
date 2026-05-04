import torch
import torch.nn.functional as F
import cv2

class GradCAM:

    def __init__(self, model):
        self.model = model

    def generate(self, image, class_idx):

        output = self.model(image)

        self.model.zero_grad()
        output[0, class_idx].backward()

        gradients = self.model.feature_maps.grad
        activations = self.model.feature_maps

        weights = torch.mean(gradients, dim=(2,3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1).squeeze()

        cam = F.relu(cam).detach().cpu().numpy()
        cam = cv2.resize(cam, (28,28))

        return (cam - cam.min()) / (cam.max() + 1e-8)