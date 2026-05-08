from typing import Callable

import torch
import torch.nn as nn
from torchvision import models


# =========================
# SIMPLE CNN (FIXED)
# =========================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.conv = nn.Conv2d(1, 16, 3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(26 * 26 * 16, num_classes)

        # 🔥 Needed for Grad-CAM
        self.feature_maps = None

    def forward(self, x):
        x = self.conv(x)

        # 🔥 store feature maps
        self.feature_maps = x

        # 🔥 ensure gradients are retained
        if x.requires_grad:
            x.retain_grad()

        x = self.relu(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# =========================
# HELPERS
# =========================
def _replace_linear(module: nn.Module, attr: str, num_classes: int) -> nn.Module:
    old = getattr(module, attr)
    setattr(module, attr, nn.Linear(old.in_features, num_classes))
    return module


# =========================
# BUILDERS
# =========================
def build_simple_cnn(num_classes: int, channels: int) -> nn.Module:
    if channels != 1:
        raise ValueError("simple_cnn supports only grayscale (1 channel).")
    return SimpleCNN(num_classes=num_classes)


def build_resnet18(num_classes: int, channels: int) -> nn.Module:
    model = models.resnet18(weights=None)
    if channels != 3:
        model.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return _replace_linear(model, "fc", num_classes)


def build_resnet50(num_classes: int, channels: int) -> nn.Module:
    model = models.resnet50(weights=None)
    if channels != 3:
        model.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return _replace_linear(model, "fc", num_classes)


def build_mobilenet_v2(num_classes: int, channels: int) -> nn.Module:
    model = models.mobilenet_v2(weights=None)
    if channels != 3:
        model.features[0][0] = nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def build_vgg16(num_classes: int, channels: int) -> nn.Module:
    model = models.vgg16(weights=None)
    if channels != 3:
        model.features[0] = nn.Conv2d(channels, 64, kernel_size=3, padding=1)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes: int, channels: int) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    if channels != 3:
        model.features[0][0] = nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


# =========================
# REGISTRY
# =========================
ARCHITECTURE_BUILDERS: dict[str, Callable[[int, int], nn.Module]] = {
    "simple_cnn": build_simple_cnn,   # ✅ USE THIS EXACT NAME
    "resnet18": build_resnet18,
    "resnet50": build_resnet50,
    "mobilenet_v2": build_mobilenet_v2,
    "vgg16": build_vgg16,
    "efficientnet_b0": build_efficientnet_b0,
}


# =========================
# MAIN ENTRY
# =========================
def build_architecture(name: str, num_classes: int, channels: int) -> nn.Module:
    name = name.lower()

    if name not in ARCHITECTURE_BUILDERS:
        raise ValueError(f"Unsupported architecture: {name}")

    return ARCHITECTURE_BUILDERS[name](num_classes, channels)