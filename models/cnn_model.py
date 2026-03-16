# ============================================
# cnn_model.py
# Updated for Grad-CAM support
# ============================================

import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # convolution layer
        self.conv = nn.Conv2d(1, 16, 3)

        # activation
        self.relu = nn.ReLU()

        # classifier
        self.fc = nn.Linear(26 * 26 * 16, 10)

    def forward(self, x):

        # save conv output (important for Grad-CAM)
        x = self.conv(x)
        x = self.relu(x)

        # flatten
        x = x.reshape(x.size(0), -1)

        # classification
        x = self.fc(x)

        return x