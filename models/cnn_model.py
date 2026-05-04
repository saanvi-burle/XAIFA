import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1, 16, 3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(26*26*16, 10)

        self.feature_maps = None

    def forward(self, x):
        x = self.conv(x)

        self.feature_maps = x
        x.retain_grad()

        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x