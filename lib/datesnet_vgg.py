import torch
import torch.nn as nn
import torch.nn.functional as F


class DatesNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.l1_mlp = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
        )

        self.l2_mlp = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
        )

        self.final_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12 * 12 * 64, 64),
            nn.ReLU(),
            nn.Linear(64, cfg.ModelConfig.n_classes)
        )

    def forward(self, x):
        x = self.l1_mlp(x)
        x = self.l2_mlp(x)
        x = self.final_linear(x)
        out = F.log_softmax(x, dim=1)
        return out