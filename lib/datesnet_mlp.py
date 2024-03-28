from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F


class DatesNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # self.mlp_layer = nn.Sequential(
        self.l1 = nn.Conv2d(cfg.ModelConfig.in_channels, out_channels=32, kernel_size=3, padding=1)
        self.l2 = nn.MaxPool2d(kernel_size=2)
        self.l3 = nn.Conv2d(32, out_channels=64, kernel_size=3, padding=1)
        self.l4 = nn.MaxPool2d(kernel_size=2)
        self.l5 = nn.Flatten()
        self.l6 = nn.Linear(12 * 12 * 64, 64)
        self.l7 = nn.ReLU()
        self.l8 = nn.Linear(64, cfg.ModelConfig.n_classes)
        # )

    def forward(self, x):
        # out = F.log_softmax(self.mlp_layer(x), dim=1)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        out = F.log_softmax(x, dim=1)
        return out
