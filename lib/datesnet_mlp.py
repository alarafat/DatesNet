import torch
import torch.nn as nn
import torch.nn.functional as F


class DatesNet(nn.Module):
    """
    One of the world's smallest and simplest MLP model with a couple of conv layers with maxpools in between
    The model has a couple of Linear layers on the head to map the features to the number output classes

    Model output is log_softmax to return log probabilities to use with KLDivLoss
    """
    def __init__(self, cfg):
        super().__init__()

        self.mlp_layer = nn.Sequential(
            nn.Conv2d(cfg.ModelConfig.in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(12 * 12 * 64, 64),
            nn.ReLU(),
            nn.Linear(64, cfg.ModelConfig.n_classes),
        )

    def forward(self, x):
        out = F.log_softmax(self.mlp_layer(x), dim=1)
        return out
