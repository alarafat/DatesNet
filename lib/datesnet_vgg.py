import torch
import torch.nn as nn
import torch.nn.functional as F


class DatesNet(nn.Module):
    """
    Simplest customized VGGNet with
    - 2 MLP layers with shrinking feature layers using maxpools but increasing in depth
    - couple of Linear layers as model head to map the features to the number of output classes

    Model output is log_softmax to return log probabilities to use with KLDivLoss
    """
    def __init__(self, cfg):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels=cfg.ModelConfig.in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.mlp_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
        )

        self.mlp_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12 * 12 * 64, 64),
            nn.ReLU(),
            nn.Linear(64, cfg.ModelConfig.n_classes)
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=0.001)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.mlp_layer1(x)
        x = self.mlp_layer2(x)
        x = self.head(x)
        out = F.log_softmax(x, dim=1)
        return out
