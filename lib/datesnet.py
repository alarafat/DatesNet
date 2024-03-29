from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.attention import SelfAttention


class DatesNet(nn.Module):
    """
    DatesNet
    Heavy-loading UNet with Residual blocks and Attention blocks.
    """
    def __init__(self,
                 cfg,
                 start_hidden_channels: int = 64,
                 n_hidden_expansion: int = 3):
        super().__init__()

        hidden_channels = [start_hidden_channels * (2 ** idx) for idx in range(n_hidden_expansion)]  # [64, 128, 256]
        decoder_in_channels = list()  # [512, 256, 128]
        for idx in reversed(range(hidden_channels.__len__())):
            decoder_in_channels.append(hidden_channels[idx] + hidden_channels[idx])

        # ToDo: Code cleaning to reduce the code size by adding the encoder layers in loop to the ModuleList
        # self.encoder = nn.ModuleList()
        # for idx in n_hidden_expansion:
        #     self.encoder.extend([
        #         nn.Sequential(
        #             nn.Conv2d(in_channels=cfg.ModelConfig.in_channels, out_channels=hidden_channels[0], kernel_size=3, padding=1),  # (b, 64, h, w)
        #             ResidualBlock(in_channels=hidden_channels[0], out_channels=hidden_channels[0]),  # (b, 64, h, w)
        #             # ResidualBlock(in_channels=hidden_channels[0], out_channels=hidden_channels[0]),  # (b, 64, h, w)
        #         ),
        #     ])

        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=cfg.ModelConfig.in_channels, out_channels=hidden_channels[0], kernel_size=3, padding=1),  # (b, 64, h, w)
                ResidualBlock(in_channels=hidden_channels[0], out_channels=hidden_channels[0]),  # (b, 64, h, w)
                # ResidualBlock(in_channels=hidden_channels[0], out_channels=hidden_channels[0]),  # (b, 64, h, w)
            ),

            nn.Sequential(
                nn.Conv2d(in_channels=hidden_channels[0], out_channels=hidden_channels[0], kernel_size=3, stride=2, padding=1),  # (b, 64, h/2, w/2)
                ResidualBlock(in_channels=hidden_channels[0], out_channels=hidden_channels[1]),  # -> (b, 128, h/2, w/2)
                # ResidualBlock(in_channels=hidden_channels[1], out_channels=hidden_channels[1]),  # (b, 128, h/2, w/2)
            ),

            nn.Sequential(
                nn.Conv2d(in_channels=hidden_channels[1], out_channels=hidden_channels[1], kernel_size=3, stride=2, padding=1),  # (b, 128, h/4, w/4)
                ResidualBlock(in_channels=hidden_channels[1], out_channels=hidden_channels[2]),  # -> (b, 256, h/4, w/4)
                # AttentionBlock(hidden_channels[2]),
                # ResidualBlock(in_channels=hidden_channels[2], out_channels=hidden_channels[2]),  # (b, 256, h/4, w/4)
            ),

            # nn.Sequential(
            #     nn.Conv2d(in_channels=hidden_channels[2], out_channels=hidden_channels[2], kernel_size=3, stride=2, padding=1),  # (b, 256, h/8, w/8)
            #     ResidualBlock(in_channels=hidden_channels[2], out_channels=hidden_channels[2]),             # (b, 256, h/8, w/8)
            # )
        ])

        self.bottleneck = ResidualBlock(in_channels=hidden_channels[2], out_channels=hidden_channels[2])  # (b, 256, h/4, w/4)

        self.decoders = nn.ModuleList([
            # nn.Sequential(
            #     ResidualBlock(in_channels=decoder_in_channels[0], out_channels=hidden_channels[2]),  # (b, 512, h/8, w/8) -> (b, 256, h/8, w/8)
            #     Upsample(hidden_channels[2]),  # (b, 256, h/4, w/4)
            # ),

            nn.Sequential(
                ResidualBlock(in_channels=decoder_in_channels[0], out_channels=hidden_channels[2]),  # (b, 512, h/4, w/4) -> (b, 256, h/4, w/4)
                # AttentionBlock(hidden_channels[2]),
                # ResidualBlock(in_channels=hidden_channels[2], out_channels=hidden_channels[2]),  # (b, 256, h/4, w/4)
                ResidualBlock(in_channels=hidden_channels[2], out_channels=hidden_channels[1]),  # (b, 128, h/4, w/4)
                Upsample(hidden_channels[1]),  # (b, 128, h/2, w/2)
            ),

            nn.Sequential(
                ResidualBlock(in_channels=decoder_in_channels[1], out_channels=hidden_channels[1]),  # (b, 256, h/4, w/4) -> (b, 128, h/4, w/4)
                ResidualBlock(in_channels=hidden_channels[1], out_channels=hidden_channels[0]),  # (b, 128, h/2, w/2) -> (b, 64, h/2, w/2)
                # ResidualBlock(in_channels=hidden_channels[0], out_channels=hidden_channels[0]),  # (b, 64, h/2, w/2) -> (b, 64, h/2, w/2)
                Upsample(hidden_channels[0]),  # (b, 64, h, w)
            ),

            nn.Sequential(
                ResidualBlock(in_channels=decoder_in_channels[2], out_channels=hidden_channels[0]),  # (b, 128, h/4, w/4) -> (b, 64, h/4, w/4)
                # ResidualBlock(in_channels=hidden_channels[0], out_channels=hidden_channels[0]),  # (b, 64, h/2, w/2) -> (b, 64, h/2, w/2)
                ResidualBlock(in_channels=hidden_channels[0], out_channels=hidden_channels[0]),  # (b, 64, h/2, w/2) -> (b, 64, h/2, w/2)
            ),
        ])

        self.head = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels[0], out_channels=cfg.ModelConfig.n_classes, kernel_size=3, padding=1),
            nn.Flatten(),
            nn.Linear(in_features=cfg.DatasetConfig.image_shape[0] * cfg.DatasetConfig.image_shape[1] * cfg.ModelConfig.n_classes,
                      out_features=cfg.ModelConfig.n_classes, bias=False),
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
        skips = []
        for layers in self.encoders:
            x = layers(x)
            skips.append(x)

        x = self.bottleneck(x)  # (b, 256, h/8, w/8)

        for layers in self.decoders:
            x = torch.cat((x, skips.pop()), dim=1)
            x = layers(x)

        # We are using log-softmax since we are KLDiv Loss, otherwise use softmax if we use cross-entropy
        out = F.log_softmax(self.head(x), dim=1)

        return out


class AttentionBlock(nn.Module):
    """"
    Attention Block
    This block calls the self attention layer with ra image, rather than using image embeddings
    ToDo: Add image embeddings
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.self_attention_layer = SelfAttention(n_embd=in_channels, n_head=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connection = x

        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.self_attention_layer(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        out = x + skip_connection
        return out


class ResidualBlock(nn.Module):
    """"
    Residual block with GroupNorm instead of using BatchNorm and SilU rather than ReLU.
    ToDo: Test the impact of the layer options
    """
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 16):
        super().__init__()
        self.layer_latent = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            # nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            # nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if in_channels == out_channels:
            self.extra_layer = nn.Identity()
        else:
            self.extra_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = self.extra_layer(x)

        out = self.layer_latent(x)

        return out + skips


class Upsample(nn.Module):
    """
    A module to upsample the input tensor by a factor of 2 using nearest neighbor interpolation,
    followed by a convolutional layer to process the upsampled output.

    """
    def __init__(self, in_channels):
        super().__init__()
        # A Conv layer to refine the features post-upsampling, adding depth to the model without altering the tensor's depth.
        self.layer = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Upsample the input tensor to twice its size using nearest neighbor interpolation
        out = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.layer(out)
        return out
