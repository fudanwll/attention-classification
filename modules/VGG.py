import torch.nn as nn
from modules.SKAttention import SKAttention

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernels, batch_norm=False):
        super(VGGBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        self.conv = nn.Sequential(*layers)
        self.attention = SKAttention(channel=out_channels, kernels=kernels)

    def forward(self, x):
        x = self.conv(x)
        x = self.attention(x)
        return x