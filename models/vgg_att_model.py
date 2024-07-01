import torch
import torch.nn as nn
from modules.VGG import VGGBlock

class VggAttentionModel(nn.Module):
    def __init__(self, num_classes=100):
        super(VggAttentionModel, self).__init__()
        self.features = nn.Sequential(
            VGGBlock(3, 64, [3], batch_norm=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGGBlock(64, 128, [3], batch_norm=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGGBlock(128, 256, [3, 5], batch_norm=True),
            VGGBlock(256, 256, [3, 5], batch_norm=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGGBlock(256, 512, [3, 5, 7], batch_norm=True),
            VGGBlock(512, 512, [3, 5, 7], batch_norm=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VGGBlock(512, 512, [3, 5, 7], batch_norm=True),
            VGGBlock(512, 512, [3, 5, 7], batch_norm=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)  # 打印特征图的大小
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x