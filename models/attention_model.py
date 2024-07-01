import torch
import torch.nn as nn
from modules.CNNLayer import ConvBlock, ResBlock
from modules.SelfAttention import SelfAttention

# debug for the NANs value
# def check_for_nans(tensor, name="Tensor"):
#     if torch.isnan(tensor).any():
#         print(f"NaNs detected in {name}")

# class ImageAttentionModel(nn.Module):
#     def __init__(
#         self, in_channels, num_classes, embed_size, num_heads, patch_size=(8, 8)  # 改变patch_size为8x8
#     ):
#         super(ImageAttentionModel, self).__init__()

#         self.patch_size = patch_size
#         self.embed_size = embed_size
#         # Define net
#         self.conv1 = ConvBlock(in_channels, out_channels=64)  # b, 64, w, h
#         self.conv2 = ResBlock(64, out_channels=128)  # b, 128, w, h
#         self.to_patch_embedding = nn.Linear(
#             patch_size[0] * patch_size[1] * 128, embed_size
#         )  # Adjusted for patch embedding b, 16, c * 64 -> b, 16, e
#         self.attention = SelfAttention(embed_size=embed_size, heads=num_heads)
#         self.fc = nn.Linear(embed_size, num_classes)

#     def create_patches(self, x: torch.Tensor):
#         """ 
#         Create patches from the input images
#         """
#         batch_size, channels, _, _ = x.size()

#         x = x.unfold(2, self.patch_size[0], self.patch_size[0]).unfold(
#             3, self.patch_size[1], self.patch_size[1]
#         )  # b, c, 4, 4, 8, 8
#         x = x.contiguous().view(
#             batch_size, channels, -1, self.patch_size[0] * self.patch_size[1]
#         )  # b, c, 16, 64

#         x = (
#             x.permute(0, 2, 1, 3)  # b, 16, c, 64
#             .contiguous()
#             .view(
#                 batch_size, -1, channels * self.patch_size[0] * self.patch_size[1]
#             )  # b, 16, c*64
#         )
#         return x  # b, 16, c*64

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.create_patches(x)
#         x = self.to_patch_embedding(x)
#         x = self.attention(x, x, x)
#         x = x.mean(dim=1)
#         x = self.fc(x)

#         return x


# debug for the NANs value
# def check_for_nans(tensor, name="Tensor"):
#     if torch.isnan(tensor).any():
#         print(f"NaNs detected in {name}")

class ImageAttentionModel(nn.Module):
    def __init__(
        self, in_channels, num_classes, embed_size, num_heads, patch_size=(16, 16)
    ):
        super(ImageAttentionModel, self).__init__()

        self.patch_size = patch_size
        self.embed_size = embed_size
        # Define net
        self.conv1 = ConvBlock(in_channels, out_channels=64)  # b, 64, w, h
        self.conv2 = ResBlock(64, out_channels=128)  # b, 128, w, h
        self.to_patch_embedding = nn.Linear(
            patch_size[0] * patch_size[1] * 128, embed_size
        )  # Adjusted for patch embedding b, 4, c * 256 -> b, 4, e
        self.attention = SelfAttention(embed_size=embed_size, heads=num_heads)
        self.fc = nn.Linear(embed_size, num_classes)

    def create_patches(self, x: torch.Tensor):
        """ 
        Create patches from the input images
        """
        batch_size, channels, _, _ = x.size()

        x = x.unfold(2, self.patch_size[0], self.patch_size[0]).unfold(
            3, self.patch_size[1], self.patch_size[1]
        )  # b, c, 2, 2, 16, 16
        x = x.contiguous().view(
            batch_size, channels, -1, self.patch_size[0] * self.patch_size[1]
        )  # b, c, 4, 256

        x = (
            x.permute(0, 2, 1, 3)  # b, 4, c, 256
            .contiguous()
            .view(
                batch_size, -1, channels * self.patch_size[0] * self.patch_size[1]
            )  # b, 4, c*256
        )
        return x  # b, 4, c * 256

    def forward(self, x):
        # ConvLayer
        x = self.conv1(x)
        x = self.conv2(x)
        # Transform dimensions to fit the self-attention layer
        # batch_size, channels, height, width = x.size()
        x = self.create_patches(x)
        x = self.to_patch_embedding(x)
        # x = x.view(batch_size, channels * height * width)
        x = self.attention(x, x, x)
        x = x.mean(dim=1)
        x = self.fc(x)

        return x

# class ImageAttentionModel(nn.Module):
#     def __init__(self, in_channels, num_classes, embed_size, num_heads):
#         super(ImageAttentionModel, self).__init__()
#         self.embed_size = embed_size
#         self.patch_size = 8  # 设定补丁大小为8x8
#         self.num_patches = (32 // self.patch_size) * (32 // self.patch_size)  # 一个图像中的补丁数量
#         self.heads = num_heads
#         self.head_dim = embed_size // num_heads

#         self.initial_conv = nn.Conv2d(in_channels, self.embed_size, kernel_size=self.patch_size, stride=self.patch_size, bias=False)  # 初始卷积生成补丁并嵌入到embed_size
#         self.attention = SelfAttention(embed_size=self.embed_size, heads=self.heads)
#         self.final_conv = ConvBlock(embed_size, embed_size)
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.classifier = nn.Linear(embed_size, num_classes)

#     def forward(self, x):
#         # 初始卷积用于生成补丁并将其嵌入到embed_size
#         x = self.initial_conv(x)  # 输出维度 [batch_size, embed_size, num_patches_height, num_patches_width]
        
#         # 调整张量的形状以适配自注意力模块
#         x = x.flatten(2)  # 将每个补丁展平 [batch_size, embed_size, num_patches]
#         x = x.transpose(1, 2)  # 交换embed_size和num_patches的维度，以适配自注意力 [batch_size, num_patches, embed_size]

#         # 自注意力层
#         x = self.attention(x, x, x)

#         # 逆操作，准备进入最后的卷积层
#         x = x.transpose(1, 2).contiguous().view(-1, self.embed_size, int(self.num_patches**0.5), int(self.num_patches**0.5))  # 重塑回特征图形状

#         x = self.final_conv(x)
#         x = self.pool(x).view(x.size(0), -1)
#         x = self.classifier(x)
#         return x

# class ImageAttentionModel(nn.Module):
#     def __init__(self, in_channels, num_classes, embed_size, num_heads):
#         super(ImageAttentionModel, self).__init__()
#         self.embed_size = embed_size
#         # 设置补丁大小为8
#         self.patch_size = 8  
#         # 计算在32x32图像中8x8补丁的数量
#         self.num_patches = (32 // self.patch_size) ** 2  # 4*4=16
#         self.heads = num_heads
#         self.head_dim = embed_size // num_heads

#         # 初始卷积层用于适当调整通道数
#         self.initial_conv = nn.Conv2d(in_channels, embed_size, kernel_size=1)  # 用于调整输入的通道数
#         # 注意力机制
#         self.attention = SelfAttention(embed_size=embed_size, heads=num_heads)
#         # 卷积和残差块用于进一步处理注意力后的特征
#         self.final_conv = ConvBlock(embed_size, embed_size)
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.classifier = nn.Linear(embed_size, num_classes)

#     def forward(self, x):
#         x = self.initial_conv(x)  # [batch_size, embed_size, 32, 32]

#         # 将图片转换为8x8的补丁
#         x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
#         x = x.contiguous().view(x.size(0), -1, self.embed_size, self.patch_size * self.patch_size)  # 重塑为 [batch_size, num_patches, embed_size, patch_area]
#         x = x.permute(0, 1, 3, 2).reshape(x.size(0), -1, self.embed_size)  # [batch_size, num_patches, embed_size * patch_area]

#         # 应用注意力机制
#         x = self.attention(x, x, x)  # 注意力层输入输出形状 [batch_size, num_patches, embed_size * patch_area]

#         # 重塑回卷积层可以处理的形状
#         x = x.view(x.size(0), self.embed_size, self.patch_size, self.patch_size * self.num_patches)  # 修正这里的视图变换
#         x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), self.embed_size, int(self.num_patches**0.5) * self.patch_size, int(self.num_patches**0.5) * self.patch_size)

#         x = self.final_conv(x)
#         x = self.pool(x).view(x.size(0), -1)
#         x = self.classifier(x)

#         return x





