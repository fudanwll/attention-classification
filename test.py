import torch
import torch.nn as nn
from modules.CNNLayer import ConvBlock, ResBlock  
from modules.SelfAttention import SelfAttention  
from models.attention_model import ImageAttentionModel
from torch.cuda.amp import autocast

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = ImageAttentionModel(in_channels=3, num_classes=10, embed_size=128, num_heads=4).to(device)


x = torch.randn(1, 3, 32, 32).to(device).half()

model.half()

x = model.conv1(x)
print(x.shape)

x = model.conv2(x)
print(x.shape)

x = model.create_patches(x)
print(x.shape)

if torch.isnan(x).any(): 
    print("xxxxxxxxxx")

x = model.to_patch_embedding(x)
if torch.isnan(x).any(): 
    print("xxxxxxxxxx")

# batch_size, channels, height, width = x.size()

# x = x.view(batch_size, channels * height * width)
# print(x.shape)

# x = x.view(batch_size, -1, model.embed_size)
# print(x.shape)

# x = model.attention(x, x, x)
# print(x.shape)

# x = x.mean(dim=1)
# print(x.shape)

# x = model.fc(x)
# print(x.shape)


# # 使用 make_dot 从模型输出生成图
# y = model(x)
# vis_graph = make_dot(y, params=dict(list(model.named_parameters()) + [('input', x)]))

# # 保存可视化的图到文件
# vis_graph.render('model_visualization', format='png')
