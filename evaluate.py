import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from models.attention_model import ImageAttentionModel

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = ImageAttentionModel(in_channels=3, num_classes=10, embed_size=128, num_heads=4)
model.load_state_dict(torch.load('./result/best_model.pth', map_location=device))
model = model.to(device)
model.eval()

# 定义变换和加载测试数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

# 选择100个随机样本
indices = np.random.choice(len(test_dataset), 100, replace=False)
subset = Subset(test_dataset, indices)
test_loader = DataLoader(subset, batch_size=10, shuffle=False)

# 测试模型
correct = 0
total = 0
with torch.no_grad():
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the 100 test images: {accuracy:.2f}%')
