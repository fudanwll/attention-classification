# 基于卷积Self-Attention的图像分类模型Image_Classification

## 项目概述
这个项目的目标是使用深度学习技术识别CIFAR-10数据集中的图像。我们采用了一个基于注意力机制的图像识别模型，旨在提高分类的准确率。

## 环境要求
运行此项目需要以下环境：
- Python 3.9
- PyTorch 2.0.0
- torchvision 0.15.0
- tqdm

## 安装指南
首先，确保你的Python环境已经安装。然后安装必要的库：

```bash
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

## 如何运行代码
1. 克隆仓库到本地：
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```
2. 运行训练脚本：
   ```bash
   python train.py
   ```

3. 运行评估脚本：
   评估训练之后的模型在随机100张图片上识别的准确度
   ```bash
   python evaluate.py
   ```

## 训练方法
    模型: 使用了一个基于注意力机制的卷积神经网络。
    
    数据增强: 包括随机裁剪和水平翻转。
    
    优化器: SGD，学习率为0.01，动量为0.9。
    
    损失函数: 交叉熵损失。

## 模型优化
    使用更复杂的模型结构
    增加更多的数据增强技术
    调整优化器的参数，如学习率
    使用学习率衰减和其他正则化技术


