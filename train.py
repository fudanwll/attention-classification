import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.attention_model import ImageAttentionModel
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
print(f"Available GPUs: {num_gpus}")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train)
train_loader = DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=32)

test_dataset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(
    test_dataset, batch_size=100, shuffle=False, num_workers=32)

# classes = ('plane', 'car', 'bird', 'cat', 'deer',
#            'dog', 'frog', 'horse', 'ship', 'truck')

model = ImageAttentionModel(in_channels=3, num_classes=10, embed_size=128, num_heads=4).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=5e-4)


def train(model, device, train_loader, criterion, optimizer, epochs=100):
    model.train()
    best_accuracy = 0
    for epoch in range(epochs):
        total_loss = 0
        for data, target in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader)}, Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        model.eval()
        total_test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc=f"Testing Epoch {epoch+1}"):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                total_test_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()

        test_acc = 100 * test_correct / test_total
        test_loss = total_test_loss / len(test_loader)
        print(f'Epoch {epoch+1}, Test Loss: {test_loss}, Test Accuracy: {test_acc:.2f}%')

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), './result/best_model.pth')
            print(f"New best model saved with accuracy: {best_accuracy:.2f}%")

train(model, device, train_loader, criterion, optimizer)

