# train.py
# PyTorch 训练脚本 - 芯片表面缺陷分类（4分类）

import os
import sys
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms, models

# 假设 models.cnn_model 中存在 ChipDefectCNN 类
# 如果不存在，可以在此处定义，但题目要求使用该模块，故尝试导入
try:
    from models.cnn_model import ChipDefectCNN
except ImportError:
    # 如果导入失败，给出提示并定义简单的替代模型（仅用于演示，实际应调整）
    print("警告：无法从 models.cnn_model 导入 ChipDefectCNN，使用内置简化版本")
    class ChipDefectCNN(nn.Module):
        """简化的CNN模型，用于芯片缺陷分类"""
        def __init__(self, num_classes=4):
            super(ChipDefectCNN, self).__init__()
            # 使用预训练的ResNet18作为骨干，调整最后的全连接层
            self.backbone = models.resnet18(pretrained=True)
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_ftrs, num_classes)
        
        def forward(self, x):
            return self.backbone(x)

# ===================== 配置 =====================
DATA_ROOT = r"D:\DXB1\Baird\Lizhipeng1\data\半导体芯片表面缺陷检测(预处理后)"
BATCH_SIZE = 32
IMG_SIZE = 224
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
NUM_CLASSES = 4
CLASS_NAMES = ['ZF-scratch', 'scratch', 'broken', 'pinbreak']  # 按字母顺序，与ImageFolder一致

# 归一化参数（ImageNet标准）
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# 模型保存路径
MODEL_DIR = 'models'
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pth')
LOSS_CURVE_PATH = os.path.join(MODEL_DIR, 'loss_curve.png')
ACC_CURVE_PATH = os.path.join(MODEL_DIR, 'acc_curve.png')

# 创建模型保存目录
os.makedirs(MODEL_DIR, exist_ok=True)

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ===================== 数据变换 =====================
# 训练集数据增强
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# 验证集和测试集：仅调整大小和归一化
val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# ===================== 加载数据集 =====================
train_dir = os.path.join(DATA_ROOT, 'train')
val_dir = os.path.join(DATA_ROOT, 'val')
test_dir = os.path.join(DATA_ROOT, 'test')

# 使用 ImageFolder 加载数据集
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=val_test_transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=val_test_transform)

# 打印类别对应关系
print("类别映射:", train_dataset.class_to_idx)
print("类别名称:", train_dataset.classes)

# 统计每个类别的样本数量
def print_class_counts(dataset, name):
    """打印数据集中每个类别的样本数"""
    # 获取所有样本的标签索引
    labels = [label for _, label in dataset.samples]  # dataset.samples 是 (path, label) 列表
    counter = Counter(labels)
    total = sum(counter.values())
    print(f"\n{name} 数据集样本统计 (总数: {total}):")
    for idx, class_name in enumerate(dataset.classes):
        count = counter.get(idx, 0)
        print(f"  {class_name}: {count} 张")
    print()

print_class_counts(train_dataset, "训练集")
print_class_counts(val_dataset, "验证集")
print_class_counts(test_dataset, "测试集")

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

dataloaders = {
    'train': train_loader,
    'val': val_loader,
    'test': test_loader
}
dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset),
    'test': len(test_dataset)
}

# ===================== 模型 =====================
model = ChipDefectCNN(num_classes=NUM_CLASSES)
model = model.to(device)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 优化器
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 学习率调度器：每10轮衰减为原来的0.5
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ===================== 训练与验证函数 =====================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 使用 tqdm 显示进度条
    loop = tqdm(dataloader, desc='Training', leave=False)
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # 更新进度条信息
        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_one_epoch(model, dataloader, criterion, device):
    """验证一个epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        loop = tqdm(dataloader, desc='Validating', leave=False)
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# ===================== 训练主流程 =====================
def train():
    print("\n开始训练...")
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    # 记录损失和准确率
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print("-" * 30)

        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # 验证
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        # 更新学习率
        scheduler.step()

        # 记录
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # 打印当前轮次结果
        print(f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, BEST_MODEL_PATH)
            print(f"  -> 保存最佳模型 (验证准确率: {best_acc:.4f})")

    print(f"\n训练完成！最佳验证准确率: {best_acc:.4f}")
    print(f"最佳模型已保存至: {BEST_MODEL_PATH}")

    # 加载最佳模型用于测试
    model.load_state_dict(best_model_wts)

    # ===================== 测试集评估 =====================
    print("\n在测试集上评估最佳模型...")
    test_loss, test_acc = validate_one_epoch(model, test_loader, criterion, device)
    print(f"测试集 Loss: {test_loss:.4f}  Acc: {test_acc:.4f}")

    # ===================== 绘制曲线 =====================
    epochs_range = range(1, NUM_EPOCHS + 1)

    # Loss 曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)

    # Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, label='Train Acc')
    plt.plot(epochs_range, val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH, dpi=300)
    plt.savefig(ACC_CURVE_PATH, dpi=300)   # 也可以分开保存，这里分别保存两个图，但上面是子图，我们另外保存
    # 为符合题目要求分别保存两个单独的图
    plt.close()

    # 单独保存 Loss 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs_range, val_losses, label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(LOSS_CURVE_PATH, dpi=300)
    plt.close()

    # 单独保存 Accuracy 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, train_accs, label='Train Acc', marker='o')
    plt.plot(epochs_range, val_accs, label='Val Acc', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(ACC_CURVE_PATH, dpi=300)
    plt.close()

    print(f"Loss 曲线已保存至: {LOSS_CURVE_PATH}")
    print(f"Accuracy 曲线已保存至: {ACC_CURVE_PATH}")

    return model

# ===================== 执行训练 =====================
if __name__ == '__main__':
    # 自动运行训练流程
    model = train()