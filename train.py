"""
项目名称：芯片缺陷检测CNN训练脚本
模块功能：使用预训练ResNet18迁移学习，微调部分骨干层，结合增强的数据增强、
          早停法、分组学习率等策略，减少过拟合，提高测试准确率。
优化要点：
    - 解冻ResNet18的layer4和全连接层，进行微调
    - 分组优化器（全连接层学习率1e-3，骨干层学习率1e-4）
    - 使用RandomResizedCrop增强多样性
    - 早停法避免过拟合
    - 权重衰减和标签平滑
    - 余弦退火学习率调度
    - 测试时增强
对应课程章节：
    - 第1章：数据预处理与数据增强
    - 第7章：损失函数与反向传播
    - 第8章：迁移学习、微调、正则化、学习率调度
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


def main():
    # ==================== 配置参数 ====================
    BATCH_SIZE = 32
    MAX_EPOCHS = 100                    # 最大训练轮数（配合早停）
    LEARNING_RATE_FC = 1e-3             # 全连接层学习率
    LEARNING_RATE_BACKBONE = 1e-4       # 骨干层微调学习率
    WEIGHT_DECAY = 5e-4                 # 权重衰减（L2正则化）
    LABEL_SMOOTHING = 0.1               # 标签平滑
    EARLY_STOP_PATIENCE = 8             # 早停耐心值
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据路径
    DATA_ROOT = r"D:\DXB1\Baird\Lizhipeng1\data\半导体芯片表面缺陷检测(预处理后)"
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    VAL_DIR = os.path.join(DATA_ROOT, "val")
    TEST_DIR = os.path.join(DATA_ROOT, "test")

    # 保存路径
    MODEL_SAVE_DIR = "models"
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.pth")
    LOSS_CURVE_PATH = os.path.join(MODEL_SAVE_DIR, "loss_curve.png")
    ACC_CURVE_PATH = os.path.join(MODEL_SAVE_DIR, "acc_curve.png")

    print(f"使用设备: {DEVICE}")

    # ==================== 数据预处理与增强 ====================
    # 训练集：更丰富的数据增强，使用RandomResizedCrop替代固定Resize
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪缩放
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),                # 适度旋转
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 平移
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),  # 轻微颜色抖动
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),   # 随机遮挡
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 验证集和测试集：不做增强
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=eval_transform)
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=2, pin_memory=True)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"类别: {train_dataset.classes}")

    # ==================== 模型构建与微调设置 ====================
    # 加载预训练ResNet18
    model = models.resnet18(weights='IMAGENET1K_V1')

    # 冻结所有参数，之后解冻layer4和fc
    for param in model.parameters():
        param.requires_grad = False

    # 解冻layer4（ResNet18的最后一个残差块）
    for name, param in model.named_parameters():
        if "layer4" in name:
            param.requires_grad = True

    # 替换最后的全连接层，并添加Dropout增强正则化
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),               # Dropout防止过拟合
        nn.Linear(num_features, 4)     # 输出4类
    )
    # 新加的层默认requires_grad=True

    model = model.to(DEVICE)

    # 分组优化器：不同学习率
    optimizer = optim.Adam([
        {'params': model.fc.parameters(), 'lr': LEARNING_RATE_FC},
        {'params': [p for n, p in model.named_parameters() if 'layer4' in n], 'lr': LEARNING_RATE_BACKBONE}
    ], weight_decay=WEIGHT_DECAY)

    # 余弦退火调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    # 损失函数：交叉熵 + 标签平滑
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # ==================== 训练循环（含早停） ====================
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_acc = 0.0
    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(MAX_EPOCHS):
        print(f"\nEpoch {epoch+1}/{MAX_EPOCHS}")
        print("-" * 40)

        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc="训练", leave=False)
        for images, labels in train_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            train_bar.set_postfix(loss=loss.item(), acc=correct/total)

        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc = correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="验证", leave=False)
            for images, labels in val_bar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = correct / total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        print(f"训练损失: {epoch_train_loss:.4f}, 训练准确率: {epoch_train_acc:.4f}")
        print(f"验证损失: {epoch_val_loss:.4f}, 验证准确率: {epoch_val_acc:.4f}")

        # 保存最佳模型（基于验证准确率）
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"验证准确率提升至 {best_val_acc:.4f}，模型已保存")
            early_stop_counter = 0  # 重置早停计数器
        else:
            # 检查早停条件（基于验证损失）
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                print(f"验证损失未改善，早停计数器: {early_stop_counter}/{EARLY_STOP_PATIENCE}")
                if early_stop_counter >= EARLY_STOP_PATIENCE:
                    print("触发早停，训练终止")
                    break

        scheduler.step()

    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.4f}")

    # ==================== 测试集评估（含TTA） ====================
    print("在测试集上评估最佳模型（使用测试时增强）...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    # 定义水平翻转变换
    flip_transform = transforms.RandomHorizontalFlip(p=1.0)

    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="测试"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # 原始预测
            outputs_orig = model(images)
            # 水平翻转预测
            images_flipped = flip_transform(images)
            outputs_flipped = model(images_flipped)
            # 平均
            outputs = (outputs_orig + outputs_flipped) / 2.0

            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = correct / total
    test_loss = test_loss / len(test_dataset)
    print(f"测试集损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}")

    # ==================== 绘制曲线 ====================
    # 只绘制实际训练的轮数
    epochs_actual = len(train_losses)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs_actual + 1), train_losses, label='训练损失')
    plt.plot(range(1, epochs_actual + 1), val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和验证损失曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig(LOSS_CURVE_PATH, dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs_actual + 1), train_accs, label='训练准确率')
    plt.plot(range(1, epochs_actual + 1), val_accs, label='验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('训练和验证准确率曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig(ACC_CURVE_PATH, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"曲线已保存到 {MODEL_SAVE_DIR} 目录")


if __name__ == "__main__":
    main()