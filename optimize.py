"""
模型微调脚本（基于错误样本）- 支持混合训练 + 多版本管理
功能：
    1. 从数据库读取已标记真实类别的错误样本
    2. 混合训练：原始训练集 + 错误样本（防止过拟合）
    3. 每次优化保存为独立版本（带时间戳）
    4. 记录版本元信息（时间、准确率、样本数等）
    5. 对比原始模型和最新优化模型的测试准确率
对应课程：第8章（模型微调）、第5/6章（模型评估）
"""

import os
import csv
import shutil
import json
import sqlite3
import random
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==================== 配置 ====================
DATA_ROOT = r"D:\DXB1\Baird\Lizhipeng1\data\半导体芯片表面缺陷检测(预处理后)"
ERROR_DIR = "error_samples"
DB_PATH = "predictions.db"
MODEL_DIR = "models"
VERSIONS_DIR = os.path.join(MODEL_DIR, "versions")
ORIGINAL_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
VERSIONS_JSON = os.path.join(VERSIONS_DIR, "versions.json")

CLASS_NAMES = ["ZF-scratch", "broken", "pinbreak", "scratch"]

# 微调参数
BATCH_SIZE = 32
EPOCHS = 8                      # 少量轮数，防止过拟合
LEARNING_RATE = 0.0001          # 低学习率，温和微调
MIX_RATIO = 0.3                 # 错误样本混合比例（30%）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(VERSIONS_DIR, exist_ok=True)


# ==================== 第1步：从数据库读取错误样本 ====================
def load_error_samples_from_db():
    """从数据库读取已标记真实类别的错误样本"""
    samples = []
    
    if not os.path.exists(DB_PATH):
        print(f"⚠️ 数据库不存在: {DB_PATH}")
        return samples
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT image_name, image_path, predicted_class, true_class
            FROM predictions
            WHERE is_correct = 0 AND true_class IS NOT NULL AND true_class != ''
            ORDER BY reported_at DESC
        ''')
        rows = cursor.fetchall()
        conn.close()
        
        print(f"📋 从数据库查询到 {len(rows)} 条已标记的错误记录")
        
        for row in rows:
            image_name, image_path, predicted_class, true_class = row
            # 在 error_samples 目录中查找对应的图片
            error_dir = os.path.join(ERROR_DIR, predicted_class)
            if os.path.exists(error_dir):
                found = False
                for f in os.listdir(error_dir):
                    if image_name in f or f.endswith(image_name):
                        error_path = os.path.join(error_dir, f)
                        samples.append({
                            "image_path": error_path,
                            "true_class": true_class,
                            "predicted_class": predicted_class,
                            "original_name": image_name
                        })
                        print(f"✅ 读取样本: {image_name} → 真实类别: {true_class}")
                        found = True
                        break
                if not found:
                    print(f"⚠️ 未找到错误图片: {image_name}")
        
        print(f"📊 共找到 {len(samples)} 个可用的错误样本")
    except Exception as e:
        print(f"❌ 读取数据库失败: {e}")
    
    return samples


# ==================== 第2步：混合训练数据准备 ====================
def prepare_mixed_dataset(samples):
    """
    准备混合训练数据：原始训练集 + 错误样本
    按比例从每个类别中采样错误样本，避免类别不平衡
    """
    train_dir = os.path.join(DATA_ROOT, "train")
    
    # 统计原始训练集各类别数量
    class_counts = {}
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(train_dir, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) 
                        if f.endswith(('.jpg', '.png', '.jpeg')) and not f.startswith('error_')])
            class_counts[class_name] = count
    
    print("\n📊 原始训练集各类别数量:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count} 张")
    
    # 按类别分组错误样本
    error_by_class = defaultdict(list)
    for s in samples:
        error_by_class[s["true_class"]].append(s)
    
    # 计算每个类别应采样的错误样本数量
    sampled_samples = []
    print("\n📊 错误样本采样计划:")
    for class_name in CLASS_NAMES:
        original_count = class_counts.get(class_name, 0)
        # 错误样本数量 = 原始数量 × 混合比例
        target_count = int(original_count * MIX_RATIO)
        available = len(error_by_class.get(class_name, []))
        actual_count = min(target_count, available)
        
        if actual_count > 0:
            # 随机选择
            selected = random.sample(error_by_class[class_name], actual_count)
            sampled_samples.extend(selected)
            print(f"  {class_name}: 原始 {original_count} 张 + 错误样本 {actual_count} 张 (目标 {target_count})")
        else:
            print(f"  {class_name}: 原始 {original_count} 张 + 错误样本 0 张 (无可用样本)")
    
    print(f"\n✅ 共选择 {len(sampled_samples)} 个错误样本用于混合训练")
    return sampled_samples


# ==================== 第3步：复制错误样本到训练集 ====================
def merge_error_samples_to_train(samples):
    """将错误样本复制到训练集对应类别目录下（带 error_ 前缀）"""
    train_dir = os.path.join(DATA_ROOT, "train")
    copied_count = 0
    
    for sample in samples:
        true_class = sample["true_class"]
        src_path = sample["image_path"]
        
        if not os.path.exists(src_path):
            print(f"⚠️ 文件不存在: {src_path}")
            continue
        
        dst_dir = os.path.join(train_dir, true_class)
        os.makedirs(dst_dir, exist_ok=True)
        
        base_name = os.path.basename(src_path)
        dst_name = f"error_{base_name}"
        dst_path = os.path.join(dst_dir, dst_name)
        
        try:
            shutil.copy2(src_path, dst_path)
            copied_count += 1
            print(f"✅ 复制: {base_name} → {true_class}/error_{base_name}")
        except Exception as e:
            print(f"❌ 复制失败: {e}")
    
    print(f"✅ 已将 {copied_count} 个错误样本复制到训练集")
    return copied_count


# ==================== 第4步：清理旧的错误样本副本 ====================
def clean_error_copies():
    """清理训练集中之前复制的错误样本（error_* 文件），避免重复"""
    train_dir = os.path.join(DATA_ROOT, "train")
    cleaned_count = 0
    
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(train_dir, class_name)
        if os.path.exists(class_dir):
            for f in os.listdir(class_dir):
                if f.startswith('error_'):
                    file_path = os.path.join(class_dir, f)
                    try:
                        os.remove(file_path)
                        cleaned_count += 1
                    except Exception as e:
                        print(f"⚠️ 删除失败: {file_path} - {e}")
    
    if cleaned_count > 0:
        print(f"🧹 清理了 {cleaned_count} 个旧错误样本副本")
    return cleaned_count


# ==================== 第5步：获取数据加载器 ====================
def get_transforms():
    """获取训练集和验证集的预处理变换"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def load_data():
    """加载数据集"""
    train_transform, val_transform = get_transforms()
    
    train_dir = os.path.join(DATA_ROOT, "train")
    val_dir = os.path.join(DATA_ROOT, "val")
    test_dir = os.path.join(DATA_ROOT, "test")
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)
    
    print(f"\n📊 数据集加载完成:")
    print(f"  训练集: {len(train_dataset)} 张")
    print(f"  验证集: {len(val_dataset)} 张")
    print(f"  测试集: {len(test_dataset)} 张")
    
    return train_loader, val_loader, test_loader


# ==================== 第6步：微调模型（带版本保存） ====================
def fine_tune_model(train_loader, val_loader, sample_count):
    """在错误样本增强后的数据集上微调模型（ResNet18）"""
    
    print("\n📥 加载原始模型...")
    model = models.resnet18()
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 4)
    )
    model.load_state_dict(torch.load(ORIGINAL_MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    
    # 优化器：低学习率，温和微调
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_val_acc = 0.0
    best_epoch = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    print(f"\n🚀 开始微调...")
    print(f"  训练轮数: {EPOCHS}")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  错误样本数: {sample_count}")
    print("=" * 50)
    
    for epoch in range(1, EPOCHS + 1):
        # 训练阶段
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
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
            train_bar.set_postfix({'loss': loss.item(), 'acc': correct/total})
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total
        
        # 验证阶段
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            # 保存版本（带时间戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version_name = f"v_{timestamp}"
            version_path = os.path.join(VERSIONS_DIR, f"best_model_{version_name}.pth")
            torch.save(model.state_dict(), version_path)
            print(f"✅ 保存优化模型版本: {version_name} (验证准确率: {best_val_acc:.4f})")
        
        scheduler.step()
        print("-" * 40)
    
    print(f"\n✅ 微调完成！")
    print(f"  最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")
    
    # 记录版本信息到 JSON
    version_info = {
        "name": version_name,
        "path": version_path,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "val_acc": round(best_val_acc, 4),
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "sample_count": sample_count,
        "best_epoch": best_epoch
    }
    
    # 读取已有的版本信息
    all_versions = []
    if os.path.exists(VERSIONS_JSON):
        with open(VERSIONS_JSON, 'r', encoding='utf-8') as f:
            try:
                all_versions = json.load(f)
            except:
                all_versions = []
    
    all_versions.append(version_info)
    with open(VERSIONS_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_versions, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 版本信息已记录: {VERSIONS_JSON}")
    
    return model, train_losses, val_losses, train_accs, val_accs, version_name, version_path


# ==================== 第7步：评估模型 ====================
def evaluate_model(model_path, test_loader):
    """评估指定模型在测试集上的准确率"""
    model = models.resnet18()
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 4)
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="评估"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total


def compare_models(test_loader):
    """对比原始模型和最新优化模型的测试准确率"""
    print("\n" + "=" * 50)
    print("📊 模型效果对比")
    print("=" * 50)
    
    # 评估原始模型
    print("\n🔍 评估原始模型...")
    old_acc = evaluate_model(ORIGINAL_MODEL_PATH, test_loader)
    print(f"原始模型测试准确率: {old_acc:.4f} ({old_acc*100:.2f}%)")
    
    # 读取所有版本
    if not os.path.exists(VERSIONS_JSON):
        print("⚠️ 未找到优化模型版本")
        return
    
    with open(VERSIONS_JSON, 'r', encoding='utf-8') as f:
        all_versions = json.load(f)
    
    if not all_versions:
        print("⚠️ 未找到优化模型版本")
        return
    
    # 评估最新版本
    latest = all_versions[-1]
    if os.path.exists(latest["path"]):
        print(f"\n🔍 评估最新优化模型 ({latest['name']})...")
        new_acc = evaluate_model(latest["path"], test_loader)
        print(f"优化模型测试准确率: {new_acc:.4f} ({new_acc*100:.2f}%)")
        
        improvement = (new_acc - old_acc) * 100
        print(f"\n📈 提升: {improvement:+.2f} 个百分点")
        if improvement > 0:
            print("🎉 优化成功！错误样本微调有效！")
        else:
            print("⚠️ 准确率未提升，建议收集更多错误样本或调整参数")
    
    # 打印所有历史版本
    print("\n📋 历史版本记录:")
    print("-" * 50)
    for i, v in enumerate(all_versions):
        print(f"  [{i+1}] {v['name']}")
        print(f"      时间: {v['timestamp']}")
        print(f"      验证准确率: {v['val_acc']:.4f}")
        print(f"      错误样本数: {v['sample_count']}")
        print()


# ==================== 第8步：绘制曲线 ====================
def plot_curves(train_losses, val_losses, train_accs, val_accs, version_name):
    """绘制微调过程的曲线"""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'微调过程 Loss 曲线 ({version_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(VERSIONS_DIR, f"fine_tune_loss_{version_name}.png"), dpi=300)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Train Acc')
    plt.plot(epochs, val_accs, 'r-', label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'微调过程 Accuracy 曲线 ({version_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(VERSIONS_DIR, f"fine_tune_acc_{version_name}.png"), dpi=300)
    
    plt.tight_layout()
    plt.show()
    print(f"📊 微调曲线已保存到 {VERSIONS_DIR}/fine_tune_*_{version_name}.png")


# ==================== 主程序 ====================
def main():
    print("=" * 60)
    print("🔧 错误样本优化微调工具（混合训练 + 多版本管理）")
    print("=" * 60)
    print(f"使用设备: {DEVICE}")
    print(f"混合比例: {MIX_RATIO*100:.0f}% 错误样本")
    
    # 第1步：读取错误样本
    print("\n📖 [1/6] 从数据库读取错误样本...")
    samples = load_error_samples_from_db()
    print(f"找到 {len(samples)} 个已标记真实类别的错误样本")
    
    if len(samples) < 3:
        print("⚠️ 错误样本太少（至少需要 3 个），建议多收集一些再优化")
        response = input("是否继续？(y/n): ")
        if response.lower() != 'y':
            return
    
    # 第2步：清理旧的错误样本副本
    print("\n🧹 [2/6] 清理旧的错误样本副本...")
    clean_error_copies()
    
    # 第3步：混合训练数据准备
    print("\n📊 [3/6] 准备混合训练数据...")
    if len(samples) > 10:
        sampled_samples = prepare_mixed_dataset(samples)
    else:
        print("⚠️ 错误样本较少（≤10个），使用全部样本进行微调")
        sampled_samples = samples
    
    # 第4步：复制到训练集
    print("\n📂 [4/6] 复制错误样本到训练集...")
    copied = merge_error_samples_to_train(sampled_samples)
    
    if copied == 0:
        print("❌ 没有成功复制任何错误样本")
        return
    
    # 第5步：加载数据
    print("\n📊 [5/6] 加载数据集...")
    train_loader, val_loader, test_loader = load_data()
    
    # 第6步：微调模型
    print("\n🔧 [6/6] 开始微调模型...")
    model, train_losses, val_losses, train_accs, val_accs, version_name, version_path = fine_tune_model(
        train_loader, val_loader, len(sampled_samples)
    )
    
    # 绘制曲线
    plot_curves(train_losses, val_losses, train_accs, val_accs, version_name)
    
    # 对比评估
    compare_models(test_loader)
    
    print("\n" + "=" * 60)
    print("✅ 优化完成！")
    print("=" * 60)
    print(f"原始模型: {ORIGINAL_MODEL_PATH}")
    print(f"本次优化版本: {version_name}")
    print(f"保存路径: {version_path}")
    print(f"版本信息: {VERSIONS_JSON}")


if __name__ == "__main__":
    main()