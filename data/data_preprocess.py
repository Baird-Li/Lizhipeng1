"""
数据预处理模块
课程对应：第1章 绪论 - 数据预处理
功能：将 YOLO 格式的数据集转换为分类格式，并进行归一化、数据增强、创建DataLoader
"""


import os
import shutil
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import yaml


# ==================== 1. 配置路径 ====================
# 你的数据集根目录
DATA_ROOT = r"D:\DXB1\Baird\Lizhipeng1\data\半导体芯片表面缺陷检测"

# 输出目录（分类格式）
OUTPUT_ROOT = r"D:\DXB1\Baird\Lizhipeng1\data\chip_classification"

# 类别名称（从 names.yaml 读取）
CLASS_NAMES = ['ZF-scratch', 'scratch', 'broken', 'pinbreak']


# ==================== 2. 读取 YOLO 格式标注文件 ====================
def read_yolo_label(label_path):
    """
    读取 YOLO 格式的标注文件
    每行格式: class_id x_center y_center width height
    返回: 该图片中所有目标的类别ID列表
    """
    class_ids = []
    if not os.path.exists(label_path):
        return class_ids
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                class_id = int(parts[0])
                class_ids.append(class_id)
    
    return class_ids


# ==================== 3. 转换 YOLO 格式 → 分类格式 ====================
def convert_yolo_to_classification():
    """
    将 YOLO 格式的数据集转换为分类格式
    按类别将图片复制到对应的文件夹中
    """
    print("=" * 60)
    print("🔄 开始转换：YOLO格式 → 分类格式")
    print("=" * 60)
    
    # 创建输出目录
    for split in ['train', 'val', 'test']:
        for class_name in CLASS_NAMES:
            dir_path = os.path.join(OUTPUT_ROOT, split, class_name)
            os.makedirs(dir_path, exist_ok=True)
    
    # 处理 train, test, valid 三个数据集
    splits = {
        'train': 'train',
        'test': 'test',
        'valid': 'val'  # 将 valid 映射为 val
    }
    
    total_count = 0
    
    for src_split, dst_split in splits.items():
        src_images_dir = os.path.join(DATA_ROOT, src_split, 'images')
        src_labels_dir = os.path.join(DATA_ROOT, src_split, 'labels')
        
        if not os.path.exists(src_images_dir):
            print(f"⚠️ 跳过 {src_split}：目录不存在")
            continue
        
        # 获取所有图片文件
        image_files = [f for f in os.listdir(src_images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"\n📁 处理 {src_split} 集：{len(image_files)} 张图片")
        
        for img_file in image_files:
            img_path = os.path.join(src_images_dir, img_file)
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(src_labels_dir, label_file)
            
            # 读取标注，获取类别ID
            class_ids = read_yolo_label(label_path)
            
            if len(class_ids) == 0:
                # 没有标注，跳过
                print(f"   ⚠️ 跳过 {img_file}：无标注")
                continue
            
            # 如果一张图片有多个缺陷，只取第一个类别（分类任务只需要一个标签）
            # 也可以选择出现次数最多的类别
            class_id = class_ids[0]
            
            # 目标路径
            dst_dir = os.path.join(OUTPUT_ROOT, dst_split, CLASS_NAMES[class_id])
            dst_path = os.path.join(dst_dir, img_file)
            
            # 复制图片
            shutil.copy2(img_path, dst_path)
            total_count += 1
        
        print(f"   ✅ {src_split} 转换完成")
    
    print("\n" + "=" * 60)
    print(f"✅ 转换完成！共处理 {total_count} 张图片")
    print(f"📂 输出目录: {OUTPUT_ROOT}")
    print("=" * 60)
    
    # 打印各目录统计
    print("\n📊 各数据集统计：")
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(OUTPUT_ROOT, split)
        if os.path.exists(split_path):
            total = 0
            for class_name in CLASS_NAMES:
                class_path = os.path.join(split_path, class_name)
                if os.path.exists(class_path):
                    count = len([f for f in os.listdir(class_path) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    total += count
                    print(f"   {split}/{class_name}: {count} 张")
            print(f"   {split} 合计: {total} 张\n")
    
    return OUTPUT_ROOT


# ==================== 4. 自定义数据集类 ====================
class ChipDataset(Dataset):
    """芯片缺陷数据集类"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = CLASS_NAMES
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        self._load_data()
    
    def _load_data(self):
        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(class_idx)
        
        print(f"✅ 从 {self.data_dir} 加载了 {len(self.images)} 张图片")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ==================== 5. 数据预处理变换 ====================
def get_data_transforms():
    """
    获取数据预处理变换
    对应课程第1章：最小-最大归一化 + 数据增强
    """
    
    # 训练集：归一化 + 数据增强
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # 自动将 [0,255] 归一化到 [0,1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 验证集/测试集：只做归一化
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return train_transform, val_transform


# ==================== 6. 创建DataLoader ====================
def create_data_loaders(batch_size=32, num_workers=0):
    """
    创建训练集、验证集、测试集的 DataLoader
    """
    
    train_dir = os.path.join(OUTPUT_ROOT, 'train')
    val_dir = os.path.join(OUTPUT_ROOT, 'val')
    test_dir = os.path.join(OUTPUT_ROOT, 'test')
    
    train_transform, val_transform = get_data_transforms()
    
    # 检查目录是否存在
    if not os.path.exists(train_dir):
        print(f"❌ 训练集目录不存在: {train_dir}")
        print("   请先运行转换函数 convert_yolo_to_classification()")
        return None, None, None
    
    # 创建数据集
    train_dataset = ChipDataset(train_dir, transform=train_transform)
    val_dataset = ChipDataset(val_dir, transform=val_transform) if os.path.exists(val_dir) else None
    test_dataset = ChipDataset(test_dir, transform=val_transform) if os.path.exists(test_dir) else None
    
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # 打印统计信息
    print("\n" + "=" * 50)
    print("📊 数据集加载完成")
    print("=" * 50)
    print(f"训练集: {len(train_dataset)} 张图片, {len(train_loader)} 批次")
    if val_dataset:
        print(f"验证集: {len(val_dataset)} 张图片, {len(val_loader)} 批次")
    if test_dataset:
        print(f"测试集: {len(test_dataset)} 张图片, {len(test_loader)} 批次")
    print("=" * 50)
    
    return train_loader, val_loader, test_loader


# ==================== 7. 检查数据集结构 ====================
def check_dataset():
    """检查 YOLO 格式数据集目录结构"""
    print("\n🔍 检查数据集目录结构...")
    print(f"数据根目录: {DATA_ROOT}")
    print("-" * 40)
    
    if not os.path.exists(DATA_ROOT):
        print(f"❌ 目录不存在: {DATA_ROOT}")
        return False
    
    for sub in ['train', 'test', 'valid']:
        sub_path = os.path.join(DATA_ROOT, sub)
        if os.path.exists(sub_path):
            images_path = os.path.join(sub_path, 'images')
            labels_path = os.path.join(sub_path, 'labels')
            img_count = len([f for f in os.listdir(images_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(images_path) else 0
            lbl_count = len([f for f in os.listdir(labels_path) 
                           if f.endswith('.txt')]) if os.path.exists(labels_path) else 0
            print(f"✅ {sub}: images={img_count} 张, labels={lbl_count} 个")
        else:
            print(f"⚠️ {sub}: 不存在")
    
    return True


# ==================== 8. 检查分类格式数据集 ====================
def check_classification_dataset():
    """检查转换后的分类格式数据集"""
    print("\n🔍 检查分类格式数据集...")
    print(f"输出目录: {OUTPUT_ROOT}")
    print("-" * 40)
    
    if not os.path.exists(OUTPUT_ROOT):
        print(f"❌ 目录不存在: {OUTPUT_ROOT}")
        print("   请先运行: convert_yolo_to_classification()")
        return False
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(OUTPUT_ROOT, split)
        if os.path.exists(split_path):
            total = 0
            for class_name in CLASS_NAMES:
                class_path = os.path.join(split_path, class_name)
                count = len([f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(class_path) else 0
                total += count
                print(f"   {split}/{class_name}: {count} 张")
            print(f"   {split} 合计: {total} 张\n")
        else:
            print(f"⚠️ {split}: 不存在")
    
    return True


# ==================== 9. 测试加载一个批次 ====================
def test_load():
    """测试数据加载"""
    print("\n📥 正在加载数据...")
    train_loader, val_loader, test_loader = create_data_loaders(batch_size=32)
    
    if train_loader:
        images, labels = next(iter(train_loader))
        print(f"\n✅ 测试成功！")
        print(f"   批次大小: {images.shape[0]} 张图片")
        print(f"   图片形状: {images.shape}")
        print(f"   标签: {labels.tolist()}")
    else:
        print("❌ 加载失败，请检查数据集路径")


# ==================== 10. 主程序 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("芯片缺陷检测 - 数据预处理模块")
    print("=" * 60)
    
    # 第1步：检查原始 YOLO 格式数据集
    check_dataset()
    
    # 第2步：检查是否已经转换过
    if os.path.exists(OUTPUT_ROOT):
        print(f"\n📂 检测到已转换的数据: {OUTPUT_ROOT}")
        check_classification_dataset()
        choice = input("\n是否重新转换？(y/n): ").strip().lower()
        if choice == 'y':
            shutil.rmtree(OUTPUT_ROOT)
            convert_yolo_to_classification()
    else:
        # 第3步：转换 YOLO → 分类格式
        convert_yolo_to_classification()
    
    # 第4步：检查转换后的数据
    check_classification_dataset()
    
    # 第5步：测试加载
    test_load()
    
    print("\n" + "=" * 60)
    print("✅ 数据预处理模块运行完成！")
    print("=" * 60)