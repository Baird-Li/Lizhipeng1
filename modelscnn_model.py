"""
项目名称：芯片缺陷检测CNN模型 (Chip Defect CNN Model)
模块功能：定义用于芯片缺陷分类的卷积神经网络结构，包含前向传播与参数统计
对应课程章节：
    - 第7章：前向传播与反向传播
    - 第8章：卷积神经网络、池化层、Dropout防止过拟合
"""

import torch
import torch.nn as nn


class ChipDefectCNN(nn.Module):
    """芯片缺陷检测卷积神经网络模型"""

    def __init__(self):
        """初始化网络各层（对应课程第8章：卷积神经网络结构定义）"""
        super(ChipDefectCNN, self).__init__()  # 调用父类构造函数初始化模块

        # 卷积层1：输入3通道，输出32通道，卷积核3x3，padding=1保持尺寸，后接ReLU和最大池化2x2
        self.conv1 = nn.Sequential(  # 使用Sequential组合卷积、激活、池化（第8章：典型CNN模块）
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 卷积操作，输出尺寸不变 (第8章：卷积层)
            nn.ReLU(),  # ReLU激活函数引入非线性 (第8章：激活函数)
            nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化，尺寸减半 (第8章：池化层)
        )

        # 卷积层2：输入32通道，输出64通道，同样保持尺寸后池化
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 卷积层3：输入64通道，输出128通道，保持尺寸后池化，最终特征图尺寸为28x28
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 展平操作：将三维特征图转换为一维向量（第8章：全连接层之前的展平）
        self.flatten = nn.Flatten()  # 自动计算展平后的维度

        # 全连接层1：输入维度为128*28*28，输出256个神经元，后接ReLU
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 28 * 28, 256),  # 全连接层 (第8章：全连接层)
            nn.ReLU()  # 激活函数
        )

        # Dropout层：训练时随机丢弃50%神经元，防止过拟合（第8章：Dropout正则化）
        self.dropout = nn.Dropout(p=0.5)  # p为丢弃概率

        # 全连接层2：输出4个类别的得分（不加Softmax，因为使用CrossEntropyLoss自动包含）
        self.fc2 = nn.Linear(256, 4)  # 输出层 (第8章：输出层设计)

    def forward(self, x):
        """
        前向传播过程（对应课程第7章：前向传播计算）
        参数:
            x: 输入张量，形状为 (batch_size, 3, 224, 224)
        返回:
            输出张量，形状为 (batch_size, 4)
        """
        # 依次通过三个卷积模块
        x = self.conv1(x)  # 输出形状 (batch,32,112,112)
        x = self.conv2(x)  # 输出形状 (batch,64,56,56)
        x = self.conv3(x)  # 输出形状 (batch,128,28,28)

        # 展平为二维张量 (batch, 128*28*28)
        x = self.flatten(x)

        # 通过第一个全连接层和ReLU
        x = self.fc1(x)  # 输出形状 (batch,256)

        # 应用Dropout（训练时生效，测试时自动关闭）
        x = self.dropout(x)

        # 通过输出层得到各类别得分
        x = self.fc2(x)  # 输出形状 (batch,4)

        return x

    def count_parameters(self):
        """
        统计模型可训练参数数量（对应课程第8章：模型复杂度评估）
        返回:
            参数量（整数）
        """
        # 遍历所有参数，计算元素总数
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params


# 测试代码（仅在直接运行本文件时执行）
if __name__ == "__main__":
    # 创建模型实例（对应课程第8章：模型实例化）
    model = ChipDefectCNN()
    print("模型结构：")
    print(model)  # 打印模型结构

    # 统计并打印参数量
    num_params = model.count_parameters()
    print(f"\n模型可训练参数量: {num_params:,}")  # 千位分隔符显示

    # 生成随机输入张量，模拟一张3x224x224的彩色图片（batch_size=1）
    dummy_input = torch.randn(1, 3, 224, 224)
    print(f"\n随机输入张量形状: {dummy_input.shape}")

    # 前向传播测试（对应课程第7章：前向传播验证）
    with torch.no_grad():  # 测试阶段不计算梯度
        output = model(dummy_input)
    print(f"输出张量形状: {output.shape}")  # 应为 [1,4]
    print("前向传播测试成功！")