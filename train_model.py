# -*- coding: utf-8 -*-
"""
模型训练脚本 - 训练嵌套字符识别神经网络
"""

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import argparse
from training_data_generator import TrainingDataGenerator, visualize_samples
from nested_char_detector import NestedCharDetector, NestedCharDataset
import matplotlib.pyplot as plt


def plot_training_history(train_losses: list, val_losses: list, save_path: str):
    """绘制训练历史"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"训练历史图已保存到: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='训练嵌套字符识别模型')
    
    # 数据生成参数
    parser.add_argument('--generate-data', action='store_true',
                       help='是否生成新的训练数据')
    parser.add_argument('--samples-per-class', type=int, default=500,
                       help='每类样本数量')
    parser.add_argument('--font-path', type=str, default='C:/Windows/Fonts/simhei.ttf',
                       help='字体文件路径')
    parser.add_argument('--data-dir', type=str, default='training_data',
                       help='数据集目录')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--num-epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='数据加载线程数')
    
    # 模型参数
    parser.add_argument('--model-save-path', type=str, default='models/nested_char_model.pth',
                       help='模型保存路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的模型路径')
    
    args = parser.parse_args()
    
    # 创建模型保存目录
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    
    # 步骤1：生成训练数据（如果需要）
    if args.generate_data:
        print("=" * 60)
        print("步骤1: 生成训练数据集")
        print("=" * 60)
        
        generator = TrainingDataGenerator(args.font_path, output_dir=args.data_dir)
        metadata = generator.generate_dataset(samples_per_class=args.samples_per_class)
        
        print("\n生成样本可视化...")
        visualize_samples(generator)
    
    # 步骤2：加载数据集
    print("\n" + "=" * 60)
    print("步骤2: 加载数据集")
    print("=" * 60)
    
    generator = TrainingDataGenerator(args.font_path, output_dir=args.data_dir)
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = generator.load_dataset_paths()
    
    print(f"训练集: {len(train_paths)}张图像")
    print(f"验证集: {len(val_paths)}张图像")
    print(f"测试集: {len(test_paths)}张图像")
    
    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 创建数据集和数据加载器
    train_dataset = NestedCharDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = NestedCharDataset(val_paths, val_labels, transform=val_transform)
    test_dataset = NestedCharDataset(test_paths, test_labels, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    
    # 步骤3：初始化模型
    print("\n" + "=" * 60)
    print("步骤3: 初始化模型")
    print("=" * 60)
    
    detector = NestedCharDetector(model_path=args.resume)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 步骤4：训练模型
    print("\n" + "=" * 60)
    print("步骤4: 训练模型")
    print("=" * 60)
    
    train_losses, val_losses = detector.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_path=args.model_save_path
    )
    
    # 绘制训练历史
    history_plot_path = os.path.join(os.path.dirname(args.model_save_path), 'training_history.png')
    plot_training_history(train_losses, val_losses, history_plot_path)
    
    # 步骤5：测试模型
    print("\n" + "=" * 60)
    print("步骤5: 测试模型")
    print("=" * 60)
    
    detector.model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = detector.model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_accuracy = 100 * test_correct / test_total
    print(f"测试集准确率: {test_accuracy:.2f}%")
    
    # 保存测试结果
    results_path = os.path.join(os.path.dirname(args.model_save_path), 'test_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(f"测试集准确率: {test_accuracy:.2f}%\n")
        f.write(f"测试样本数: {test_total}\n")
        f.write(f"正确预测数: {test_correct}\n")
    
    print(f"测试结果已保存到: {results_path}")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()