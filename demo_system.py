# -*- coding: utf-8 -*-
"""
嵌套字符识别系统完整演示
展示：生成、变换、识别、OCR框检测的完整流程
"""

import cv2
import numpy as np
import os
import argparse
from training_data_generator import TrainingDataGenerator
from nested_char_detector import NestedCharDetector
import matplotlib.pyplot as plt
from matplotlib import patches


def visualize_detection_result(image_path: str, result: dict, save_path: str):
    """可视化检测结果"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图像: {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左图：原始图像
    axes[0].imshow(image_rgb)
    axes[0].set_title('原始图像', fontsize=14, fontproperties='SimHei')
    axes[0].axis('off')
    
    # 右图：检测结果
    axes[1].imshow(image_rgb)
    
    # 绘制OCR框
    for box in result['ocr_boxes']:
        x, y, w, h = box
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                 edgecolor='red', facecolor='none')
        axes[1].add_patch(rect)
    
    # 添加检测信息
    info_text = f"嵌套字: {'是' if result['is_nested'] else '否'}\n"
    info_text += f"层数: {result['estimated_layers']}\n"
    info_text += f"置信度: {result['confidence']:.3f}\n"
    info_text += f"检测框数: {result['num_boxes']}\n"
    info_text += f"旋转角度: {result['transform_params']['rotation']:.2f}°\n"
    info_text += f"缩放: ({result['transform_params']['scale_x']:.2f}, {result['transform_params']['scale_y']:.2f})"
    
    axes[1].text(10, 30, info_text, fontsize=10, color='yellow',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                verticalalignment='top', fontproperties='SimHei')
    
    axes[1].set_title(f'检测结果 (检测到{result["num_boxes"]}个字符框)', 
                     fontsize=14, fontproperties='SimHei')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"检测结果可视化已保存到: {save_path}")


def demo_generate_and_detect():
    """演示：生成嵌套字符并进行检测"""
    print("=" * 80)
    print("演示1: 生成嵌套字符并进行检测")
    print("=" * 80)
    
    font_path = 'C:/Windows/Fonts/simhei.ttf'
    output_dir = 'demo_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成不同层数的嵌套字符
    generator = TrainingDataGenerator(font_path, output_dir=output_dir)
    
    print("\n步骤1: 生成测试图像...")
    test_images = []
    for layers in [0, 1, 2, 3, 4, 5]:
        print(f"  生成{layers}层嵌套字符...")
        image = generator.generate_nested_char_image(layers, apply_transform=True)
        
        filename = f"demo_layer{layers}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, image)
        test_images.append((filepath, layers))
        print(f"    已保存: {filepath}")
    
    # 初始化检测器
    print("\n步骤2: 初始化检测器...")
    model_path = 'models/nested_char_model.pth'
    if os.path.exists(model_path):
        detector = NestedCharDetector(model_path=model_path)
        print(f"  已加载训练好的模型: {model_path}")
    else:
        detector = NestedCharDetector()
        print(f"  警告: 未找到训练好的模型，使用未训练的模型")
    
    # 检测每个图像
    print("\n步骤3: 检测嵌套字符...")
    results = []
    for filepath, true_layers in test_images:
        print(f"\n  检测: {os.path.basename(filepath)}")
        print(f"    真实层数: {true_layers}")
        
        result = detector.detect(filepath)
        results.append((filepath, true_layers, result))
        
        print(f"    预测层数: {result['estimated_layers']}")
        print(f"    置信度: {result['confidence']:.3f}")
        print(f"    OCR框数: {result['num_boxes']}")
        print(f"    变换参数: 旋转{result['transform_params']['rotation']:.1f}°, "
              f"缩放({result['transform_params']['scale_x']:.2f}, {result['transform_params']['scale_y']:.2f})")
        
        # 可视化结果
        vis_path = filepath.replace('.png', '_result.png')
        visualize_detection_result(filepath, result, vis_path)
    
    # 统计准确率
    print("\n步骤4: 统计结果...")
    correct = sum(1 for _, true, result in results if true == result['estimated_layers'])
    total = len(results)
    accuracy = 100 * correct / total if total > 0 else 0
    
    print(f"\n检测准确率: {accuracy:.2f}% ({correct}/{total})")
    
    return results


def demo_transform_recovery():
    """演示：线性变换恢复"""
    print("\n" + "=" * 80)
    print("演示2: 线性变换恢复")
    print("=" * 80)
    
    font_path = 'C:/Windows/Fonts/simhei.ttf'
    output_dir = 'demo_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成一个嵌套字符
    generator = TrainingDataGenerator(font_path, output_dir=output_dir)
    print("\n生成测试图像（带变换）...")
    image = generator.generate_nested_char_image(layers=3, apply_transform=True)
    
    original_path = os.path.join(output_dir, 'transform_original.png')
    cv2.imwrite(original_path, image)
    print(f"已保存原始图像: {original_path}")
    
    # 检测变换参数
    detector = NestedCharDetector()
    print("\n检测变换参数...")
    transform_params = detector.transform_recovery.detect_transform_params(image)
    
    print(f"  旋转角度: {transform_params.rotation:.2f}°")
    print(f"  X轴缩放: {transform_params.scale_x:.2f}")
    print(f"  Y轴缩放: {transform_params.scale_y:.2f}")
    print(f"  水平翻转: {transform_params.flip_horizontal}")
    print(f"  垂直翻转: {transform_params.flip_vertical}")
    
    # 应用逆变换
    print("\n应用逆变换恢复图像...")
    recovered_image = detector.transform_recovery.apply_inverse_transform(image, transform_params)
    
    recovered_path = os.path.join(output_dir, 'transform_recovered.png')
    cv2.imwrite(recovered_path, recovered_image)
    print(f"已保存恢复后的图像: {recovered_path}")
    
    # 可视化对比
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('变换后的图像', fontsize=12, fontproperties='SimHei')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(recovered_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title('恢复后的图像', fontsize=12, fontproperties='SimHei')
    axes[1].axis('off')
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'transform_comparison.png')
    plt.savefig(comparison_path, dpi=150)
    plt.close()
    
    print(f"对比图已保存: {comparison_path}")


def demo_pyramid_analysis():
    """演示：高斯金字塔分析"""
    print("\n" + "=" * 80)
    print("演示3: 高斯金字塔和边缘密度分析")
    print("=" * 80)
    
    font_path = 'C:/Windows/Fonts/simhei.ttf'
    output_dir = 'demo_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成测试图像
    generator = TrainingDataGenerator(font_path, output_dir=output_dir)
    print("\n生成测试图像...")
    image = generator.generate_nested_char_image(layers=3)
    
    test_path = os.path.join(output_dir, 'pyramid_test.png')
    cv2.imwrite(test_path, image)
    
    # 金字塔分析
    detector = NestedCharDetector()
    print("\n构建高斯金字塔...")
    pyramid_features = detector.pyramid_analyzer.analyze(image)
    
    print(f"\n金字塔层数: {len(pyramid_features.levels)}")
    for i, (level, density, char_size) in enumerate(zip(
        pyramid_features.levels, 
        pyramid_features.edge_densities,
        pyramid_features.char_sizes
    )):
        print(f"  层{i}: 尺寸{level.shape}, 边缘密度{density:.4f}, 字符大小{char_size:.2f}")
    
    # 可视化金字塔
    num_levels = len(pyramid_features.levels)
    fig, axes = plt.subplots(2, (num_levels + 1) // 2, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, level in enumerate(pyramid_features.levels):
        axes[i].imshow(level, cmap='gray')
        axes[i].set_title(f'层{i}\n密度:{pyramid_features.edge_densities[i]:.4f}',
                         fontsize=10, fontproperties='SimHei')
        axes[i].axis('off')
    
    plt.tight_layout()
    pyramid_path = os.path.join(output_dir, 'pyramid_visualization.png')
    plt.savefig(pyramid_path, dpi=150)
    plt.close()
    
    print(f"\n金字塔可视化已保存: {pyramid_path}")


def main():
    parser = argparse.ArgumentParser(description='嵌套字符识别系统演示')
    parser.add_argument('--demo', type=str, 
                       choices=['all', 'detect', 'transform', 'pyramid'],
                       default='all',
                       help='选择演示模式')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("嵌套字符识别系统 - 完整演示")
    print("=" * 80)
    
    if args.demo in ['all', 'detect']:
        demo_generate_and_detect()
    
    if args.demo in ['all', 'transform']:
        demo_transform_recovery()
    
    if args.demo in ['all', 'pyramid']:
        demo_pyramid_analysis()
    
    print("\n" + "=" * 80)
    print("演示完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()