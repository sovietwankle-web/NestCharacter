# -*- coding: utf-8 -*-
"""
系统测试脚本 - 验证各个模块的功能
"""

import os
import sys
import numpy as np
import cv2
from typing import Tuple


def test_imports() -> bool:
    """测试所有模块导入"""
    print("=" * 60)
    print("测试1: 模块导入")
    print("=" * 60)
    
    try:
        import torch
        print(f"[OK] PyTorch {torch.__version__}")
        print(f"  CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        
        import torchvision
        print(f"[OK] TorchVision {torchvision.__version__}")
        
        import cv2
        print(f"[OK] OpenCV {cv2.__version__}")
        
        import numpy as np
        print(f"[OK] NumPy {np.__version__}")
        
        from PIL import Image
        print(f"[OK] Pillow {Image.__version__}")
        
        import scipy
        print(f"[OK] SciPy {scipy.__version__}")
        
        import sklearn
        print(f"[OK] Scikit-learn {sklearn.__version__}")
        
        import matplotlib
        print(f"[OK] Matplotlib {matplotlib.__version__}")
        
        from training_data_generator import TrainingDataGenerator
        print("[OK] TrainingDataGenerator")
        
        from nested_char_detector import (
            NestedCharDetector, LinearTransformRecovery,
            GaussianPyramidAnalyzer, OCRBoxGenerator
        )
        print("[OK] NestedCharDetector及相关模块")
        
        print("\n所有模块导入成功！")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] 导入失败: {str(e)}")
        return False


def test_data_generator() -> bool:
    """测试数据生成器"""
    print("\n" + "=" * 60)
    print("测试2: 数据生成器")
    print("=" * 60)
    
    try:
        from training_data_generator import TrainingDataGenerator
        
        font_path = 'C:/Windows/Fonts/simhei.ttf'
        if not os.path.exists(font_path):
            print(f"警告: 字体文件不存在 {font_path}")
            print("尝试使用默认字体...")
        
        generator = TrainingDataGenerator(font_path, output_dir='test_output')
        print("[OK] 生成器初始化成功")
        
        # 测试生成不同层数的图像
        for layers in [0, 1, 2, 3]:
            image = generator.generate_nested_char_image(layers, size=(256, 256))
            assert image is not None, f"{layers}层图像生成失败"
            assert image.shape == (256, 256, 3), f"{layers}层图像尺寸错误"
            print(f"[OK] 生成{layers}层嵌套字符: {image.shape}")
        
        # 测试带变换的生成
        image_transformed = generator.generate_nested_char_image(2, apply_transform=True)
        assert image_transformed is not None, "变换图像生成失败"
        print(f"[OK] 生成带变换的图像: {image_transformed.shape}")
        
        print("\n数据生成器测试通过！")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] 数据生成器测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_transform_recovery() -> bool:
    """测试变换恢复"""
    print("\n" + "=" * 60)
    print("测试3: 线性变换恢复")
    print("=" * 60)
    
    try:
        from nested_char_detector import LinearTransformRecovery
        from training_data_generator import TrainingDataGenerator
        
        # 生成测试图像
        generator = TrainingDataGenerator('C:/Windows/Fonts/simhei.ttf')
        image = generator.generate_nested_char_image(2, apply_transform=True)
        print(f"[OK] 生成测试图像: {image.shape}")
        
        # 测试变换检测
        recovery = LinearTransformRecovery()
        params = recovery.detect_transform_params(image)
        print(f"[OK] 检测变换参数:")
        print(f"  旋转: {params.rotation:.2f}°")
        print(f"  缩放: ({params.scale_x:.2f}, {params.scale_y:.2f})")
        print(f"  翻转: H={params.flip_horizontal}, V={params.flip_vertical}")
        
        # 测试逆变换
        recovered = recovery.apply_inverse_transform(image, params)
        assert recovered is not None, "逆变换失败"
        assert recovered.shape == image.shape, "恢复图像尺寸不匹配"
        print(f"[OK] 应用逆变换: {recovered.shape}")
        
        print("\n变换恢复测试通过！")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] 变换恢复测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_pyramid_analyzer() -> bool:
    """测试高斯金字塔分析"""
    print("\n" + "=" * 60)
    print("测试4: 高斯金字塔分析")
    print("=" * 60)
    
    try:
        from nested_char_detector import GaussianPyramidAnalyzer
        from training_data_generator import TrainingDataGenerator
        
        # 生成测试图像
        generator = TrainingDataGenerator('C:/Windows/Fonts/simhei.ttf')
        image = generator.generate_nested_char_image(3)
        print(f"[OK] 生成测试图像: {image.shape}")
        
        # 测试金字塔构建
        analyzer = GaussianPyramidAnalyzer(num_levels=5)
        features = analyzer.analyze(image)
        
        assert len(features.levels) == 5, "金字塔层数错误"
        assert len(features.edge_densities) == 5, "边缘密度数量错误"
        assert len(features.gradients) == 5, "梯度数量错误"
        assert len(features.char_sizes) == 5, "字符大小数量错误"
        
        print(f"[OK] 构建{len(features.levels)}层金字塔")
        
        for i in range(len(features.levels)):
            print(f"  层{i}: 尺寸{features.levels[i].shape}, "
                  f"边缘密度{features.edge_densities[i]:.4f}, "
                  f"字符大小{features.char_sizes[i]:.2f}")
        
        print("\n金字塔分析测试通过！")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] 金字塔分析测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_neural_network() -> bool:
    """测试神经网络"""
    print("\n" + "=" * 60)
    print("测试5: 神经网络模型")
    print("=" * 60)
    
    try:
        import torch
        from nested_char_detector import NestedCharCNN
        
        # 创建模型
        model = NestedCharCNN(num_classes=6)
        print(f"[OK] 创建模型")
        
        # 测试前向传播
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        print(f"[OK] 模型移至设备: {device}")
        
        # 创建随机输入
        batch_size = 4
        input_tensor = torch.randn(batch_size, 1, 256, 256).to(device)
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (batch_size, 6), f"输出形状错误: {output.shape}"
        print(f"[OK] 前向传播成功: 输入{input_tensor.shape} -> 输出{output.shape}")
        
        # 测试softmax
        probabilities = torch.nn.functional.softmax(output, dim=1)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(batch_size).to(device), atol=1e-5), \
            "概率和不为1"
        print(f"[OK] Softmax输出正确")
        
        print("\n神经网络测试通过！")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] 神经网络测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_ocr_box_generator() -> bool:
    """测试OCR框生成"""
    print("\n" + "=" * 60)
    print("测试6: OCR框生成器")
    print("=" * 60)
    
    try:
        from nested_char_detector import OCRBoxGenerator, GaussianPyramidAnalyzer
        from training_data_generator import TrainingDataGenerator
        
        # 生成测试图像
        generator = TrainingDataGenerator('C:/Windows/Fonts/simhei.ttf')
        image = generator.generate_nested_char_image(2)
        print(f"[OK] 生成测试图像: {image.shape}")
        
        # 分析金字塔
        pyramid_analyzer = GaussianPyramidAnalyzer()
        features = pyramid_analyzer.analyze(image)
        print(f"[OK] 金字塔分析完成")
        
        # 生成OCR框
        box_generator = OCRBoxGenerator()
        boxes = box_generator.generate_boxes(image, features)
        
        print(f"[OK] 生成{len(boxes)}个OCR框")
        
        for i, box in enumerate(boxes[:5]):  # 只显示前5个
            x, y, w, h = box
            print(f"  框{i}: x={x}, y={y}, w={w}, h={h}")
        
        print("\nOCR框生成测试通过！")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] OCR框生成测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_full_detector() -> bool:
    """测试完整检测器"""
    print("\n" + "=" * 60)
    print("测试7: 完整检测器（不含训练模型）")
    print("=" * 60)
    
    try:
        from nested_char_detector import NestedCharDetector
        from training_data_generator import TrainingDataGenerator
        
        # 生成测试图像
        generator = TrainingDataGenerator('C:/Windows/Fonts/simhei.ttf')
        test_dir = 'test_output'
        os.makedirs(test_dir, exist_ok=True)
        
        test_image_path = os.path.join(test_dir, 'test_image.png')
        image = generator.generate_nested_char_image(3, apply_transform=True)
        cv2.imwrite(test_image_path, image)
        print(f"[OK] 生成测试图像: {test_image_path}")
        
        # 初始化检测器（不加载模型）
        detector = NestedCharDetector()
        print(f"[OK] 初始化检测器")
        
        # 运行检测
        result = detector.detect(test_image_path)
        
        print(f"[OK] 检测完成:")
        print(f"  是否为嵌套字: {result['is_nested']}")
        print(f"  估计层数: {result['estimated_layers']}")
        print(f"  置信度: {result['confidence']:.3f}")
        print(f"  OCR框数: {result['num_boxes']}")
        print(f"  变换参数: 旋转{result['transform_params']['rotation']:.1f}°")
        
        assert 'is_nested' in result, "缺少is_nested字段"
        assert 'estimated_layers' in result, "缺少estimated_layers字段"
        assert 'confidence' in result, "缺少confidence字段"
        assert 'ocr_boxes' in result, "缺少ocr_boxes字段"
        
        print("\n完整检测器测试通过！")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] 完整检测器测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("嵌套字符识别系统 - 完整测试")
    print("=" * 60)
    
    tests = [
        ("模块导入", test_imports),
        ("数据生成器", test_data_generator),
        ("变换恢复", test_transform_recovery),
        ("金字塔分析", test_pyramid_analyzer),
        ("神经网络", test_neural_network),
        ("OCR框生成", test_ocr_box_generator),
        ("完整检测器", test_full_detector),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n[FAIL] 测试 '{name}' 异常: {str(e)}")
            results.append((name, False))
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "[OK] 通过" if success else "[FAIL] 失败"
        print(f"{status} - {name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n[SUCCESS] 所有测试通过！系统运行正常。")
        return True
    else:
        print(f"\n[WARNING]  {total - passed} 个测试失败，请检查错误信息。")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)