# 快速开始指南

本指南将帮助您快速上手嵌套字符识别系统。

## 📋 前置要求

- Python 3.8+
- pip包管理器
- （可选）NVIDIA GPU + CUDA 11.8+

## 🚀 5分钟快速开始

### 步骤1：安装依赖

```bash
# 克隆或下载项目后，进入项目目录
cd CharacterForMulti

# 安装依赖
pip install -r requirements.txt

# 如果有GPU（推荐）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 步骤2：运行演示

```bash
# 运行完整系统演示
python demo_system.py --demo all
```

这将：
- 生成不同层数的嵌套字符样本
- 应用各种线性变换
- 使用检测器识别嵌套结构
- 生成OCR检测框
- 保存可视化结果到 `demo_output/` 目录

### 步骤3：查看结果

演示完成后，查看 `demo_output/` 目录：
- `demo_layer*.png` - 生成的测试图像
- `*_result.png` - 检测结果可视化
- `transform_*.png` - 变换恢复演示
- `pyramid_*.png` - 金字塔分析可视化

## 📚 进阶使用

### 训练自己的模型

```bash
# 生成训练数据（每类500个样本）
python train_model.py --generate-data --samples-per-class 500

# 训练模型（50轮）
python train_model.py --num-epochs 50 --batch-size 32

# 查看训练结果
# - models/nested_char_model.pth (模型文件)
# - models/training_history.png (训练曲线)
# - models/test_results.txt (测试结果)
```

### 使用Python API

```python
from nested_char_detector import NestedCharDetector

# 初始化检测器（使用训练好的模型）
detector = NestedCharDetector(model_path='models/nested_char_model.pth')

# 检测图像
result = detector.detect('your_image.png')

# 打印结果
print(f"是否为嵌套字: {result['is_nested']}")
print(f"嵌套层数: {result['estimated_layers']}")
print(f"置信度: {result['confidence']:.3f}")
print(f"检测到 {result['num_boxes']} 个字符框")

# 访问详细信息
print(f"变换参数: {result['transform_params']}")
print(f"金字塔分析: {result['pyramid_analysis']}")
print(f"OCR框坐标: {result['ocr_boxes']}")
```

### 生成自定义嵌套字符

```python
from training_data_generator import TrainingDataGenerator
import cv2

# 创建生成器
generator = TrainingDataGenerator(
    font_path='C:/Windows/Fonts/simhei.ttf',
    output_dir='my_output'
)

# 生成3层嵌套字符（带变换）
image = generator.generate_nested_char_image(
    layers=3,
    size=(512, 512),
    apply_transform=True
)

# 保存图像
cv2.imwrite('my_nested_char.png', image)
```

## 🎯 常见任务

### 任务1：批量检测图像

```python
from nested_char_detector import NestedCharDetector
import os

detector = NestedCharDetector(model_path='models/nested_char_model.pth')

# 批量检测目录中的所有图像
image_dir = 'your_images/'
results = []

for filename in os.listdir(image_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        filepath = os.path.join(image_dir, filename)
        result = detector.detect(filepath)
        results.append((filename, result))
        
        print(f"{filename}: {result['estimated_layers']}层, "
              f"置信度{result['confidence']:.2f}")

# 统计
nested_count = sum(1 for _, r in results if r['is_nested'])
print(f"\n检测到 {nested_count}/{len(results)} 个嵌套字符")
```

### 任务2：只做变换恢复

```python
from nested_char_detector import LinearTransformRecovery
import cv2

# 加载图像
image = cv2.imread('transformed_image.png')

# 创建恢复器
recovery = LinearTransformRecovery()

# 检测变换参数
params = recovery.detect_transform_params(image)
print(f"检测到的变换: 旋转{params.rotation:.1f}°, "
      f"缩放({params.scale_x:.2f}, {params.scale_y:.2f})")

# 恢复图像
recovered = recovery.apply_inverse_transform(image, params)

# 保存
cv2.imwrite('recovered_image.png', recovered)
```

### 任务3：分析金字塔特征

```python
from nested_char_detector import GaussianPyramidAnalyzer
import cv2

# 加载图像
image = cv2.imread('test_image.png')

# 创建分析器
analyzer = GaussianPyramidAnalyzer(num_levels=5)

# 分析
features = analyzer.analyze(image)

# 打印各层特征
for i in range(len(features.levels)):
    print(f"层{i}:")
    print(f"  尺寸: {features.levels[i].shape}")
    print(f"  边缘密度: {features.edge_densities[i]:.4f}")
    print(f"  字符大小: {features.char_sizes[i]:.2f}像素")
```

## ⚙️ 配置选项

### 训练参数调优

```bash
# 小数据集快速训练（测试）
python train_model.py --generate-data --samples-per-class 100 --num-epochs 10

# 大数据集高精度训练
python train_model.py --generate-data --samples-per-class 1000 --num-epochs 100

# 调整学习率
python train_model.py --learning-rate 0.0001

# 使用更大的批次（需要更多GPU内存）
python train_model.py --batch-size 64
```

### 检测参数调整

在 `nested_char_detector.py` 中可以调整：

```python
# 金字塔层数
pyramid_analyzer = GaussianPyramidAnalyzer(num_levels=5)  # 改为3-7

# OCR框合并阈值
def _merge_overlapping_boxes(self, boxes, iou_threshold=0.3):  # 改为0.1-0.5
```

## 🐛 故障排除

### 问题1：CUDA不可用

```bash
# 检查CUDA是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 如果返回False，安装CUDA版本的PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 问题2：内存不足

```bash
# 减小批次大小
python train_model.py --batch-size 16

# 减少样本数量
python train_model.py --samples-per-class 200
```

### 问题3：字体文件找不到

```python
# Windows
font_path = 'C:/Windows/Fonts/simhei.ttf'

# macOS
font_path = '/System/Library/Fonts/STHeiti Light.ttc'

# Linux
font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
```

### 问题4：训练速度慢

```bash
# 使用GPU
# 检查是否使用GPU：训练时会显示 "使用设备: cuda"

# 增加数据加载线程
python train_model.py --num-workers 8

# 减少训练轮数
python train_model.py --num-epochs 20
```

## 📊 性能基准

在标准配置下（500样本/类，50轮）：

| 指标 | CPU | GPU (RTX 3060) |
|------|-----|----------------|
| 训练时间 | ~8小时 | ~2小时 |
| 推理速度 | ~2秒/图 | ~0.5秒/图 |
| 内存占用 | ~2GB | ~2GB |
| 测试准确率 | ~92% | ~92% |

## 🎓 学习资源

1. **分形理论基础**：阅读 `分形算法技术文档.md`
2. **代码结构**：查看各模块的docstring
3. **API文档**：使用 `help(NestedCharDetector)` 查看
4. **示例代码**：参考 `demo_system.py`

## 💡 最佳实践

1. **数据生成**：
   - 每类至少500个样本
   - 50%样本应用变换
   - 使用多种字体

2. **模型训练**：
   - 使用GPU加速
   - 监控验证集损失
   - 保存最佳模型

3. **检测优化**：
   - 先恢复变换
   - 使用训练好的模型
   - 结合分形特征

4. **生产部署**：
   - 批量处理提高效率
   - 缓存模型避免重复加载
   - 使用多进程并行

## 📞 获取帮助

- 查看完整文档：`README.md`
- 查看技术文档：`分形算法技术文档.md`
- 提交Issue：GitHub Issues
- 查看示例：`demo_system.py`

## 🎉 下一步

现在您已经掌握了基础使用，可以：

1. 训练自己的模型
2. 处理实际的嵌套字符图像
3. 调整参数优化性能
4. 扩展功能满足特定需求

祝您使用愉快！