# NestCharacter：嵌套字符生成、识别与对抗分析

本项目围绕“嵌套字符（大字内填充小字）”构建了一个完整流程：  
**生成数据 → 训练模型 → 检测层数 → OCR框提取 → 文本识别/对抗评估**。

当前仓库主入口为 `ui_demo.py`（桌面交互界面），核心推理模块在 `nested_char_detector.py`。

---

## 1. 项目目标与问题定义

项目解决两类问题：

1. **结构识别**：输入一张图，判断是否存在嵌套结构，并估计嵌套层数（0~5层）。
2. **内容提取**：定位可能包含字符的区域（OCR框），并输出候选文本（EasyOCR/模型OCR头）。

在应用层还扩展了“图像隐写式对抗”场景：  
通过“密钥大字 + 密文小字”的方式进行信息隐藏，并评估可检测性。

---

## 2. 技术架构（实现逻辑）

### 2.1 端到端推理流程

`NestedCharDetector.detect()` 的主流程：

1. **图像预处理**（CLAHE + 双边滤波）  
2. **线性变换参数估计与逆变换恢复**（旋转/缩放/翻转）
3. **高斯金字塔分析**（5层）  
4. **层数分类**（CNN + TTA）
5. **OCR框生成**（梯度 + 形态学 + 轮廓 + IoU合并）
6. **字符识别**（双头模型OCR分支，或UI中走EasyOCR）
7. **结果融合输出**

### 2.2 模块说明

- `NestCharacter.py`：基础“以小字填充大字笔画”的图像生成器。
- `training_data_generator.py`：批量生成训练数据（嵌套任务 + OCR任务）。
- `nested_char_detector.py`：核心检测器与模型定义（含单头/双头网络）。
- `ui_demo.py`：Tkinter UI，集成生成、检测、加解密演示、EasyOCR增强识别。
- `train_model.py`：训练脚本（单任务/复合任务入口）。

---

## 3. 数据来源与处理过程（重点）

### 3.1 数据来源

### A. 嵌套层数识别数据（合成）

来源：`TrainingDataGenerator.generate_dataset()` 自动生成。  
类别：`layer0`~`layer5`（共6类，0层作为负样本）。

当前仓库已有统计（`training_data/metadata.json`）：

- 每类 500 张
- 共 3000 张
- 划分：训练/验证/测试 = 70% / 15% / 15%
  - train: 2100
  - val: 450
  - test: 450

### B. OCR字符识别数据（合成）

来源：`TrainingDataGenerator.generate_ocr_dataset()`。  
字符池由**常用汉字 + 英文字母 + 数字**构成，映射文件见 `training_data/ocr_data/char_vocab.json`。

当前仓库已有统计（`training_data/ocr_data/ocr_metadata.json`）：

- 类别数：562
- 每类样本：30
- 总样本：16860
- 划分：
  - train: 11802
  - val: 2248
  - test: 2810

### 3.2 数据选取策略（“选了哪部分数据”）

### 嵌套任务

- 层数类别平衡（0~5每类等量），避免“某层过多”导致偏置。
- 每类样本内部：
  - 前50%：干净样本
  - 后50%：随机线性变换样本（旋转/缩放/翻转/噪声）

### OCR任务

- 每个字符类别样本固定数目。
- 每类前25%为相对干净样本，后75%做增强（旋转、透视、粗细、模糊、噪声）。
- 特意**不做水平翻转**（避免汉字语义失真）。

### 3.3 特征提取（“得到了什么特征”）

### 结构特征（用于检测与框生成）

- 旋转角：HoughLines从边缘直线估计。
- 缩放比：基于轮廓宽高比中位数估计 `scale_x/scale_y`。
- 翻转标志：投影与梯度不对称性估计 `flip_h/flip_v`。
- 金字塔特征（5层）：
  - 边缘密度（Canny）
  - 梯度幅值（Sobel）
  - 字符尺度估计（连通域面积中位数开方）

### 深度特征（用于层数/字符分类）

- 主干网络：ResNet风格残差块 + SE注意力（单通道输入）。
- 单头模型：只输出6类层数。
- 双头模型：共享主干，分别输出
  - 嵌套层数分类头（6类）
  - OCR字符分类头（562类）

### 推理增强

- TTA：原图 + 水平翻转 + ±5°旋转，概率平均后输出层数与置信度。

### 3.4 信息产出与后续使用（“得到了什么信息，如何用”）

模型最终输出：

- `estimated_layers`：层数预测（0~5）
- `confidence`：预测置信度
- `transform_params`：检测到的变换参数
- `pyramid_analysis`：各层边缘密度与字符尺度
- `ocr_boxes`：候选文字框
- `recognized_text`：识别文本（双头模型/外部OCR）

这些信息在后续中的用途：

1. **层数结果**用于“是否嵌套”判断与风险分级。
2. **变换参数**用于还原图像，提升后续识别稳定性。
3. **金字塔特征**用于选择最优尺度并自适应生成OCR框。
4. **OCR结果**用于提取主字/密文并做可读性评估。
5. **UI报告**整合结构检测 + OCR候选 + 字符尺寸分层统计。

---

## 4. 核心算法细节

### 4.1 最优金字塔层选择

框选层级根据边缘密度与字符尺寸打分：

- 密度目标约 0.1
- 字符尺寸目标约 30 像素
- 综合分数：`0.6 * density_score + 0.4 * size_score`

### 4.2 OCR框生成策略

- 梯度图归一化后用 Otsu 二值化
- 形态学闭操作连接断裂笔画
- 根据字符尺度自适应面积阈值：
  - `min_area ≈ (0.5 * char_size)^2`
  - `max_area ≈ max((3.0 * char_size)^2, 0.35 * level_area)`
- 用 IoU 合并重叠框（默认阈值 0.3）

---

## 5. 当前效果（基于仓库现有结果文件）

数据来源：
- `models/test_results.txt`
- `models/composite_test_results.txt`
- `test_output/export_20260416_201219/*`

### 5.1 单头层数模型（`nested_char_model.pth`）

- 测试准确率：**89.56%**（403/450）

说明：该结果对应“层数分类”主任务，效果稳定、可直接用于结构判别。

### 5.2 双头复合模型（`nested_char_model_dual.pth`）

- 嵌套检测准确率：50.22%（226/450）
- OCR识别准确率：0.53%（15/2810）

说明：当前复合训练结果明显低于单头层数模型，OCR头在大类别场景下尚未收敛充分。

### 5.3 UI导出样例（2026-04-16）

- `05_nested_char_1_2layers`：预测2层，置信度0.7961，主字识别为“试”
- `06_nested_char_2_3layers`：预测1层，置信度0.3142（低置信误判）
- `07_nested_char_3_4layers`：预测1层，置信度0.3897（高层样本识别困难）

结论：  
当前版本**更适合“是否存在嵌套结构/粗层数估计”**，对高复杂嵌套与细粒度OCR仍需继续优化。

---

## 6. 环境与安装

```bash
pip install -r requirements.txt
```

依赖见 `requirements.txt`：
- Pillow
- numpy
- opencv-python
- torch
- torchvision
- easyocr

---

## 7. 使用方式

### 7.1 启动图形界面（推荐）

```bash
python ui_demo.py
```

UI包含两大页签：
- 生成与检测
- 加密与解密分析

### 7.2 仅调用检测API

```python
from nested_char_detector import NestedCharDetector

detector = NestedCharDetector(model_path="models/nested_char_model.pth")
result = detector.detect("test_output/test_image.png")

print(result["estimated_layers"], result["confidence"], result["recognized_text"])
```

### 7.3 生成嵌套字符图

```bash
python NestCharacter.py -lt "海内存知己" -st "填充文本内容..." -lfs 300 -sfs 10 -w 6 -o output_art.png
```

---

## 8. 项目目录（核心部分）

```text
NestCharacter/
├─ ui_demo.py                         # 主UI入口（生成/检测/加解密）
├─ nested_char_detector.py            # 推理核心：变换恢复+金字塔+模型+OCR框
├─ training_data_generator.py         # 训练数据生成器（嵌套/OCR）
├─ train_model.py                     # 训练脚本入口
├─ NestCharacter.py                   # 命令行生成器
├─ models/
│  ├─ nested_char_model.pth
│  ├─ nested_char_model_dual.pth
│  ├─ test_results.txt
│  └─ composite_test_results.txt
├─ training_data/
│  ├─ metadata.json
│  └─ ocr_data/
│     ├─ char_vocab.json
│     └─ ocr_metadata.json
└─ test_output/export_20260416_201219/   # UI批量导出结果样例
```

---

## 9. 已知问题与建议

1. `train_model.py` 当前依赖 `NestedCharDataset / OCRPatchDataset`，但在现版本 `nested_char_detector.py` 中未找到对应实现，直接运行会报导入错误。  
2. 双头模型效果显著弱于单头模型，建议优先使用单头模型做层数识别。  
3. 若要提升OCR效果，建议：
   - 扩充每字符样本量（例如 100+）
   - 调整OCR损失权重与训练轮数
   - 先做OCR单任务预训练，再进行复合微调

---

## 10. 版本说明

- README更新时间：2026-04-17
- 结果说明基于仓库中可复核文件，不依赖外部数据源。
