# -*- coding: utf-8 -*-
"""
嵌套字符检测器 - 基于深度学习和分形理论的综合识别系统
包含：线性变换恢复、高斯金字塔、边缘密度分析、神经网络识别
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


@dataclass
class TransformParams:
    """线性变换参数"""
    rotation: float  # 旋转角度
    scale_x: float  # X轴缩放
    scale_y: float  # Y轴缩放
    shear_x: float  # X轴剪切
    shear_y: float  # Y轴剪切
    flip_horizontal: bool  # 水平翻转
    flip_vertical: bool  # 垂直翻转


@dataclass
class PyramidFeatures:
    """高斯金字塔特征"""
    levels: List[np.ndarray]  # 各层图像
    edge_densities: List[float]  # 各层边缘密度
    gradients: List[np.ndarray]  # 各层梯度
    char_sizes: List[float]  # 估计的字符大小


class LinearTransformRecovery:
    """线性变换恢复模块 - 将变换后的图像恢复到正常状态"""
    
    def __init__(self):
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def detect_transform_params(self, image: np.ndarray) -> TransformParams:
        """检测图像的线性变换参数"""
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 检测边缘
        edges = cv2.Canny(gray, 50, 150)
        
        # 使用霍夫变换检测直线来估计旋转角度
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        rotation = 0.0
        
        if lines is not None:
            angles = []
            for line in lines[:10]:  # 只取前10条线
                rho, theta = line[0]
                angle = np.degrees(theta)
                # 将角度归一化到[-45, 45]范围
                if angle > 135:
                    angle -= 180
                elif angle > 45:
                    angle -= 90
                angles.append(angle)
            
            if angles:
                rotation = np.median(angles)
        
        # 检测缩放比例 - 通过分析字符的宽高比
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        aspect_ratios = []
        
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # 过滤小轮廓
                x, y, w, h = cv2.boundingRect(contour)
                if h > 0:
                    aspect_ratios.append(w / h)
        
        scale_x = 1.0
        scale_y = 1.0
        
        if aspect_ratios:
            median_ratio = np.median(aspect_ratios)
            # 假设正常字符的宽高比约为0.8-1.2
            if median_ratio > 1.5:  # 横向拉伸
                scale_x = 1.0 / np.sqrt(median_ratio)
                scale_y = np.sqrt(median_ratio)
            elif median_ratio < 0.5:  # 纵向拉伸
                scale_x = np.sqrt(1.0 / median_ratio)
                scale_y = 1.0 / np.sqrt(1.0 / median_ratio)
        
        # 检测翻转 - 通过梯度方向直方图和投影分析
        flip_h = False
        flip_v = False

        h, w = gray.shape

        # 使用水平/垂直投影分析检测翻转
        # 计算水平投影（每行像素和）
        h_proj = np.sum(edges, axis=1).astype(np.float64)
        # 计算垂直投影（每列像素和）
        v_proj = np.sum(edges, axis=0).astype(np.float64)

        # 分析投影的不对称性 - 比较上下/左右半部分的梯度能量
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # 水平翻转检测：比较左右半部分的梯度方向分布
        left_grad = np.mean(np.abs(grad_x[:, :w//2]))
        right_grad = np.mean(np.abs(grad_x[:, w//2:]))
        if left_grad > 0 and right_grad > 0:
            h_asymmetry = abs(left_grad - right_grad) / max(left_grad, right_grad)
            # 结合投影重心偏移
            if len(v_proj) > 0 and np.sum(v_proj) > 0:
                v_center = np.average(np.arange(len(v_proj)), weights=v_proj)
                v_offset = abs(v_center - w / 2) / (w / 2)
                if h_asymmetry > 0.3 and v_offset > 0.15:
                    flip_h = True

        # 垂直翻转检测：比较上下半部分的梯度方向分布
        top_grad = np.mean(np.abs(grad_y[:h//2, :]))
        bottom_grad = np.mean(np.abs(grad_y[h//2:, :]))
        if top_grad > 0 and bottom_grad > 0:
            v_asymmetry = abs(top_grad - bottom_grad) / max(top_grad, bottom_grad)
            if len(h_proj) > 0 and np.sum(h_proj) > 0:
                h_center = np.average(np.arange(len(h_proj)), weights=h_proj)
                h_offset = abs(h_center - h / 2) / (h / 2)
                if v_asymmetry > 0.3 and h_offset > 0.15:
                    flip_v = True
        
        return TransformParams(
            rotation=rotation,
            scale_x=scale_x,
            scale_y=scale_y,
            shear_x=0.0,
            shear_y=0.0,
            flip_horizontal=flip_h,
            flip_vertical=flip_v
        )
    
    def apply_inverse_transform(self, image: np.ndarray, params: TransformParams) -> np.ndarray:
        """应用逆变换恢复图像"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # 处理翻转
        if params.flip_horizontal:
            image = cv2.flip(image, 1)
        if params.flip_vertical:
            image = cv2.flip(image, 0)
        
        # 处理旋转
        if abs(params.rotation) > 0.5:
            rotation_matrix = cv2.getRotationMatrix2D(center, -params.rotation, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        
        # 处理缩放
        if abs(params.scale_x - 1.0) > 0.05 or abs(params.scale_y - 1.0) > 0.05:
            new_w = int(w * params.scale_x)
            new_h = int(h * params.scale_y)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            # 裁剪或填充回原始大小
            if new_w > w or new_h > h:
                start_x = (new_w - w) // 2
                start_y = (new_h - h) // 2
                image = image[start_y:start_y+h, start_x:start_x+w]
            else:
                pad_x = (w - new_w) // 2
                pad_y = (h - new_h) // 2
                image = cv2.copyMakeBorder(image, pad_y, h-new_h-pad_y, 
                                          pad_x, w-new_w-pad_x,
                                          cv2.BORDER_REPLICATE)
        
        return image


class GaussianPyramidAnalyzer:
    """高斯金字塔分析器 - 多尺度边缘密度计算"""
    
    def __init__(self, num_levels: int = 5):
        self.num_levels: int = num_levels
        
    def build_pyramid(self, image: np.ndarray) -> List[np.ndarray]:
        """构建高斯金字塔"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        pyramid = [gray]
        current = gray
        
        for _ in range(self.num_levels - 1):
            current = cv2.pyrDown(current)
            pyramid.append(current)
        
        return pyramid
    
    def calculate_edge_density(self, image: np.ndarray) -> float:
        """计算局部边缘密度"""
        # 使用Canny边缘检测
        edges = cv2.Canny(image, 50, 150)
        
        # 计算边缘密度（边缘像素占比）
        edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
        
        return edge_density
    
    def calculate_gradient_magnitude(self, image: np.ndarray) -> np.ndarray:
        """计算梯度幅值"""
        # Sobel算子计算梯度
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # 梯度幅值
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return gradient_magnitude
    
    def estimate_char_size(self, image: np.ndarray, gradient: np.ndarray) -> float:
        """估计字符大小"""
        # 使用梯度信息估计字符的特征尺寸
        # 找到梯度的峰值位置
        threshold = np.percentile(gradient, 90)
        strong_edges = gradient > threshold
        
        # 使用连通组件分析
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            strong_edges.astype(np.uint8), connectivity=8
        )
        
        if num_labels <= 1:
            return 0.0
        
        # 计算平均组件大小（跳过背景）
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) == 0:
            return 0.0
        
        # 使用中位数作为特征尺寸
        char_size = np.sqrt(np.median(areas))
        
        return char_size
    
    def analyze(self, image: np.ndarray) -> PyramidFeatures:
        """完整的金字塔分析"""
        pyramid = self.build_pyramid(image)
        
        edge_densities = []
        gradients = []
        char_sizes = []
        
        for level_img in pyramid:
            # 计算边缘密度
            edge_density = self.calculate_edge_density(level_img)
            edge_densities.append(edge_density)
            
            # 计算梯度
            gradient = self.calculate_gradient_magnitude(level_img)
            gradients.append(gradient)
            
            # 估计字符大小
            char_size = self.estimate_char_size(level_img, gradient)
            char_sizes.append(char_size)
        
        return PyramidFeatures(
            levels=pyramid,
            edge_densities=edge_densities,
            gradients=gradients,
            char_sizes=char_sizes
        )


class SEBlock(nn.Module):
    """Squeeze-and-Excitation注意力模块"""

    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """带SE注意力的残差块"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class NestedCharCNN(nn.Module):
    """嵌套字符识别卷积神经网络 - 带残差连接和SE注意力"""

    def __init__(self, num_classes: int = 6):
        """
        Args:
            num_classes: 分类数量（0层到5层）
        """
        super(NestedCharCNN, self).__init__()

        # 初始卷积层
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 残差块组
        self.layer1 = self._make_layer(32, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def _make_layer(self, in_channels: int, out_channels: int,
                    num_blocks: int, stride: int) -> nn.Sequential:
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class NestedCharDataset(Dataset):
    """嵌套字符数据集"""
    
    def __init__(self, image_paths: List[str], labels: List[int], 
                 transform: Optional[transforms.Compose] = None):
        self.image_paths: List[str] = image_paths
        self.labels: List[int] = labels
        self.transform: Optional[transforms.Compose] = transform
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # 加载图像
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"无法加载图像: {self.image_paths[idx]}")
        
        # 调整大小
        image = cv2.resize(image, (256, 256))
        
        # 转换为PIL图像
        image = Image.fromarray(image)
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        label = self.labels[idx]
        
        return image, label


class OCRBoxGenerator:
    """OCR框生成器 - 基于梯度和字符大小关系"""
    
    def __init__(self):
        self.pyramid_analyzer: GaussianPyramidAnalyzer = GaussianPyramidAnalyzer()
        
    def generate_boxes(self, image: np.ndarray, 
                      pyramid_features: PyramidFeatures) -> List[Tuple[int, int, int, int]]:
        """生成OCR检测框"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 使用最优层级的梯度信息
        best_level = self._select_best_level(pyramid_features)
        gradient = pyramid_features.gradients[best_level]
        char_size = pyramid_features.char_sizes[best_level]
        
        # 根据字符大小调整检测参数
        if char_size > 0:
            min_area = int((char_size * 0.5) ** 2)
            max_area = int((char_size * 3.0) ** 2)
        else:
            min_area = 100
            max_area = 10000
        
        # 自适应二值化梯度图 - 使用Otsu替代固定百分位
        grad_normalized = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, binary = cv2.threshold(grad_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 自适应形态学核大小 - 根据字符大小调整
        kernel_size = max(3, min(7, int(char_size / 5))) if char_size > 0 else 3
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if min_area <= area <= max_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # 缩放回原始图像尺寸
                scale = 2 ** best_level
                x *= scale
                y *= scale
                w *= scale
                h *= scale
                
                boxes.append((x, y, w, h))
        
        # 合并重叠的框
        boxes = self._merge_overlapping_boxes(boxes)
        
        return boxes
    
    def _select_best_level(self, pyramid_features: PyramidFeatures) -> int:
        """选择最优的金字塔层级"""
        # 选择边缘密度适中且字符大小合理的层级
        edge_densities = pyramid_features.edge_densities
        char_sizes = pyramid_features.char_sizes
        
        scores = []
        for i, (density, size) in enumerate(zip(edge_densities, char_sizes)):
            # 边缘密度得分（0.05-0.15为最优）
            density_score = 1.0 - abs(density - 0.1) / 0.1
            density_score = max(0, density_score)
            
            # 字符大小得分（10-50像素为最优）
            size_score = 1.0 - abs(size - 30) / 30
            size_score = max(0, size_score)
            
            # 综合得分
            score = density_score * 0.6 + size_score * 0.4
            scores.append(score)
        
        best_level = np.argmax(scores)
        return best_level
    
    def _merge_overlapping_boxes(self, boxes: List[Tuple[int, int, int, int]], 
                                 iou_threshold: float = 0.3) -> List[Tuple[int, int, int, int]]:
        """合并重叠的检测框"""
        if len(boxes) == 0:
            return []
        
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        merged = []
        
        while boxes:
            current = boxes.pop(0)
            merged.append(current)
            
            # 检查剩余框是否与当前框重叠
            i = 0
            while i < len(boxes):
                if self._calculate_iou(current, boxes[i]) > iou_threshold:
                    # 合并框
                    current = self._merge_two_boxes(current, boxes[i])
                    merged[-1] = current
                    boxes.pop(i)
                else:
                    i += 1
        
        return merged
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], 
                      box2: Tuple[int, int, int, int]) -> float:
        """计算两个框的IoU"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # 计算交集
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # 计算并集
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_two_boxes(self, box1: Tuple[int, int, int, int], 
                        box2: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """合并两个框"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        x = min(x1, x2)
        y = min(y1, y2)
        w = max(x1 + w1, x2 + w2) - x
        h = max(y1 + h1, y2 + h2) - y
        
        return (x, y, w, h)


class NestedCharDetector:
    """嵌套字符检测器 - 整合所有模块的主类"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.transform_recovery: LinearTransformRecovery = LinearTransformRecovery()
        self.pyramid_analyzer: GaussianPyramidAnalyzer = GaussianPyramidAnalyzer()
        self.box_generator: OCRBoxGenerator = OCRBoxGenerator()
        
        # 初始化神经网络模型
        self.model: NestedCharCNN = NestedCharCNN(num_classes=6).to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"已加载模型: {model_path}")
        
        self.model.eval()
        
        # 图像预处理
        self.transform: transforms.Compose = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """图像预处理 - CLAHE对比度增强 + 双边滤波去噪"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # CLAHE自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 双边滤波 - 去噪同时保留边缘
        denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)

        # 转回BGR
        if len(image.shape) == 3:
            result = image.copy()
            result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            # 用增强后的灰度图替换亮度通道
            ratio = np.where(result_gray > 0,
                             denoised.astype(np.float32) / result_gray.astype(np.float32),
                             1.0)
            for c in range(3):
                result[:, :, c] = np.clip(result[:, :, c] * ratio, 0, 255).astype(np.uint8)
            return result
        return denoised

    def _tta_inference(self, pil_image: Image.Image) -> Tuple[int, float]:
        """测试时增强(TTA) - 多次推理取平均提高鲁棒性"""
        augmented_images = []

        # 原始图像
        augmented_images.append(self.transform(pil_image))

        # 水平翻转
        flipped_h = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
        augmented_images.append(self.transform(flipped_h))

        # 轻微旋转 +5度
        rotated_pos = pil_image.rotate(-5, fillcolor=255)
        augmented_images.append(self.transform(rotated_pos))

        # 轻微旋转 -5度
        rotated_neg = pil_image.rotate(5, fillcolor=255)
        augmented_images.append(self.transform(rotated_neg))

        # 批量推理
        batch = torch.stack(augmented_images).to(self.device)

        with torch.no_grad():
            outputs = self.model(batch)
            all_probs = F.softmax(outputs, dim=1)
            # 平均所有增强版本的概率
            avg_probs = torch.mean(all_probs, dim=0)
            predicted_layers = torch.argmax(avg_probs).item()
            confidence = avg_probs[predicted_layers].item()

        return predicted_layers, confidence

    def detect(self, image_path: str) -> Dict:
        """完整的检测流程"""
        # 1. 加载图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")

        # 2. 图像预处理 - 对比度增强和去噪
        image = self._preprocess_image(image)

        # 3. 检测并恢复线性变换
        transform_params = self.transform_recovery.detect_transform_params(image)
        recovered_image = self.transform_recovery.apply_inverse_transform(image, transform_params)

        # 4. 构建高斯金字塔并分析
        pyramid_features = self.pyramid_analyzer.analyze(recovered_image)

        # 5. 使用TTA增强的神经网络识别嵌套层数
        gray = cv2.cvtColor(recovered_image, cv2.COLOR_BGR2GRAY)
        pil_image = Image.fromarray(gray)
        predicted_layers, confidence = self._tta_inference(pil_image)

        # 6. 生成OCR检测框
        ocr_boxes = self.box_generator.generate_boxes(recovered_image, pyramid_features)
        
        # 7. 整合结果
        result = {
            'image_path': image_path,
            'is_nested': predicted_layers > 0,
            'estimated_layers': predicted_layers,
            'confidence': confidence,
            'transform_params': {
                'rotation': transform_params.rotation,
                'scale_x': transform_params.scale_x,
                'scale_y': transform_params.scale_y,
                'flip_horizontal': transform_params.flip_horizontal,
                'flip_vertical': transform_params.flip_vertical
            },
            'pyramid_analysis': {
                'edge_densities': pyramid_features.edge_densities,
                'char_sizes': pyramid_features.char_sizes
            },
            'ocr_boxes': ocr_boxes,
            'num_boxes': len(ocr_boxes)
        }
        
        return result
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
             num_epochs: int = 50, learning_rate: float = 0.001,
             save_path: str = 'models/nested_char_model.pth'):
        """训练神经网络模型"""
        self.model.train()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * train_correct / train_total:.2f}%'
                })
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = 100 * train_correct / train_total
            train_losses.append(avg_train_loss)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * val_correct / val_total
            val_losses.append(avg_val_loss)
            
            print(f'\nEpoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
            print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
            
            # 学习率调整
            scheduler.step(avg_val_loss)
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
                print(f'  保存最佳模型到: {save_path}')
        
        return train_losses, val_losses