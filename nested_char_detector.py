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
        
        # 检测翻转 - 通过分析文字方向
        flip_h = False
        flip_v = False
        
        # 简单的翻转检测：检查图像的质心分布
        moments = cv2.moments(edges)
        if moments['m00'] > 0:
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
            
            h, w = gray.shape
            # 如果质心偏离中心较多，可能有翻转
            if cx < w * 0.3 or cx > w * 0.7:
                flip_h = True
            if cy < h * 0.3 or cy > h * 0.7:
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


class NestedCharCNN(nn.Module):
    """嵌套字符识别卷积神经网络"""
    
    def __init__(self, num_classes: int = 6):
        """
        Args:
            num_classes: 分类数量（0层到5层）
        """
        super(NestedCharCNN, self).__init__()
        
        # 卷积层 - 提取多尺度特征
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 卷积块1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # 卷积块2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # 卷积块3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # 卷积块4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # 全局平均池化
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
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
        
        # 二值化梯度图
        threshold = np.percentile(gradient, 85)
        binary = (gradient > threshold).astype(np.uint8) * 255
        
        # 形态学操作连接断裂的笔画
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
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
    
    def detect(self, image_path: str) -> Dict:
        """完整的检测流程"""
        # 1. 加载图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        # 2. 检测并恢复线性变换
        transform_params = self.transform_recovery.detect_transform_params(image)
        recovered_image = self.transform_recovery.apply_inverse_transform(image, transform_params)
        
        # 3. 构建高斯金字塔并分析
        pyramid_features = self.pyramid_analyzer.analyze(recovered_image)
        
        # 4. 使用神经网络识别嵌套层数
        gray = cv2.cvtColor(recovered_image, cv2.COLOR_BGR2GRAY)
        pil_image = Image.fromarray(gray)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_layers = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_layers].item()
        
        # 5. 生成OCR检测框
        ocr_boxes = self.box_generator.generate_boxes(recovered_image, pyramid_features)
        
        # 6. 整合结果
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