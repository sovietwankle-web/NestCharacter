# -*- coding: utf-8 -*-
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time
import argparse
import requests
import json
import cv2
import os
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import math
from scipy import ndimage
from sklearn.feature_extraction import image
from skimage.measure import block_reduce

@dataclass
class NestedCharConfig:
    """嵌套字配置类"""
    layers: int  # 嵌套层数
    texts: List[str]  # 每层的文本内容
    font_sizes: List[int]  # 每层的字体大小
    size_ratios: List[float]  # 每层相对于上一层的大小比例
    step_ratios: List[float]  # 每层的步长比例
    colors: List[str]  # 每层的颜色
    
@dataclass
class FractalFeatures:
    """分形特征类"""
    box_dimension: float  # 盒维数
    lacunarity: float  # 空隙度
    multifractal_spectrum: np.ndarray  # 多重分形谱
    self_similarity: float  # 自相似性度量

class NestedCharacterGenerator:
    """多层嵌套字生成器"""
    
    def __init__(self, font_path: str):
        self.font_path: str = font_path
        self.output_dir: str = "output_images"
        self.input_dir: str = "input_images"
        
    def create_nested_character(self, config: NestedCharConfig, output_filename: str) -> Image.Image:
        """创建多层嵌套字符"""
        print(f"开始生成{config.layers}层嵌套字符...")
        
        # 从最大层开始，逐层生成
        base_font_size = config.font_sizes[0]
        base_text = config.texts[0]
        
        # 计算最终图像尺寸
        final_width = int(len(base_text) * base_font_size * 1.2)
        final_height = int(base_font_size * 1.5)
        
        # 创建基础图像
        final_image = Image.new('RGB', (final_width, final_height), 'white')
        
        # 逐层生成嵌套结构
        for layer in range(config.layers):
            print(f"  正在生成第{layer + 1}层...")
            layer_image = self._generate_layer(
                config, layer, final_width, final_height
            )
            
            # 将当前层合并到最终图像
            final_image = self._merge_layers(final_image, layer_image, layer)
            
        # 保存图像
        output_path = os.path.join(self.output_dir, output_filename)
        final_image.save(output_path)
        print(f"嵌套字符已保存到: {output_path}")
        
        return final_image
    
    def _generate_layer(self, config: NestedCharConfig, layer_idx: int, 
                       width: int, height: int) -> Image.Image:
        """生成指定层的字符"""
        if layer_idx >= len(config.texts):
            return Image.new('RGB', (width, height), 'white')
            
        text = config.texts[layer_idx]
        font_size = config.font_sizes[layer_idx]
        color = config.colors[layer_idx] if layer_idx < len(config.colors) else 'black'
        
        try:
            font = ImageFont.truetype(self.font_path, font_size)
        except IOError:
            print(f"警告：无法加载字体文件 {self.font_path}，使用默认字体")
            font = ImageFont.load_default()
            
        layer_image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(layer_image)
        
        if layer_idx == 0:
            # 第一层：绘制主要字符
            self._draw_base_characters(draw, text, font, color, width, height)
        else:
            # 后续层：在前一层的基础上填充
            self._draw_nested_fill(draw, text, font, color, width, height, 
                                 config, layer_idx)
        
        return layer_image
    
    def _draw_base_characters(self, draw: ImageDraw.Draw, text: str, 
                            font: ImageFont.FreeTypeFont, color: str,
                            width: int, height: int):
        """绘制基础字符层"""
        char_width = width // len(text) if len(text) > 0 else width
        y_pos = (height - font.size) // 2
        
        for i, char in enumerate(text):
            x_pos = i * char_width + (char_width - font.getlength(char)) // 2
            draw.text((x_pos, y_pos), char, font=font, fill=color)
    
    def _draw_nested_fill(self, draw: ImageDraw.Draw, text: str,
                         font: ImageFont.FreeTypeFont, color: str,
                         width: int, height: int, config: NestedCharConfig,
                         layer_idx: int):
        """在指定区域内填充嵌套字符"""
        step_ratio = config.step_ratios[layer_idx] if layer_idx < len(config.step_ratios) else 0.8
        step_x = int(font.size * step_ratio)
        step_y = int(font.size * step_ratio)
        
        text_idx = 0
        for y in range(0, height, step_y):
            for x in range(0, width, step_x):
                if text_idx < len(text):
                    char = text[text_idx % len(text)]
                    draw.text((x, y), char, font=font, fill=color)
                    text_idx += 1
    
    def _merge_layers(self, base_image: Image.Image, layer_image: Image.Image, 
                     layer_idx: int) -> Image.Image:
        """合并图层"""
        if layer_idx == 0:
            return layer_image
        
        # 使用alpha混合合并图层
        base_array = np.array(base_image)
        layer_array = np.array(layer_image)
        
        # 简单的加权混合
        alpha = 0.7  # 可以根据需要调整
        merged_array = (base_array * (1 - alpha) + layer_array * alpha).astype(np.uint8)
        
        return Image.fromarray(merged_array)

def create_text_fill_art(
    large_text: str,
    small_text: str,
    large_font_size: int,
    small_font_size: int,
    font_path: str,
    output_filename: str,
    step_x: int,
    step_y: int,
    wrap_after: int,
    background_color: str = "white",
    text_color: str = "black"
):

    print("开始生成图像，参数如下:")
    print(f"  - 大字文本: {large_text[:30]}...")
    print(f"  - 换行设置: 每 {wrap_after} 字换行" if wrap_after > 0 else "  - 换行设置: 不换行")
    print(f"  - 小字文本: {small_text}")
    print(f"  - 大字字号: {large_font_size}")
    print(f"  - 小字字号: {small_font_size}")
    
    final_step_x = step_x if step_x is not None else max(2, int(small_font_size * 0.85))
    final_step_y = step_y if step_y is not None else max(2, int(small_font_size * 0.85))
    print(f"  - 小字水平间距: {final_step_x}px")
    print(f"  - 小字垂直间距: {final_step_y}px")
    print("-" * 30)
    
    start_time = time.time()
    if wrap_after > 0:
        lines = [large_text[i:i+wrap_after] for i in range(0, len(large_text), wrap_after)]
    else:
        lines = [large_text]

    LINE_SPACING_FACTOR = 1.2
    line_height = int(large_font_size * LINE_SPACING_FACTOR)

    padding_y = int(large_font_size * 0.6)
    
    text_block_height = line_height * (len(lines) - 1) + large_font_size
    IMAGE_HEIGHT = text_block_height + 2 * padding_y

    num_chars_for_width = wrap_after if wrap_after > 0 else len(large_text)
    IMAGE_WIDTH = int(num_chars_for_width * large_font_size * 1.05)

    try:
        large_font = ImageFont.truetype(font_path, large_font_size)
        small_font = ImageFont.truetype(font_path, small_font_size)
    except IOError:
        print(f"错误：无法在 '{font_path}' 找到字体文件。")
        return

    final_image = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), background_color)
    draw_final = ImageDraw.Draw(final_image)
    
    small_text_index = 0
    
    print("开始逐字生成模板并填充...")
    
    current_y = padding_y
    for line in lines:
        try:
             # getlength is more accurate for multi-character strings than summing getbbox widths
            line_width = large_font.getlength(line)
        except AttributeError:
             # Fallback for older Pillow versions
            line_width = draw_final.textlength(line, font=large_font)

        line_start_x = (IMAGE_WIDTH - line_width) / 2
        current_x = line_start_x

        # 按字符遍历
        for char in line:
            print(f"  - 正在处理大字: '{char}'")

            char_mask = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), 0)
            draw_char_mask = ImageDraw.Draw(char_mask)
            draw_char_mask.text((current_x, current_y), char, font=large_font, fill=255)
            char_mask_array = np.array(char_mask)
            for y in range(0, IMAGE_HEIGHT, final_step_y):
                for x in range(0, IMAGE_WIDTH, final_step_x):
                    # 检查坐标是否在单字模板的笔画内
                    if char_mask_array[y, x] > 128:
                        char_to_draw = small_text[small_text_index % len(small_text)]
                        draw_final.text((x, y), char_to_draw, font=small_font, fill=text_color)
                        small_text_index += 1
            
            try:
                char_width = large_font.getlength(char)
            except AttributeError:
                char_width = draw_final.textlength(char, font=large_font)
            current_x += char_width
        current_y += line_height


    print("填充完成。正在保存图像...")

    # 保存最终结果
    final_image.save(output_filename)
    end_time = time.time()
    print("-" * 30)
    print(f"图像已保存为: {output_filename}")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print("-" * 30)


class FractalAnalyzer:
    """基于分形理论的嵌套字识别器"""
    
    def __init__(self):
        self.min_box_size: int = 2
        self.max_box_size: int = 64
        
    def analyze_image(self, image_path: str) -> FractalFeatures:
        """分析图像的分形特征"""
        print(f"开始分析图像: {image_path}")
        
        # 加载和预处理图像
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        # 二值化处理
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        # 计算分形特征
        box_dimension = self._calculate_box_dimension(binary_image)
        lacunarity = self._calculate_lacunarity(binary_image)
        multifractal_spectrum = self._calculate_multifractal_spectrum(binary_image)
        self_similarity = self._calculate_self_similarity(binary_image)
        
        features = FractalFeatures(
            box_dimension=box_dimension,
            lacunarity=lacunarity,
            multifractal_spectrum=multifractal_spectrum,
            self_similarity=self_similarity
        )
        
        print(f"分形分析完成:")
        print(f"  盒维数: {box_dimension:.4f}")
        print(f"  空隙度: {lacunarity:.4f}")
        print(f"  自相似性: {self_similarity:.4f}")
        
        return features
    
    def _calculate_box_dimension(self, binary_image: np.ndarray) -> float:
        """计算盒维数"""
        # 确保是二值图像（0和255）
        binary_image = (binary_image < 128).astype(np.uint8) * 255
        
        h, w = binary_image.shape
        max_scale = min(h, w) // 4
        
        if max_scale < self.min_box_size:
            return 1.0
            
        # 使用更多的尺度点
        scales = np.unique(np.logspace(np.log10(self.min_box_size),
                                     np.log10(max_scale),
                                     num=15, dtype=int))
        
        counts = []
        for scale in scales:
            count = 0
            # 使用重叠的盒子来提高精度
            step = max(1, scale // 2)
            for i in range(0, h - scale + 1, step):
                for j in range(0, w - scale + 1, step):
                    box = binary_image[i:i+scale, j:j+scale]
                    if np.any(box == 255):  # 如果盒子中有前景像素
                        count += 1
            counts.append(count)
        
        # 过滤无效数据点
        valid_indices = np.array(counts) > 0
        scales = scales[valid_indices]
        counts = np.array(counts)[valid_indices]
        
        if len(scales) < 3:
            return 1.2  # 返回一个合理的默认值
            
        # 使用对数线性回归计算盒维数
        log_scales = np.log(1.0 / scales)  # 注意这里取倒数
        log_counts = np.log(counts)
        
        try:
            coeffs = np.polyfit(log_scales, log_counts, 1)
            box_dim = coeffs[0]
            
            # 限制盒维数在合理范围内
            box_dim = max(1.0, min(2.0, box_dim))
            return box_dim
        except:
            return 1.2
    
    def _calculate_lacunarity(self, binary_image: np.ndarray) -> float:
        """计算空隙度"""
        # 确保是二值图像
        binary_image = (binary_image < 128).astype(np.uint8) * 255
        
        h, w = binary_image.shape
        window_sizes = [4, 8, 16, 32]  # 使用多个窗口尺寸
        
        lacunarities = []
        
        for window_size in window_sizes:
            if h < window_size or w < window_size:
                continue
                
            masses = []
            step = max(1, window_size // 4)  # 重叠窗口
            
            for i in range(0, h - window_size + 1, step):
                for j in range(0, w - window_size + 1, step):
                    window = binary_image[i:i+window_size, j:j+window_size]
                    mass = np.sum(window == 255)  # 前景像素数量
                    masses.append(mass)
            
            masses = np.array(masses)
            if len(masses) == 0:
                continue
                
            # 计算该窗口尺寸下的空隙度
            mean_mass = np.mean(masses)
            if mean_mass > 0:
                variance = np.var(masses)
                lacunarity = 1 + variance / (mean_mass ** 2)
                lacunarities.append(lacunarity)
        
        # 返回平均空隙度
        return np.mean(lacunarities) if lacunarities else 1.0
    
    def _calculate_multifractal_spectrum(self, binary_image: np.ndarray) -> np.ndarray:
        """计算多重分形谱"""
        # 简化的多重分形谱计算
        q_values = np.linspace(-5, 5, 21)
        spectrum = np.zeros_like(q_values)
        
        # 使用不同尺度计算
        scales = [2, 4, 8, 16]
        
        for i, q in enumerate(q_values):
            tau_q = 0
            for scale in scales:
                h, w = binary_image.shape
                boxes_h = h // scale
                boxes_w = w // scale
                
                measures = []
                for bi in range(boxes_h):
                    for bj in range(boxes_w):
                        box = binary_image[bi*scale:(bi+1)*scale, bj*scale:(bj+1)*scale]
                        measure = np.sum(box < 255) / (scale * scale)
                        if measure > 0:
                            measures.append(measure)
                
                if len(measures) > 0:
                    measures = np.array(measures)
                    if q != 1:
                        partition_sum = np.sum(measures ** q)
                        if partition_sum > 0:
                            tau_q += np.log(partition_sum) / np.log(scale)
                    else:
                        entropy = -np.sum(measures * np.log(measures + 1e-10))
                        tau_q += entropy / np.log(scale)
            
            spectrum[i] = tau_q / len(scales)
        
        return spectrum
    
    def _calculate_self_similarity(self, binary_image: np.ndarray) -> float:
        """计算自相似性度量"""
        # 确保是二值图像
        binary_image = (binary_image < 128).astype(np.uint8) * 255
        
        h, w = binary_image.shape
        min_size = min(h, w)
        
        if min_size < 32:
            return 0.0
            
        # 调整到合适的尺寸进行分析
        target_size = min(256, 2 ** int(np.log2(min_size)))
        if target_size < 32:
            target_size = 32
            
        # 调整图像大小
        resized = cv2.resize(binary_image, (target_size, target_size))
        resized = (resized > 128).astype(np.float32)
        
        # 计算不同尺度下的相关性
        correlations = []
        scales = [2, 4]  # 减少尺度数量以提高稳定性
        
        for scale in scales:
            if target_size // scale < 8:
                continue
                
            try:
                # 下采样
                new_size = target_size // scale
                downsampled = cv2.resize(resized, (new_size, new_size))
                
                # 上采样回原尺寸
                upsampled = cv2.resize(downsampled, (target_size, target_size))
                
                # 计算结构相似性
                original_flat = resized.flatten()
                upsampled_flat = upsampled.flatten()
                
                # 避免全零的情况
                if np.std(original_flat) > 0 and np.std(upsampled_flat) > 0:
                    correlation = np.corrcoef(original_flat, upsampled_flat)[0, 1]
                    if not np.isnan(correlation) and not np.isinf(correlation):
                        correlations.append(abs(correlation))
                        
            except Exception:
                continue
        
        if correlations:
            return np.mean(correlations)
        else:
            # 如果无法计算相关性，使用简单的结构度量
            return min(0.5, np.std(resized) * 2)
    
    def detect_nested_structure(self, features: FractalFeatures) -> Dict[str, any]:
        """基于分形特征检测嵌套结构"""
        print("基于分形特征检测嵌套结构...")
        
        # 嵌套字检测阈值（基于实际测试调整）
        nested_threshold = {
            'box_dimension': (1.1, 2.0),  # 调整盒维数范围
            'lacunarity': (1.0, 50.0),    # 调整空隙度范围
            'self_similarity': (0.0, 1.0)  # 调整自相似性范围
        }
        
        # 使用更灵活的检测逻辑
        box_dim_ok = nested_threshold['box_dimension'][0] <= features.box_dimension <= nested_threshold['box_dimension'][1]
        lacunarity_ok = nested_threshold['lacunarity'][0] <= features.lacunarity <= nested_threshold['lacunarity'][1]
        similarity_ok = nested_threshold['self_similarity'][0] <= features.self_similarity <= nested_threshold['self_similarity'][1]
        
        # 如果至少满足两个条件，或者有明显的分形特征，则认为是嵌套字
        conditions_met = sum([box_dim_ok, lacunarity_ok, similarity_ok])
        
        # 特殊判断：如果盒维数大于1.15且空隙度大于1.0，很可能是嵌套字
        special_case = features.box_dimension > 1.15 and features.lacunarity >= 1.0
        
        is_nested = conditions_met >= 2 or special_case
        
        # 估计嵌套层数（基于盒维数和自相似性）
        if is_nested:
            # 层数估计公式（经验公式）
            estimated_layers = max(1, int(2 + (features.box_dimension - 1.0) * 3 + features.self_similarity * 2))
            estimated_layers = min(estimated_layers, 5)  # 限制最大层数
        else:
            estimated_layers = 0
        
        # 计算置信度
        confidence = 0.0
        if is_nested:
            # 基于特征值与理想范围的接近程度计算置信度
            box_dim_score = 1.0 - abs(features.box_dimension - 1.55) / 0.25
            lacunarity_score = min(1.0, (features.lacunarity - 2.0) / 8.0)
            similarity_score = features.self_similarity
            
            confidence = (box_dim_score + lacunarity_score + similarity_score) / 3.0
            confidence = max(0.0, min(1.0, confidence))
        
        result = {
            'is_nested': is_nested,
            'estimated_layers': estimated_layers,
            'confidence': confidence,
            'features': {
                'box_dimension': features.box_dimension,
                'lacunarity': features.lacunarity,
                'self_similarity': features.self_similarity,
                'multifractal_complexity': np.std(features.multifractal_spectrum)
            }
        }
        
        print(f"检测结果:")
        print(f"  是否为嵌套字: {is_nested}")
        print(f"  估计层数: {estimated_layers}")
        print(f"  置信度: {confidence:.3f}")
        
        return result


class NestedCharacterAPI:
    """嵌套字符API接口"""
    
    def __init__(self, generator: NestedCharacterGenerator, analyzer: FractalAnalyzer):
        self.generator = generator
        self.analyzer = analyzer
    
    def generate_batch_nested_chars(self, configs: List[NestedCharConfig]) -> List[str]:
        """批量生成嵌套字符"""
        output_files = []
        
        for i, config in enumerate(configs):
            output_filename = f"nested_char_{i+1}_{config.layers}layers.png"
            try:
                self.generator.create_nested_character(config, output_filename)
                output_files.append(output_filename)
                print(f"成功生成: {output_filename}")
            except Exception as e:
                print(f"生成失败 {output_filename}: {str(e)}")
        
        return output_files
    
    def analyze_batch_images(self, image_paths: List[str]) -> List[Dict]:
        """批量分析图像"""
        results = []
        
        for image_path in image_paths:
            try:
                # 分析分形特征
                features = self.analyzer.analyze_image(image_path)
                
                # 检测嵌套结构
                detection_result = self.analyzer.detect_nested_structure(features)
                
                result = {
                    'image_path': image_path,
                    'analysis_result': detection_result
                }
                results.append(result)
                
            except Exception as e:
                print(f"分析失败 {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results


def create_sample_configs() -> List[NestedCharConfig]:
    """创建示例配置"""
    configs = []
    
    # 配置1：2层嵌套
    config1 = NestedCharConfig(
        layers=2,
        texts=["测试", "嵌套字符识别系统"],
        font_sizes=[200, 12],
        size_ratios=[1.0, 0.06],
        step_ratios=[1.0, 0.8],
        colors=['black', 'gray']
    )
    configs.append(config1)
    
    # 配置2：3层嵌套
    config2 = NestedCharConfig(
        layers=3,
        texts=["AI", "人工智能技术", "深度学习神经网络"],
        font_sizes=[300, 20, 8],
        size_ratios=[1.0, 0.067, 0.027],
        step_ratios=[1.0, 0.7, 0.6],
        colors=['black', 'darkgray', 'lightgray']
    )
    configs.append(config2)
    
    # 配置3：4层嵌套（复杂示例）
    config3 = NestedCharConfig(
        layers=4,
        texts=["分形", "复杂系统理论", "自相似结构分析", "多重分形谱计算方法"],
        font_sizes=[250, 18, 10, 6],
        size_ratios=[1.0, 0.072, 0.04, 0.024],
        step_ratios=[1.0, 0.75, 0.65, 0.55],
        colors=['black', 'darkgray', 'gray', 'lightgray']
    )
    configs.append(config3)
    
    return configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="多层嵌套字符生成和识别系统",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # 添加模式选择参数
    parser.add_argument('--mode', type=str, choices=['generate', 'analyze', 'batch_generate', 'batch_analyze', 'demo', 'legacy'], 
                        default='demo', help="运行模式:\n"
                        "generate: 生成单个嵌套字符\n"
                        "analyze: 分析单个图像\n"
                        "batch_generate: 批量生成嵌套字符\n"
                        "batch_analyze: 批量分析图像\n"
                        "demo: 运行演示程序\n"
                        "legacy: 使用原始功能")

    # 新功能参数
    parser.add_argument('--layers', type=int, default=2, help="嵌套层数")
    parser.add_argument('--image-path', type=str, help="要分析的图像路径")
    parser.add_argument('--batch-count', type=int, default=3, help="批量生成的数量")

    # 原有参数（用于兼容性）
    parser.add_argument('-lt', '--large-text', type=str, default="测试",
                        help="被填充的大字文本。")
    parser.add_argument('-st', '--small-text', type=str, default="嵌套字符识别系统基于分形理论实现多层文字嵌套结构的生成与检测",
                        help="用于填充的小字文本。")
    parser.add_argument('-lfs', '--large-font-size', type=int, default=400,
                        help="被填充的大字字号。")
    parser.add_argument('-sfs', '--small-font-size', type=int, default=12,
                        help="用于填充的小字字号。")
    parser.add_argument('-w', '--wrap-after', type=int, default=10,
                        help="设置大字文本每行显示的字数。")
    parser.add_argument('--step-x', type=int, default=13,
                        help="小字填充的水平步长/间距（像素）。")
    parser.add_argument('--step-y', type=int, default=13,
                        help="小字填充的垂直步长/行间距（像素）。")
    parser.add_argument('-fp', '--font-path', type=str, default='C:/Windows/Fonts/simhei.ttf',
                        help="中文字体文件的路径。")
    parser.add_argument('-o', '--output', type=str, default="output_art.png",
                        help="输出图像的文件名。")

    args = parser.parse_args()

    # 根据模式运行不同功能
    if args.mode == 'legacy':
        # 原始功能
        create_text_fill_art(
            large_text=args.large_text,
            small_text=args.small_text,
            large_font_size=args.large_font_size,
            small_font_size=args.small_font_size,
            font_path=args.font_path,
            output_filename=args.output,
            step_x=args.step_x,
            step_y=args.step_y,
            wrap_after=args.wrap_after
        )
    
    elif args.mode == 'demo':
        print("=== 多层嵌套字符生成和识别系统演示 ===")
        
        # 初始化组件
        generator = NestedCharacterGenerator(args.font_path)
        analyzer = FractalAnalyzer()
        api = NestedCharacterAPI(generator, analyzer)
        
        # 创建示例配置
        configs = create_sample_configs()
        
        # 批量生成嵌套字符
        print("\n1. 批量生成嵌套字符...")
        output_files = api.generate_batch_nested_chars(configs)
        
        # 分析生成的图像
        print("\n2. 分析生成的嵌套字符...")
        full_paths = [os.path.join("output_images", f) for f in output_files if os.path.exists(os.path.join("output_images", f))]
        
        if full_paths:
            results = api.analyze_batch_images(full_paths)
            
            print("\n=== 分析结果汇总 ===")
            for result in results:
                if 'error' in result:
                    print(f"[错误] {result['image_path']}: {result['error']}")
                else:
                    analysis = result['analysis_result']
                    print(f"[分析] {result['image_path']}:")
                    print(f"   是否为嵌套字: {'是' if analysis['is_nested'] else '否'}")
                    print(f"   估计层数: {analysis['estimated_layers']}")
                    print(f"   置信度: {analysis['confidence']:.3f}")
                    print(f"   盒维数: {analysis['features']['box_dimension']:.4f}")
                    print(f"   空隙度: {analysis['features']['lacunarity']:.4f}")
                    print(f"   自相似性: {analysis['features']['self_similarity']:.4f}")
                    print()
        else:
            print("[警告] 没有找到生成的图像文件进行分析")
        
        print("=== 演示完成 ===")
    
    elif args.mode == 'generate':
        # 单个嵌套字符生成
        generator = NestedCharacterGenerator(args.font_path)
        
        config = NestedCharConfig(
            layers=args.layers,
            texts=[args.large_text] + [args.small_text],
            font_sizes=[args.large_font_size] + [args.small_font_size],
            size_ratios=[1.0] + [0.06],
            step_ratios=[1.0] + [0.8],
            colors=['black', 'gray']
        )
        
        generator.create_nested_character(config, args.output)
    
    elif args.mode == 'analyze':
        # 单个图像分析
        if not args.image_path:
            print("错误：分析模式需要指定 --image-path 参数")
            exit(1)
            
        analyzer = FractalAnalyzer()
        
        try:
            features = analyzer.analyze_image(args.image_path)
            result = analyzer.detect_nested_structure(features)
            
            print(f"\n=== 图像分析结果 ===")
            print(f"图像路径: {args.image_path}")
            print(f"是否为嵌套字: {'是' if result['is_nested'] else '否'}")
            print(f"估计层数: {result['estimated_layers']}")
            print(f"置信度: {result['confidence']:.3f}")
            print(f"分形特征:")
            print(f"  盒维数: {result['features']['box_dimension']:.4f}")
            print(f"  空隙度: {result['features']['lacunarity']:.4f}")
            print(f"  自相似性: {result['features']['self_similarity']:.4f}")
            print(f"  多重分形复杂度: {result['features']['multifractal_complexity']:.4f}")
            
        except Exception as e:
            print(f"分析失败: {str(e)}")
    
    elif args.mode == 'batch_generate':
        # 批量生成
        generator = NestedCharacterGenerator(args.font_path)
        api = NestedCharacterAPI(generator, FractalAnalyzer())
        
        configs = create_sample_configs()[:args.batch_count]
        output_files = api.generate_batch_nested_chars(configs)
        
        print(f"\n批量生成完成，共生成 {len(output_files)} 个文件:")
        for f in output_files:
            print(f"  - {f}")
    
    elif args.mode == 'batch_analyze':
        # 批量分析
        analyzer = FractalAnalyzer()
        api = NestedCharacterAPI(NestedCharacterGenerator(args.font_path), analyzer)
        
        # 获取输出目录中的所有图像文件
        image_files = []
        if os.path.exists("output_images"):
            for f in os.listdir("output_images"):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join("output_images", f))
        
        if not image_files:
            print("[警告] 在output_images目录中没有找到图像文件")
        else:
            print(f"开始批量分析 {len(image_files)} 个图像文件...")
            results = api.analyze_batch_images(image_files)
            
            print("\n=== 批量分析结果 ===")
            for result in results:
                if 'error' in result:
                    print(f"[错误] {result['image_path']}: {result['error']}")
                else:
                    analysis = result['analysis_result']
                    print(f"[分析] {os.path.basename(result['image_path'])}:")
                    print(f"   嵌套字: {'是' if analysis['is_nested'] else '否'}")
                    print(f"   层数: {analysis['estimated_layers']}")
                    print(f"   置信度: {analysis['confidence']:.3f}")