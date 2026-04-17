# -*- coding: utf-8 -*-
"""
训练数据生成器 - 生成不同层数、变换的嵌套字符作为训练集
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
from typing import List, Tuple, Dict
import random
from tqdm import tqdm
import json


class TrainingDataGenerator:
    """训练数据生成器"""
    
    def __init__(self, font_path: str, output_dir: str = 'training_data'):
        self.font_path: str = font_path
        self.output_dir: str = output_dir
        self.train_dir: str = os.path.join(output_dir, 'train')
        self.val_dir: str = os.path.join(output_dir, 'val')
        self.test_dir: str = os.path.join(output_dir, 'test')

        # 创建目录
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # 中文字符集
        self.char_pool: List[str] = self._create_char_pool()

        # OCR字符映射表
        self.char_vocab: Dict[str, int] = {char: idx for idx, char in enumerate(self.char_pool)}
        self.id_to_char: Dict[int, str] = {idx: char for idx, char in enumerate(self.char_pool)}
        
    def _create_char_pool(self) -> List[str]:
        """创建字符池"""
        chars = []
        
        # 常用汉字
        common_chars = "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严龙飞"
        
        for char in common_chars:
            chars.append(char)
        
        # 英文字母
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
            chars.append(c)
        
        # 数字
        for d in "0123456789":
            chars.append(d)
        
        return chars
    
    def generate_nested_char_image(self, layers: int, size: Tuple[int, int] = (256, 256),
                                   apply_transform: bool = False) -> np.ndarray:
        """生成嵌套字符图像"""
        width, height = size
        
        # 创建白色背景
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        
        # 根据层数选择字符和字体大小
        if layers == 0:
            # 纯背景或简单图案
            return np.array(image)
        
        # 第一层：主字符
        main_char = random.choice(self.char_pool)
        main_font_size = random.randint(80, 150)
        
        try:
            main_font = ImageFont.truetype(self.font_path, main_font_size)
        except:
            main_font = ImageFont.load_default()
        
        # 绘制主字符
        bbox = draw.textbbox((0, 0), main_char, font=main_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        draw.text((x, y), main_char, font=main_font, fill='black')
        
        # 后续层：嵌套填充
        if layers > 1:
            # 创建主字符的mask
            mask = Image.new('L', (width, height), 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.text((x, y), main_char, font=main_font, fill=255)
            mask_array = np.array(mask)
            
            # 在mask区域内填充小字符
            for layer in range(1, layers):
                fill_text = ''.join(random.choices(self.char_pool, k=100))
                fill_font_size = max(6, main_font_size // (2 ** layer))
                
                try:
                    fill_font = ImageFont.truetype(self.font_path, fill_font_size)
                except:
                    fill_font = ImageFont.load_default()
                
                step = max(fill_font_size // 2, 3)
                char_idx = 0
                
                for py in range(0, height, step):
                    for px in range(0, width, step):
                        if mask_array[py, px] > 128:
                            char = fill_text[char_idx % len(fill_text)]
                            # 使用不同灰度
                            gray_value = 50 + (layer * 30) % 150
                            color = (gray_value, gray_value, gray_value)
                            draw.text((px, py), char, font=fill_font, fill=color)
                            char_idx += 1
        
        # 转换为numpy数组
        result = np.array(image)
        
        # 应用线性变换
        if apply_transform:
            result = self._apply_random_transform(result)
        
        return result
    
    def _apply_random_transform(self, image: np.ndarray) -> np.ndarray:
        """应用随机线性变换"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # 随机旋转
        if random.random() > 0.5:
            angle = random.uniform(-30, 30)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (w, h),
                                  borderValue=(255, 255, 255))
        
        # 随机缩放
        if random.random() > 0.5:
            scale_x = random.uniform(0.7, 1.3)
            scale_y = random.uniform(0.7, 1.3)
            image = cv2.resize(image, None, fx=scale_x, fy=scale_y)
            
            # 裁剪或填充回原始大小
            new_h, new_w = image.shape[:2]
            if new_h > h or new_w > w:
                start_y = (new_h - h) // 2
                start_x = (new_w - w) // 2
                image = image[start_y:start_y+h, start_x:start_x+w]
            else:
                pad_y = (h - new_h) // 2
                pad_x = (w - new_w) // 2
                image = cv2.copyMakeBorder(image, pad_y, h-new_h-pad_y,
                                          pad_x, w-new_w-pad_x,
                                          cv2.BORDER_CONSTANT, value=(255, 255, 255))
        
        # 随机翻转
        if random.random() > 0.7:
            image = cv2.flip(image, 1)  # 水平翻转
        
        if random.random() > 0.7:
            image = cv2.flip(image, 0)  # 垂直翻转
        
        # 添加噪声
        if random.random() > 0.5:
            noise = np.random.normal(0, 10, image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def generate_ocr_patch(self, char: str, size: Tuple[int, int] = (64, 64),
                           apply_augment: bool = True) -> Tuple[np.ndarray, int]:
        """生成单个字符的灰度图像块用于OCR训练

        Args:
            char: 要渲染的字符
            size: 输出图像尺寸 (宽, 高)
            apply_augment: 是否应用数据增强

        Returns:
            (灰度图像uint8, 类别ID)
        """
        width, height = size
        class_id = self.char_vocab[char]

        image = Image.new('L', (width, height), color=255)
        draw = ImageDraw.Draw(image)

        font_size = random.randint(int(height * 0.55), int(height * 0.85))
        try:
            font = ImageFont.truetype(self.font_path, font_size)
        except Exception:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), char, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        # 随机位置偏移 (±10%) 增强位置不变性
        offset_x = random.randint(-width // 10, width // 10)
        offset_y = random.randint(-height // 10, height // 10)
        x = (width - tw) // 2 + offset_x
        y = (height - th) // 2 + offset_y

        fill_gray = random.randint(0, 60)
        draw.text((x, y), char, font=font, fill=fill_gray)

        img_np = np.array(image)

        if apply_augment:
            img_np = self._augment_ocr_patch(img_np)

        return img_np, class_id

    def _augment_ocr_patch(self, image: np.ndarray) -> np.ndarray:
        """OCR图像块专用数据增强
        轻旋转、透视变形、笔画粗细变化、模糊、噪声；不做水平翻转（汉字翻转后含义改变）
        """
        h, w = image.shape[:2]

        # 旋转 ±15度
        if random.random() > 0.4:
            angle = random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h),
                                   borderValue=255, flags=cv2.INTER_CUBIC)

        # 轻微透视变形
        if random.random() > 0.6:
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            d = w * 0.05
            pts2 = np.float32([
                [random.uniform(0, d), random.uniform(0, d)],
                [w - random.uniform(0, d), random.uniform(0, d)],
                [random.uniform(0, d), h - random.uniform(0, d)],
                [w - random.uniform(0, d), h - random.uniform(0, d)],
            ])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            image = cv2.warpPerspective(image, M, (w, h), borderValue=255)

        # 形态学变换模拟笔画粗细变化
        if random.random() > 0.5:
            k = random.choice([2, 3])
            kernel = np.ones((k, k), np.uint8)
            if random.random() > 0.5:
                image = cv2.erode(image, kernel, iterations=1)
            else:
                image = cv2.dilate(image, kernel, iterations=1)

        # 高斯模糊
        if random.random() > 0.5:
            ksize = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)

        # 背景噪声
        if random.random() > 0.5:
            noise = np.random.normal(0, 8, image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return image

    def generate_ocr_dataset(self, samples_per_char: int = 50,
                             patch_size: Tuple[int, int] = (64, 64),
                             train_ratio: float = 0.7,
                             val_ratio: float = 0.15,
                             test_ratio: float = 0.15) -> dict:
        """生成OCR训练数据集——为字符池中的每个字符生成多个样本

        目录结构:
            output_dir/ocr_data/{train,val,test}/char_{class_id}_{i}.png

        Returns:
            包含统计信息的元数据字典
        """
        ocr_dir = os.path.join(self.output_dir, 'ocr_data')
        train_dir = os.path.join(ocr_dir, 'train')
        val_dir = os.path.join(ocr_dir, 'val')
        test_dir = os.path.join(ocr_dir, 'test')
        for d in [train_dir, val_dir, test_dir]:
            os.makedirs(d, exist_ok=True)

        train_n = int(samples_per_char * train_ratio)
        val_n = int(samples_per_char * val_ratio)

        metadata = {
            'patch_size': list(patch_size),
            'samples_per_char': samples_per_char,
            'num_classes': len(self.char_vocab),
            'splits': {'train': 0, 'val': 0, 'test': 0},
        }

        # 保存字符映射表
        vocab_path = os.path.join(ocr_dir, 'char_vocab.json')
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.char_vocab, f, ensure_ascii=False, indent=2)

        for char, class_id in tqdm(self.char_vocab.items(), desc='生成OCR字符块'):
            for i in range(samples_per_char):
                # 前25%干净样本，其余带增强
                apply_aug = (i >= samples_per_char // 4)
                img, _ = self.generate_ocr_patch(char, size=patch_size,
                                                 apply_augment=apply_aug)
                if i < train_n:
                    out_dir = train_dir
                    metadata['splits']['train'] += 1
                elif i < train_n + val_n:
                    out_dir = val_dir
                    metadata['splits']['val'] += 1
                else:
                    out_dir = test_dir
                    metadata['splits']['test'] += 1

                filename = f"char_{class_id:04d}_{i:04d}.png"
                cv2.imwrite(os.path.join(out_dir, filename), img)

        meta_path = os.path.join(ocr_dir, 'ocr_metadata.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"OCR数据集生成完成: {metadata['splits']}")
        return metadata

    def generate_dataset(self, samples_per_class: int = 500,
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.15,
                        test_ratio: float = 0.15):
        """生成完整数据集"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "比例之和必须为1"
        
        metadata = {
            'samples_per_class': samples_per_class,
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'classes': {}
        }
        
        # 为每个层数生成样本（0-5层）
        for layers in range(6):
            print(f"\n生成{layers}层嵌套字符样本...")
            
            class_metadata = {
                'layer': layers,
                'train_samples': 0,
                'val_samples': 0,
                'test_samples': 0
            }
            
            # 生成样本
            samples = []
            for i in tqdm(range(samples_per_class), desc=f"Layer {layers}"):
                # 50%的样本应用变换
                apply_transform = i >= samples_per_class // 2
                
                image = self.generate_nested_char_image(layers, apply_transform=apply_transform)
                samples.append(image)
            
            # 划分数据集
            random.shuffle(samples)
            
            train_count = int(samples_per_class * train_ratio)
            val_count = int(samples_per_class * val_ratio)
            
            train_samples = samples[:train_count]
            val_samples = samples[train_count:train_count + val_count]
            test_samples = samples[train_count + val_count:]
            
            # 保存训练集
            for i, img in enumerate(train_samples):
                filename = f"layer{layers}_train_{i:04d}.png"
                filepath = os.path.join(self.train_dir, filename)
                cv2.imwrite(filepath, img)
            
            class_metadata['train_samples'] = len(train_samples)
            
            # 保存验证集
            for i, img in enumerate(val_samples):
                filename = f"layer{layers}_val_{i:04d}.png"
                filepath = os.path.join(self.val_dir, filename)
                cv2.imwrite(filepath, img)
            
            class_metadata['val_samples'] = len(val_samples)
            
            # 保存测试集
            for i, img in enumerate(test_samples):
                filename = f"layer{layers}_test_{i:04d}.png"
                filepath = os.path.join(self.test_dir, filename)
                cv2.imwrite(filepath, img)
            
            class_metadata['test_samples'] = len(test_samples)
            
            metadata['classes'][f'layer_{layers}'] = class_metadata
            
            print(f"  训练集: {len(train_samples)}张")
            print(f"  验证集: {len(val_samples)}张")
            print(f"  测试集: {len(test_samples)}张")
        
        # 保存元数据
        metadata_path = os.path.join(self.output_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n数据集生成完成！")
        print(f"元数据已保存到: {metadata_path}")
        
        return metadata
    
    def load_dataset_paths(self) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
        """加载数据集路径和标签"""
        train_paths = []
        train_labels = []
        val_paths = []
        val_labels = []
        test_paths = []
        test_labels = []
        
        # 训练集
        for filename in os.listdir(self.train_dir):
            if filename.endswith('.png'):
                filepath = os.path.join(self.train_dir, filename)
                layer = int(filename.split('_')[0].replace('layer', ''))
                train_paths.append(filepath)
                train_labels.append(layer)
        
        # 验证集
        for filename in os.listdir(self.val_dir):
            if filename.endswith('.png'):
                filepath = os.path.join(self.val_dir, filename)
                layer = int(filename.split('_')[0].replace('layer', ''))
                val_paths.append(filepath)
                val_labels.append(layer)
        
        # 测试集
        for filename in os.listdir(self.test_dir):
            if filename.endswith('.png'):
                filepath = os.path.join(self.test_dir, filename)
                layer = int(filename.split('_')[0].replace('layer', ''))
                test_paths.append(filepath)
                test_labels.append(layer)
        
        return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def visualize_samples(generator: TrainingDataGenerator, num_samples: int = 6):
    """可视化生成的样本"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, layers in enumerate(range(6)):
        image = generator.generate_nested_char_image(layers, apply_transform=True)
        
        axes[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f'{layers}层嵌套字符')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(generator.output_dir, 'sample_visualization.png'), dpi=150)
    plt.close()
    
    print(f"样本可视化已保存到: {os.path.join(generator.output_dir, 'sample_visualization.png')}")


if __name__ == '__main__':
    # 示例用法
    font_path = 'C:/Windows/Fonts/simhei.ttf'
    
    generator = TrainingDataGenerator(font_path, output_dir='training_data')
    
    # 生成数据集（每类100个样本用于测试）
    print("开始生成训练数据集...")
    metadata = generator.generate_dataset(samples_per_class=100)
    
    # 可视化样本
    print("\n生成样本可视化...")
    visualize_samples(generator)
    
    print("\n完成！")