# -*- coding: utf-8 -*-
"""
嵌套字符交互式演示 - tkinter UI
功能：生成嵌套字符图 → 展示OCR检测能力 → 加密对抗
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
import os
import sys
import tempfile
from PIL import Image, ImageDraw, ImageFont, ImageTk
from nested_char_detector import NestedCharDetector
from NestCharacter import create_text_fill_art

try:
    import easyocr
except Exception:
    easyocr = None

# ==================== 资源路径（兼容 PyInstaller 打包） ====================
def resource_path(relative_path):
    """获取资源文件的绝对路径，兼容 PyInstaller 打包"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), relative_path)

FONT_PATH = 'C:/Windows/Fonts/simhei.ttf'
MODEL_PATH = resource_path('models/nested_char_model.pth')
DUAL_MODEL_PATH = resource_path('models/nested_char_model_dual.pth')
CHAR_VOCAB_PATH = resource_path('training_data/ocr_data/char_vocab.json')

# 各层参数配置
LAYER_CONFIG = {
    1: {"large_fs": 400, "small_fs": 12, "step": 13,
        "main_max": 20, "fill_min": 10,
        "desc": "1层：大字400px，小字12px | 主字≤20字，填充≥10字"},
    2: {"large_fs": 300, "small_fs": 10, "step": 11,
        "main_max": 10, "fill_min": 50,
        "desc": "2层：大字300px，小字10px→5px | 主字≤10字，填充≥50字"},
    3: {"large_fs": 250, "small_fs": 8, "step": 9,
        "main_max": 5, "fill_min": 200,
        "desc": "3层：大字250px，小字8px→4px | 主字≤5字，填充≥200字"},
    4: {"large_fs": 200, "small_fs": 7, "step": 8,
        "main_max": 3, "fill_min": 500,
        "desc": "4层：大字200px，小字7px→3px | 主字≤3字，填充≥500字"},
    5: {"large_fs": 180, "small_fs": 6, "step": 7,
        "main_max": 2, "fill_min": 1000,
        "desc": "5层：大字180px，小字6px→3px | 主字≤2字，填充≥1000字"},
}

# 全局检测器（延迟初始化）
detector = None
ocr_reader = None


def get_detector():
    global detector
    if detector is None:
        if os.path.exists(DUAL_MODEL_PATH) and os.path.exists(CHAR_VOCAB_PATH):
            detector = NestedCharDetector(
                model_path=DUAL_MODEL_PATH,
                use_dual_head=True,
                char_vocab_path=CHAR_VOCAB_PATH
            )
        elif os.path.exists(MODEL_PATH):
            detector = NestedCharDetector(model_path=MODEL_PATH)
        else:
            detector = NestedCharDetector()
    return detector


def get_ocr_reader():
    """Init EasyOCR lazily (mature OCR backend)."""
    global ocr_reader
    if ocr_reader is None:
        if easyocr is None:
            return None
        try:
            # Keep gpu disabled by default for startup stability on diverse machines.
            ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
        except Exception:
            return None
    return ocr_reader


def recognize_text_with_easyocr(image_bgr, boxes, max_boxes=80):
    """Recognize text in detected boxes using EasyOCR (main-char first)."""
    reader = get_ocr_reader()
    if reader is None:
        return [], "EasyOCR 未安装，已回退到内置识别结果。"

    if image_bgr is None:
        return [], "图像为空，无法OCR。"

    h_img, w_img = image_bgr.shape[:2]

    # Rank boxes so likely-main characters are processed first.
    candidates = []
    if boxes:
        clean_boxes = []
        for b in boxes:
            if len(b) != 4:
                continue
            x, y, w, h = [int(v) for v in b]
            if w <= 2 or h <= 2:
                continue
            clean_boxes.append((x, y, w, h))

        if clean_boxes:
            areas = np.array([w * h for (_, _, w, h) in clean_boxes], dtype=np.float32)
            max_area = float(max(np.max(areas), 1.0))
            img_cx, img_cy = w_img * 0.5, h_img * 0.5
            max_dist = float(np.hypot(img_cx, img_cy) + 1e-6)

            scored = []
            for (x, y, w, h) in clean_boxes:
                area = float(w * h)
                area_score = area / max_area

                cx = x + w * 0.5
                cy = y + h * 0.5
                center_dist = float(np.hypot(cx - img_cx, cy - img_cy))
                center_score = max(0.0, 1.0 - center_dist / max_dist)

                ratio = max(float(w) / max(float(h), 1.0), 1e-6)
                shape_score = max(0.0, 1.0 - abs(np.log(ratio)) / 1.5)

                main_score = area_score * 0.70 + center_score * 0.20 + shape_score * 0.10
                scored.append(((x, y, w, h), main_score))

            scored.sort(key=lambda it: it[1], reverse=True)
            ranked = scored[:max_boxes]

            # Top-N as main candidates; rest as detail candidates.
            main_top_n = max(1, min(4, (len(ranked) + 2) // 3))
            for idx, (box, score) in enumerate(ranked):
                priority = 'main' if idx < main_top_n else 'detail'
                candidates.append((box, score, priority))
    else:
        candidates.append(((0, 0, w_img, h_img), 0.0, 'full'))

    if not candidates:
        return [], "检测框为空，无法执行主字优先OCR。"

    results = []
    for box, score, priority in candidates:
        x, y, w, h = box
        pad = max(2, int(min(w, h) * 0.08))
        x1 = max(0, int(x - pad))
        y1 = max(0, int(y - pad))
        x2 = min(w_img, int(x + w + pad))
        y2 = min(h_img, int(y + h + pad))
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image_bgr[y1:y2, x1:x2]
        try:
            rec = reader.readtext(crop, detail=1, paragraph=False)
        except Exception as e:
            return [], f"EasyOCR 调用失败：{e}"
        if not rec:
            continue

        # Keep best candidate per box for report readability.
        best = max(rec, key=lambda it: float(it[2]))
        text = str(best[1]).strip()
        conf = float(best[2])
        if text:
            results.append({
                'box': tuple(int(v) for v in box),
                'text': text,
                'confidence': conf,
                'priority': priority,
                'main_score': float(score)
            })

    # Keep "main" first, then by confidence.
    priority_order = {'main': 0, 'detail': 1, 'full': 2}
    results.sort(key=lambda r: (priority_order.get(r.get('priority', 'detail'), 9), -r['confidence']))
    return results, ""


# ==================== 核心逻辑 ====================

def generate_nested_image(main_text, fill_text, layers, wrap_after):
    """生成多层嵌套字符图像"""
    if not main_text.strip():
        return None, "请输入主字文本"
    if not fill_text.strip():
        return None, "请输入填充文本"

    cfg = LAYER_CONFIG[layers]
    warnings = []

    if len(main_text) > cfg["main_max"]:
        warnings.append(f"主字建议≤{cfg['main_max']}字，当前{len(main_text)}字，已截断")
        main_text = main_text[:cfg["main_max"]]
    if len(fill_text) < cfg["fill_min"]:
        warnings.append(f"填充文本建议≥{cfg['fill_min']}字，当前{len(fill_text)}字，效果可能不佳")

    wrap = int(wrap_after) if wrap_after > 0 else 0

    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        create_text_fill_art(
            large_text=main_text,
            small_text=fill_text,
            large_font_size=cfg["large_fs"],
            small_font_size=cfg["small_fs"],
            font_path=FONT_PATH,
            output_filename=tmp_path,
            step_x=cfg["step"],
            step_y=cfg["step"],
            wrap_after=wrap,
        )

        image = Image.open(tmp_path).copy()

        if layers >= 2:
            draw = ImageDraw.Draw(image)
            img_w, img_h = image.size

            for layer_idx in range(2, layers + 1):
                smaller_size = max(3, cfg["small_fs"] // (2 ** (layer_idx - 1)))
                step = max(smaller_size, 3)
                gray_val = 60 + (layer_idx * 35) % 130
                color = (gray_val, gray_val, gray_val)

                img_array = np.array(image)
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

                try:
                    tiny_font = ImageFont.truetype(FONT_PATH, smaller_size)
                except Exception:
                    tiny_font = ImageFont.load_default()

                char_idx = 0
                for y in range(0, img_h, step):
                    for x in range(0, img_w, step):
                        if y < mask.shape[0] and x < mask.shape[1] and mask[y, x] > 128:
                            ch = fill_text[char_idx % len(fill_text)]
                            draw.text((x, y), ch, font=tiny_font, fill=color)
                            char_idx += 1

        warn_msg = " | ".join(warnings) if warnings else ""
        return image, warn_msg

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def run_detection(image):
    """对图像执行嵌套字符检测"""
    if image is None:
        return None, ""

    det = get_detector()

    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        if isinstance(image, Image.Image):
            image.save(tmp_path)
        else:
            cv2.imwrite(tmp_path, cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

        result = det.detect(tmp_path)

        img_cv = cv2.imread(tmp_path)
        img_draw = img_cv.copy()
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        # 按高度分层：收集所有有效连通区域的高度
        all_heights = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            if area > 5:
                all_heights.append(h)

        # 用聚类把字符分成不同层次（按高度降序排列）
        # 层颜色：红(L1) -> 橙(L2) -> 黄(L3) -> 绿(L4) -> 青(L5)
        LAYER_COLORS_BGR = [
            (0, 0, 255),      # L1 红
            (0, 165, 255),    # L2 橙
            (0, 255, 255),    # L3 黄
            (255, 255, 0),    # L4 青
            (255, 0, 255),    # L5 粉
        ]
        LAYER_COLORS_RGB = [
            (255, 0, 0),      # L1 红
            (255, 165, 0),    # L2 橙
            (255, 255, 0),    # L3 黄
            (0, 255, 255),    # L4 青
            (255, 0, 255),    # L5 粉
        ]

        # 用高度排序 + 自然间隙切割来分层
        layers_data = {}  # layer_index -> [(w, h, area), ...]
        if all_heights:
            sorted_heights = sorted(all_heights, reverse=True)

            # 找到自然断裂点：相邻高度差 > 较小值的50% 就算新层
            splits = [0]
            for i in range(1, len(sorted_heights)):
                gap = sorted_heights[i - 1] - sorted_heights[i]
                if gap > sorted_heights[i] * 0.5 and gap > 3:
                    splits.append(i)
            splits.append(len(sorted_heights))

            # 最多分5层
            num_layers = min(len(splits) - 1, 5)
            if num_layers > 1:
                # 重新计算分割点，只保留最大的几个间隙
                gaps = []
                for i in range(1, len(splits) - 1):
                    gap_val = sorted_heights[splits[i] - 1] - sorted_heights[splits[i]]
                    gaps.append((gap_val, splits[i]))
                gaps.sort(reverse=True)
                top_gaps = sorted([g[1] for g in gaps[:num_layers - 1]])
                splits = [0] + top_gaps + [len(sorted_heights)]

            # 计算每层的阈值范围
            layer_ranges = []
            for i in range(len(splits) - 1):
                h_min = sorted_heights[splits[i + 1] - 1] if splits[i + 1] > splits[i] else 0
                h_max = sorted_heights[splits[i]]
                layer_ranges.append((h_min, h_max))
        else:
            layer_ranges = []

        # 对每个连通区域分配层次并画框
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            if area <= 5:
                continue

            # 分配层次
            layer_idx = len(layer_ranges) - 1  # 默认最小层
            for li, (h_min, h_max) in enumerate(layer_ranges):
                if h >= h_min:
                    layer_idx = li
                    break

            color_idx = min(layer_idx, len(LAYER_COLORS_BGR) - 1)
            thickness = max(1, 4 - color_idx)  # 大层粗框，小层细框
            color = LAYER_COLORS_BGR[color_idx]
            cv2.rectangle(img_draw, (x, y), (x + w, y + h), color, thickness)

            if layer_idx not in layers_data:
                layers_data[layer_idx] = []
            layers_data[layer_idx].append((w, h, area))

        img_with_boxes = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)

        report = f"===== 检测结果 =====\n"
        report += f"是否嵌套：{'是' if result['is_nested'] else '否'}\n"
        report += f"预测层数：{result['estimated_layers']} 层\n"

        easy_ocr_results, ocr_msg = recognize_text_with_easyocr(
            img_cv, result.get('ocr_boxes', [])
        )
        report += f"\n===== OCR识别结果（EasyOCR） =====\n"
        if ocr_msg:
            report += f"{ocr_msg}\n"
            fallback_text = result.get('recognized_text', '')
            if fallback_text:
                report += f"内置识别文本：{fallback_text}\n"
        elif easy_ocr_results:
            main_hits = [r for r in easy_ocr_results if r.get('priority') == 'main']
            detail_hits = [r for r in easy_ocr_results if r.get('priority') == 'detail']
            if not main_hits:
                main_hits = easy_ocr_results[:2]

            main_text = ''.join(r['text'] for r in main_hits[:12])
            all_text = ''.join(r['text'] for r in easy_ocr_results[:30])
            report += f"主字优先文本：{main_text}\n"
            report += f"综合文本：{all_text}\n"
            report += f"主字候选数：{len(main_hits)}，细节候选数：{len(detail_hits)}\n"
            report += "主字候选 (前6)：\n"
            for r in main_hits[:6]:
                report += (
                    f"  '{r['text']}' ({r['confidence']:.0%})"
                    f" score={r.get('main_score', 0.0):.2f} box={r['box']}\n"
                )
            if detail_hits:
                report += "细节候选 (前6)：\n"
                for r in detail_hits[:6]:
                    report += f"  '{r['text']}' ({r['confidence']:.0%}) box={r['box']}\n"
        else:
            report += "未识别出可用文本。\n"

        total_chars = sum(len(v) for v in layers_data.values())
        report += f"\n===== 字的大小 =====\n"
        report += f"总字数：{total_chars} 个\n\n"

        LAYER_NAMES = ['L1 大字', 'L2 中字', 'L3 小字', 'L4 微字', 'L5 极小']
        for li in sorted(layers_data.keys()):
            items = layers_data[li]
            ci = min(li, len(LAYER_COLORS_RGB) - 1)
            r, g, b = LAYER_COLORS_RGB[ci]
            name = LAYER_NAMES[ci] if ci < len(LAYER_NAMES) else f'L{ci+1}'
            avg_w = sum(w for w, h, a in items) / len(items)
            avg_h = sum(h for w, h, a in items) / len(items)
            max_h_val = max(h for w, h, a in items)
            min_h_val = min(h for w, h, a in items)
            report += (
                f"{name}（{r},{g},{b}框）：{len(items)} 个\n"
                f"  平均：{avg_w:.0f}x{avg_h:.0f}px\n"
                f"  高度：{min_h_val}~{max_h_val}px\n\n"
            )

        return Image.fromarray(img_with_boxes), report

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def crypto_encode(key_text, secret_text, font_size, small_font_size, step, wrap_after):
    """加密：用密钥（大字）隐藏密文（小字）"""
    if not key_text.strip():
        return None, "请输入密钥（大字）"
    if not secret_text.strip():
        return None, "请输入密文（小字）"

    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        wrap = int(wrap_after) if wrap_after > 0 else 0
        create_text_fill_art(
            large_text=key_text,
            small_text=secret_text,
            large_font_size=int(font_size),
            small_font_size=int(small_font_size),
            font_path=FONT_PATH,
            output_filename=tmp_path,
            step_x=int(step),
            step_y=int(step),
            wrap_after=wrap,
        )
        image = Image.open(tmp_path).copy()

        info = (
            f"加密完成\n"
            f"密钥（大字）：{key_text}\n"
            f"密文长度：{len(secret_text)} 字\n"
            f"图像尺寸：{image.size[0]}x{image.size[1]}\n"
            f"大字号：{int(font_size)}px，小字号：{int(small_font_size)}px\n"
            f"\n传输此图即传输密文。解密方需要知道密钥和字体参数。"
        )
        return image, info
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def crypto_decode(image, key_text, font_size, wrap_after):
    """解密：用密钥重建大字mask，提取小字区域"""
    if image is None:
        return None, None, "请先上传或生成加密图像"
    if not key_text.strip():
        return None, None, "请输入密钥（大字）才能解密"

    if isinstance(image, Image.Image):
        img_pil = image.copy()
    else:
        img_pil = Image.fromarray(image)

    img_w, img_h = img_pil.size

    try:
        large_font = ImageFont.truetype(FONT_PATH, int(font_size))
    except Exception:
        large_font = ImageFont.load_default()

    wrap = int(wrap_after) if wrap_after > 0 else 0
    if wrap > 0:
        lines = [key_text[i:i + wrap] for i in range(0, len(key_text), wrap)]
    else:
        lines = [key_text]

    line_height = int(int(font_size) * 1.2)
    padding_y = int(int(font_size) * 0.6)

    key_mask = Image.new('L', (img_w, img_h), 0)
    mask_draw = ImageDraw.Draw(key_mask)

    current_y = padding_y
    for line in lines:
        try:
            line_width = large_font.getlength(line)
        except Exception:
            line_width = len(line) * int(font_size)

        current_x = (img_w - line_width) / 2
        for char in line:
            mask_draw.text((current_x, current_y), char, font=large_font, fill=255)
            try:
                char_w = large_font.getlength(char)
            except Exception:
                char_w = int(font_size)
            current_x += char_w
        current_y += line_height

    mask_arr = np.array(key_mask)

    img_array = np.array(img_pil)
    highlight = img_array.copy()
    highlight[mask_arr < 128] = (highlight[mask_arr < 128] * 0.3).astype(np.uint8)
    edges = cv2.Canny(mask_arr, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    highlight[edges_dilated > 0] = [0, 255, 0]

    ys, xs = np.where(mask_arr > 128)
    if len(ys) > 0:
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        secret_region = img_array[y1:y2, x1:x2].copy()
        mask_region = mask_arr[y1:y2, x1:x2]
        secret_region[mask_region < 128] = 255
    else:
        secret_region = img_array.copy()

    fill_pixels = np.sum(mask_arr > 128)
    total_pixels = img_w * img_h
    fill_ratio = fill_pixels / total_pixels if total_pixels > 0 else 0

    det = get_detector()
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        img_pil.save(tmp_path)
        result = det.detect(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    report = (
        f"========== 解密分析 ==========\n"
        f"密钥：{key_text}\n"
        f"密钥字数：{len(key_text)}\n"
        f"图像尺寸：{img_w}x{img_h}\n"
        f"密文区域占比：{fill_ratio:.1%}\n"
        f"\n"
        f"----- AI检测结果 -----\n"
        f"检测到嵌套：{'是' if result['is_nested'] else '否'}\n"
        f"预测层数：{result['estimated_layers']} 层\n"
        f"置信度：{result['confidence']:.2%}\n"
        f"OCR检测框：{result['num_boxes']} 个\n"
        f"\n"
        f"----- 对抗评估 -----\n"
    )

    if result['is_nested'] and result['confidence'] > 0.7:
        report += (
            f"安全等级：低\n"
            f"AI能以{result['confidence']:.0%}置信度检测到嵌套结构。\n"
            f"建议：增加嵌套层数或添加噪声干扰。"
        )
    elif result['is_nested'] and result['confidence'] <= 0.7:
        report += (
            f"安全等级：中\n"
            f"AI检测到嵌套但置信度较低({result['confidence']:.0%})。\n"
            f"有一定隐蔽性，但仍可被检测。"
        )
    else:
        report += (
            f"安全等级：高\n"
            f"AI未能检测到嵌套结构。\n"
            f"密文具有较好隐蔽性。"
        )

    report += "\n=============================="

    return Image.fromarray(highlight), Image.fromarray(secret_region), report


# ==================== tkinter UI ====================

DEFAULT_FILL = (
    "床前明月光疑是地上霜举头望明月低头思故乡"
    "春眠不觉晓处处闻啼鸟夜来风雨声花落知多少"
    "白日依山尽黄河入海流欲穷千里目更上一层楼"
    "红豆生南国春来发几枝愿君多采撷此物最相思"
    "千山鸟飞绝万径人踪灭孤舟蓑笠翁独钓寒江雪"
    "松下问童子言师采药去只在此山中云深不知处"
    "锄禾日当午汗滴禾下土谁知盘中餐粒粒皆辛苦"
    "向晚意不适驱车登古原夕阳无限好只是近黄昏"
    "远上寒山石径斜白云深处有人家停车坐爱枫林晚霜叶红于二月花"
    "月落乌啼霜满天江枫渔火对愁眠姑苏城外寒山寺夜半钟声到客船"
    "空山不见人但闻人语响返景入深林复照青苔上"
    "独坐幽篁里弹琴复长啸深林人不知明月来相照"
    "葡萄美酒夜光杯欲饮琵琶马上催醉卧沙场君莫笑古来征战几人回"
    "秦时明月汉时关万里长征人未还但使龙城飞将在不教胡马度阴山"
    "朝辞白帝彩云间千里江陵一日还两岸猿声啼不住轻舟已过万重山"
    "李白乘舟将欲行忽闻岸上踏歌声桃花潭水深千尺不及汪伦送我情"
    "故人西辞黄鹤楼烟花三月下扬州孤帆远影碧空尽唯见长江天际流"
    "日照香炉生紫烟遥看瀑布挂前川飞流直下三千尺疑是银河落九天"
)

# 主题配色
BG_DARK = "#1e1e2e"
BG_MID = "#2a2a3d"
BG_LIGHT = "#363650"
FG_MAIN = "#cdd6f4"
FG_DIM = "#a6adc8"
ACCENT = "#89b4fa"
ACCENT2 = "#a6e3a1"
ACCENT3 = "#f9e2af"
DANGER = "#f38ba8"


class NestedCharApp:
    def __init__(self, root):
        self.root = root
        self.root.title("嵌套字符工坊")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 650)
        self.root.configure(bg=BG_DARK)

        # 状态
        self.gen_image = None
        self.detect_image = None
        self.enc_image = None
        self.dec_highlight = None
        self.dec_extracted = None

        self._setup_styles()
        self._build_ui()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        style.configure('.', background=BG_DARK, foreground=FG_MAIN, borderwidth=0)
        style.configure('TNotebook', background=BG_DARK, borderwidth=0)
        style.configure('TNotebook.Tab',
                        background=BG_MID, foreground=FG_DIM,
                        padding=[18, 8], font=('Microsoft YaHei UI', 11, 'bold'),
                        borderwidth=0)
        style.map('TNotebook.Tab',
                  background=[('selected', BG_LIGHT)],
                  foreground=[('selected', ACCENT)])

        style.configure('TFrame', background=BG_DARK)
        style.configure('TLabelframe', background=BG_MID, foreground=ACCENT,
                        borderwidth=1, relief='groove',
                        font=('Microsoft YaHei UI', 10, 'bold'))
        style.configure('TLabelframe.Label', background=BG_MID, foreground=ACCENT,
                        font=('Microsoft YaHei UI', 10, 'bold'))

        style.configure('TLabel', background=BG_DARK, foreground=FG_MAIN,
                        font=('Microsoft YaHei UI', 10))
        style.configure('Hint.TLabel', background=BG_DARK, foreground=FG_DIM,
                        font=('Microsoft YaHei UI', 9))
        style.configure('Warn.TLabel', background=BG_DARK, foreground=ACCENT3,
                        font=('Microsoft YaHei UI', 9))
        style.configure('Success.TLabel', background=BG_DARK, foreground=ACCENT2,
                        font=('Microsoft YaHei UI', 9))

        style.configure('Accent.TButton', background=ACCENT, foreground=BG_DARK,
                        font=('Microsoft YaHei UI', 10, 'bold'),
                        padding=[16, 6], borderwidth=0)
        style.map('Accent.TButton',
                  background=[('active', '#74c7ec'), ('pressed', '#89dceb')])

        style.configure('Secondary.TButton', background=BG_LIGHT, foreground=FG_MAIN,
                        font=('Microsoft YaHei UI', 9),
                        padding=[10, 4], borderwidth=0)
        style.map('Secondary.TButton',
                  background=[('active', '#45475a')])

        style.configure('TScale', background=BG_DARK, troughcolor=BG_MID,
                        sliderthickness=16)
        style.configure('TSpinbox', background=BG_LIGHT, foreground=FG_MAIN,
                        fieldbackground=BG_MID, arrowcolor=FG_DIM)

        style.configure('Img.TLabel', background='#11111b', foreground=FG_DIM,
                        font=('Microsoft YaHei UI', 9), relief='sunken',
                        borderwidth=1)

    # ---------- 图片显示工具 ----------
    def _pil_to_photo(self, pil_image):
        if pil_image is None:
            return None
        return ImageTk.PhotoImage(pil_image.copy())

    def _create_image_canvas(self, parent, placeholder='等待图片...'):
        frame = ttk.Frame(parent)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        canvas = tk.Canvas(frame, bg='#11111b', highlightthickness=0, bd=0)
        hbar = ttk.Scrollbar(frame, orient='horizontal', command=canvas.xview)
        vbar = ttk.Scrollbar(frame, orient='vertical', command=canvas.yview)
        canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

        canvas.grid(row=0, column=0, sticky='nsew')
        vbar.grid(row=0, column=1, sticky='ns')
        hbar.grid(row=1, column=0, sticky='ew')

        canvas._photo = None
        canvas.create_text(
            12, 12, anchor='nw', text=placeholder, fill=FG_DIM,
            font=('Microsoft YaHei UI', 10)
        )
        canvas.configure(scrollregion=(0, 0, 1, 1))
        return frame, canvas

    def _show_image(self, widget, pil_image, max_w=480, max_h=320):
        if pil_image is None:
            if isinstance(widget, tk.Canvas):
                widget.delete('all')
                widget.create_text(
                    12, 12, anchor='nw', text='暂无图片', fill=FG_DIM,
                    font=('Microsoft YaHei UI', 10)
                )
                widget._photo = None
                widget.configure(scrollregion=(0, 0, 1, 1))
            else:
                widget.configure(image='')
                widget.image = None
            return

        # Do not downscale: keep original resolution and rely on scrollbars.
        photo = self._pil_to_photo(pil_image)
        if isinstance(widget, tk.Canvas):
            widget.delete('all')
            widget.create_image(0, 0, anchor='nw', image=photo)
            widget._photo = photo
            widget.configure(scrollregion=(0, 0, pil_image.width, pil_image.height))
        else:
            widget.configure(image=photo)
            widget.image = photo

    # ---------- 构建UI ----------
    def _build_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        self._build_gen_tab(notebook)
        self._build_crypto_tab(notebook)

    def _make_text(self, parent, height=2, width=28, **kw):
        """创建统一风格的Text控件"""
        txt = tk.Text(parent, height=height, width=width,
                      font=('Microsoft YaHei UI', 10),
                      bg=BG_MID, fg=FG_MAIN, insertbackground=FG_MAIN,
                      selectbackground=ACCENT, selectforeground=BG_DARK,
                      relief='flat', borderwidth=4,
                      padx=6, pady=4, **kw)
        return txt

    def _make_entry(self, parent, var, width=20, **kw):
        """创建统一风格的Entry控件"""
        ent = ttk.Entry(parent, textvariable=var, width=width,
                        font=('Microsoft YaHei UI', 10))
        return ent

    def _make_spinbox(self, parent, from_, to, var, width=5, increment=1):
        """创建统一风格的Spinbox"""
        spn = ttk.Spinbox(parent, from_=from_, to=to, increment=increment,
                          textvariable=var, width=width)
        return spn

    def _build_gen_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text='  生成与检测  ')

        # ---- 左侧：参数面板 ----
        left = ttk.LabelFrame(tab, text=' 生成参数 ', padding=12)
        left.pack(side='left', fill='y', padx=(8, 4), pady=8)

        ttk.Label(left, text='嵌套层数', style='TLabel').pack(anchor='w')
        self.layers_var = tk.IntVar(value=2)
        layers_frame = ttk.Frame(left)
        layers_frame.pack(fill='x', pady=(2, 0))
        layers_scale = ttk.Scale(layers_frame, from_=1, to=5, variable=self.layers_var,
                                  orient='horizontal', command=self._on_layer_change)
        layers_scale.pack(side='left', fill='x', expand=True)
        self.layers_label = ttk.Label(layers_frame, text='2', width=3,
                                       font=('Consolas', 14, 'bold'),
                                       foreground=ACCENT, background=BG_DARK)
        self.layers_label.pack(side='right')

        self.layer_info_var = tk.StringVar(value=LAYER_CONFIG[2]["desc"])
        ttk.Label(left, textvariable=self.layer_info_var, style='Hint.TLabel',
                  wraplength=230).pack(anchor='w', pady=4)

        ttk.Label(left, text='主字文本').pack(anchor='w', pady=(10, 0))
        self.main_text = self._make_text(left, height=2, width=28)
        self.main_text.pack(fill='x', pady=2)
        self.main_text.insert('1.0', '海内存知己天涯若比邻')

        ttk.Label(left, text='填充文本').pack(anchor='w', pady=(10, 0))
        self.fill_text = self._make_text(left, height=5, width=28)
        self.fill_text.pack(fill='x', pady=2)
        self.fill_text.insert('1.0', DEFAULT_FILL)

        ttk.Label(left, text='每行字数 (0=不换行)').pack(anchor='w', pady=(10, 0))
        self.wrap_var = tk.IntVar(value=5)
        self._make_spinbox(left, 0, 20, self.wrap_var, width=5).pack(anchor='w', pady=2)

        self.gen_warn_var = tk.StringVar(value='')
        ttk.Label(left, textvariable=self.gen_warn_var, style='Warn.TLabel',
                  wraplength=230).pack(anchor='w', pady=2)

        ttk.Button(left, text='生成嵌套图', style='Accent.TButton',
                    command=self._on_generate).pack(fill='x', pady=(12, 2))

        # ---- 右侧：结果 ----
        right = ttk.Frame(tab)
        right.pack(side='right', fill='both', expand=True, padx=(4, 8), pady=8)

        # 生成结果
        gen_frame = ttk.LabelFrame(right, text=' 生成结果 ', padding=6)
        gen_frame.pack(fill='x', pady=(0, 6))

        gen_canvas_frame, self.gen_img_label = self._create_image_canvas(
            gen_frame, placeholder='点击「生成嵌套图」开始'
        )
        gen_canvas_frame.pack(fill='both', expand=True)

        gen_action = ttk.Frame(gen_frame)
        gen_action.pack(fill='x', pady=6)
        ttk.Button(gen_action, text='运行检测', style='Accent.TButton',
                    command=self._on_detect).pack(side='left', padx=2)
        ttk.Button(gen_action, text='保存图片', style='Secondary.TButton',
                    command=self._save_gen_image).pack(side='right', padx=2)

        # 检测结果
        det_frame = ttk.LabelFrame(right, text=' 检测结果 ', padding=6)
        det_frame.pack(fill='both', expand=True)

        det_inner = ttk.Frame(det_frame)
        det_inner.pack(fill='both', expand=True)

        det_canvas_frame, self.det_img_label = self._create_image_canvas(
            det_inner, placeholder='等待检测...'
        )
        det_canvas_frame.pack(side='left', fill='both', expand=True)

        report_frame = ttk.Frame(det_inner)
        report_frame.pack(side='right', fill='both', expand=True)
        self.det_report = tk.Text(report_frame, width=34, height=14,
                                   font=('Consolas', 9), wrap='word', state='disabled',
                                   bg=BG_MID, fg=FG_MAIN, insertbackground=FG_MAIN,
                                   selectbackground=ACCENT, selectforeground=BG_DARK,
                                   relief='flat', borderwidth=4, padx=6, pady=4)
        det_scroll = ttk.Scrollbar(report_frame, command=self.det_report.yview)
        self.det_report.configure(yscrollcommand=det_scroll.set)
        det_scroll.pack(side='right', fill='y')
        self.det_report.pack(side='left', fill='both', expand=True)

    def _build_crypto_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text='  加密对抗  ')

        ttk.Label(tab, text='原理：大字=密钥（定位密文区域），小字=密文（隐藏在大字笔画中的秘密信息）',
                  style='Hint.TLabel', wraplength=900).pack(anchor='w', padx=12, pady=(6, 0))

        # ---- 上半：加密 ----
        enc_outer = ttk.LabelFrame(tab, text=' 加密（发送方） ', padding=10)
        enc_outer.pack(fill='x', padx=12, pady=6)

        enc_inner = ttk.Frame(enc_outer)
        enc_inner.pack(fill='x')

        enc_params = ttk.Frame(enc_inner)
        enc_params.pack(side='left', fill='y', padx=(0, 12))

        ttk.Label(enc_params, text='密钥（大字）').pack(anchor='w')
        self.enc_key_var = tk.StringVar(value='天下大同')
        self._make_entry(enc_params, self.enc_key_var, width=20).pack(fill='x', pady=2)

        ttk.Label(enc_params, text='密文（小字）').pack(anchor='w', pady=(8, 0))
        self.enc_secret = self._make_text(enc_params, height=3, width=28)
        self.enc_secret.pack(fill='x', pady=2)
        self.enc_secret.insert('1.0', '明日午时三刻于城南老槐树下接头暗号风紧扯呼')

        param_row = ttk.Frame(enc_params)
        param_row.pack(fill='x', pady=4)
        ttk.Label(param_row, text='大字号:').pack(side='left')
        self.enc_fontsize_var = tk.IntVar(value=400)
        self._make_spinbox(param_row, 100, 500, self.enc_fontsize_var, width=5, increment=50).pack(side='left', padx=2)
        ttk.Label(param_row, text='小字号:').pack(side='left', padx=(12, 0))
        self.enc_smallsize_var = tk.IntVar(value=10)
        self._make_spinbox(param_row, 4, 20, self.enc_smallsize_var, width=4).pack(side='left', padx=2)

        param_row2 = ttk.Frame(enc_params)
        param_row2.pack(fill='x', pady=2)
        ttk.Label(param_row2, text='填充步长:').pack(side='left')
        self.enc_step_var = tk.IntVar(value=11)
        self._make_spinbox(param_row2, 3, 20, self.enc_step_var, width=4).pack(side='left', padx=2)
        ttk.Label(param_row2, text='每行字数:').pack(side='left', padx=(12, 0))
        self.enc_wrap_var = tk.IntVar(value=2)
        self._make_spinbox(param_row2, 0, 10, self.enc_wrap_var, width=4).pack(side='left', padx=2)

        ttk.Button(enc_params, text='加密生成', style='Accent.TButton',
                    command=self._on_encrypt).pack(fill='x', pady=(10, 2))

        self.enc_info_var = tk.StringVar(value='')
        ttk.Label(enc_params, textvariable=self.enc_info_var, style='Success.TLabel',
                  wraplength=230).pack(anchor='w')

        enc_img_frame = ttk.Frame(enc_inner)
        enc_img_frame.pack(side='right', fill='both', expand=True)
        enc_canvas_frame, self.enc_img_label = self._create_image_canvas(
            enc_img_frame, placeholder='等待加密...'
        )
        enc_canvas_frame.pack(fill='both', expand=True)

        # ---- 下半：解密 ----
        dec_outer = ttk.LabelFrame(tab, text=' 解密（接收方） ', padding=10)
        dec_outer.pack(fill='both', expand=True, padx=12, pady=6)

        dec_inner = ttk.Frame(dec_outer)
        dec_inner.pack(fill='both', expand=True)

        dec_params = ttk.Frame(dec_inner)
        dec_params.pack(side='left', fill='y', padx=(0, 12))

        ttk.Label(dec_params, text='上传加密图').pack(anchor='w')
        upload_row = ttk.Frame(dec_params)
        upload_row.pack(fill='x', pady=2)
        self.dec_image_path = tk.StringVar(value='')
        ttk.Entry(upload_row, textvariable=self.dec_image_path, width=20).pack(side='left', fill='x', expand=True)
        ttk.Button(upload_row, text='浏览', style='Secondary.TButton',
                    command=self._browse_dec_image).pack(side='right', padx=2)
        ttk.Button(dec_params, text='使用上方加密图', style='Secondary.TButton',
                    command=self._use_enc_image).pack(fill='x', pady=2)

        ttk.Label(dec_params, text='密钥').pack(anchor='w', pady=(8, 0))
        self.dec_key_var = tk.StringVar(value='天下大同')
        self._make_entry(dec_params, self.dec_key_var, width=20).pack(fill='x', pady=2)

        dec_param_row = ttk.Frame(dec_params)
        dec_param_row.pack(fill='x', pady=4)
        ttk.Label(dec_param_row, text='大字号:').pack(side='left')
        self.dec_fontsize_var = tk.IntVar(value=400)
        self._make_spinbox(dec_param_row, 100, 500, self.dec_fontsize_var, width=5, increment=50).pack(side='left', padx=2)
        ttk.Label(dec_param_row, text='每行字数:').pack(side='left', padx=(12, 0))
        self.dec_wrap_var = tk.IntVar(value=2)
        self._make_spinbox(dec_param_row, 0, 10, self.dec_wrap_var, width=4).pack(side='left', padx=2)

        ttk.Button(dec_params, text='解密分析', style='Accent.TButton',
                    command=self._on_decrypt).pack(fill='x', pady=(10, 2))

        # 解密结果区域
        dec_result = ttk.Frame(dec_inner)
        dec_result.pack(side='right', fill='both', expand=True)

        dec_imgs = ttk.Frame(dec_result)
        dec_imgs.pack(fill='both', expand=True)

        hl_frame = ttk.LabelFrame(dec_imgs, text=' 密文区域高亮 ', padding=4)
        hl_frame.pack(side='left', fill='both', expand=True)
        hl_canvas_frame, self.dec_hl_label = self._create_image_canvas(
            hl_frame, placeholder='等待解密...'
        )
        hl_canvas_frame.pack(fill='both', expand=True)

        ex_frame = ttk.LabelFrame(dec_imgs, text=' 提取的密文区域 ', padding=4)
        ex_frame.pack(side='left', fill='both', expand=True, padx=4)
        ex_canvas_frame, self.dec_ex_label = self._create_image_canvas(
            ex_frame, placeholder='等待解密...'
        )
        ex_canvas_frame.pack(fill='both', expand=True)

        self.dec_report = tk.Text(dec_result, height=10, width=60,
                                    font=('Consolas', 9), wrap='word', state='disabled',
                                    bg=BG_MID, fg=FG_MAIN, insertbackground=FG_MAIN,
                                    selectbackground=ACCENT, selectforeground=BG_DARK,
                                    relief='flat', borderwidth=4, padx=6, pady=4)
        dec_scroll = ttk.Scrollbar(dec_result, command=self.dec_report.yview)
        self.dec_report.configure(yscrollcommand=dec_scroll.set)
        dec_scroll.pack(side='right', fill='y')
        self.dec_report.pack(side='left', fill='both', expand=True, pady=(6, 0))

    # ---------- 回调 ----------

    def _on_layer_change(self, val=None):
        layer = int(float(val)) if val else self.layers_var.get()
        self.layers_var.set(layer)
        self.layers_label.configure(text=str(layer))
        self.layer_info_var.set(LAYER_CONFIG[layer]["desc"])

    def _on_generate(self):
        main = self.main_text.get('1.0', 'end').strip()
        fill = self.fill_text.get('1.0', 'end').strip()
        layers = self.layers_var.get()
        wrap = self.wrap_var.get()

        self.gen_warn_var.set('生成中...')
        self.root.update_idletasks()

        try:
            image, warn = generate_nested_image(main, fill, layers, wrap)
            if image is None:
                self.gen_warn_var.set(warn)
                return
            self.gen_image = image
            self._show_image(self.gen_img_label, image)
            self.gen_warn_var.set(warn)
        except Exception as e:
            self.gen_warn_var.set(f'生成失败: {e}')

    def _on_detect(self):
        if self.gen_image is None:
            messagebox.showinfo('提示', '请先生成嵌套图')
            return

        self.det_report.configure(state='normal')
        self.det_report.delete('1.0', 'end')
        self.det_report.insert('1.0', '检测中...')
        self.det_report.configure(state='disabled')
        self.root.update_idletasks()

        try:
            det_img, report = run_detection(self.gen_image)
            self.detect_image = det_img
            if det_img:
                self._show_image(self.det_img_label, det_img)

            self.det_report.configure(state='normal')
            self.det_report.delete('1.0', 'end')
            self.det_report.insert('1.0', report)
            self.det_report.configure(state='disabled')
        except Exception as e:
            self.det_report.configure(state='normal')
            self.det_report.delete('1.0', 'end')
            self.det_report.insert('1.0', f'检测失败: {e}')
            self.det_report.configure(state='disabled')

    def _save_gen_image(self):
        if self.gen_image is None:
            messagebox.showinfo('提示', '请先生成图片')
            return
        path = filedialog.asksaveasfilename(defaultextension='.png',
                                              filetypes=[('PNG', '*.png'), ('JPEG', '*.jpg')])
        if path:
            self.gen_image.save(path)
            messagebox.showinfo('保存成功', f'图片已保存到: {path}')

    def _on_encrypt(self):
        key = self.enc_key_var.get().strip()
        secret = self.enc_secret.get('1.0', 'end').strip()
        fs = self.enc_fontsize_var.get()
        sfs = self.enc_smallsize_var.get()
        step = self.enc_step_var.get()
        wrap = self.enc_wrap_var.get()

        self.enc_info_var.set('加密中...')
        self.root.update_idletasks()

        try:
            image, info = crypto_encode(key, secret, fs, sfs, step, wrap)
            if image is None:
                self.enc_info_var.set(info)
                return
            self.enc_image = image
            self._show_image(self.enc_img_label, image, max_w=400, max_h=280)
            self.enc_info_var.set(info)
        except Exception as e:
            self.enc_info_var.set(f'加密失败: {e}')

    def _browse_dec_image(self):
        path = filedialog.askopenfilename(filetypes=[('图片', '*.png *.jpg *.jpeg *.bmp')])
        if path:
            self.dec_image_path.set(path)
            try:
                self.enc_image = Image.open(path)
            except Exception:
                pass

    def _use_enc_image(self):
        if self.enc_image is None:
            messagebox.showinfo('提示', '请先生成加密图')
            return
        self.dec_image_path.set('（使用上方加密图）')

    def _on_decrypt(self):
        path = self.dec_image_path.get()
        dec_img = None
        if path and not path.startswith('（'):
            try:
                dec_img = Image.open(path)
            except Exception as e:
                messagebox.showerror('错误', f'无法加载图片: {e}')
                return
        elif self.enc_image is not None:
            dec_img = self.enc_image

        if dec_img is None:
            messagebox.showinfo('提示', '请先上传或生成加密图')
            return

        key = self.dec_key_var.get().strip()
        fs = self.dec_fontsize_var.get()
        wrap = self.dec_wrap_var.get()

        self.dec_report.configure(state='normal')
        self.dec_report.delete('1.0', 'end')
        self.dec_report.insert('1.0', '解密分析中...')
        self.dec_report.configure(state='disabled')
        self.root.update_idletasks()

        try:
            hl, extracted, report = crypto_decode(dec_img, key, fs, wrap)
            if hl is None:
                self.dec_report.configure(state='normal')
                self.dec_report.delete('1.0', 'end')
                self.dec_report.insert('1.0', report)
                self.dec_report.configure(state='disabled')
                return

            self.dec_highlight = hl
            self.dec_extracted = extracted
            self._show_image(self.dec_hl_label, hl, max_w=280, max_h=200)
            self._show_image(self.dec_ex_label, extracted, max_w=280, max_h=200)

            self.dec_report.configure(state='normal')
            self.dec_report.delete('1.0', 'end')
            self.dec_report.insert('1.0', report)
            self.dec_report.configure(state='disabled')
        except Exception as e:
            self.dec_report.configure(state='normal')
            self.dec_report.delete('1.0', 'end')
            self.dec_report.insert('1.0', f'解密失败: {e}')
            self.dec_report.configure(state='disabled')


def main():
    try:
        root = tk.Tk()
        app = NestedCharApp(root)
        root.mainloop()
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        try:
            from tkinter import messagebox
            messagebox.showerror('启动错误', tb)
        except Exception:
            input(f'Error:\n{tb}\nPress Enter to exit...')


if __name__ == '__main__':
    main()
