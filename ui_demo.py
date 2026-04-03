# -*- coding: utf-8 -*-
"""
嵌套字符交互式演示 - Gradio Web UI
功能：生成嵌套字符图 → 展示OCR检测能力
"""

import gradio as gr
import numpy as np
import cv2
import os
import tempfile
from PIL import Image, ImageDraw, ImageFont
from nested_char_detector import NestedCharDetector

# ==================== 配置 ====================
FONT_PATH = 'C:/Windows/Fonts/simhei.ttf'
MODEL_PATH = 'models/nested_char_model.pth'

# 各层字数限制建议
LAYER_LIMITS = {
    1: {"main_max": 20, "fill_min": 10,   "desc": "1层：主字≤20字，填充≥10字"},
    2: {"main_max": 10, "fill_min": 50,   "desc": "2层：主字≤10字，填充≥50字"},
    3: {"main_max": 5,  "fill_min": 200,  "desc": "3层：主字≤5字，填充≥200字"},
    4: {"main_max": 3,  "fill_min": 500,  "desc": "4层：主字≤3字，填充≥500字"},
    5: {"main_max": 2,  "fill_min": 1000, "desc": "5层：主字≤2字，填充≥1000字"},
}

# 全局检测器（延迟初始化）
detector = None


def get_detector():
    global detector
    if detector is None:
        detector = NestedCharDetector(model_path=MODEL_PATH if os.path.exists(MODEL_PATH) else None)
    return detector


# ==================== 嵌套图生成 ====================

def generate_nested_image(main_text, fill_text, layers, wrap_after):
    """生成多层嵌套字符图像"""
    if not main_text.strip():
        return None, "请输入主字文本"
    if layers >= 2 and not fill_text.strip():
        return None, "2层及以上需要填充文本"

    limits = LAYER_LIMITS[layers]
    warnings = []
    if len(main_text) > limits["main_max"]:
        warnings.append(f"主字建议≤{limits['main_max']}字，当前{len(main_text)}字，已自动截断")
        main_text = main_text[:limits["main_max"]]
    if layers >= 2 and len(fill_text) < limits["fill_min"]:
        warnings.append(f"填充文本建议≥{limits['fill_min']}字，当前{len(fill_text)}字，效果可能不佳")

    wrap = max(1, wrap_after) if wrap_after > 0 else len(main_text)

    # 根据层数选择字号
    large_font_size = max(60, 300 // max(1, wrap))
    base_small_size = max(6, large_font_size // (2 ** (layers - 1))) if layers >= 2 else 12

    try:
        large_font = ImageFont.truetype(FONT_PATH, large_font_size)
    except:
        large_font = ImageFont.load_default()

    # 计算画布大小
    line_spacing = 1.2
    if wrap > 0:
        text_lines = [main_text[i:i + wrap] for i in range(0, len(main_text), wrap)]
    else:
        text_lines = [main_text]

    line_height = int(large_font_size * line_spacing)
    padding = int(large_font_size * 0.5)
    img_h = line_height * len(text_lines) + 2 * padding
    img_w = int(wrap * large_font_size * 1.05) + 2 * padding

    image = Image.new('RGB', (img_w, img_h), 'white')
    draw = ImageDraw.Draw(image)

    # --- 第1层：绘制主字轮廓（用填充字或直接绘制） ---
    if layers == 1:
        # 单层：直接用小字填充大字轮廓
        _draw_filled_text(image, draw, text_lines, large_font, large_font_size,
                          fill_text if fill_text.strip() else main_text,
                          base_small_size, padding, line_height, img_w, img_h,
                          text_color='black')
    else:
        # 多层嵌套
        _draw_filled_text(image, draw, text_lines, large_font, large_font_size,
                          fill_text, base_small_size, padding, line_height,
                          img_w, img_h, text_color='black')

        # 后续层：在填充字的笔画内再填充更小的字
        for layer_idx in range(2, layers):
            smaller_size = max(4, base_small_size // (2 ** (layer_idx - 1)))
            gray_val = 80 + (layer_idx * 40) % 120
            color = (gray_val, gray_val, gray_val)

            # 创建当前图像的笔画mask
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

            try:
                tiny_font = ImageFont.truetype(FONT_PATH, smaller_size)
            except:
                tiny_font = ImageFont.load_default()

            step = max(smaller_size // 2, 2)
            char_idx = 0
            for y in range(0, img_h, step):
                for x in range(0, img_w, step):
                    if y < mask.shape[0] and x < mask.shape[1] and mask[y, x] > 128:
                        ch = fill_text[char_idx % len(fill_text)]
                        draw.text((x, y), ch, font=tiny_font, fill=color)
                        char_idx += 1

    warn_msg = " | ".join(warnings) if warnings else ""
    return image, warn_msg


def _draw_filled_text(image, draw, text_lines, large_font, large_font_size,
                      fill_text, small_font_size, padding, line_height,
                      img_w, img_h, text_color='black'):
    """用小字填充大字轮廓"""
    try:
        small_font = ImageFont.truetype(FONT_PATH, small_font_size)
    except:
        small_font = ImageFont.load_default()

    step = max(small_font_size // 2, 2)
    fill_idx = 0

    current_y = padding
    for line in text_lines:
        try:
            line_w = large_font.getlength(line)
        except:
            line_w = len(line) * large_font_size
        start_x = (img_w - line_w) / 2

        for ch in line:
            # 创建单字mask
            char_mask = Image.new('L', (img_w, img_h), 0)
            mask_draw = ImageDraw.Draw(char_mask)
            mask_draw.text((start_x, current_y), ch, font=large_font, fill=255)
            mask_arr = np.array(char_mask)

            for y in range(0, img_h, step):
                for x in range(0, img_w, step):
                    if mask_arr[y, x] > 128:
                        fc = fill_text[fill_idx % len(fill_text)] if fill_text else ch
                        draw.text((x, y), fc, font=small_font, fill=text_color)
                        fill_idx += 1

            try:
                char_w = large_font.getlength(ch)
            except:
                char_w = large_font_size
            start_x += char_w

        current_y += line_height


# ==================== OCR检测 ====================

def run_detection(image):
    """对图像执行嵌套字符检测"""
    if image is None:
        return None, "请先生成图像"

    det = get_detector()

    # 保存临时文件供检测器使用
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        if isinstance(image, Image.Image):
            image.save(tmp_path)
        else:
            cv2.imwrite(tmp_path, cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))

        result = det.detect(tmp_path)

        # 在图像上绘制检测框
        img_cv = cv2.imread(tmp_path)
        for (x, y, w, h) in result['ocr_boxes']:
            cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # 转回RGB
        img_with_boxes = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        # 构建结果文本
        report = f"""
========== 检测结果 ==========
是否嵌套：{'是' if result['is_nested'] else '否'}
预测层数：{result['estimated_layers']} 层
置信度　：{result['confidence']:.2%}
检测框数：{result['num_boxes']} 个

----- 变换参数 -----
旋转角度：{result['transform_params']['rotation']:.2f}°
X轴缩放：{result['transform_params']['scale_x']:.2f}
Y轴缩放：{result['transform_params']['scale_y']:.2f}
水平翻转：{'是' if result['transform_params']['flip_horizontal'] else '否'}
垂直翻转：{'是' if result['transform_params']['flip_vertical'] else '否'}

----- 金字塔分析 -----
边缘密度：{', '.join([f'L{i}={d:.4f}' for i, d in enumerate(result['pyramid_analysis']['edge_densities'])])}
字符大小：{', '.join([f'L{i}={s:.1f}' for i, s in enumerate(result['pyramid_analysis']['char_sizes'])])}
==============================
"""
        return img_with_boxes, report

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ==================== UI回调 ====================

def on_layer_change(layers):
    """层数变化时更新提示"""
    info = LAYER_LIMITS[layers]
    return info["desc"]


def on_generate(main_text, fill_text, layers, wrap_after):
    """生成按钮回调"""
    image, warn = generate_nested_image(main_text, fill_text, layers, wrap_after)
    return image, warn


def on_detect(image):
    """检测按钮回调"""
    return run_detection(image)


# ==================== Gradio界面 ====================

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
    "独坐幽篁���弹琴复长啸深林人不知明月来相照"
    "葡萄美酒夜光杯欲饮琵琶马上催醉卧沙场君莫笑古来征战几人回"
    "秦时明月汉时关万里长征人未还但使龙城飞将在不教胡马度阴山"
    "朝辞白帝彩云间千里江陵一日还两岸猿声啼不住轻舟已过万重山"
    "李白乘舟将欲行忽闻岸上踏歌声桃花潭水深千尺不及汪伦送我情"
    "故人西辞黄鹤楼烟花三月下扬州孤帆远影碧空尽唯见长江天际流"
    "日照香炉生紫烟遥看瀑布挂前川飞流直下三千尺疑是银河落九天"
)

with gr.Blocks(title="嵌套字符生成与OCR检测演示") as demo:
    gr.Markdown("# 嵌套字符生成与OCR检测演示")
    gr.Markdown("输入文本生成嵌套字符图，然后用AI模型检测嵌套层数并展示OCR识别框。")

    with gr.Row():
        # ===== 左侧：生成控制 =====
        with gr.Column(scale=1):
            gr.Markdown("### 生成参数")
            layers_slider = gr.Slider(1, 5, value=2, step=1, label="嵌套层数")
            layer_info = gr.Textbox(value=LAYER_LIMITS[2]["desc"], label="字数建议",
                                    interactive=False)
            main_input = gr.Textbox(value="海内存知己天涯若比邻", label="主字文本",
                                    placeholder="输入要显示的大字...", lines=2)
            fill_input = gr.Textbox(value=DEFAULT_FILL, label="填充文本",
                                    placeholder="用于填充笔画的小字...", lines=4)
            wrap_slider = gr.Slider(0, 20, value=5, step=1,
                                    label="每行字数 (0=不换行)")
            gen_btn = gr.Button("生成嵌套图", variant="primary", size="lg")
            gen_warn = gr.Textbox(label="提示", interactive=False, visible=True)

        # ===== 右侧：图像展示与检测 =====
        with gr.Column(scale=2):
            gr.Markdown("### 生成结果")
            output_image = gr.Image(label="嵌套字符图", type="pil", height=400)

            gr.Markdown("### OCR检测")
            detect_btn = gr.Button("运行OCR检测", variant="secondary", size="lg")

            with gr.Row():
                detect_image = gr.Image(label="检测结果（红框=OCR检测框）", height=400)
                detect_report = gr.Textbox(label="检测报告", lines=18, interactive=False)

    # ===== 事件绑定 =====
    layers_slider.change(on_layer_change, inputs=[layers_slider], outputs=[layer_info])

    gen_btn.click(on_generate,
                  inputs=[main_input, fill_input, layers_slider, wrap_slider],
                  outputs=[output_image, gen_warn])

    detect_btn.click(on_detect,
                     inputs=[output_image],
                     outputs=[detect_image, detect_report])


if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
