# -*- coding: utf-8 -*-
"""
嵌套字符交互式演示 - Gradio Web UI
功能：生成嵌套字符图 → 展示OCR检测能力 → 加密对抗
"""

import gradio as gr
import numpy as np
import cv2
import os
import tempfile
from PIL import Image, ImageDraw, ImageFont
from nested_char_detector import NestedCharDetector
from NestCharacter import create_text_fill_art

# ==================== 配置 ====================
FONT_PATH = 'C:/Windows/Fonts/simhei.ttf'
MODEL_PATH = 'models/nested_char_model.pth'

# 各层参数配置：大字号、小字号、步长
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


def get_detector():
    global detector
    if detector is None:
        detector = NestedCharDetector(model_path=MODEL_PATH if os.path.exists(MODEL_PATH) else None)
    return detector


# ==================== 嵌套图生成 ====================

def generate_nested_image(main_text, fill_text, layers, wrap_after):
    """生成多层嵌套字符图像，直接调用NestCharacter.create_text_fill_art"""
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

    # 第1层：用NestCharacter的原版逻辑生成高质量填充图
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

        # 多层嵌套：在已有图像的笔画内继续用更小字符填充
        if layers >= 2:
            draw = ImageDraw.Draw(image)
            img_w, img_h = image.size

            for layer_idx in range(2, layers + 1):
                smaller_size = max(3, cfg["small_fs"] // (2 ** (layer_idx - 1)))
                step = max(smaller_size, 3)
                gray_val = 60 + (layer_idx * 35) % 130
                color = (gray_val, gray_val, gray_val)

                # 用当前图像的笔画区域作为mask
                img_array = np.array(image)
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

                try:
                    tiny_font = ImageFont.truetype(FONT_PATH, smaller_size)
                except:
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


# ==================== OCR检测 ====================

def run_detection(image):
    """对图像执行嵌套字符检测，蓝框大字、绿框小字、红框逐字，输出字的大小"""
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

        # 1. 嵌套结构检测
        result = det.detect(tmp_path)

        # 2. 图像标注
        img_cv = cv2.imread(tmp_path)
        img_draw = img_cv.copy()
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # === 蓝色粗框：大字整体轮廓（闭操作合并小字成大字区块） ===
        kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_large, iterations=3)
        contours_large, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_boxes = []
        for cnt in contours_large:
            if cv2.contourArea(cnt) > 2000:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img_draw, (x, y), (x + w, y + h), (255, 0, 0), 3)
                large_boxes.append((w, h))

        # === 红框：逐字标注（连通区域分析） ===
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        char_sizes = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area <= 5:
                continue
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 0, 255), 1)
            char_sizes.append((w, h, area))

        # === 绿色框：小字区域（在大字内部的小连通区域） ===
        # 用小核膨胀把相邻小字合并成小字簇
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        dilated_small = cv2.dilate(binary, kernel_small, iterations=1)
        contours_small, _ = cv2.findContours(dilated_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        small_cluster_count = 0
        for cnt in contours_small:
            area = cv2.contourArea(cnt)
            if 100 < area < 50000:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
                small_cluster_count += 1

        img_with_boxes = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)

        # 3. 报告：字的大小统计
        report = f"===== 检测结果 =====\n"
        report += f"是否嵌套：{'是' if result['is_nested'] else '否'}\n"
        report += f"预测层数：{result['estimated_layers']} 层\n\n"

        report += f"===== 标注说明 =====\n"
        report += f"蓝色粗框：大字区域（{len(large_boxes)} 个）\n"
        report += f"绿色框　：小字簇（{small_cluster_count} 个）\n"
        report += f"红色细框：逐字标注（{len(char_sizes)} 个）\n\n"

        report += f"===== 字的大小 =====\n"
        if large_boxes:
            avg_w = sum(w for w, h in large_boxes) / len(large_boxes)
            avg_h = sum(h for w, h in large_boxes) / len(large_boxes)
            report += f"大字平均尺寸：{avg_w:.0f} x {avg_h:.0f} px\n"

        if char_sizes:
            heights = sorted([h for w, h, a in char_sizes], reverse=True)
            # 按高度分大小字：取最大高度的30%为分界线
            big_threshold = heights[0] * 0.3
            big = [(w, h) for w, h, a in char_sizes if h >= big_threshold]
            small = [(w, h) for w, h, a in char_sizes if h < big_threshold]

            if big:
                bw = sum(w for w, h in big) / len(big)
                bh = sum(h for w, h in big) / len(big)
                report += f"大字符：{len(big)} 个，平均 {bw:.0f}x{bh:.0f} px\n"
            if small:
                sw = sum(w for w, h in small) / len(small)
                sh = sum(h for w, h in small) / len(small)
                report += f"小字符：{len(small)} 个，平均 {sw:.0f}x{sh:.0f} px\n"

        return img_with_boxes, report

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ==================== UI回调 ====================

def on_layer_change(layers):
    """层数变化时更新提示"""
    info = LAYER_CONFIG[layers]
    return info["desc"]


def on_generate(main_text, fill_text, layers, wrap_after):
    """生成按钮回调"""
    image, warn = generate_nested_image(main_text, fill_text, layers, wrap_after)
    return image, warn


def on_detect(image):
    """检测按钮回调"""
    return run_detection(image)


# ==================== 加密对抗：大字为密钥，小字为密文 ====================

def crypto_encode(key_text, secret_text, font_size, small_font_size, step, wrap_after):
    """加密：用密钥（大字）隐藏密文（小字）到嵌套字符图中"""
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
    """解密：用密钥重建大字mask，提取小字区域，与OCR检测对比"""
    if image is None:
        return None, None, "请先上传或生成加密图像"
    if not key_text.strip():
        return None, None, "请输入密钥（大字）才能解密"

    # 获取图像尺寸
    if isinstance(image, Image.Image):
        img_pil = image.copy()
    else:
        img_pil = Image.fromarray(image)

    img_w, img_h = img_pil.size

    # 用密钥重建大字mask
    try:
        large_font = ImageFont.truetype(FONT_PATH, int(font_size))
    except:
        large_font = ImageFont.load_default()

    wrap = int(wrap_after) if wrap_after > 0 else 0
    if wrap > 0:
        lines = [key_text[i:i + wrap] for i in range(0, len(key_text), wrap)]
    else:
        lines = [key_text]

    line_height = int(int(font_size) * 1.2)
    padding_y = int(int(font_size) * 0.6)

    # 重建完整mask
    key_mask = Image.new('L', (img_w, img_h), 0)
    mask_draw = ImageDraw.Draw(key_mask)

    current_y = padding_y
    for line in lines:
        try:
            line_width = large_font.getlength(line)
        except:
            line_width = len(line) * int(font_size)

        current_x = (img_w - line_width) / 2
        for char in line:
            mask_draw.text((current_x, current_y), char, font=large_font, fill=255)
            try:
                char_w = large_font.getlength(char)
            except:
                char_w = int(font_size)
            current_x += char_w
        current_y += line_height

    mask_arr = np.array(key_mask)

    # 可视化：将mask区域高亮（密文所在区域标绿）
    img_array = np.array(img_pil)
    highlight = img_array.copy()
    # mask外的区域变暗
    highlight[mask_arr < 128] = (highlight[mask_arr < 128] * 0.3).astype(np.uint8)
    # mask内区域加绿色边框
    edges = cv2.Canny(mask_arr, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    highlight[edges_dilated > 0] = [0, 255, 0]

    # 提取密文区域并裁剪
    # 找到mask的bounding box
    ys, xs = np.where(mask_arr > 128)
    if len(ys) > 0:
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        # 裁剪密文区域
        secret_region = img_array[y1:y2, x1:x2].copy()
        mask_region = mask_arr[y1:y2, x1:x2]
        # mask外设为白色，只保留密文
        secret_region[mask_region < 128] = 255
    else:
        secret_region = img_array.copy()

    # 统计密文区域信息
    fill_pixels = np.sum(mask_arr > 128)
    total_pixels = img_w * img_h
    fill_ratio = fill_pixels / total_pixels if total_pixels > 0 else 0

    # OCR检测
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

    return highlight, secret_region, report


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
    "独坐幽篁里弹琴复长啸深林人不知明月来相照"
    "葡萄美酒夜光杯欲饮琵琶马上催醉卧沙场君莫笑古来征战几人回"
    "秦时明月汉时关万里长征人未还但使龙城飞将在不教胡马度阴山"
    "朝辞白帝彩云间千里江陵一日还两岸猿声啼不住轻舟已过万重山"
    "李白乘舟将欲行忽闻岸上踏歌声桃花潭水深千尺不及汪伦送我情"
    "故人西辞黄鹤楼烟花三月下扬州孤帆远影碧空尽唯见长江天际流"
    "日照香炉生紫烟遥看瀑布挂前川飞流直下三千尺疑是银河落九天"
)

with gr.Blocks(title="嵌套字符生成与OCR检测演示") as demo:
    gr.Markdown("# 嵌套字符生成 · OCR检测 · 加密对抗")

    with gr.Tabs():
        # ==================== Tab 1: 生成与检测 ====================
        with gr.Tab("生成与检测"):
            gr.Markdown("输入文本生成嵌套字符图，然后用AI模型检测嵌套层数并展示OCR识别框。")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 生成参数")
                    layers_slider = gr.Slider(1, 5, value=2, step=1, label="嵌套层数")
                    layer_info = gr.Textbox(value=LAYER_CONFIG[2]["desc"], label="字数建议",
                                            interactive=False)
                    main_input = gr.Textbox(value="海内存知己天涯若比邻", label="主字文本",
                                            placeholder="输入要显示的大字...", lines=2)
                    fill_input = gr.Textbox(value=DEFAULT_FILL, label="填充文本",
                                            placeholder="用于填充笔画的小字...", lines=4)
                    wrap_slider = gr.Slider(0, 20, value=5, step=1,
                                            label="每行字数 (0=不换行)")
                    gen_btn = gr.Button("生成嵌套图", variant="primary", size="lg")
                    gen_warn = gr.Textbox(label="提示", interactive=False)

                with gr.Column(scale=2):
                    gr.Markdown("### 生成结果")
                    output_image = gr.Image(label="嵌套字符图", type="pil", height=400)
                    gr.Markdown("### OCR检测")
                    detect_btn = gr.Button("运行检测", variant="secondary", size="lg")
                    with gr.Row():
                        detect_image = gr.Image(label="蓝框=大字 绿框=小字 红框=梯度检测", height=400)
                        detect_report = gr.Textbox(label="检测报告", lines=14, interactive=False)

        # ==================== Tab 2: 加密对抗 ====================
        with gr.Tab("加密对抗"):
            gr.Markdown(
                "### 原理\n"
                "- **大字 = 密钥**：只有知道密钥的人才能定位密文区域\n"
                "- **小字 = 密文**：隐藏在大字笔画中的秘密信息\n"
                "- **传输**：发送嵌套图片即传输密文，旁观者只看到大字内容\n"
                "- **对抗**：AI检测器会尝试识别嵌套结构，评估隐蔽性\n"
            )

            with gr.Row():
                # ===== 加密侧 =====
                with gr.Column(scale=1):
                    gr.Markdown("### 加密（发送方）")
                    enc_key = gr.Textbox(value="天下大同", label="密钥（大字）",
                                         placeholder="输入密钥文字...", lines=1)
                    enc_secret = gr.Textbox(
                        value="明日午时三刻于城南老槐树下接头暗号风紧扯呼",
                        label="密文（小字）",
                        placeholder="输入要隐藏的秘密信息...", lines=3)
                    with gr.Row():
                        enc_fontsize = gr.Slider(100, 500, value=400, step=50,
                                                 label="大字号")
                        enc_smallsize = gr.Slider(4, 20, value=10, step=1,
                                                  label="小字号")
                    with gr.Row():
                        enc_step = gr.Slider(3, 20, value=11, step=1, label="填充步长")
                        enc_wrap = gr.Slider(0, 10, value=2, step=1, label="每行字数")
                    enc_btn = gr.Button("加密生成", variant="primary", size="lg")
                    enc_info = gr.Textbox(label="加密信息", interactive=False, lines=6)

                # ===== 加密结果 =====
                with gr.Column(scale=1):
                    gr.Markdown("### 加密图（传输载体）")
                    enc_image = gr.Image(label="加密后的嵌套字符图", type="pil", height=400)

            gr.Markdown("---")

            with gr.Row():
                # ===== 解密侧 =====
                with gr.Column(scale=1):
                    gr.Markdown("### 解密（接收方）")
                    dec_image = gr.Image(label="上传加密图（或从上方自动获取）",
                                         type="pil", height=300)
                    dec_key = gr.Textbox(value="天下大同", label="输入密钥",
                                         placeholder="输入大字密钥...", lines=1)
                    dec_fontsize = gr.Slider(100, 500, value=400, step=50,
                                             label="大字号（需与加密时一致）")
                    dec_wrap = gr.Slider(0, 10, value=2, step=1,
                                         label="每行字数（需与加密时一致）")
                    dec_btn = gr.Button("解密分析", variant="secondary", size="lg")

                with gr.Column(scale=2):
                    gr.Markdown("### 解密结果")
                    with gr.Row():
                        dec_highlight = gr.Image(label="密文区域高亮（绿框=密钥mask边界）",
                                                 height=300)
                        dec_extracted = gr.Image(label="提取的密文区域", height=300)
                    dec_report = gr.Textbox(label="解密分析报告", lines=12, interactive=False)

    # ===== 事件绑定 =====
    layers_slider.change(on_layer_change, inputs=[layers_slider], outputs=[layer_info])

    gen_btn.click(on_generate,
                  inputs=[main_input, fill_input, layers_slider, wrap_slider],
                  outputs=[output_image, gen_warn])

    detect_btn.click(on_detect,
                     inputs=[output_image],
                     outputs=[detect_image, detect_report])

    # 加密
    enc_btn.click(crypto_encode,
                  inputs=[enc_key, enc_secret, enc_fontsize, enc_smallsize,
                          enc_step, enc_wrap],
                  outputs=[enc_image, enc_info])

    # 加密完自动填入解密侧
    enc_image.change(lambda img: img, inputs=[enc_image], outputs=[dec_image])

    # 解密
    dec_btn.click(crypto_decode,
                  inputs=[dec_image, dec_key, dec_fontsize, dec_wrap],
                  outputs=[dec_highlight, dec_extracted, dec_report])


if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
