# -*- coding: utf-8 -*-
# coding:utf-8
"""
独立数据集生成脚本（不与训练脚本耦合）。
要求：
1) create_text_fill_art 逻辑按 NestCharacter.py 复制。
2) 不缩放图片。
3) 画布处理为“仅居中裁剪/居中贴入”，不做 resize。
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time
import argparse
import os
import random
import json
import tempfile
import contextlib
import io
import cv2
import NestCharacter as nest_original

_STEP_CACHE = {}


# ====== 以下函数按 NestCharacter.py 逻辑复制 ======
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
        text_color: str = "black",
        collect_boxes: bool = False,
):
    print("开始生成图像，参数如下:")
    print(f"  - 大字文本: {large_text[:30]}...")
    if wrap_after > 0:
        print(f"  - 换行设置: 每{wrap_after}字换行")
    else:
        print("  - 换行设置: 不换行")
    print(f"  - 小字文本: {small_text[:30]}...")
    print(f"  - 大字字号: {large_font_size}")
    print(f"  - 小字字号: {small_font_size}")

    final_step_x = step_x if step_x is not None else max(2, int(small_font_size * 0.85))
    final_step_y = step_y if step_y is not None else max(2, int(small_font_size * 0.85))
    print(f"  - 小字水平间距: {final_step_x}px")
    print(f"  - 小字垂直间距: {final_step_y}px")
    print("-" * 30)

    start_time = time.time()
    if wrap_after > 0:
        lines = [large_text[i:i + wrap_after] for i in range(0, len(large_text), wrap_after)]
    else:
        lines = [large_text]

    line_spacing_factor = 1.2
    line_height = int(large_font_size * line_spacing_factor)

    padding_y = int(large_font_size * 0.6)

    text_block_height = line_height * (len(lines) - 1) + large_font_size
    image_height = text_block_height + 2 * padding_y

    num_chars_for_width = wrap_after if wrap_after > 0 else len(large_text)
    image_width = int(num_chars_for_width * large_font_size * 1.05)

    try:
        large_font = ImageFont.truetype(font_path, large_font_size)
        small_font = ImageFont.truetype(font_path, small_font_size)
    except IOError:
        print(f"错误：无法在 '{font_path}' 找到字体文件。")
        return

    # 按“将要生成的实际字符”获取该字符在当前字号下的最大包围盒，不画想当然的大块区域
    char_box_cache = {}

    final_image = Image.new('RGB', (image_width, image_height), background_color)
    draw_final = ImageDraw.Draw(final_image)

    small_text_index = 0
    # 防止同一采样点被不同大字重复分配（同区域二次写入）
    point_used = np.zeros((image_height, image_width), dtype=np.uint8)
    # 区域占用（按每个实际字符包围盒）：某区域已占用则后续字符不再插入该区域
    region_used = np.zeros((image_height, image_width), dtype=np.uint8)
    # 调试计数：用于最终验证是否出现“同区域被写入多次”
    region_counter = np.zeros((image_height, image_width), dtype=np.uint16)
    # 大字模板区域占用：同一模板区域只分配一次，防止同一块区域被多个大字重复分配
    template_region_claimed = np.zeros((image_height, image_width), dtype=np.uint8)
    overlap_blocked_count = 0
    placed_count = 0
    placed_boxes = []
    large_boxes = []

    print("开始逐字生成模板并填充...")

    current_y = padding_y
    for line in lines:
        try:
            line_width = large_font.getlength(line)
        except AttributeError:
            line_width = draw_final.textlength(line, font=large_font)

        line_start_x = (image_width - line_width) / 2
        current_x = line_start_x

        for char in line:
            print(f"  - 正在处理大字: '{char}'")

            if collect_boxes:
                lb = large_font.getbbox(char)
                if lb is not None:
                    ll, lt, lr, lbm = lb
                    lx0 = int(current_x + ll)
                    ly0 = int(current_y + lt)
                    lx1 = int(current_x + lr)
                    ly1 = int(current_y + lbm)
                    clx0 = max(0, lx0)
                    cly0 = max(0, ly0)
                    clx1 = min(image_width, lx1)
                    cly1 = min(image_height, ly1)
                    if clx0 < clx1 and cly0 < cly1:
                        large_boxes.append((clx0, cly0, clx1, cly1))

            char_mask = Image.new('L', (image_width, image_height), 0)
            draw_char_mask = ImageDraw.Draw(char_mask)
            draw_char_mask.text((current_x, current_y), char, font=large_font, fill=255)
            char_mask_array = np.array(char_mask) > 128
            char_fill_mask = char_mask_array & (template_region_claimed == 0)

            for y in range(0, image_height, final_step_y):
                for x in range(0, image_width, final_step_x):
                    if char_fill_mask[y, x] and point_used[y, x] == 0:
                        char_to_draw = small_text[small_text_index % len(small_text)]

                        if char_to_draw not in char_box_cache:
                            bbox = small_font.getbbox(char_to_draw)
                            if bbox is None:
                                point_used[y, x] = 1
                                continue
                            char_box_cache[char_to_draw] = bbox

                        left, top, right, bottom = char_box_cache[char_to_draw]
                        x0 = x + left
                        y0 = y + top
                        x1 = x + right
                        y1 = y + bottom

                        rx0 = max(0, x0)
                        ry0 = max(0, y0)
                        rx1 = min(image_width, x1)
                        ry1 = min(image_height, y1)
                        if rx0 >= rx1 or ry0 >= ry1:
                            point_used[y, x] = 1
                            continue

                        if np.any(region_used[ry0:ry1, rx0:rx1] > 0):
                            point_used[y, x] = 1
                            overlap_blocked_count += 1
                            continue

                        draw_final.text((x, y), char_to_draw, font=small_font, fill=text_color)
                        region_used[ry0:ry1, rx0:rx1] = 1
                        region_counter[ry0:ry1, rx0:rx1] += 1
                        if collect_boxes:
                            placed_boxes.append((int(rx0), int(ry0), int(rx1), int(ry1)))
                        small_text_index += 1
                        placed_count += 1
                        point_used[y, x] = 1

            # 无论该区域是否成功插入小字，模板区域只允许第一次被分配
            template_region_claimed[char_mask_array] = 1

            try:
                char_width = large_font.getlength(char)
            except AttributeError:
                char_width = draw_final.textlength(char, font=large_font)
            current_x += char_width

        current_y += line_height

    print("填充完成，正在保存图像...")
    overlap_area = int(np.count_nonzero(region_counter > 1))
    print(f"  - 已放置小字数量: {placed_count}")
    print(f"  - 因区域占用被拦截次数: {overlap_blocked_count}")
    print(f"  - 区域重叠检测面积(应为0): {overlap_area}")
    if overlap_area > 0:
        raise RuntimeError(f"检测到区域重叠面积 {overlap_area}，已中止保存。")

    final_image.save(output_filename)
    end_time = time.time()
    print("-" * 30)
    print(f"图像已保存为: {output_filename}")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print("-" * 30)
    if collect_boxes:
        return {
            "region_used": region_used,
            "placed_boxes": placed_boxes,
            "large_region": template_region_claimed,
            "large_boxes": large_boxes,
        }
    return region_used


COMMON_CHARS = (
    "的一是在不了有人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说"
    "种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从"
    "业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又"
    "么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位"
    "入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放"
    "决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受"
    "联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离"
    "名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复"
    "容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支船史感劳便团往酸历市克何除消构府称太准精值"
    "号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满"
    "细引听该铁价严龙飞ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
)


def _read_image_no_scale(path: str) -> np.ndarray:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        return np.array(rgb, dtype=np.uint8)


def _safe_non_overlap_step(font_path: str, small_font_size: int) -> int:
    """
    计算“同层小字不重叠”所需的最小统一步长。
    用所有候选字符的 bbox 极值估计安全网格跨度。
    """
    key = (font_path, small_font_size)
    if key in _STEP_CACHE:
        return _STEP_CACHE[key]

    font = ImageFont.truetype(font_path, small_font_size)
    left_min = 0
    top_min = 0
    right_max = 0
    bottom_max = 0

    for ch in set(COMMON_CHARS):
        bbox = font.getbbox(ch)
        if bbox is None:
            continue
        left, top, right, bottom = bbox
        if left < left_min:
            left_min = left
        if top < top_min:
            top_min = top
        if right > right_max:
            right_max = right
        if bottom > bottom_max:
            bottom_max = bottom

    span_x = right_max - left_min
    span_y = bottom_max - top_min
    safe_step = max(2, span_x + 1, span_y + 1)
    _STEP_CACHE[key] = safe_step
    return safe_step


def _center_crop_or_pad_no_resize(img: np.ndarray, canvas_size: int) -> np.ndarray:
    """
    严格不缩放：
    - 原图比画布大：居中裁剪。
    - 原图比画布小：居中贴入白底。
    """
    h, w = img.shape[:2]
    canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)

    if w >= canvas_size:
        src_x = (w - canvas_size) // 2
        dst_x = 0
        copy_w = canvas_size
    else:
        src_x = 0
        dst_x = (canvas_size - w) // 2
        copy_w = w

    if h >= canvas_size:
        src_y = (h - canvas_size) // 2
        dst_y = 0
        copy_h = canvas_size
    else:
        src_y = 0
        dst_y = (canvas_size - h) // 2
        copy_h = h

    canvas[dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = img[src_y:src_y + copy_h, src_x:src_x + copy_w]
    return canvas


def _center_crop_or_pad_mask_no_resize(mask: np.ndarray, canvas_size: int) -> np.ndarray:
    h, w = mask.shape[:2]
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    if w >= canvas_size:
        src_x = (w - canvas_size) // 2
        dst_x = 0
        copy_w = canvas_size
    else:
        src_x = 0
        dst_x = (canvas_size - w) // 2
        copy_w = w

    if h >= canvas_size:
        src_y = (h - canvas_size) // 2
        dst_y = 0
        copy_h = canvas_size
    else:
        src_y = 0
        dst_y = (canvas_size - h) // 2
        copy_h = h

    canvas[dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = mask[src_y:src_y + copy_h, src_x:src_x + copy_w]
    return canvas


def _crop_pad_params(src_w: int, src_h: int, canvas_size: int):
    if src_w >= canvas_size:
        src_x = (src_w - canvas_size) // 2
        dst_x = 0
        copy_w = canvas_size
    else:
        src_x = 0
        dst_x = (canvas_size - src_w) // 2
        copy_w = src_w

    if src_h >= canvas_size:
        src_y = (src_h - canvas_size) // 2
        dst_y = 0
        copy_h = canvas_size
    else:
        src_y = 0
        dst_y = (canvas_size - src_h) // 2
        copy_h = src_h

    return src_x, src_y, dst_x, dst_y, copy_w, copy_h


def _map_boxes_to_canvas(boxes, src_w: int, src_h: int, canvas_size: int):
    src_x, src_y, dst_x, dst_y, copy_w, copy_h = _crop_pad_params(src_w, src_h, canvas_size)
    sx1 = src_x + copy_w
    sy1 = src_y + copy_h
    mapped = []
    for x0, y0, x1, y1 in boxes:
        ix0 = max(x0, src_x)
        iy0 = max(y0, src_y)
        ix1 = min(x1, sx1)
        iy1 = min(y1, sy1)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        mx0 = dst_x + (ix0 - src_x)
        my0 = dst_y + (iy0 - src_y)
        mx1 = dst_x + (ix1 - src_x)
        my1 = dst_y + (iy1 - src_y)
        mapped.append((int(mx0), int(my0), int(mx1), int(my1)))
    return mapped


def _bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    return x0, y0, x1, y1


def _crop_plan_by_mask(img: np.ndarray, region: np.ndarray, max_region: np.ndarray, small_boxes):
    bbox = _bbox_from_mask(region)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    img_c = img[y0:y1, x0:x1].copy()
    region_c = region[y0:y1, x0:x1].copy()
    max_region_c = max_region[y0:y1, x0:x1].copy()
    boxes_c = []
    for bx0, by0, bx1, by1 in small_boxes:
        ix0 = max(bx0, x0)
        iy0 = max(by0, y0)
        ix1 = min(bx1, x1)
        iy1 = min(by1, y1)
        if ix0 < ix1 and iy0 < iy1:
            boxes_c.append((int(ix0 - x0), int(iy0 - y0), int(ix1 - x0), int(iy1 - y0)))
    return {
        "img": img_c,
        "region": region_c,
        "max_region": max_region_c,
        "boxes": boxes_c,
        "w": int(x1 - x0),
        "h": int(y1 - y0),
    }


def _boxes_to_slot(boxes, slot_x: int, slot_y: int):
    out = []
    for x0, y0, x1, y1 in boxes:
        out.append((int(slot_x + x0), int(slot_y + y0), int(slot_x + x1), int(slot_y + y1)))
    return out


def _component_boxes(mask: np.ndarray, min_area: int = 1):
    m = (mask.astype(np.uint8) * 255)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    boxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return boxes


def _draw_region_boxes(
    base_rgb: np.ndarray,
    small_boxes=None,
    large_boxes_a=None,
    large_boxes_b=None,
    large_mask_a=None,
    large_mask_b=None,
    step: int = 1,
    width: int = 1,
    large_width: int = 2,
) -> np.ndarray:
    out = base_rgb.copy()
    pil_img = Image.fromarray(out, mode="RGB")
    draw = ImageDraw.Draw(pil_img)
    safe_step = max(1, int(step))
    if small_boxes is None:
        small_boxes = []
    if large_boxes_a is None:
        large_boxes_a = []
    if large_boxes_b is None:
        large_boxes_b = []
    for i, (x0, y0, x1, y1) in enumerate(small_boxes):
        if i % safe_step == 0:
            draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(255, 0, 0), width=width)
    for i, (x0, y0, x1, y1) in enumerate(large_boxes_a):
        if i % safe_step == 0:
            draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(0, 128, 255), width=large_width)
    for i, (x0, y0, x1, y1) in enumerate(large_boxes_b):
        if i % safe_step == 0:
            draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(0, 200, 0), width=large_width)

    out = np.array(pil_img, dtype=np.uint8)

    # 用真实区域轮廓画大字边界，避免包围框视觉上“看起来重叠”
    if large_mask_a is not None:
        ma = (large_mask_a.astype(np.uint8) * 255)
        contours_a, _ = cv2.findContours(ma, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours_a, -1, (255, 128, 0), thickness=max(1, large_width))
    if large_mask_b is not None:
        mb = (large_mask_b.astype(np.uint8) * 255)
        contours_b, _ = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours_b, -1, (0, 220, 0), thickness=max(1, large_width))

    return out


def _sanitize_debug_report(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str) and k.startswith("_"):
                continue
            out[k] = _sanitize_debug_report(v)
        return out
    if isinstance(obj, list):
        return [_sanitize_debug_report(x) for x in obj]
    if isinstance(obj, tuple):
        return [_sanitize_debug_report(x) for x in obj]
    return obj


def _ink_mask(img: np.ndarray, threshold: int = 255) -> np.ndarray:
    return np.any(img < threshold, axis=2)


def _composite_without_overlap(dst: np.ndarray, src: np.ndarray, occupied: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_ink = _ink_mask(src)
    place_mask = src_ink & (~occupied)
    dst[place_mask] = src[place_mask]
    occupied |= place_mask
    return dst, occupied


def _generate_one_layer_image(
    font_path: str,
    canvas_size: int,
    wrap_after: int,
    large_font_size: int,
    small_font_size: int,
    step: int,
    large_text: str,
    small_text: str,
    silent: bool = True,
    collect_boxes: bool = False,
) -> tuple[np.ndarray, np.ndarray, list, np.ndarray, list]:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        region_local = None
        layer_boxes_local = []
        large_region_local = None
        large_boxes_local = []
        if silent:
            with contextlib.redirect_stdout(io.StringIO()):
                render_ret = create_text_fill_art(
                    large_text=large_text,
                    small_text=small_text,
                    large_font_size=large_font_size,
                    small_font_size=small_font_size,
                    font_path=font_path,
                    output_filename=tmp_path,
                    step_x=step,
                    step_y=step,
                    wrap_after=wrap_after,
                    collect_boxes=collect_boxes,
                )
        else:
            render_ret = create_text_fill_art(
                large_text=large_text,
                small_text=small_text,
                large_font_size=large_font_size,
                small_font_size=small_font_size,
                font_path=font_path,
                output_filename=tmp_path,
                step_x=step,
                step_y=step,
                wrap_after=wrap_after,
                collect_boxes=collect_boxes,
            )

        if isinstance(render_ret, dict):
            region_local = render_ret.get("region_used")
            layer_boxes_local = render_ret.get("placed_boxes", [])
            large_region_local = render_ret.get("large_region")
            large_boxes_local = render_ret.get("large_boxes", [])
        else:
            region_local = render_ret

        img = _read_image_no_scale(tmp_path)
        img_canvas = _center_crop_or_pad_no_resize(img, canvas_size)
        if region_local is None:
            region_local = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        if large_region_local is None:
            large_region_local = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        region_canvas = _center_crop_or_pad_mask_no_resize(region_local.astype(np.uint8), canvas_size) > 0
        large_region_canvas = _center_crop_or_pad_mask_no_resize(large_region_local.astype(np.uint8), canvas_size) > 0
        layer_boxes_canvas = _map_boxes_to_canvas(layer_boxes_local, img.shape[1], img.shape[0], canvas_size)
        large_boxes_canvas = _map_boxes_to_canvas(large_boxes_local, img.shape[1], img.shape[0], canvas_size)
        return img_canvas, region_canvas, layer_boxes_canvas, large_region_canvas, large_boxes_canvas
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def generate_nested_image(
    layers: int,
    font_path: str,
    canvas_size: int,
    wrap_after: int,
    base_large_font_size: int,
    depth_scale: float,
    small_ratio_min: float,
    small_ratio_max: float,
    silent: bool = True,
    debug_report: dict | None = None,
    collect_boxes: bool = False,
    separate_3_2: bool = True,
) -> np.ndarray:
    def _char_count_for_layout() -> int:
        line_count = random.choice([2, 3])
        if wrap_after > 0:
            return wrap_after * line_count
        return max(20, line_count * 10)

    def _render_depth_plan(
        depth_indices, base_scale: float = 1.0, plan_name: str = "plan"
    ) -> tuple[np.ndarray, np.ndarray, list, np.ndarray, list]:
        group_canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
        group_region_used = np.zeros((canvas_size, canvas_size), dtype=bool)
        group_region_counter = np.zeros((canvas_size, canvas_size), dtype=np.uint16)
        char_count = _char_count_for_layout()
        plan_report = []
        group_small_boxes = []
        plan_max_large_region = np.zeros((canvas_size, canvas_size), dtype=bool)
        plan_max_large_boxes = []

        for depth in depth_indices:
            large_font_size = max(12, int(base_large_font_size * base_scale * (depth_scale ** depth)))
            small_font_size = max(2, int(large_font_size * random.uniform(small_ratio_min, small_ratio_max)))

            # 原始下限 + 无重叠安全下限（保证同层小字不叠）
            raw_step = max(2, int(small_font_size * 0.85))
            safe_step = _safe_non_overlap_step(font_path, small_font_size)
            step = max(raw_step, safe_step)

            large_text = "".join(random.choices(COMMON_CHARS, k=char_count))
            small_text = "".join(random.choices(COMMON_CHARS, k=2200))

            layer_img, layer_region, layer_boxes, layer_large_region, layer_large_boxes = _generate_one_layer_image(
                font_path=font_path,
                canvas_size=canvas_size,
                wrap_after=wrap_after,
                large_font_size=large_font_size,
                small_font_size=small_font_size,
                step=step,
                large_text=large_text,
                small_text=small_text,
                silent=silent,
                collect_boxes=collect_boxes,
            )
            if depth == 0:
                plan_max_large_region = layer_large_region.copy()
                plan_max_large_boxes = [(int(a), int(b), int(c), int(d)) for a, b, c, d in layer_large_boxes]
            overlap_before_place = int(np.count_nonzero(layer_region & group_region_used))
            place_region = layer_region & (~group_region_used)
            group_canvas[place_region] = layer_img[place_region]
            group_region_used |= place_region
            group_region_counter[place_region] += 1
            if collect_boxes:
                for bx0, by0, bx1, by1 in layer_boxes:
                    if np.any(place_region[by0:by1, bx0:bx1]):
                        group_small_boxes.append((int(bx0), int(by0), int(bx1), int(by1)))
            plan_report.append({
                "depth": int(depth),
                "large_font_size": int(large_font_size),
                "small_font_size": int(small_font_size),
                "step": int(step),
                "layer_region_area": int(np.count_nonzero(layer_region)),
                "overlap_with_existing_before_place": overlap_before_place,
                "placed_area": int(np.count_nonzero(place_region)),
            })

        if np.any(group_region_counter > 1):
            raise RuntimeError("检测到 depth plan 内区域重复分配。")

        if debug_report is not None:
            debug_report.setdefault("plans", []).append({
                "plan_name": plan_name,
                "depths": [int(d) for d in depth_indices],
                "plan_region_area": int(np.count_nonzero(group_region_used)),
                "details": plan_report,
            })

        return group_canvas, group_region_used, group_small_boxes, plan_max_large_region, plan_max_large_boxes

    # 按你指定的定义生成：
    # 3层=2层基础上再加1层；4层=2+2；5层=2+3
    if layers == 2:
        img, _, small_boxes, max_large_region, max_large_boxes = _render_depth_plan([0, 1], base_scale=1.0, plan_name="L2")
        if collect_boxes and debug_report is not None:
            debug_report["final_small_boxes"] = small_boxes
            debug_report["final_large_boxes_a"] = _component_boxes(max_large_region, min_area=1)
            debug_report["final_large_boxes_b"] = []
            debug_report["max_region_area"] = int(np.count_nonzero(max_large_region))
            debug_report["_final_large_mask_a"] = max_large_region
            debug_report["_final_large_mask_b"] = np.zeros_like(max_large_region, dtype=bool)
        return img
    if layers == 3:
        img, _, small_boxes, max_large_region, max_large_boxes = _render_depth_plan([0, 1, 2], base_scale=1.0, plan_name="L3")
        if collect_boxes and debug_report is not None:
            debug_report["final_small_boxes"] = small_boxes
            debug_report["final_large_boxes_a"] = _component_boxes(max_large_region, min_area=1)
            debug_report["final_large_boxes_b"] = []
            debug_report["max_region_area"] = int(np.count_nonzero(max_large_region))
            debug_report["_final_large_mask_a"] = max_large_region
            debug_report["_final_large_mask_b"] = np.zeros_like(max_large_region, dtype=bool)
        return img
    if layers == 4:
        img_2a, reg_2a, boxes_2a, max_2a, max_boxes_2a = _render_depth_plan([0, 1], base_scale=1.0, plan_name="L2A")
        img_2b, reg_2b, boxes_2b, max_2b, max_boxes_2b = _render_depth_plan([0, 1], base_scale=random.uniform(0.80, 0.92), plan_name="L2B")
        cross_overlap = int(np.count_nonzero(reg_2a & reg_2b))
        cross_max_overlap_before = int(np.count_nonzero(max_2a & max_2b))
        out = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
        occ = np.zeros((canvas_size, canvas_size), dtype=bool)
        occ_counter = np.zeros((canvas_size, canvas_size), dtype=np.uint16)
        out_small_boxes = []
        out_large_boxes_a = _component_boxes(max_2a, min_area=1)
        place_2a = reg_2a & (~occ)
        out[place_2a] = img_2a[place_2a]
        occ |= place_2a
        occ_counter[place_2a] += 1
        if collect_boxes:
            for bx0, by0, bx1, by1 in boxes_2a:
                if np.any(place_2a[by0:by1, bx0:bx1]):
                    out_small_boxes.append((int(bx0), int(by0), int(bx1), int(by1)))
        # 关键约束：第二个 plan 的“最大大字区域”不能和第一个 plan 最大区域重叠
        place_2b = reg_2b & (~occ) & (~max_2a)
        out[place_2b] = img_2b[place_2b]
        occ |= place_2b
        occ_counter[place_2b] += 1
        max_2b_filtered = max_2b & (~max_2a)
        cross_max_overlap_after = int(np.count_nonzero(max_2a & max_2b_filtered))
        out_large_boxes_b = _component_boxes(max_2b_filtered, min_area=1)
        if collect_boxes:
            for bx0, by0, bx1, by1 in boxes_2b:
                if np.any(place_2b[by0:by1, bx0:bx1]):
                    out_small_boxes.append((int(bx0), int(by0), int(bx1), int(by1)))
        if np.any(occ_counter > 1):
            raise RuntimeError("检测到 4层(2+2) 组合区域重复分配。")
        if cross_max_overlap_after > 0:
            raise RuntimeError(f"检测到 4层(2+2) 最大大字区域重叠: {cross_max_overlap_after}")
        if debug_report is not None:
            debug_report["cross_plan_overlap_area_before_filter"] = cross_overlap
            debug_report["cross_max_overlap_area_before_filter"] = cross_max_overlap_before
            debug_report["cross_max_overlap_area_after_filter"] = cross_max_overlap_after
            if collect_boxes:
                debug_report["final_small_boxes"] = out_small_boxes
                debug_report["final_large_boxes_a"] = out_large_boxes_a
                debug_report["final_large_boxes_b"] = out_large_boxes_b
                debug_report["_final_large_mask_a"] = max_2a
                debug_report["_final_large_mask_b"] = max_2b_filtered
        return out
    if layers == 5:
        # 按用户口径：5层按 3+2 组合
        img_3, reg_3, boxes_3, max_3, max_boxes_3 = _render_depth_plan([0, 1, 2], base_scale=1.0, plan_name="L3")
        img_2, reg_2, boxes_2, max_2, max_boxes_2 = _render_depth_plan([0, 1], base_scale=random.uniform(0.75, 0.90), plan_name="L2")
        cross_overlap = int(np.count_nonzero(reg_2 & reg_3))
        cross_max_overlap_before = int(np.count_nonzero(max_3 & max_2))
        out = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
        out_small_boxes = []
        out_large_boxes_a = []
        out_large_boxes_b = []

        if separate_3_2:
            crop_3 = _crop_plan_by_mask(img_3, reg_3, max_3, boxes_3)
            crop_2 = _crop_plan_by_mask(img_2, reg_2, max_2, boxes_2)
            if crop_3 is None or crop_2 is None:
                raise RuntimeError("5层(3+2) 计划中有空区域，无法分离放置。")

            margin = max(100, canvas_size // 50)
            gap = max(140, canvas_size // 30)
            half_w = (canvas_size - 2 * margin - gap) // 2
            avail_h = canvas_size - 2 * margin
            if crop_3["w"] > half_w or crop_2["w"] > half_w or crop_3["h"] > avail_h or crop_2["h"] > avail_h:
                raise RuntimeError(
                    f"当前画布不足以分离3+2（canvas={canvas_size}, L3={crop_3['w']}x{crop_3['h']}, "
                    f"L2={crop_2['w']}x{crop_2['h']}），请增大 --canvas-size。"
                )

            slot3_x = margin + (half_w - crop_3["w"]) // 2
            slot2_x = margin + half_w + gap + (half_w - crop_2["w"]) // 2
            slot3_y = margin + (avail_h - crop_3["h"]) // 2
            slot2_y = margin + (avail_h - crop_2["h"]) // 2

            r3 = crop_3["region"]
            r2 = crop_2["region"]
            m3 = crop_3["max_region"]
            m2 = crop_2["max_region"]

            roi3 = out[slot3_y:slot3_y + crop_3["h"], slot3_x:slot3_x + crop_3["w"]]
            roi3[r3] = crop_3["img"][r3]
            out[slot3_y:slot3_y + crop_3["h"], slot3_x:slot3_x + crop_3["w"]] = roi3

            roi2 = out[slot2_y:slot2_y + crop_2["h"], slot2_x:slot2_x + crop_2["w"]]
            roi2[r2] = crop_2["img"][r2]
            out[slot2_y:slot2_y + crop_2["h"], slot2_x:slot2_x + crop_2["w"]] = roi2

            final_max_3 = np.zeros((canvas_size, canvas_size), dtype=bool)
            final_max_2 = np.zeros((canvas_size, canvas_size), dtype=bool)
            final_max_3[slot3_y:slot3_y + crop_3["h"], slot3_x:slot3_x + crop_3["w"]] = m3
            final_max_2[slot2_y:slot2_y + crop_2["h"], slot2_x:slot2_x + crop_2["w"]] = m2
            cross_max_overlap_after = int(np.count_nonzero(final_max_3 & final_max_2))
            if cross_max_overlap_after > 0:
                raise RuntimeError(f"检测到 5层(3+2) 最大大字区域重叠: {cross_max_overlap_after}")

            if collect_boxes:
                out_small_boxes.extend(_boxes_to_slot(crop_3["boxes"], slot3_x, slot3_y))
                out_small_boxes.extend(_boxes_to_slot(crop_2["boxes"], slot2_x, slot2_y))
                out_large_boxes_a = _component_boxes(final_max_3, min_area=1)
                out_large_boxes_b = _component_boxes(final_max_2, min_area=1)
        else:
            occ = np.zeros((canvas_size, canvas_size), dtype=bool)
            occ_counter = np.zeros((canvas_size, canvas_size), dtype=np.uint16)
            out_large_boxes_a = _component_boxes(max_3, min_area=1)
            place_3 = reg_3 & (~occ)
            out[place_3] = img_3[place_3]
            occ |= place_3
            occ_counter[place_3] += 1
            if collect_boxes:
                for bx0, by0, bx1, by1 in boxes_3:
                    if np.any(place_3[by0:by1, bx0:bx1]):
                        out_small_boxes.append((int(bx0), int(by0), int(bx1), int(by1)))

            place_2 = reg_2 & (~occ) & (~max_3)
            out[place_2] = img_2[place_2]
            occ |= place_2
            occ_counter[place_2] += 1
            max_2_filtered = max_2 & (~max_3)
            cross_max_overlap_after = int(np.count_nonzero(max_3 & max_2_filtered))
            out_large_boxes_b = _component_boxes(max_2_filtered, min_area=1)
            if collect_boxes:
                for bx0, by0, bx1, by1 in boxes_2:
                    if np.any(place_2[by0:by1, bx0:bx1]):
                        out_small_boxes.append((int(bx0), int(by0), int(bx1), int(by1)))
            if np.any(occ_counter > 1):
                raise RuntimeError("检测到 5层(3+2) 组合区域重复分配。")
            if cross_max_overlap_after > 0:
                raise RuntimeError(f"检测到 5层(3+2) 最大大字区域重叠: {cross_max_overlap_after}")
            final_max_3 = max_3
            final_max_2 = max_2_filtered

        if debug_report is not None:
            debug_report["cross_plan_overlap_area_before_filter"] = cross_overlap
            debug_report["cross_max_overlap_area_before_filter"] = cross_max_overlap_before
            debug_report["cross_max_overlap_area_after_filter"] = cross_max_overlap_after
            if collect_boxes:
                debug_report["final_small_boxes"] = out_small_boxes
                debug_report["final_large_boxes_a"] = out_large_boxes_a
                debug_report["final_large_boxes_b"] = out_large_boxes_b
                debug_report["_final_large_mask_a"] = final_max_3
                debug_report["_final_large_mask_b"] = final_max_2
        return out

    # 兜底：非2-5层时，按连续层数生成
    img, _, boxes, max_large_region, max_large_boxes = _render_depth_plan(list(range(layers)), base_scale=1.0, plan_name=f"L{layers}")
    if collect_boxes and debug_report is not None:
        debug_report["final_small_boxes"] = boxes
        debug_report["final_large_boxes_a"] = _component_boxes(max_large_region, min_area=1)
        debug_report["final_large_boxes_b"] = []
        debug_report["max_region_area"] = int(np.count_nonzero(max_large_region))
        debug_report["_final_large_mask_a"] = max_large_region
        debug_report["_final_large_mask_b"] = np.zeros_like(max_large_region, dtype=bool)
    return img


def ensure_dirs(base_dir: str):
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(base_dir, split), exist_ok=True)


def clear_existing(base_dir: str, layer_min: int, layer_max: int):
    for split in ("train", "val", "test"):
        d = os.path.join(base_dir, split)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith(".png"):
                continue
            for layer in range(layer_min, layer_max + 1):
                if fn.startswith(f"layer{layer}_{split}_"):
                    os.remove(os.path.join(d, fn))
                    break


def _save_png(path: str, rgb_array: np.ndarray):
    Image.fromarray(rgb_array, mode="RGB").save(path)


def _generate_layer2_by_original_file(
    out_path: str,
    font_path: str,
    silent: bool = True,
) -> None:
    """
    二层样本严格走 NestCharacter.py 原始函数，不经过二次图像处理。
    仅随机替换文本内容，其他关键参数保持原文件默认值。
    """
    wrap_after = 10
    large_font_size = 400
    small_font_size = 12
    step_x = 13
    step_y = 13

    line_count = random.choice([2, 3])
    large_text = "".join(random.choices(COMMON_CHARS, k=wrap_after * line_count))
    small_text = "".join(random.choices(COMMON_CHARS, k=2200))

    if silent:
        with contextlib.redirect_stdout(io.StringIO()):
            nest_original.create_text_fill_art(
                large_text=large_text,
                small_text=small_text,
                large_font_size=large_font_size,
                small_font_size=small_font_size,
                font_path=font_path,
                output_filename=out_path,
                step_x=step_x,
                step_y=step_y,
                wrap_after=wrap_after,
            )
    else:
        nest_original.create_text_fill_art(
            large_text=large_text,
            small_text=small_text,
            large_font_size=large_font_size,
            small_font_size=small_font_size,
            font_path=font_path,
            output_filename=out_path,
            step_x=step_x,
            step_y=step_y,
            wrap_after=wrap_after,
        )

    if not os.path.exists(out_path):
        raise RuntimeError(f"NestCharacter.py 未生成输出文件: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="独立嵌套文字训练数据生成（复制版）")
    parser.add_argument("--font-path", type=str, default="C:/Windows/Fonts/msyh.ttc")
    parser.add_argument("--out-dir", type=str, default="training_data")
    parser.add_argument("--layer-min", type=int, default=3)
    parser.add_argument("--layer-max", type=int, default=5)
    parser.add_argument("--samples-per-class", type=int, default=500)

    # 画布默认自动计算，不再固定 1536（避免三层大字被强裁剪）
    parser.add_argument("--canvas-size", type=int, default=0)
    parser.add_argument("--wrap-after", type=int, default=10)

    # 修复：删除 base_large 动态限制，改为显式基础大字字号
    parser.add_argument("--base-large-font-size", type=int, default=400)
    parser.add_argument("--depth-scale", type=float, default=0.70)
    parser.add_argument("--small-ratio-min", type=float, default=0.030)
    parser.add_argument("--small-ratio-max", type=float, default=0.045)

    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--export-debug-json", action="store_true", help="导出每张图的区域分配对照json")
    parser.add_argument("--debug-json-name", type=str, default="overlap_debug_report.json")
    parser.add_argument("--export-box-overlay", action="store_true", help="导出带区域框的可视化图")
    parser.add_argument("--box-overlay-dir", type=str, default="box_overlay")
    parser.add_argument("--box-draw-step", type=int, default=1, help="每隔N个框画一个，减小视觉拥挤")
    parser.add_argument("--box-outline-width", type=int, default=1)
    parser.add_argument("--no-separate-3-2", action="store_true", help="关闭5层时3+2物理分离放置")
    args = parser.parse_args()

    if args.seed is None:
        runtime_seed = int.from_bytes(os.urandom(8), "big") & 0x7FFFFFFF
    else:
        runtime_seed = int(args.seed)

    random.seed(runtime_seed)
    np.random.seed(runtime_seed)

    if args.canvas_size and args.canvas_size > 0:
        runtime_canvas_size = int(args.canvas_size)
    else:
        # 依据 NestCharacter 原始尺寸公式估算：取 3 行时的最大边，避免裁剪
        if args.wrap_after > 0:
            est_width = int(args.wrap_after * args.base_large_font_size * 1.05)
        else:
            est_width = int(20 * args.base_large_font_size * 1.05)
        est_height = int(args.base_large_font_size * (1.2 * (3 - 1) + 2.2))
        base_size = int(max(est_width, est_height))
        separate_3_2 = (not args.no_separate_3_2)
        if separate_3_2 and (args.layer_min <= 5 <= args.layer_max):
            runtime_canvas_size = int(base_size * 2 + 400)
            print(f"自动画布尺寸: {runtime_canvas_size} (5层3+2分离模式，自动加大画布)")
        else:
            runtime_canvas_size = base_size
            print(f"自动画布尺寸: {runtime_canvas_size} (由大字字号/换行估算，避免1536裁剪)")
    separate_3_2 = (not args.no_separate_3_2)

    ensure_dirs(args.out_dir)
    if not args.keep_existing:
        clear_existing(args.out_dir, args.layer_min, args.layer_max)
    if args.export_box_overlay:
        for split in ("train", "val", "test"):
            os.makedirs(os.path.join(args.out_dir, args.box_overlay_dir, split), exist_ok=True)

    train_n = int(args.samples_per_class * args.train_ratio)
    val_n = int(args.samples_per_class * args.val_ratio)

    metadata = {
        "generator": "generate_dataset_copypaste.py",
        "source_reference": "NestCharacter.py",
        "layer2_source": "direct_NestCharacter.py_no_postprocess",
        "layer_min": args.layer_min,
        "layer_max": args.layer_max,
        "samples_per_class": args.samples_per_class,
        "canvas_size": runtime_canvas_size,
        "wrap_after": args.wrap_after,
        "base_large_font_size": args.base_large_font_size,
        "depth_scale": args.depth_scale,
        "small_ratio_min": args.small_ratio_min,
        "small_ratio_max": args.small_ratio_max,
        "runtime_seed": runtime_seed,
        "separate_3_2": separate_3_2,
        "classes": {},
    }
    debug_records = []

    for layer in range(args.layer_min, args.layer_max + 1):
        layer_meta = {"train_samples": 0, "val_samples": 0, "test_samples": 0}
        print(f"\n生成 layer={layer} ...")

        for i in range(args.samples_per_class):
            if i < train_n:
                split = "train"
                idx = i
                layer_meta["train_samples"] += 1
            elif i < train_n + val_n:
                split = "val"
                idx = i - train_n
                layer_meta["val_samples"] += 1
            else:
                split = "test"
                idx = i - train_n - val_n
                layer_meta["test_samples"] += 1

            out_name = f"layer{layer}_{split}_{idx:04d}.png"
            out_path = os.path.join(args.out_dir, split, out_name)

            # 用户要求：二层样本由原文件直接生成，且除文本外不做任何改动
            if layer == 2:
                _generate_layer2_by_original_file(
                    out_path=out_path,
                    font_path=args.font_path,
                    silent=(not args.verbose),
                )
                if args.export_debug_json:
                    debug_records.append({
                        "layer": int(layer),
                        "split": split,
                        "index": int(idx),
                        "filename": out_name,
                        "overlay_filename": None,
                        "debug": {
                            "mode": "layer2_direct_from_NestCharacter.py",
                            "postprocess": "none",
                            "note": "only_text_randomized",
                        },
                    })
                continue

            sample_debug = {} if (args.export_debug_json or args.export_box_overlay) else None
            img = generate_nested_image(
                layers=layer,
                font_path=args.font_path,
                canvas_size=runtime_canvas_size,
                wrap_after=args.wrap_after,
                base_large_font_size=args.base_large_font_size,
                depth_scale=args.depth_scale,
                small_ratio_min=args.small_ratio_min,
                small_ratio_max=args.small_ratio_max,
                silent=(not args.verbose),
                debug_report=sample_debug,
                collect_boxes=args.export_box_overlay,
                separate_3_2=separate_3_2,
            )
            _save_png(out_path, img)
            if args.export_box_overlay and sample_debug is not None:
                small_boxes = sample_debug.get("final_small_boxes", [])
                large_boxes_a = sample_debug.get("final_large_boxes_a", [])
                large_boxes_b = sample_debug.get("final_large_boxes_b", [])
                large_mask_a = sample_debug.get("_final_large_mask_a", None)
                large_mask_b = sample_debug.get("_final_large_mask_b", None)
                if args.verbose:
                    print(f"  - overlay框数量: small={len(small_boxes)} blue={len(large_boxes_a)} green={len(large_boxes_b)}")
                overlay_img = _draw_region_boxes(
                    img.copy(),
                    small_boxes=small_boxes,
                    large_boxes_a=large_boxes_a,
                    large_boxes_b=large_boxes_b,
                    large_mask_a=large_mask_a,
                    large_mask_b=large_mask_b,
                    step=args.box_draw_step,
                    width=args.box_outline_width,
                    large_width=max(2, args.box_outline_width),
                )
                overlay_name = out_name.replace(".png", "_boxes.png")
                overlay_path = os.path.join(args.out_dir, args.box_overlay_dir, split, overlay_name)
                _save_png(overlay_path, overlay_img)
            if args.export_debug_json:
                overlay_name = out_name.replace(".png", "_boxes.png") if args.export_box_overlay else None
                debug_records.append({
                    "layer": int(layer),
                    "split": split,
                    "index": int(idx),
                    "filename": out_name,
                    "overlay_filename": overlay_name,
                    "debug": _sanitize_debug_report(sample_debug),
                })

        metadata["classes"][f"layer_{layer}"] = layer_meta
        print(f"完成 layer={layer}: {layer_meta}")

    meta_path = os.path.join(args.out_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    if args.export_debug_json:
        debug_payload = {
            "source_reference": "NestCharacter.py",
            "runtime_seed": runtime_seed,
            "canvas_size": runtime_canvas_size,
            "wrap_after": args.wrap_after,
            "base_large_font_size": args.base_large_font_size,
            "depth_scale": args.depth_scale,
            "small_ratio_min": args.small_ratio_min,
            "small_ratio_max": args.small_ratio_max,
            "records": debug_records,
        }
        debug_path = os.path.join(args.out_dir, args.debug_json_name)
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(debug_payload, f, ensure_ascii=False, indent=2)
        print(f"对照json已导出: {debug_path}")

    print(f"\n数据集生成完成，metadata: {meta_path}")


if __name__ == "__main__":
    main()
