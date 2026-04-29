# -*- coding: utf-8 -*-
"""
自动化夜间训练脚本：
1) 先生成训练样例预览图，人工确认风格是否合理
2) 可选一键执行 2-5 层 + OCR + 复合微调 全流程训练

默认只生成预览，不会自动开训。
"""

import argparse
import contextlib
import gc
import io
import json
import os
import random
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from training_data_generator import TrainingDataGenerator
from nested_char_detector import (
    NestedCharCNN,
    DualHeadNestedCharCNN,
    load_nesting_weights_from_legacy,
    freeze_backbone,
    unfreeze_backbone,
)
from NestCharacter_copypaste import create_text_fill_art


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimpleLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def info(self, msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


class NestedLayerDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], image_size: int, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.image_size = image_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取图片: {self.image_paths[idx]}")
        # Resize handles any source size -> target image_size
        pil = Image.fromarray(img)
        if self.transform is not None:
            pil = self.transform(pil)
        return pil, self.labels[idx]


class OCRPatchDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        for p in sorted(self.data_dir.glob("char_*.png")):
            parts = p.stem.split("_")
            if len(parts) >= 3:
                label = int(parts[1])
                self.samples.append((p, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取OCR图片: {path}")
        pil = Image.fromarray(img)
        if self.transform is not None:
            pil = self.transform(pil)
        return pil, label


def save_image_grid(images: List[np.ndarray], out_path: Path, cols: int = 4, padding: int = 8) -> None:
    if not images:
        return
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    h = max(heights)
    w = max(widths)

    rows = int(np.ceil(len(images) / cols))
    canvas_h = rows * h + (rows + 1) * padding
    canvas_w = cols * w + (cols + 1) * padding
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        y = padding + r * (h + padding)
        x = padding + c * (w + padding)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        hh, ww = img.shape[:2]
        canvas[y:y + hh, x:x + ww] = img

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


class NestCharacterStrictGenerator:
    """严格参考 NestCharacter.py 的笔画填充逻辑，用于训练集生成。"""

    def __init__(
        self,
        font_path: str,
        char_pool: List[str],
        canvas_size: int = 512,
        large_font_ratio_min: float = 0.68,
        large_font_ratio_max: float = 0.92,
        depth_scale: float = 0.67,
        small_font_ratio_min: float = 0.028,
        small_font_ratio_max: float = 0.045,
        step_ratio_min: float = 1.02,
        step_ratio_max: float = 1.28,
        no_overlap_same_layer: bool = True,
        large_char_spacing_scale: float = 1.12,
        layer_char_min: int = 0,
        layer_char_max: int = 0,
        wrap_after: int = 12,
        use_original_logic_for_layer2: bool = True,
    ):
        self.font_path = font_path
        self.char_pool = char_pool
        self.canvas_size = canvas_size
        self.large_font_ratio_min = large_font_ratio_min
        self.large_font_ratio_max = large_font_ratio_max
        self.depth_scale = depth_scale
        self.small_font_ratio_min = small_font_ratio_min
        self.small_font_ratio_max = small_font_ratio_max
        self.step_ratio_min = step_ratio_min
        self.step_ratio_max = step_ratio_max
        self.no_overlap_same_layer = no_overlap_same_layer
        self.large_char_spacing_scale = large_char_spacing_scale
        self.layer_char_min = layer_char_min
        self.layer_char_max = layer_char_max
        self.wrap_after = wrap_after
        self.last_layout_overlap_pixels = 0
        self.use_original_logic_for_layer2 = use_original_logic_for_layer2

    def _load_font(self, font_size: int):
        try:
            return ImageFont.truetype(self.font_path, font_size)
        except Exception:
            return ImageFont.load_default()

    def _render_text_fill(self, large_text: str, small_text: str,
                          large_font_size: int, small_font_size: int,
                          step_x: int, step_y: int, wrap_after: int) -> np.ndarray:
        """直接调用完整复制文件 NestCharacter_copypaste.py 里的原函数。"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                create_text_fill_art(
                    large_text=large_text,
                    small_text=small_text,
                    large_font_size=large_font_size,
                    small_font_size=small_font_size,
                    font_path=self.font_path,
                    output_filename=tmp_path,
                    step_x=step_x,
                    step_y=step_y,
                    wrap_after=wrap_after,
                    background_color="white",
                    text_color="black",
                )
            final_image = cv2.imread(tmp_path, cv2.IMREAD_COLOR)
            if final_image is None:
                raise ValueError(f"原函数已执行但无法读取输出图像: {tmp_path}")
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        self.last_layout_overlap_pixels = -1
        return final_image

    def _fit_to_canvas_no_scale(self, image: np.ndarray) -> np.ndarray:
        """将图放进画布，不缩放；若超出则扩画布，不裁剪。"""
        ch = self.canvas_size
        cw = self.canvas_size
        h, w = image.shape[:2]
        out_h = max(ch, h)
        out_w = max(cw, w)
        canvas = np.full((out_h, out_w, 3), 255, dtype=np.uint8)
        dst_y = (out_h - h) // 2
        dst_x = (out_w - w) // 2
        canvas[dst_y:dst_y + h, dst_x:dst_x + w] = image
        return canvas

    def _apply_random_transform(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        out = image.copy()

        if random.random() > 0.5:
            angle = random.uniform(-18, 18)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            out = cv2.warpAffine(out, M, (w, h), borderValue=(255, 255, 255))

        # 按用户要求：不做缩放，不调用resize，避免任何压缩/拉伸。

        if random.random() > 0.7:
            out = cv2.flip(out, 1)

        if random.random() > 0.5:
            noise = np.random.normal(0, 8, out.shape).astype(np.float32)
            out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return out

    def _generate_two_layer_original_logic(self) -> np.ndarray:
        """严格走 NestCharacter.py 的单次大字填小字逻辑，作为2层样本。"""
        line_count = random.choice([2, 3])
        wrap_after = max(1, self.wrap_after)
        char_count = wrap_after * line_count
        large_text = "".join(random.choices(self.char_pool, k=char_count))
        small_text = "".join(random.choices(self.char_pool, k=2200))

        # 只做“可放下画布”的字号约束，不做缩放
        w_cap = int(self.canvas_size / (wrap_after * 1.05))
        h_cap = int(self.canvas_size / (1.2 * (line_count - 1) + 2.2))
        large_font_size = max(20, int(min(w_cap, h_cap) * random.uniform(0.90, 0.98)))

        small_font_size = max(4, int(large_font_size * random.uniform(0.030, 0.045)))
        step = max(1, int(small_font_size * 0.85))

        rendered = self._render_text_fill(
            large_text=large_text,
            small_text=small_text,
            large_font_size=large_font_size,
            small_font_size=small_font_size,
            step_x=step,
            step_y=step,
            wrap_after=wrap_after,
        )
        return self._fit_to_canvas_no_scale(rendered)

    def generate_layer_image(self, layers: int, apply_transform: bool = False) -> np.ndarray:
        # 按原程序逻辑逐层叠加（每层都是“大字掩膜内填小字”）
        if layers == 2 and self.use_original_logic_for_layer2:
            canvas = self._generate_two_layer_original_logic()
            if apply_transform:
                canvas = self._apply_random_transform(canvas)
            return canvas

        canvas = np.full((self.canvas_size, self.canvas_size, 3), 255, dtype=np.uint8)
        for depth in range(layers):
            wrap_after = max(1, self.wrap_after)
            line_count = random.choice([2, 3])
            large_text = "".join(random.choices(self.char_pool, k=wrap_after * line_count))
            small_text = "".join(random.choices(self.char_pool, k=1800))

            w_cap = int(self.canvas_size / (wrap_after * 1.05))
            h_cap = int(self.canvas_size / (1.2 * (line_count - 1) + 2.2))
            base_font = max(18, int(min(w_cap, h_cap) * random.uniform(0.90, 0.98)))
            large_font_size = max(14, int(base_font * (self.depth_scale ** depth)))
            small_font_size = max(4, int(large_font_size * random.uniform(self.small_font_ratio_min, self.small_font_ratio_max)))
            step = max(1, int(small_font_size * 0.85))

            rendered = self._render_text_fill(
                large_text=large_text,
                small_text=small_text,
                large_font_size=large_font_size,
                small_font_size=small_font_size,
                step_x=step,
                step_y=step,
                wrap_after=wrap_after,
            )
            rendered = self._fit_to_canvas_no_scale(rendered)
            canvas = np.minimum(canvas, rendered)

        if apply_transform:
            canvas = self._apply_random_transform(canvas)
        return canvas


def _clear_split_layer_files(data_dir: Path, layer_min: int, layer_max: int) -> None:
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for layer in range(layer_min, layer_max + 1):
            for p in split_dir.glob(f"layer{layer}_{split}_*.png"):
                p.unlink(missing_ok=True)


def generate_nesting_dataset_strict(
    strict_gen: NestCharacterStrictGenerator,
    data_dir: Path,
    samples_per_class: int,
    layer_min: int,
    layer_max: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Dict:
    _clear_split_layer_files(data_dir, layer_min, layer_max)
    train_n = int(samples_per_class * train_ratio)
    val_n = int(samples_per_class * val_ratio)

    metadata = {
        "generator": "NestCharacter.py strict style",
        "samples_per_class": samples_per_class,
        "layer_min": layer_min,
        "layer_max": layer_max,
        "classes": {},
    }

    for layer in range(layer_min, layer_max + 1):
        layer_meta = {"train_samples": 0, "val_samples": 0, "test_samples": 0}
        for idx in range(samples_per_class):
            apply_tf = idx >= samples_per_class // 2
            img = strict_gen.generate_layer_image(layer, apply_transform=apply_tf)

            if idx < train_n:
                split = "train"
                split_idx = idx
                layer_meta["train_samples"] += 1
            elif idx < train_n + val_n:
                split = "val"
                split_idx = idx - train_n
                layer_meta["val_samples"] += 1
            else:
                split = "test"
                split_idx = idx - train_n - val_n
                layer_meta["test_samples"] += 1

            filename = f"layer{layer}_{split}_{split_idx:04d}.png"
            out_path = data_dir / split / filename
            cv2.imwrite(str(out_path), img)
        metadata["classes"][f"layer_{layer}"] = layer_meta

    with (data_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata


def generate_preview_samples(
    strict_gen: NestCharacterStrictGenerator,
    generator: TrainingDataGenerator,
    preview_dir: Path,
    per_layer: int,
    ocr_count: int,
    layer_min: int,
    layer_max: int,
    logger: SimpleLogger,
) -> Dict[str, str]:
    preview_dir.mkdir(parents=True, exist_ok=True)
    layer_images: List[np.ndarray] = []

    for layer in range(layer_min, layer_max + 1):
        layer_folder = preview_dir / f"layer_{layer}"
        layer_folder.mkdir(parents=True, exist_ok=True)
        for i in range(per_layer):
            img = strict_gen.generate_layer_image(layer, apply_transform=(i % 2 == 1))
            out_path = layer_folder / f"sample_{i:02d}.png"
            cv2.imwrite(str(out_path), img)
            layer_images.append(img)
            if strict_gen.last_layout_overlap_pixels > 0:
                logger.info(
                    f"警告: layer={layer} sample={i} 同层大字发生重叠像素={strict_gen.last_layout_overlap_pixels}"
                )

    layer_grid_path = preview_dir / "preview_layers_grid.png"
    save_image_grid(layer_images, layer_grid_path, cols=per_layer)

    selected_chars = random.sample(generator.char_pool, k=min(ocr_count, len(generator.char_pool)))
    ocr_images: List[np.ndarray] = []
    for i, ch in enumerate(selected_chars):
        patch, class_id = generator.generate_ocr_patch(ch, size=(64, 64), apply_augment=(i % 2 == 1))
        patch_bgr = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        pil = Image.fromarray(patch_bgr)
        draw = ImageDraw.Draw(pil)
        draw.rectangle([(0, 0), (63, 63)], outline=(180, 180, 180), width=1)
        draw.text((3, 2), ch, fill=(0, 0, 255))
        draw.text((3, 50), str(class_id), fill=(80, 80, 80))
        ocr_images.append(np.array(pil))
    ocr_grid_path = preview_dir / "preview_ocr_grid.png"
    save_image_grid(ocr_images, ocr_grid_path, cols=8)

    manifest = {
        "layer_grid": str(layer_grid_path),
        "ocr_grid": str(ocr_grid_path),
        "preview_dir": str(preview_dir),
    }
    with (preview_dir / "preview_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info(f"预览图已生成: {layer_grid_path}")
    logger.info(f"OCR预览图已生成: {ocr_grid_path}")
    return manifest


def build_nested_dataloaders(
    generator: TrainingDataGenerator,
    batch_size: int,
    num_workers: int,
    layer_min: int,
    layer_max: int,
    image_size: int,
):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(1.0, 1.0), shear=5),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.25),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = generator.load_dataset_paths()

    def keep_selected(paths: List[str], labels: List[int]):
        p_out, y_out = [], []
        for p, y in zip(paths, labels):
            if layer_min <= y <= layer_max:
                p_out.append(p)
                y_out.append(y - layer_min)
        return p_out, y_out

    train_paths, train_labels = keep_selected(train_paths, train_labels)
    val_paths, val_labels = keep_selected(val_paths, val_labels)
    test_paths, test_labels = keep_selected(test_paths, test_labels)

    train_ds = NestedLayerDataset(train_paths, train_labels, image_size=image_size, transform=train_tf)
    val_ds = NestedLayerDataset(val_paths, val_labels, image_size=image_size, transform=val_tf)
    test_ds = NestedLayerDataset(test_paths, test_labels, image_size=image_size, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def build_ocr_dataloaders(
    ocr_data_dir: Path,
    batch_size: int,
    num_workers: int,
    image_size: int = 256,
):
    tf_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_ds = OCRPatchDataset(str(ocr_data_dir / "train"), transform=tf_train)
    val_ds = OCRPatchDataset(str(ocr_data_dir / "val"), transform=tf_val)
    test_ds = OCRPatchDataset(str(ocr_data_dir / "test"), transform=tf_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def eval_nesting(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total, correct = 0, 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += criterion(logits, y).item()
            pred = logits.argmax(dim=1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    acc = 100.0 * correct / max(total, 1)
    return loss_sum / max(len(loader), 1), acc


def eval_dual(model: DualHeadNestedCharCNN, loader: DataLoader, device: str, task: str) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total, correct = 0, 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, task=task)
            loss_sum += criterion(logits, y).item()
            pred = logits.argmax(dim=1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    acc = 100.0 * correct / max(total, 1)
    return loss_sum / max(len(loader), 1), acc


def train_nesting_stage(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    epochs: int,
    lr: float,
    device: str,
    save_path: Path,
    logger: SimpleLogger,
    early_stop_patience: int = 0,
    early_stop_acc: float = 99.9,
) -> Dict:
    model = NestedCharCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 20.0)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    best_acc = -1.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    streak = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=(device == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        scheduler.step()
        val_loss, val_acc = eval_nesting(model, val_loader, device)
        train_loss = running_loss / max(len(train_loader), 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        logger.info(f"[NEST] epoch {epoch + 1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"[NEST] 保存最佳模型: {save_path}")

        if early_stop_patience > 0:
            if val_acc >= early_stop_acc:
                streak += 1
            else:
                streak = 0
            if streak >= early_stop_patience:
                logger.info(
                    f"[NEST] 早停触发：连续 {streak} 个 epoch val_acc>={early_stop_acc:.2f}%"
                )
                break

    del model, optimizer, scheduler, scaler
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return {"best_val_acc": best_acc, "history": history, "model_path": str(save_path)}


def train_ocr_stage(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_ocr_classes: int,
    epochs: int,
    lr: float,
    device: str,
    save_path: Path,
    logger: SimpleLogger,
) -> Dict:
    model = DualHeadNestedCharCNN(num_nesting_classes=4, num_ocr_classes=num_ocr_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 20.0)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    best_acc = -1.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=(device == "cuda")):
                logits = model(x, task="ocr")
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        scheduler.step()
        val_loss, val_acc = eval_dual(model, val_loader, device, task="ocr")
        train_loss = running_loss / max(len(train_loader), 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        logger.info(f"[OCR ] epoch {epoch + 1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_ocr_classes": num_ocr_classes,
                },
                save_path,
            )
            logger.info(f"[OCR ] 保存最佳模型: {save_path}")

    del model, optimizer, scheduler, scaler
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return {"best_val_acc": best_acc, "history": history, "model_path": str(save_path)}


def train_composite_stage(
    nesting_train_loader: DataLoader,
    nesting_val_loader: DataLoader,
    ocr_train_loader: DataLoader,
    ocr_val_loader: DataLoader,
    num_nesting_classes: int,
    num_ocr_classes: int,
    nesting_state_path: Path,
    ocr_ckpt_path: Path,
    epochs: int,
    warmup_epochs: int,
    lr: float,
    ocr_loss_weight: float,
    nesting_loss_weight: float,
    device: str,
    save_path: Path,
    logger: SimpleLogger,
) -> Dict:
    model = DualHeadNestedCharCNN(
        num_nesting_classes=num_nesting_classes,
        num_ocr_classes=num_ocr_classes,
    ).to(device)

    # 权重加载顺序（修正版）：
    # 1) 先加载 NEST ckpt 全量（backbone + nesting_head），保留训练到 100% 的 NEST 特征
    # 2) 再只用 OCR ckpt 的 ocr_head 覆盖
    # 反过来加载会导致 nesting_head 与 OCR backbone 特征不匹配（实测 epoch0 即降至 20%）
    nesting_state = torch.load(nesting_state_path, map_location=device, weights_only=False)
    current_state = model.state_dict()
    nest_loaded = 0
    for old_key, tensor in nesting_state.items():
        # 旧版 NestedCharCNN 保存的是 classifier.* ，映射到 nesting_head.*
        new_key = old_key.replace('classifier.', 'nesting_head.', 1) if old_key.startswith('classifier.') else old_key
        if new_key in current_state and current_state[new_key].shape == tensor.shape:
            current_state[new_key] = tensor
            nest_loaded += 1
    model.load_state_dict(current_state)
    logger.info(f"[COMP] 从 NEST ckpt 加载 {nest_loaded} 个权重 tensor（backbone + nesting_head）")

    ocr_ckpt = torch.load(ocr_ckpt_path, map_location=device, weights_only=False)
    ocr_state = ocr_ckpt["model_state_dict"] if "model_state_dict" in ocr_ckpt else ocr_ckpt
    current_state = model.state_dict()
    ocr_loaded = 0
    for k, v in ocr_state.items():
        if k.startswith('ocr_head.') and k in current_state and current_state[k].shape == v.shape:
            current_state[k] = v
            ocr_loaded += 1
    model.load_state_dict(current_state)
    logger.info(f"[COMP] 从 OCR ckpt 仅加载 ocr_head（{ocr_loaded} 个 tensor），保留 NEST backbone")

    nesting_criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    ocr_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    backbone_params = (
        list(model.stem.parameters())
        + list(model.layer1.parameters())
        + list(model.layer2.parameters())
        + list(model.layer3.parameters())
        + list(model.layer4.parameters())
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr},
            {"params": model.nesting_head.parameters(), "lr": lr * 2},
            {"params": model.ocr_head.parameters(), "lr": lr * 2},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 20.0)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    best_score = -1.0
    history = {
        "val_nesting_acc": [],
        "val_ocr_acc": [],
        "val_nesting_loss": [],
        "val_ocr_loss": [],
    }

    for epoch in range(epochs):
        if epoch < warmup_epochs:
            freeze_backbone(model)
            stage = "warmup(ocr-only)"
        else:
            if epoch == warmup_epochs:
                unfreeze_backbone(model)
            stage = "joint"

        model.train()
        nest_iter = iter(nesting_train_loader)
        ocr_iter = iter(ocr_train_loader)
        steps = max(len(nesting_train_loader), len(ocr_train_loader))

        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            total_loss = torch.tensor(0.0, device=device)

            if epoch >= warmup_epochs:
                try:
                    nx, ny = next(nest_iter)
                except StopIteration:
                    nest_iter = iter(nesting_train_loader)
                    nx, ny = next(nest_iter)
                nx, ny = nx.to(device), ny.to(device)
                with torch.amp.autocast(device, enabled=(device == "cuda")):
                    nest_logits = model(nx, task="nesting")
                    nest_loss = nesting_criterion(nest_logits, ny)
                total_loss = total_loss + nesting_loss_weight * nest_loss

            try:
                ox, oy = next(ocr_iter)
            except StopIteration:
                ocr_iter = iter(ocr_train_loader)
                ox, oy = next(ocr_iter)
            ox, oy = ox.to(device), oy.to(device)
            with torch.amp.autocast(device, enabled=(device == "cuda")):
                ocr_logits = model(ox, task="ocr")
                ocr_loss = ocr_criterion(ocr_logits, oy)
            total_loss = total_loss + ocr_loss_weight * ocr_loss

            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        val_nesting_loss, val_nesting_acc = eval_dual(model, nesting_val_loader, device, task="nesting")
        val_ocr_loss, val_ocr_acc = eval_dual(model, ocr_val_loader, device, task="ocr")
        history["val_nesting_acc"].append(val_nesting_acc)
        history["val_ocr_acc"].append(val_ocr_acc)
        history["val_nesting_loss"].append(val_nesting_loss)
        history["val_ocr_loss"].append(val_ocr_loss)
        logger.info(
            f"[COMP] epoch {epoch + 1}/{epochs} stage={stage} "
            f"val_nesting_acc={val_nesting_acc:.2f}% val_ocr_acc={val_ocr_acc:.2f}% "
            f"val_nesting_loss={val_nesting_loss:.4f} val_ocr_loss={val_ocr_loss:.4f}"
        )

        score = val_nesting_acc + val_ocr_acc
        if score > best_score:
            best_score = score
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_nesting_classes": num_nesting_classes,
                    "num_ocr_classes": num_ocr_classes,
                    "history": history,
                },
                save_path,
            )
            logger.info(f"[COMP] 保存最佳模型: {save_path}")

    del model, optimizer, scheduler, scaler
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return {"best_score": best_score, "history": history, "model_path": str(save_path)}


def run_full_training(args, logger: SimpleLogger, run_dir: Path) -> Dict:
    logger.info("开始自动训练流程")
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"训练设备: {device}")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    generator = TrainingDataGenerator(font_path=args.font_path, output_dir=args.data_dir)

    if args.generate_data or args.generate_ocr_data:
        if args.generate_data:
            logger.info("生成嵌套训练数据集...")
            generator.generate_dataset(samples_per_class=args.samples_per_class)
        if args.generate_ocr_data:
            logger.info("生成OCR训练数据集...")
            generator.generate_ocr_dataset(samples_per_char=args.samples_per_char)

    nesting_train, nesting_val, nesting_test = build_nested_dataloaders(
        generator=generator,
        batch_size=args.batch_size_nesting,
        num_workers=args.num_workers,
        layer_min=args.layer_min,
        layer_max=args.layer_max,
        image_size=args.nest_image_size,
    )
    ocr_train, ocr_val, ocr_test = build_ocr_dataloaders(
        ocr_data_dir=Path(args.ocr_data_dir),
        batch_size=args.batch_size_ocr,
        num_workers=args.num_workers,
        image_size=args.nest_image_size,
    )

    num_nesting_classes = args.layer_max - args.layer_min + 1
    vocab_path = Path(args.ocr_data_dir) / "char_vocab.json"
    with vocab_path.open("r", encoding="utf-8") as f:
        char_vocab = json.load(f)
    num_ocr_classes = len(char_vocab)
    logger.info(f"类别数: nesting={num_nesting_classes}, ocr={num_ocr_classes}")

    nesting_path = run_dir / "models" / "nesting_best.pth"
    ocr_path = run_dir / "models" / "ocr_best.pth"
    composite_path = run_dir / "models" / "composite_best.pth"

    nest_stats = train_nesting_stage(
        train_loader=nesting_train,
        val_loader=nesting_val,
        num_classes=num_nesting_classes,
        epochs=args.epochs_nesting,
        lr=args.lr_nesting,
        device=device,
        save_path=nesting_path,
        logger=logger,
        early_stop_patience=args.nest_early_stop_patience,
        early_stop_acc=args.nest_early_stop_acc,
    )

    ocr_stats = train_ocr_stage(
        train_loader=ocr_train,
        val_loader=ocr_val,
        num_ocr_classes=num_ocr_classes,
        epochs=args.epochs_ocr,
        lr=args.lr_ocr,
        device=device,
        save_path=ocr_path,
        logger=logger,
    )

    comp_stats = train_composite_stage(
        nesting_train_loader=nesting_train,
        nesting_val_loader=nesting_val,
        ocr_train_loader=ocr_train,
        ocr_val_loader=ocr_val,
        num_nesting_classes=num_nesting_classes,
        num_ocr_classes=num_ocr_classes,
        nesting_state_path=nesting_path,
        ocr_ckpt_path=ocr_path,
        epochs=args.epochs_composite,
        warmup_epochs=args.warmup_epochs,
        lr=args.lr_composite,
        ocr_loss_weight=args.ocr_loss_weight,
        nesting_loss_weight=args.nesting_loss_weight,
        device=device,
        save_path=composite_path,
        logger=logger,
    )

    model = DualHeadNestedCharCNN(
        num_nesting_classes=num_nesting_classes,
        num_ocr_classes=num_ocr_classes,
    ).to(device)
    ckpt = torch.load(composite_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    test_nesting_loss, test_nesting_acc = eval_dual(model, nesting_test, device, task="nesting")
    test_ocr_loss, test_ocr_acc = eval_dual(model, ocr_test, device, task="ocr")
    logger.info(
        f"[TEST] nesting_acc={test_nesting_acc:.2f}% nesting_loss={test_nesting_loss:.4f} "
        f"ocr_acc={test_ocr_acc:.2f}% ocr_loss={test_ocr_loss:.4f}"
    )

    summary = {
        "run_dir": str(run_dir),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "params": vars(args),
        "nesting_stage": nest_stats,
        "ocr_stage": ocr_stats,
        "composite_stage": comp_stats,
        "test": {
            "nesting_acc": round(test_nesting_acc, 4),
            "nesting_loss": round(test_nesting_loss, 6),
            "ocr_acc": round(test_ocr_acc, 4),
            "ocr_loss": round(test_ocr_loss, 6),
        },
    }
    with (run_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"训练总结已保存: {run_dir / 'training_summary.json'}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="夜间自动化训练脚本（先预览，后开训）")
    parser.add_argument("--font-path", type=str, default="C:/Windows/Fonts/simhei.ttf")
    parser.add_argument("--data-dir", type=str, default="training_data")
    parser.add_argument("--ocr-data-dir", type=str, default="training_data/ocr_data")
    parser.add_argument("--out-root", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default=datetime.now().strftime("autotrain_%Y%m%d_%H%M%S"))

    parser.add_argument("--layer-min", type=int, default=2, help="层数最小值（默认2层）")
    parser.add_argument("--layer-max", type=int, default=5, help="层数最大值（默认5层）")
    parser.add_argument("--nest-image-size", type=int, default=1536, help="嵌套训练图固定边长（不缩放）")
    parser.add_argument("--large-font-ratio-min", type=float, default=0.68, help="首层大字字号下限占画布比例")
    parser.add_argument("--large-font-ratio-max", type=float, default=0.92, help="首层大字字号上限占画布比例")
    parser.add_argument("--depth-scale", type=float, default=0.67, help="层间字号缩放因子")
    parser.add_argument("--small-font-ratio-min", type=float, default=0.028, help="小字字号下限占当前大字比例")
    parser.add_argument("--small-font-ratio-max", type=float, default=0.045, help="小字字号上限占当前大字比例")
    parser.add_argument("--step-ratio-min", type=float, default=1.02, help="小字步进下限占小字字号比例")
    parser.add_argument("--step-ratio-max", type=float, default=1.28, help="小字步进上限占小字字号比例")
    parser.add_argument("--large-char-spacing-scale", type=float, default=1.12, help="同层多大字的字间距倍率")
    parser.add_argument("--allow-overlap-same-layer", action="store_true", help="允许同层字符叠压（默认不允许）")
    parser.add_argument("--layer-char-min", type=int, default=0, help="每层大字数量下限（0表示自动）")
    parser.add_argument("--layer-char-max", type=int, default=0, help="每层大字数量上限（0表示自动）")
    parser.add_argument("--wrap-after", type=int, default=12, help="大字每行字符数上限（用于防重叠）")
    parser.add_argument("--disable-original-layer2", action="store_true", help="关闭2层样本的原逻辑生成")

    parser.add_argument("--preview-per-layer", type=int, default=4)
    parser.add_argument("--preview-ocr-count", type=int, default=24)
    parser.add_argument("--preview-only", action="store_true", help="只生成预览图并退出")
    parser.add_argument("--run", action="store_true", help="确认执行完整训练流程")

    parser.add_argument("--generate-data", action="store_true", help="生成嵌套数据")
    parser.add_argument("--generate-ocr-data", action="store_true", help="生成OCR数据")
    parser.add_argument("--samples-per-class", type=int, default=2800)
    parser.add_argument("--samples-per-char", type=int, default=120)

    parser.add_argument("--epochs-nesting", type=int, default=24)
    parser.add_argument("--epochs-ocr", type=int, default=18)
    parser.add_argument("--epochs-composite", type=int, default=16)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--nest-early-stop-patience", type=int, default=10,
                        help="NEST 连续多少个 epoch val_acc>=阈值则早停；<=0 关闭")
    parser.add_argument("--nest-early-stop-acc", type=float, default=99.9,
                        help="NEST 早停的 val_acc 阈值（百分比）")
    parser.add_argument("--batch-size-nesting", type=int, default=4)
    parser.add_argument("--batch-size-ocr", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument("--lr-nesting", type=float, default=1e-3)
    parser.add_argument("--lr-ocr", type=float, default=8e-4)
    parser.add_argument("--lr-composite", type=float, default=3e-4)
    parser.add_argument("--ocr-loss-weight", type=float, default=3.0)
    parser.add_argument("--nesting-loss-weight", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.layer_char_min < 0:
        args.layer_char_min = 0
    if args.layer_char_max < 0:
        args.layer_char_max = 0
    # 只有两个值都大于0时才使用固定范围，否则走自动字数策略
    if args.layer_char_min > 0 and args.layer_char_max > 0 and args.layer_char_min > args.layer_char_max:
        args.layer_char_min, args.layer_char_max = args.layer_char_max, args.layer_char_min
    args.wrap_after = max(1, args.wrap_after)
    set_seed(args.seed)

    run_dir = Path(args.out_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = SimpleLogger(run_dir / "train.log")
    logger.info("脚本启动")

    generator = TrainingDataGenerator(font_path=args.font_path, output_dir=args.data_dir)
    strict_gen = NestCharacterStrictGenerator(
        font_path=args.font_path,
        char_pool=generator.char_pool,
        canvas_size=args.nest_image_size,
        large_font_ratio_min=args.large_font_ratio_min,
        large_font_ratio_max=args.large_font_ratio_max,
        depth_scale=args.depth_scale,
        small_font_ratio_min=args.small_font_ratio_min,
        small_font_ratio_max=args.small_font_ratio_max,
        step_ratio_min=args.step_ratio_min,
        step_ratio_max=args.step_ratio_max,
        no_overlap_same_layer=(not args.allow_overlap_same_layer),
        large_char_spacing_scale=args.large_char_spacing_scale,
        layer_char_min=args.layer_char_min,
        layer_char_max=args.layer_char_max,
        wrap_after=args.wrap_after,
        use_original_logic_for_layer2=(not args.disable_original_layer2),
    )
    preview_manifest = generate_preview_samples(
        strict_gen=strict_gen,
        generator=generator,
        preview_dir=run_dir / "preview",
        per_layer=args.preview_per_layer,
        ocr_count=args.preview_ocr_count,
        layer_min=args.layer_min,
        layer_max=args.layer_max,
        logger=logger,
    )

    if args.preview_only or not args.run:
        logger.info("当前为预览模式，未执行训练。")
        logger.info(f"请先检查预览图: {preview_manifest['layer_grid']} 和 {preview_manifest['ocr_grid']}")
        logger.info(
            "确认后可执行: "
            f"python overnight_autotrain.py --run --generate-data --generate-ocr-data --run-name {args.run_name}_full"
        )
        return

    start = time.time()
    summary = run_full_training(args, logger, run_dir)
    elapsed = (time.time() - start) / 3600.0
    logger.info(f"训练完成，总耗时: {elapsed:.2f} 小时")
    logger.info(f"最终测试: nesting_acc={summary['test']['nesting_acc']} ocr_acc={summary['test']['ocr_acc']}")


if __name__ == "__main__":
    main()
