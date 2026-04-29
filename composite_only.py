# -*- coding: utf-8 -*-
"""
单跑 composite 阶段：复用 --reuse-from 指定 run 目录下的
nesting_best.pth 与 ocr_best.pth，对 composite 重训一次。

示例：
  python composite_only.py \
    --reuse-from runs/linux_auto_20260429_171309 \
    --data-dir training_data \
    --ocr-data-dir training_data/ocr_data \
    --font-path "$PWD/fonts/NotoSansCJK-Regular.ttc"
"""
import argparse
import gc
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import torch

from overnight_autotrain import (
    SimpleLogger,
    build_nested_dataloaders,
    build_ocr_dataloaders,
    eval_dual,
    set_seed,
    train_composite_stage,
)
from training_data_generator import TrainingDataGenerator
from nested_char_detector import DualHeadNestedCharCNN


def parse_args():
    p = argparse.ArgumentParser("仅跑 composite 阶段")
    p.add_argument("--reuse-from", type=str, required=True, help="已存在的 run 目录，里面要有 models/nesting_best.pth 和 models/ocr_best.pth")
    p.add_argument("--out-root", type=str, default="runs")
    p.add_argument("--run-name", type=str, default=datetime.now().strftime("composite_only_%Y%m%d_%H%M%S"))

    p.add_argument("--font-path", type=str, default=os.environ.get("FONT_PATH", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    p.add_argument("--data-dir", type=str, default="training_data")
    p.add_argument("--ocr-data-dir", type=str, default="training_data/ocr_data")
    p.add_argument("--layer-min", type=int, default=0)
    p.add_argument("--layer-max", type=int, default=5)
    p.add_argument("--nest-image-size", type=int, default=256)
    p.add_argument("--batch-size-nesting", type=int, default=32)
    p.add_argument("--batch-size-ocr", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)

    p.add_argument("--epochs-composite", type=int, default=40)
    p.add_argument("--warmup-epochs", type=int, default=6)
    p.add_argument("--lr-composite", type=float, default=3e-4)
    p.add_argument("--ocr-loss-weight", type=float, default=3.0)
    p.add_argument("--nesting-loss-weight", type=float, default=1.0)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    set_seed(args.seed)

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    reuse = Path(args.reuse_from)
    src_nest = reuse / "models" / "nesting_best.pth"
    src_ocr = reuse / "models" / "ocr_best.pth"
    if not src_nest.exists() or not src_ocr.exists():
        raise FileNotFoundError(f"复用目录缺 ckpt: {src_nest} 或 {src_ocr}")

    run_dir = Path(args.out_root) / args.run_name
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    logger = SimpleLogger(run_dir / "train.log")
    logger.info(f"composite-only 启动；复用自 {reuse}")
    logger.info(f"训练设备: {device}")

    # 把 ckpt 拷过来，方便后续单独管理
    dst_nest = run_dir / "models" / "nesting_best.pth"
    dst_ocr = run_dir / "models" / "ocr_best.pth"
    shutil.copy2(src_nest, dst_nest)
    shutil.copy2(src_ocr, dst_ocr)
    composite_path = run_dir / "models" / "composite_best.pth"

    generator = TrainingDataGenerator(font_path=args.font_path, output_dir=args.data_dir)
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

    comp_stats = train_composite_stage(
        nesting_train_loader=nesting_train,
        nesting_val_loader=nesting_val,
        ocr_train_loader=ocr_train,
        ocr_val_loader=ocr_val,
        num_nesting_classes=num_nesting_classes,
        num_ocr_classes=num_ocr_classes,
        nesting_state_path=dst_nest,
        ocr_ckpt_path=dst_ocr,
        epochs=args.epochs_composite,
        warmup_epochs=args.warmup_epochs,
        lr=args.lr_composite,
        ocr_loss_weight=args.ocr_loss_weight,
        nesting_loss_weight=args.nesting_loss_weight,
        device=device,
        save_path=composite_path,
        logger=logger,
    )

    # 测试集评估
    model = DualHeadNestedCharCNN(
        num_nesting_classes=num_nesting_classes,
        num_ocr_classes=num_ocr_classes,
    ).to(device)
    ckpt = torch.load(composite_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    test_n_loss, test_n_acc = eval_dual(model, nesting_test, device, task="nesting")
    test_o_loss, test_o_acc = eval_dual(model, ocr_test, device, task="ocr")
    logger.info(
        f"[TEST] nesting_acc={test_n_acc:.2f}% nesting_loss={test_n_loss:.4f} "
        f"ocr_acc={test_o_acc:.2f}% ocr_loss={test_o_loss:.4f}"
    )

    summary = {
        "run_dir": str(run_dir),
        "reuse_from": str(reuse),
        "composite": comp_stats,
        "test": {
            "nesting_acc": test_n_acc,
            "nesting_loss": test_n_loss,
            "ocr_acc": test_o_acc,
            "ocr_loss": test_o_loss,
        },
    }
    with (run_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"完成，summary 写入: {run_dir / 'training_summary.json'}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
