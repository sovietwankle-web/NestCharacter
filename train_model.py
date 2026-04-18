# -*- coding: utf-8 -*-
"""
模型训练脚本 - 支持嵌套层分类、OCR字符识别、复合训练三种模式
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import argparse
from training_data_generator import TrainingDataGenerator, visualize_samples
from nested_char_detector import (
    NestedCharDetector, NestedCharDataset,
    DualHeadNestedCharCNN, OCRPatchDataset,
    load_nesting_weights_from_legacy,
    freeze_backbone, unfreeze_backbone
)
from tqdm import tqdm
import matplotlib.pyplot as plt


def plot_training_history(train_losses: list, val_losses: list, save_path: str):
    """绘制训练历史"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"训练历史图已保存到: {save_path}")


def plot_composite_history(history: dict, save_path: str):
    """绘制复合训练历史——4面板图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 嵌套 loss
    axes[0, 0].plot(history['train_nesting_loss'], label='Train')
    axes[0, 0].plot(history['val_nesting_loss'], label='Val')
    axes[0, 0].set_title('Nesting Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # OCR loss
    axes[0, 1].plot(history['train_ocr_loss'], label='Train')
    axes[0, 1].plot(history['val_ocr_loss'], label='Val')
    axes[0, 1].set_title('OCR Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 嵌套 accuracy
    axes[1, 0].plot(history['train_nesting_acc'], label='Train')
    axes[1, 0].plot(history['val_nesting_acc'], label='Val')
    axes[1, 0].set_title('Nesting Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # OCR accuracy
    axes[1, 1].plot(history['train_ocr_acc'], label='Train')
    axes[1, 1].plot(history['val_ocr_acc'], label='Val')
    axes[1, 1].set_title('OCR Accuracy (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"复合训练历史图已保存到: {save_path}")


def train_composite(
    model: DualHeadNestedCharCNN,
    nesting_train_loader: DataLoader,
    nesting_val_loader: DataLoader,
    ocr_train_loader: DataLoader,
    ocr_val_loader: DataLoader,
    num_epochs: int = 60,
    learning_rate: float = 0.001,
    ocr_loss_weight: float = 2.0,
    nesting_loss_weight: float = 1.0,
    warmup_epochs: int = 10,
    save_path: str = 'models/dual_head_model.pth',
    device: str = 'cuda',
) -> dict:
    """复合训练：交替批次训练嵌套检测和OCR识别

    策略:
      - Phase 1 (前warmup_epochs轮): 冻结骨干，仅训练OCR头
      - Phase 2 (剩余轮次): 解冻全部，交替批次复合训练
      - OCR loss权重 > 嵌套 loss权重，保证OCR准确性优先

    Returns:
        训练历史字典
    """
    model = model.to(device)

    nesting_criterion = nn.CrossEntropyLoss()
    ocr_criterion = nn.CrossEntropyLoss()

    # 分参数组：骨干基础学习率，两个头2倍学习率
    backbone_params = (
        list(model.stem.parameters())
        + list(model.layer1.parameters())
        + list(model.layer2.parameters())
        + list(model.layer3.parameters())
        + list(model.layer4.parameters())
    )
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': learning_rate},
        {'params': model.nesting_head.parameters(), 'lr': learning_rate * 2},
        {'params': model.ocr_head.parameters(), 'lr': learning_rate * 2},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=learning_rate / 100
    )

    history = {
        'train_nesting_loss': [], 'val_nesting_loss': [],
        'train_ocr_loss': [], 'val_ocr_loss': [],
        'train_nesting_acc': [], 'val_nesting_acc': [],
        'train_ocr_acc': [], 'val_ocr_acc': [],
    }
    best_combined_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Phase 1: 冻结骨干，仅训练OCR头
        if epoch < warmup_epochs:
            freeze_backbone(model)
            phase_desc = f"Phase1-OCR预热 Epoch {epoch + 1}/{num_epochs}"
        else:
            if epoch == warmup_epochs:
                unfreeze_backbone(model)
                print("=" * 50)
                print("Phase 2: 解冻骨干，开始复合训练")
                print("=" * 50)
            phase_desc = f"Phase2-复合 Epoch {epoch + 1}/{num_epochs}"

        model.train()

        nesting_iter = iter(nesting_train_loader)
        ocr_iter = iter(ocr_train_loader)

        nesting_loss_accum, nesting_correct, nesting_total = 0.0, 0, 0
        ocr_loss_accum, ocr_correct, ocr_total = 0.0, 0, 0

        num_steps = max(len(nesting_train_loader), len(ocr_train_loader))

        pbar = tqdm(range(num_steps), desc=phase_desc)
        for step in pbar:
            optimizer.zero_grad()
            total_step_loss = torch.tensor(0.0, device=device, requires_grad=True)

            # --- 嵌套批次 ---
            if epoch >= warmup_epochs:
                try:
                    nest_imgs, nest_labels = next(nesting_iter)
                except StopIteration:
                    nesting_iter = iter(nesting_train_loader)
                    nest_imgs, nest_labels = next(nesting_iter)

                nest_imgs = nest_imgs.to(device)
                nest_labels = nest_labels.to(device)
                nest_logits = model(nest_imgs, task='nesting')
                nest_loss = nesting_criterion(nest_logits, nest_labels)
                total_step_loss = total_step_loss + nesting_loss_weight * nest_loss

                nesting_loss_accum += nest_loss.item()
                _, nest_pred = torch.max(nest_logits, 1)
                nesting_total += nest_labels.size(0)
                nesting_correct += (nest_pred == nest_labels).sum().item()

            # --- OCR批次 ---
            try:
                ocr_imgs, ocr_labels = next(ocr_iter)
            except StopIteration:
                ocr_iter = iter(ocr_train_loader)
                ocr_imgs, ocr_labels = next(ocr_iter)

            ocr_imgs = ocr_imgs.to(device)
            ocr_labels = ocr_labels.to(device)
            ocr_logits = model(ocr_imgs, task='ocr')
            ocr_loss = ocr_criterion(ocr_logits, ocr_labels)
            total_step_loss = total_step_loss + ocr_loss_weight * ocr_loss

            ocr_loss_accum += ocr_loss.item()
            _, ocr_pred = torch.max(ocr_logits, 1)
            ocr_total += ocr_labels.size(0)
            ocr_correct += (ocr_pred == ocr_labels).sum().item()

            total_step_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if step % 50 == 0:
                ocr_acc_so_far = 100 * ocr_correct / max(ocr_total, 1)
                nest_acc_so_far = 100 * nesting_correct / max(nesting_total, 1) if nesting_total > 0 else 0
                pbar.set_postfix({
                    'ocr_acc': f'{ocr_acc_so_far:.1f}%',
                    'nest_acc': f'{nest_acc_so_far:.1f}%'
                })

        scheduler.step()

        # 记录训练指标
        history['train_nesting_loss'].append(nesting_loss_accum / max(num_steps, 1))
        history['train_ocr_loss'].append(ocr_loss_accum / max(num_steps, 1))
        history['train_nesting_acc'].append(100 * nesting_correct / max(nesting_total, 1))
        history['train_ocr_acc'].append(100 * ocr_correct / max(ocr_total, 1))

        # 验证
        model.eval()

        def _eval_loader(loader, task):
            loss_sum, correct, total = 0.0, 0, 0
            criterion = nesting_criterion if task == 'nesting' else ocr_criterion
            with torch.no_grad():
                for imgs, labels in loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    logits = model(imgs, task=task)
                    loss_sum += criterion(logits, labels).item()
                    _, pred = torch.max(logits, 1)
                    total += labels.size(0)
                    correct += (pred == labels).sum().item()
            return loss_sum / max(len(loader), 1), 100 * correct / max(total, 1)

        val_nest_loss, val_nest_acc = _eval_loader(nesting_val_loader, 'nesting')
        val_ocr_loss, val_ocr_acc = _eval_loader(ocr_val_loader, 'ocr')

        history['val_nesting_loss'].append(val_nest_loss)
        history['val_ocr_loss'].append(val_ocr_loss)
        history['val_nesting_acc'].append(val_nest_acc)
        history['val_ocr_acc'].append(val_ocr_acc)

        print(f"\n{phase_desc}")
        print(f"  Nesting — train_loss={history['train_nesting_loss'][-1]:.4f}  "
              f"val_loss={val_nest_loss:.4f}  "
              f"train_acc={history['train_nesting_acc'][-1]:.2f}%  "
              f"val_acc={val_nest_acc:.2f}%")
        print(f"  OCR     — train_loss={history['train_ocr_loss'][-1]:.4f}  "
              f"val_loss={val_ocr_loss:.4f}  "
              f"train_acc={history['train_ocr_acc'][-1]:.2f}%  "
              f"val_acc={val_ocr_acc:.2f}%")

        # 保存最佳模型（加权组合验证loss）
        combined_val = nesting_loss_weight * val_nest_loss + ocr_loss_weight * val_ocr_loss
        if combined_val < best_combined_val_loss:
            best_combined_val_loss = combined_val
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'num_nesting_classes': 6,
                'num_ocr_classes': model.ocr_head[-1].out_features,
            }, save_path)
            print(f"  保存最佳模型 -> {save_path}")

    return history


def main():
    parser = argparse.ArgumentParser(description='训练嵌套字符识别模型')

    # 训练模式
    parser.add_argument('--mode', choices=['nesting', 'ocr', 'composite'],
                        default='nesting',
                        help='训练模式: nesting=仅嵌套检测, ocr=仅OCR, composite=复合训练')

    # 数据生成参数
    parser.add_argument('--generate-data', action='store_true',
                       help='是否生成新的嵌套训练数据')
    parser.add_argument('--generate-ocr-data', action='store_true',
                       help='是否生成新的OCR训练数据')
    parser.add_argument('--samples-per-class', type=int, default=500,
                       help='嵌套数据每类样本数量')
    parser.add_argument('--samples-per-char', type=int, default=50,
                       help='OCR数据每字符样本数量')
    parser.add_argument('--font-path', type=str, default=None,
                       help='字体文件路径 (不指定则自动检测)')
    parser.add_argument('--data-dir', type=str, default='training_data',
                       help='数据集目录')
    parser.add_argument('--ocr-data-dir', type=str, default='training_data/ocr_data',
                       help='OCR数据集目录')

    # 训练参数
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--num-epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='数据加载线程数')
    parser.add_argument('--warmup-epochs', type=int, default=10,
                       help='复合训练Phase1预热轮数(冻结骨干仅训练OCR头)')

    # 复合训练loss权重
    parser.add_argument('--ocr-loss-weight', type=float, default=2.0,
                       help='OCR loss权重(>1表示优先OCR准确性)')
    parser.add_argument('--nesting-loss-weight', type=float, default=1.0,
                       help='嵌套检测loss权重')

    # 模型参数
    parser.add_argument('--model-save-path', type=str, default='models/nested_char_model.pth',
                       help='模型保存路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的模型路径')

    args = parser.parse_args()

    # 自动检测字体路径
    if args.font_path is None:
        import platform
        if platform.system() == 'Windows':
            args.font_path = 'C:/Windows/Fonts/simhei.ttf'
        else:
            # Linux: 尝试常见CJK字体路径
            candidates = [
                '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',
                '/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc',
                '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
                '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
            ]
            args.font_path = candidates[0]  # default fallback
            for c in candidates:
                if os.path.isfile(c):
                    args.font_path = c
                    break
        print(f"使用字体: {args.font_path}")

    # 创建模型保存目录
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    generator = TrainingDataGenerator(args.font_path, output_dir=args.data_dir)

    # ========== 生成数据 ==========
    if args.generate_data:
        print("=" * 60)
        print("生成嵌套训练数据集")
        print("=" * 60)
        metadata = generator.generate_dataset(samples_per_class=args.samples_per_class)
        visualize_samples(generator)

    if args.generate_ocr_data:
        print("=" * 60)
        print("生成OCR训练数据集")
        print("=" * 60)
        generator.generate_ocr_dataset(samples_per_char=args.samples_per_char)

    # ========== 嵌套数据加载 ==========
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = generator.load_dataset_paths()

    print(f"\n嵌套数据集 — 训练: {len(train_paths)}, 验证: {len(val_paths)}, 测试: {len(test_paths)}")

    train_dataset = NestedCharDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = NestedCharDataset(val_paths, val_labels, transform=val_transform)
    test_dataset = NestedCharDataset(test_paths, test_labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # ========== 模式选择 ==========

    if args.mode == 'nesting':
        # ---- 原有模式：仅嵌套检测训练 ----
        print("\n" + "=" * 60)
        print("模式: 仅嵌套检测训练")
        print("=" * 60)

        detector = NestedCharDetector(model_path=args.resume)
        train_losses, val_losses = detector.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            save_path=args.model_save_path
        )

        history_plot_path = os.path.join(os.path.dirname(args.model_save_path), 'training_history.png')
        plot_training_history(train_losses, val_losses, history_plot_path)

        # 测试
        detector.model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = detector.model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_accuracy = 100 * test_correct / test_total
        print(f"嵌套检测测试准确率: {test_accuracy:.2f}%")

    elif args.mode in ('composite', 'ocr'):
        # ---- 复合/OCR模式 ----
        print("\n" + "=" * 60)
        print(f"模式: {'复合训练' if args.mode == 'composite' else '仅OCR训练'}")
        print("=" * 60)

        # 加载OCR数据
        ocr_train_dir = os.path.join(args.ocr_data_dir, 'train')
        ocr_val_dir = os.path.join(args.ocr_data_dir, 'val')
        ocr_test_dir = os.path.join(args.ocr_data_dir, 'test')

        ocr_patch_size = 64
        ocr_train_transform = transforms.Compose([
            transforms.Resize((ocr_patch_size, ocr_patch_size)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        ocr_val_transform = transforms.Compose([
            transforms.Resize((ocr_patch_size, ocr_patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        ocr_train_dataset = OCRPatchDataset(ocr_train_dir,
                                            transform=ocr_train_transform)
        ocr_val_dataset = OCRPatchDataset(ocr_val_dir,
                                          transform=ocr_val_transform)
        ocr_test_dataset = OCRPatchDataset(ocr_test_dir,
                                           transform=ocr_val_transform)

        print(f"OCR数据集 — 训练: {len(ocr_train_dataset)}, 验证: {len(ocr_val_dataset)}, 测试: {len(ocr_test_dataset)}")

        ocr_train_loader = DataLoader(ocr_train_dataset, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers)
        ocr_val_loader = DataLoader(ocr_val_dataset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers)
        ocr_test_loader = DataLoader(ocr_test_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.num_workers)

        # 确定OCR类别数
        vocab_path = os.path.join(args.ocr_data_dir, 'char_vocab.json')
        if os.path.exists(vocab_path):
            import json
            with open(vocab_path, encoding='utf-8') as f:
                char_vocab = json.load(f)
            num_ocr_classes = len(char_vocab)
        else:
            num_ocr_classes = len(generator.char_vocab)

        print(f"OCR类别数: {num_ocr_classes}")

        # 初始化双头模型
        model = DualHeadNestedCharCNN(
            num_nesting_classes=6,
            num_ocr_classes=num_ocr_classes
        )

        # 加载已有权重
        if args.resume and os.path.exists(args.resume):
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
                print(f"已加载双头模型权重: {args.resume}")
            else:
                # 从旧版单头模型迁移
                load_nesting_weights_from_legacy(model, ckpt)
                print(f"已从旧版模型迁移骨干+嵌套头权重: {args.resume}")

        # 复合训练
        save_path = args.model_save_path.replace('.pth', '_dual.pth') \
            if 'dual' not in args.model_save_path else args.model_save_path

        history = train_composite(
            model=model,
            nesting_train_loader=train_loader,
            nesting_val_loader=val_loader,
            ocr_train_loader=ocr_train_loader,
            ocr_val_loader=ocr_val_loader,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            ocr_loss_weight=args.ocr_loss_weight,
            nesting_loss_weight=args.nesting_loss_weight if args.mode == 'composite' else 0.0,
            warmup_epochs=args.warmup_epochs,
            save_path=save_path,
            device=device,
        )

        # 绘制训练历史
        history_plot_path = os.path.join(os.path.dirname(save_path), 'composite_training_history.png')
        plot_composite_history(history, history_plot_path)

        # 测试两个任务
        model.eval()
        model = model.to(device)

        # 嵌套检测测试
        nest_correct, nest_total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, task='nesting')
                _, predicted = torch.max(outputs.data, 1)
                nest_total += labels.size(0)
                nest_correct += (predicted == labels).sum().item()

        # OCR测试
        ocr_correct, ocr_total = 0, 0
        with torch.no_grad():
            for images, labels in ocr_test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, task='ocr')
                _, predicted = torch.max(outputs.data, 1)
                ocr_total += labels.size(0)
                ocr_correct += (predicted == labels).sum().item()

        nest_acc = 100 * nest_correct / max(nest_total, 1)
        ocr_acc = 100 * ocr_correct / max(ocr_total, 1)

        print(f"\n{'=' * 60}")
        print(f"测试结果:")
        print(f"  嵌套检测准确率: {nest_acc:.2f}% ({nest_correct}/{nest_total})")
        print(f"  OCR识别准确率:  {ocr_acc:.2f}% ({ocr_correct}/{ocr_total})")
        print(f"{'=' * 60}")

        # 保存测试结果
        results_path = os.path.join(os.path.dirname(save_path), 'composite_test_results.txt')
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write(f"嵌套检测准确率: {nest_acc:.2f}% ({nest_correct}/{nest_total})\n")
            f.write(f"OCR识别准确率: {ocr_acc:.2f}% ({ocr_correct}/{ocr_total})\n")
            f.write(f"OCR类别数: {num_ocr_classes}\n")
            f.write(f"训练轮数: {args.num_epochs}\n")
            f.write(f"OCR loss权重: {args.ocr_loss_weight}\n")
            f.write(f"嵌套 loss权重: {args.nesting_loss_weight}\n")
        print(f"测试结果已保存到: {results_path}")

    print("\n训练完成!")


if __name__ == '__main__':
    main()
