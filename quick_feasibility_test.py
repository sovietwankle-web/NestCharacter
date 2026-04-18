#!/usr/bin/env python3
"""
快速可行性测试 — 验证整条流水线能跑通，不追求结果质量。
每层只生成 2 张图，OCR 每字 2 张，训练各跑 1 epoch。
预计 Windows CPU 上 5~15 分钟内完成。
"""

import subprocess
import sys
import os
import time
import shutil
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
TEST_DIR = os.path.join(ROOT, "_feasibility_test")
DATA_DIR = os.path.join(TEST_DIR, "data")
OCR_DIR = os.path.join(DATA_DIR, "ocr_data")
RUN_DIR = os.path.join(TEST_DIR, "runs")
FONT = "C:/Windows/Fonts/msyh.ttc"

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"


def banner(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}\n")


def run_cmd(desc, cmd, timeout=300):
    """运行命令，返回 (ok, elapsed)"""
    print(f">>> {desc}")
    print(f"    cmd: {' '.join(cmd)}")
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, cwd=ROOT, timeout=timeout,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"    {FAIL} exit={result.returncode}  ({elapsed:.1f}s)")
            if result.stderr:
                # 只打印最后 30 行
                lines = result.stderr.strip().splitlines()
                for l in lines[-30:]:
                    print(f"    | {l}")
            return False, elapsed
        print(f"    {PASS}  ({elapsed:.1f}s)")
        return True, elapsed
    except subprocess.TimeoutExpired:
        print(f"    {FAIL} TIMEOUT after {timeout}s")
        return False, timeout


def check_import(module, names):
    """检查 from module import names 是否成功"""
    import_str = f"from {module} import {', '.join(names)}"
    cmd = [PYTHON, "-c", import_str]
    return run_cmd(f"Import check: {import_str}", cmd, timeout=30)


def main():
    results = {}
    t_total = time.time()

    # ---- 清理旧测试目录 ----
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR, ignore_errors=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # ==== Step 1: Import checks ====
    banner("Step 1/6: Import 检查")

    ok1, _ = check_import("nested_char_detector", [
        "NestedCharDetector", "NestedCharDataset",
        "DualHeadNestedCharCNN", "OCRPatchDataset",
    ])
    results["import_nested_char_detector"] = ok1

    ok2, _ = run_cmd("Import check: import train_model",
                      [PYTHON, "-c", "import train_model"], timeout=30)
    results["import_train_model"] = ok2

    ok3, _ = run_cmd("Import check: import overnight_autotrain",
                      [PYTHON, "-c", "import overnight_autotrain"], timeout=30)
    results["import_overnight_autotrain"] = ok3

    ok4, _ = run_cmd("Import check: import generate_dataset_copypaste",
                      [PYTHON, "-c", "import generate_dataset_copypaste"], timeout=30)
    results["import_generate_dataset_copypaste"] = ok4

    ok5, _ = check_import("training_data_generator", ["TrainingDataGenerator"])
    results["import_training_data_generator"] = ok5

    # ==== Step 2: 生成嵌套数据集 (每类 2 张) ====
    banner("Step 2/6: 生成嵌套数据集 (每类2张, layer 0~5)")
    ok_nest, t_nest = run_cmd("generate_dataset_copypaste.py", [
        PYTHON, "generate_dataset_copypaste.py",
        "--font-path", FONT,
        "--out-dir", DATA_DIR,
        "--layer-min", "0",
        "--layer-max", "5",
        "--samples-per-class", "2",
        "--base-large-font-size", "200",
        "--wrap-after", "6",
        "--seed", "42",
    ], timeout=600)
    results["generate_nesting_data"] = ok_nest

    # 检查输出文件
    if ok_nest:
        for split in ["train", "val", "test"]:
            d = os.path.join(DATA_DIR, split)
            if os.path.isdir(d):
                n = len(os.listdir(d))
                print(f"    {split}/: {n} files")
            else:
                print(f"    {split}/: NOT FOUND")

    # ==== Step 3: 生成 OCR 数据集 (每字 2 张) ====
    banner("Step 3/6: 生成 OCR 字符数据集 (每字2张)")
    ok_ocr, t_ocr = run_cmd("training_data_generator OCR", [
        PYTHON, "-c",
        f"from training_data_generator import TrainingDataGenerator; "
        f"gen = TrainingDataGenerator('{FONT}', output_dir='{DATA_DIR.replace(chr(92), '/')}'); "
        f"gen.generate_ocr_dataset(samples_per_char=2)",
    ], timeout=300)
    results["generate_ocr_data"] = ok_ocr

    if ok_ocr and os.path.isdir(OCR_DIR):
        vocab_path = os.path.join(OCR_DIR, "char_vocab.json")
        if os.path.isfile(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab = json.load(f)
            print(f"    char_vocab.json: {len(vocab)} chars")
        else:
            print(f"    char_vocab.json: NOT FOUND")
        for split in ["train", "val", "test"]:
            d = os.path.join(OCR_DIR, split)
            if os.path.isdir(d):
                n = len(os.listdir(d))
                print(f"    ocr_data/{split}/: {n} files")

    # ==== Step 4: Dataset 类实例化 ====
    banner("Step 4/6: Dataset 类实例化测试")
    ok_ds, _ = run_cmd("NestedCharDataset + OCRPatchDataset", [
        PYTHON, "-c", f"""
import glob, os, json
import torchvision.transforms as T
from nested_char_detector import NestedCharDataset, OCRPatchDataset

# Nesting dataset
data_dir = '{DATA_DIR.replace(chr(92), '/')}'
ocr_dir = '{OCR_DIR.replace(chr(92), '/')}'

train_dir = os.path.join(data_dir, 'train')
imgs = sorted(glob.glob(os.path.join(train_dir, '*.png')))
labels = []
for p in imgs:
    fname = os.path.basename(p)
    layer = int(fname.split('_')[0].replace('layer',''))
    labels.append(layer)

tf = T.Compose([T.Resize((256,256)), T.ToTensor()])
ds = NestedCharDataset(imgs, labels, transform=tf)
print(f'NestedCharDataset: len={{len(ds)}}')
if len(ds) > 0:
    x, y = ds[0]
    print(f'  sample shape={{x.shape}}, label={{y}}')

# OCR dataset
if os.path.isdir(ocr_dir):
    ods = OCRPatchDataset(ocr_dir, transform=tf)
    print(f'OCRPatchDataset: len={{len(ods)}}')
    if len(ods) > 0:
        x2, y2 = ods[0]
        print(f'  sample shape={{x2.shape}}, label={{y2}}')
else:
    print('OCR dir not found, skipping OCRPatchDataset')
""",
    ], timeout=60)
    results["dataset_instantiation"] = ok_ds

    # ==== Step 5: 模型前向传播 ====
    banner("Step 5/6: 模型前向传播测试 (CPU)")
    ok_fwd, _ = run_cmd("DualHeadNestedCharCNN forward", [
        PYTHON, "-c", f"""
import torch, json, os
from nested_char_detector import DualHeadNestedCharCNN

ocr_dir = '{OCR_DIR.replace(chr(92), '/')}'
vocab_path = os.path.join(ocr_dir, 'char_vocab.json')
if os.path.isfile(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    num_chars = len(vocab)
else:
    num_chars = 10  # fallback

num_classes = 6  # layers 0-5
model = DualHeadNestedCharCNN(num_nesting_classes=num_classes, num_ocr_classes=num_chars)
model.eval()

# Nesting forward
x = torch.randn(2, 1, 256, 256)
with torch.no_grad():
    out = model(x, task='nesting')
print(f'Nesting output: {{out.shape}}')  # expect [2, 6]

# OCR forward
with torch.no_grad():
    out2 = model(x, task='ocr')
print(f'OCR output: {{out2.shape}}')  # expect [2, num_chars]

print('Forward pass OK')
""",
    ], timeout=60)
    results["model_forward"] = ok_fwd

    # ==== Step 6: 1-epoch 微型训练 ====
    banner("Step 6/6: 微型训练 (1 epoch, CPU)")
    ok_train, t_train = run_cmd("overnight_autotrain 1-epoch", [
        PYTHON, "overnight_autotrain.py",
        "--run",
        "--data-dir", DATA_DIR,
        "--ocr-data-dir", OCR_DIR,
        "--font-path", FONT,
        "--out-root", RUN_DIR,
        "--run-name", "feasibility_test",
        "--layer-min", "0",
        "--layer-max", "5",
        "--nest-image-size", "256",
        "--epochs-nesting", "1",
        "--epochs-ocr", "1",
        "--epochs-composite", "1",
        "--batch-size-nesting", "2",
        "--batch-size-ocr", "4",
        "--num-workers", "0",
        "--seed", "42",
        "--device", "cpu",
    ], timeout=600)
    results["training_1epoch"] = ok_train

    # ==== 汇总 ====
    banner("汇总")
    all_pass = True
    for k, v in results.items():
        status = PASS if v else FAIL
        if not v:
            all_pass = False
        print(f"  {status} {k}")

    elapsed_total = time.time() - t_total
    print(f"\n总耗时: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")

    if all_pass:
        print(f"\n{PASS} 全部通过！流水线可行。")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\n{FAIL} 有 {len(failed)} 项失败: {', '.join(failed)}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
