"""测实际磁盘占用"""
import os, sys, time, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TEST_DIR = os.path.join(os.path.dirname(__file__), "_disk_test")
if os.path.exists(TEST_DIR):
    shutil.rmtree(TEST_DIR, ignore_errors=True)
os.makedirs(TEST_DIR)

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if not os.path.exists(FONT):
    FONT = "C:/Windows/Fonts/msyh.ttc"

from generate_dataset_copypaste import generate_nested_image
import cv2, random, numpy as np
random.seed(0); np.random.seed(0)

# 原配置：canvas=4200, wrap=10, font=400
CANVAS = 4200
sizes_per_layer = {}
times_per_layer = {}

for layer in [0, 1, 2, 3, 4, 5]:
    t0 = time.time()
    sizes = []
    for i in range(3):
        img = generate_nested_image(
            layers=layer,
            font_path=FONT,
            canvas_size=CANVAS,
            wrap_after=10,
            base_large_font_size=400,
            depth_scale=0.70,
            small_ratio_min=0.030,
            small_ratio_max=0.045,
        )
        if isinstance(img, tuple):
            img = img[0]
        out = os.path.join(TEST_DIR, f"l{layer}_{i}.png")
        cv2.imwrite(out, img)
        sizes.append(os.path.getsize(out))
    times_per_layer[layer] = (time.time()-t0)/3
    sizes_per_layer[layer] = sum(sizes)/len(sizes)/1024  # KB

print(f"\n==== Canvas {CANVAS}x{CANVAS} ====")
print(f"{'Layer':<6}{'Avg size (KB)':<15}{'Avg time (s)':<15}")
for l in [0,1,2,3,4,5]:
    print(f"{l:<6}{sizes_per_layer[l]:<15.1f}{times_per_layer[l]:<15.2f}")

total_per_class = sum(sizes_per_layer.values()) / 6
print(f"\n平均每张: {total_per_class:.0f} KB")
print(f"800/类 × 6 层 = 4800 张 ≈ {4800 * total_per_class / 1024 / 1024:.2f} GB")
print(f"400/类 × 6 层 = 2400 张 ≈ {2400 * total_per_class / 1024 / 1024:.2f} GB")
