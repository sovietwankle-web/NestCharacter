"""快速定位生成器慢在哪"""
import cProfile, pstats, io, time, os, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TEST_DIR = os.path.join(os.path.dirname(__file__), "_profile_out")
if os.path.exists(TEST_DIR):
    shutil.rmtree(TEST_DIR, ignore_errors=True)

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if not os.path.exists(FONT):
    FONT = "C:/Windows/Fonts/msyh.ttc"

from generate_dataset_copypaste import generate_nested_image
import random, numpy as np
random.seed(0); np.random.seed(0)

pr = cProfile.Profile()
pr.enable()

t0 = time.time()
for layer in [2, 3, 4, 5]:
    for i in range(3):
        result = generate_nested_image(
            layers=layer,
            font_path=FONT,
            canvas_size=1536,
            wrap_after=10,
            base_large_font_size=400,
            depth_scale=0.70,
            small_ratio_min=0.030,
            small_ratio_max=0.045,
        )
print(f"Total: {time.time()-t0:.2f}s for 12 images (4 layers x 3)")

pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
print(s.getvalue())
