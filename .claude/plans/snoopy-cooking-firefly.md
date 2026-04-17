# Composite Training Plan: Nesting Detection + OCR Character Recognition

## Context

Current system only detects **nesting layers** (0-5 classification) via `NestedCharCNN`. The "OCR boxes" are just bounding box coordinates with no character recognition. User wants to add actual Chinese character recognition output via composite multi-task training, with OCR accuracy as the priority.

## Architecture Overview

```
Input Image
    │
    ▼
┌──────────────────────┐
│  Shared ResNet-SE    │  (stem + layer1-4 + global_pool → 512-dim)
│  Backbone            │  ← existing, unchanged
└──────┬───────────────┘
       │ 512-dim features
       ├──────────────────────┐
       ▼                      ▼
┌──────────────┐    ┌──────────────────┐
│ Nesting Head │    │    OCR Head       │
│ 512→256→6    │    │ 512→512→N_chars   │
│ (existing)   │    │ (new)             │
└──────────────┘    └──────────────────┘
```

Key insight: `AdaptiveAvgPool2d((1,1))` makes backbone resolution-agnostic. Nesting task uses 256x256, OCR patches use 64x64 — both produce 512-dim features.

---

## Step 1: Modify `training_data_generator.py` — Add OCR Data Generation

**Changes to `TrainingDataGenerator.__init__`:**
- Add `self.char_vocab: dict` — deterministic char→class_id mapping from existing `char_pool`
- Add `self.id_to_char: dict` — reverse mapping
- Auto-save `char_vocab.json` to `output_dir`

**New methods:**
- `_build_char_vocab() → dict` — assign stable integer ID to each character in `char_pool`
- `generate_ocr_patch(char, size=(64,64), apply_augment=True) → (np.ndarray, int)` — render single character as grayscale 64x64 patch
- `_augment_ocr_patch(image) → np.ndarray` — OCR-specific augmentation (rotation ±15, perspective, stroke width variation via morphology, blur, noise; NO horizontal flip for Chinese)
- `generate_ocr_dataset(samples_per_char=50, ...) → dict` — bulk generate patches for all chars, save to `training_data/ocr_data/{train,val,test}/`

**Filename convention:** `char_{class_id:04d}_{sample_idx:04d}.png`

---

## Step 2: Modify `nested_char_detector.py` — Add Dual-Head Model + OCR Dataset + Detection Enhancement

### 2a. New class `DualHeadNestedCharCNN(nn.Module)`

- Copy backbone from `NestedCharCNN` (stem, layer1-4, global_pool)
- Rename `classifier` → `nesting_head` (512→256→6, unchanged)
- Add `ocr_head`: `Linear(512,512) → ReLU → Dropout(0.4) → Linear(512, num_ocr_classes)`
- `_extract_features(x)` — shared backbone pass → 512-dim vector
- `forward(x, task='nesting')` — routes to appropriate head. `task='both'` returns tuple

Existing `NestedCharCNN` stays **untouched** for backward compatibility.

### 2b. New class `OCRPatchDataset(Dataset)`

- Loads from `char_{class_id}_{idx}.png` files
- Resizes to configurable patch_size (default 64)
- Returns `(tensor, class_id)`

### 2c. Helper functions

- `load_nesting_weights_from_legacy(dual_model, legacy_state_dict)` — maps old `classifier.X` → new `nesting_head.X`
- `freeze_backbone(model)` / `unfreeze_backbone(model)` — for phased training

### 2d. Modify `NestedCharDetector`

- Add `__init__` params: `use_dual_head=False`, `char_vocab_path=None`
- When `use_dual_head=True`: instantiate `DualHeadNestedCharCNN`, load `char_vocab.json` for decoding
- Add `_recognize_chars_in_boxes(gray_image, boxes)` method:
  - Crop each box → pad to square → resize → run through OCR head
  - Return list of `{box, top_chars: [{char, confidence}], text}`
- Update `detect()`: call `_recognize_chars_in_boxes` after box generation, add `ocr_results` and `recognized_text` to result dict
- Update `_tta_inference()` to pass `task='nesting'` when using dual-head model

---

## Step 3: Modify `train_model.py` — Composite Training Loop

**New CLI args:**
- `--mode {nesting|ocr|composite}` — default `nesting` for backward compat
- `--generate-ocr-data` — flag to generate OCR patches
- `--ocr-data-dir` — path to OCR data (default `training_data/ocr_data`)
- `--samples-per-char` — OCR samples per character (default 50)
- `--ocr-loss-weight` — OCR loss multiplier (default 2.0)
- `--nesting-loss-weight` — nesting loss multiplier (default 1.0)

**New function `train_composite()`:**
- Two-phase training:
  - Phase 1 (first 10 epochs): freeze backbone, train OCR head only → stable convergence
  - Phase 2 (remaining epochs): unfreeze all, interleaved batches with composite loss
- Each step: one nesting batch + one OCR batch → combined gradient
- Loss: `L = α * L_nesting + β * L_ocr` (β=2.0 prioritizes OCR)
- Optimizer: AdamW with param groups (backbone base LR, heads 2x LR)
- Scheduler: CosineAnnealingLR
- Gradient clipping: `clip_grad_norm_(1.0)`
- Save best model by weighted combined val loss
- Track all 8 metrics: {train,val} × {nesting,ocr} × {loss,acc}

**Updated `plot_training_history()`:** 4-panel chart (nesting loss, OCR loss, nesting acc, OCR acc)

---

## Step 4: Update `ui_demo.py` — Display OCR Results

- Update detector initialization with `use_dual_head=True` when model exists
- In detection result display: show recognized characters alongside bounding boxes
- Add recognized text string to output

---

## Files Modified

| File | Changes |
|---|---|
| `nested_char_detector.py` | Add `DualHeadNestedCharCNN`, `OCRPatchDataset`, helpers. Extend `NestedCharDetector` with OCR recognition. Keep `NestedCharCNN` unchanged. |
| `training_data_generator.py` | Add `char_vocab`, `generate_ocr_patch()`, `_augment_ocr_patch()`, `generate_ocr_dataset()` |
| `train_model.py` | Add `--mode composite` path, `train_composite()`, updated plotting |
| `ui_demo.py` | Display OCR recognized text in detection results |

## Parameter Impact

- New OCR head: ~615K params (+22% over ~2.8M backbone)
- Checkpoint size: ~11MB → ~14MB

## Verification

1. `python train_model.py --generate-data --generate-ocr-data --mode composite --num-epochs 60`
2. Check training logs: both nesting acc and OCR acc should improve
3. Run `python ui_demo.py` → upload test image → verify recognized characters appear alongside boxes
4. Backward compat: `python train_model.py --mode nesting` still works with original `NestedCharCNN`
