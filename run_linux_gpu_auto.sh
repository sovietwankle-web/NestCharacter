#!/usr/bin/env bash
set -Eeuo pipefail

# One-click Linux GPU pipeline:
# 1) Build 3~5 layer dataset using generate_dataset_copypaste.py
# 2) Start overnight_autotrain.py with GPU (no interactive input)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ----------------------------
# Config (override via env var)
# ----------------------------
PYTHON_BIN="${PYTHON_BIN:-python3}"
USE_VENV="${USE_VENV:-1}"                    # 1=create/use .venv, 0=use system python
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"            # 1=install requirements
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"       # e.g. https://download.pytorch.org/whl/cu124
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"            # 1=exit if CUDA unavailable

FONT_PATH="${FONT_PATH:-/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc}"
OUT_DATA_DIR="${OUT_DATA_DIR:-training_data_3to5}"
RUN_ROOT="${RUN_ROOT:-runs}"
RUN_NAME="${RUN_NAME:-linux_auto_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs}"

LAYER_MIN="${LAYER_MIN:-3}"
LAYER_MAX="${LAYER_MAX:-5}"
SAMPLES_PER_CLASS="${SAMPLES_PER_CLASS:-500}"
CANVAS_SIZE="${CANVAS_SIZE:-4200}"           # keep large render, no resize/compression
BASE_LARGE_FONT_SIZE="${BASE_LARGE_FONT_SIZE:-400}"
WRAP_AFTER="${WRAP_AFTER:-10}"
SEED="${SEED:-42}"

EPOCHS_NESTING="${EPOCHS_NESTING:-12}"
EPOCHS_OCR="${EPOCHS_OCR:-10}"
EPOCHS_COMPOSITE="${EPOCHS_COMPOSITE:-8}"
BATCH_SIZE_NESTING="${BATCH_SIZE_NESTING:-1}"  # safer for large canvas
BATCH_SIZE_OCR="${BATCH_SIZE_OCR:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"

DETACH_TRAIN="${DETACH_TRAIN:-1}"            # 1=start train with nohup in background

mkdir -p "$LOG_DIR"
TRAIN_LOG="$LOG_DIR/train_${RUN_NAME}.log"

echo "[1/6] Root: $ROOT_DIR"
echo "[1/6] Run name: $RUN_NAME"

if [[ "$USE_VENV" == "1" ]]; then
  echo "[2/6] Using venv: $VENV_DIR"
  if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  PYTHON_BIN="python"
fi

if [[ "$INSTALL_DEPS" == "1" ]]; then
  echo "[3/6] Installing dependencies..."
  "$PYTHON_BIN" -m pip install -U pip setuptools wheel
  if [[ -n "$TORCH_INDEX_URL" ]]; then
    "$PYTHON_BIN" -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"
  fi
  "$PYTHON_BIN" -m pip install -r requirements.txt
fi

if [[ ! -f "$FONT_PATH" ]]; then
  if command -v fc-list >/dev/null 2>&1; then
    AUTO_FONT="$(fc-list :lang=zh file | head -n 1 | cut -d: -f1 || true)"
    if [[ -n "${AUTO_FONT:-}" && -f "$AUTO_FONT" ]]; then
      FONT_PATH="$AUTO_FONT"
      echo "[4/6] FONT_PATH not found, auto-selected: $FONT_PATH"
    else
      echo "[4/6] ERROR: FONT_PATH not found: $FONT_PATH"
      echo "       Please set FONT_PATH to a valid CJK font."
      exit 1
    fi
  else
    echo "[4/6] ERROR: FONT_PATH not found: $FONT_PATH"
    echo "       Also fc-list is unavailable, cannot auto-detect font."
    exit 1
  fi
else
  echo "[4/6] Font: $FONT_PATH"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[4/6] GPU:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "[4/6] WARNING: nvidia-smi not found."
fi

"$PYTHON_BIN" - <<'PY'
import torch
print(f"[4/6] torch={torch.__version__}, cuda_available={torch.cuda.is_available()}, gpu_count={torch.cuda.device_count()}")
PY

if [[ "$REQUIRE_CUDA" == "1" ]]; then
  "$PYTHON_BIN" - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    print("[4/6] ERROR: CUDA is required but not available.")
    sys.exit(1)
PY
fi

echo "[5/6] Generating dataset (layers ${LAYER_MIN}~${LAYER_MAX})..."
"$PYTHON_BIN" generate_dataset_copypaste.py \
  --font-path "$FONT_PATH" \
  --out-dir "$OUT_DATA_DIR" \
  --layer-min "$LAYER_MIN" \
  --layer-max "$LAYER_MAX" \
  --samples-per-class "$SAMPLES_PER_CLASS" \
  --canvas-size "$CANVAS_SIZE" \
  --base-large-font-size "$BASE_LARGE_FONT_SIZE" \
  --wrap-after "$WRAP_AFTER" \
  --seed "$SEED" \
  --export-debug-json \
  --debug-json-name compare_report.json \
  --export-box-overlay \
  --verbose

echo "[6/6] Starting training..."
TRAIN_CMD=(
  "$PYTHON_BIN" overnight_autotrain.py
  --run
  --data-dir "$OUT_DATA_DIR"
  --ocr-data-dir "$OUT_DATA_DIR/ocr_data"
  --generate-ocr-data
  --font-path "$FONT_PATH"
  --out-root "$RUN_ROOT"
  --run-name "$RUN_NAME"
  --layer-min "$LAYER_MIN"
  --layer-max "$LAYER_MAX"
  --nest-image-size "$CANVAS_SIZE"
  --epochs-nesting "$EPOCHS_NESTING"
  --epochs-ocr "$EPOCHS_OCR"
  --epochs-composite "$EPOCHS_COMPOSITE"
  --batch-size-nesting "$BATCH_SIZE_NESTING"
  --batch-size-ocr "$BATCH_SIZE_OCR"
  --num-workers "$NUM_WORKERS"
  --seed "$SEED"
  --device cuda
)

if [[ "$DETACH_TRAIN" == "1" ]]; then
  nohup "${TRAIN_CMD[@]}" >"$TRAIN_LOG" 2>&1 &
  echo "Training started in background."
  echo "PID: $!"
  echo "Log: $TRAIN_LOG"
  echo "Watch log: tail -f $TRAIN_LOG"
else
  "${TRAIN_CMD[@]}" | tee "$TRAIN_LOG"
fi

echo "Done."
