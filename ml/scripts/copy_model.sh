#!/usr/bin/env bash
# ============================================================
# Oskín ML — Copy Trained Model to Backend & Mobile
# Run from project root: bash ml/scripts/copy_model.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TFLITE_SRC="$PROJECT_ROOT/ml/export/plant_disease.tflite"
META_SRC="$PROJECT_ROOT/ml/export/export_meta.json"
TFLITE_META_SRC="$PROJECT_ROOT/ml/export/tflite_meta.json"
BACKEND_DEST="$PROJECT_ROOT/backend/ml_models"
MOBILE_DEST="$PROJECT_ROOT/mobile_app/assets/models"

if [ ! -f "$TFLITE_SRC" ]; then
  echo "[ERROR] Model not found: $TFLITE_SRC"
  echo "        Run the ML export pipeline first:"
  echo "        cd ml && python export.py && python convert_to_tflite.py"
  exit 1
fi

mkdir -p "$BACKEND_DEST" "$MOBILE_DEST"

cp "$TFLITE_SRC" "$BACKEND_DEST/plant_disease.tflite"
cp "$TFLITE_SRC" "$MOBILE_DEST/plant_disease.tflite"
echo "[OK] Copied plant_disease.tflite"

if [ -f "$TFLITE_META_SRC" ]; then
  cp "$TFLITE_META_SRC" "$BACKEND_DEST/tflite_meta.json"
  echo "[OK] Copied tflite_meta.json"
fi

if [ -f "$META_SRC" ]; then
  python3 -c "
import json
with open('$META_SRC') as f:
    meta = json.load(f)
names = meta.get('class_names', [])
if names:
    dest = '$BACKEND_DEST/class_names.txt'
    with open(dest, 'w') as f:
        f.write('\n'.join(names))
    print(f'[OK] Wrote {len(names)} class names to {dest}')
else:
    print('[WARN] No class_names in export_meta.json')
"
fi

MODEL_SIZE=$(du -sh "$TFLITE_SRC" | cut -f1)
echo ""
echo "[OK] Model deployment complete"
echo "     Size:    $MODEL_SIZE"
echo "     Backend: $BACKEND_DEST/plant_disease.tflite"
echo "     Mobile:  $MOBILE_DEST/plant_disease.tflite"
