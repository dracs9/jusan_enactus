#!/usr/bin/env bash
# ============================================================
# Oskín — One-Time Setup Script
# Run once after cloning the repository.
# ============================================================
set -euo pipefail

CYAN="\033[36m"; GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; NC="\033[0m"

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "  ██████╗ ███████╗██╗  ██╗██╗███╗   ██╗"
echo "  ██╔═══██╗██╔════╝██║ ██╔╝██║████╗  ██║"
echo "  ██║   ██║███████╗█████╔╝ ██║██╔██╗ ██║"
echo "  ██║   ██║╚════██║██╔═██╗ ██║██║╚██╗██║"
echo "  ╚██████╔╝███████║██║  ██╗██║██║ ╚████║"
echo "   ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝"
echo "  AgTech Platform — Setup"
echo ""

# --- 1. Check prerequisites ---
info "Checking prerequisites..."
command -v docker    >/dev/null 2>&1 || error "Docker not found. Install from https://docs.docker.com/get-docker/"
command -v docker compose version >/dev/null 2>&1 || \
  command -v docker-compose >/dev/null 2>&1 || \
  error "docker compose not found."
success "Docker available"

# --- 2. Copy .env ---
if [ ! -f .env ]; then
  cp .env.example .env
  success "Created .env from .env.example — review and update secrets before production use"
else
  warn ".env already exists — skipping copy"
fi

# --- 3. Create model directories ---
info "Creating model directories..."
mkdir -p backend/ml_models
mkdir -p mobile_app/assets/models
success "Model directories created"

# --- 4. Check if trained model exists ---
if [ -f ml/export/plant_disease.tflite ]; then
  info "Found trained model — copying to backend and mobile..."
  cp ml/export/plant_disease.tflite backend/ml_models/plant_disease.tflite
  cp ml/export/plant_disease.tflite mobile_app/assets/models/plant_disease.tflite
  if [ -f ml/export/tflite_meta.json ]; then
    cp ml/export/tflite_meta.json backend/ml_models/tflite_meta.json
  fi
  success "Model copied to backend/ml_models/ and mobile_app/assets/models/"
else
  warn "No trained model found at ml/export/plant_disease.tflite"
  warn "The API will start but POST /inference will return 503 until a model is provided."
  warn "To train: see README.md → 'Train the ML Model'"
fi

# --- 5. Copy class_names.txt ---
if [ -f ml/export/export_meta.json ]; then
  python3 -c "
import json, sys
with open('ml/export/export_meta.json') as f:
    meta = json.load(f)
names = meta.get('class_names', [])
if names:
    with open('backend/ml_models/class_names.txt', 'w') as f:
        f.write('\n'.join(names))
    print(f'Wrote {len(names)} class names to backend/ml_models/class_names.txt')
else:
    print('No class names in export_meta.json')
" 2>/dev/null || true
fi

if [ ! -f backend/ml_models/class_names.txt ] && [ -f mobile_app/assets/models/class_names.txt ]; then
  cp mobile_app/assets/models/class_names.txt backend/ml_models/class_names.txt
  success "Copied class_names.txt to backend/ml_models/"
fi

echo ""
success "Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Edit .env and set a strong SECRET_KEY"
echo "  2. Run: bash run_local.sh          (local dev)"
echo "     OR:  docker compose up --build  (Docker)"
echo ""
