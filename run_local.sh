#!/usr/bin/env bash
# ============================================================
# Oskín — Local Development Runner (no Docker required)
# Starts Postgres via Docker, runs migrations, seeds DB,
# then starts the FastAPI backend.
# ============================================================
set -euo pipefail

CYAN="\033[36m"; GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; NC="\033[0m"

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Load env ---
if [ -f .env ]; then
  set -a; source .env; set +a
else
  set -a; source .env.example; set +a
fi

DB_USER="${POSTGRES_USER:-oskin}"
DB_PASS="${POSTGRES_PASSWORD:-oskin_pass}"
DB_NAME="${POSTGRES_DB:-oskin_db}"
DB_PORT="${POSTGRES_PORT:-5432}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
LOCAL_DB_URL="postgresql://${DB_USER}:${DB_PASS}@localhost:${DB_PORT}/${DB_NAME}"

# --- 1. Start Postgres ---
info "Starting PostgreSQL via Docker..."
if docker ps --format '{{.Names}}' | grep -q "^oskin_postgres$"; then
  warn "oskin_postgres container already running"
else
  docker run -d \
    --name oskin_postgres \
    --rm \
    -e POSTGRES_USER="$DB_USER" \
    -e POSTGRES_PASSWORD="$DB_PASS" \
    -e POSTGRES_DB="$DB_NAME" \
    -p "${DB_PORT}:5432" \
    postgres:15-alpine \
    >/dev/null
  info "Waiting for Postgres to be ready..."
  until docker exec oskin_postgres pg_isready -U "$DB_USER" -d "$DB_NAME" >/dev/null 2>&1; do
    sleep 1
  done
fi
success "PostgreSQL ready at localhost:${DB_PORT}"

# --- 2. Python venv ---
info "Setting up Python virtualenv..."
cd backend
if [ ! -d .venv ]; then
  python3 -m venv .venv
  success "Created .venv"
fi
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
success "Python dependencies installed"

# --- 3. Alembic migrations ---
info "Running Alembic migrations..."
export DATABASE_URL="$LOCAL_DB_URL"
alembic upgrade head
success "Migrations applied"

# --- 4. Seed database ---
info "Seeding database..."
python -m app.db.seed
success "Database seeded"

# --- 5. Copy model if available ---
if [ -f ../ml/export/plant_disease.tflite ] && [ ! -f ml_models/plant_disease.tflite ]; then
  mkdir -p ml_models
  cp ../ml/export/plant_disease.tflite ml_models/plant_disease.tflite
  info "Model copied from ml/export/ to backend/ml_models/"
fi

# --- 6. Start backend ---
info "Starting Oskín backend on port ${BACKEND_PORT}..."
echo ""
success "Oskín API: http://localhost:${BACKEND_PORT}"
success "Swagger UI: http://localhost:${BACKEND_PORT}/docs"
success "Health:     http://localhost:${BACKEND_PORT}/health"
echo ""
export MODEL_PATH="$(pwd)/ml_models/plant_disease.tflite"
export MODEL_CLASS_NAMES_PATH="$(pwd)/ml_models/class_names.txt"

uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "$BACKEND_PORT" \
  --reload \
  --log-level info

# Cleanup on exit
trap 'info "Stopping Postgres..."; docker stop oskin_postgres 2>/dev/null || true' EXIT
