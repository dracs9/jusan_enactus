# Jusan — AgTech Platform

Production-grade AgTech system for Kazakhstan farmers.
Plant disease detection via mobile camera, field management, marketplace, risk assessment, economic calculator, and AI agronomist.

---

## Architecture

```
oskin/
├── backend/          FastAPI + PostgreSQL + SQLAlchemy + Alembic
├── ml/               PyTorch training → ONNX → TFLite export pipeline
├── mobile_app/       Flutter (Riverpod + GoRouter + TFLite on-device)
├── docker-compose.yml
├── .env.example
├── setup.sh          One-time project setup
└── run_local.sh      Local dev runner
```

**Inference flow:**
- **On-device (mobile):** `mobile_app` runs TFLite model locally via `tflite_flutter`
- **Server-side:** `POST /inference` on backend uses the same TFLite model via TensorFlow Lite runtime

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Docker + Compose | 24+ | Backend + DB |
| Flutter | 3.19+ (stable) | Mobile app |
| Python | 3.11+ | ML pipeline |
| Dart | 3.0+ | (included with Flutter) |

---

## Quick Start (Docker)

```bash
# 1. Clone and setup
git clone <repo>
cd oskin
bash setup.sh

# 2. Start backend + DB
docker compose up --build

# API:      http://localhost:8000
# Swagger:  http://localhost:8000/docs
# Health:   http://localhost:8000/health

# 3. (Optional) Start Adminer DB UI
docker compose --profile tools up adminer
# Adminer: http://localhost:8080
```

---

## Local Development (without Docker for backend)

```bash
bash run_local.sh
```

This script:
1. Starts a Postgres container
2. Creates Python venv and installs deps
3. Runs Alembic migrations
4. Seeds the database with diseases, suppliers, products
5. Starts uvicorn with hot-reload

---

## Train the ML Model

### 1. Install ML dependencies
```bash
cd ml
pip install -r requirements.txt
```

### 2. Download PlantVillage dataset
```bash
# From Kaggle: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
# Structure:
# ml/data/plantvillage/
#   Apple___Apple_scab/
#   Apple___Black_rot/
#   ...
```

### 3. Stage 1 — Train on PlantVillage
```bash
cd ml
python train.py --config configs/stage1_plantvillage.yaml
# Model saved to: ml/models/best_model_stage1.pth
# Class mapping:  ml/models/class_mapping.json
```

### 4. Stage 2 — Fine-tune on Kazakhstan field images
```bash
# Add field images to ml/data/kazakhstan_fields/<class_name>/
python train.py --config configs/stage2_finetune.yaml
# Model saved to: ml/models/best_model_stage2.pth
```

### 5. Evaluate
```bash
python evaluate.py \
  --config configs/stage2_finetune.yaml \
  --checkpoint models/best_model_stage2.pth \
  --output_dir evaluation_results/
```

---

## Export the Model

### Export PyTorch → ONNX
```bash
cd ml
python export.py --config configs/export.yaml
# Output: ml/export/plant_disease.onnx
```

### Convert ONNX → TFLite (float16 quantization)
```bash
python convert_to_tflite.py --config configs/export.yaml --quantization float16
# Output: ml/export/plant_disease.tflite
```

### Benchmark inference
```bash
python inference/benchmark.py --model export/plant_disease.tflite --num_runs 100
```

---

## Copy Model to Backend and Mobile

```bash
# From project root:
bash ml/scripts/copy_model.sh
```

This copies:
- `ml/export/plant_disease.tflite` → `backend/ml_models/`
- `ml/export/plant_disease.tflite` → `mobile_app/assets/models/`
- `ml/export/export_meta.json`     → extracts `class_names.txt` → `backend/ml_models/`

Or manually:
```bash
cp ml/export/plant_disease.tflite backend/ml_models/plant_disease.tflite
cp ml/export/plant_disease.tflite mobile_app/assets/models/plant_disease.tflite
```

---

## Run the Backend

### Docker (recommended)
```bash
docker compose up --build
```

### Local
```bash
cd backend
pip install -r requirements.txt
alembic upgrade head
python -m app.db.seed
uvicorn app.main:app --reload --port 8000
```

### Environment variables
Copy `.env.example` to `.env` and edit:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection | `postgresql://oskin:oskin_pass@db:5432/oskin_db` |
| `SECRET_KEY` | JWT signing key — **change in prod** | `change_me_in_production` |
| `BACKEND_PORT` | API port | `8000` |
| `MODEL_PATH` | TFLite model path | `/app/ml_models/plant_disease.tflite` |
| `MODEL_VERSION` | Model version string | `1.0.0` |

---

## Run the Mobile App

### Prerequisites
```bash
flutter pub get
```

Add Google Maps API key:
- **Android:** `mobile_app/android/app/src/main/AndroidManifest.xml` → replace `YOUR_GOOGLE_MAPS_API_KEY`
- **iOS:** `mobile_app/ios/Runner/Info.plist` → replace `YOUR_GOOGLE_MAPS_API_KEY`

### Run (development)
```bash
cd mobile_app

# Android emulator (backend at 10.0.2.2:8000)
flutter run

# iOS simulator (backend at localhost:8000)
flutter run --dart-define=BACKEND_URL=http://localhost:8000

# Physical device (replace with your machine's IP)
flutter run --dart-define=BACKEND_URL=http://192.168.1.100:8000
```

### Build release APK
```bash
flutter build apk --release --dart-define=BACKEND_URL=https://api.oskin.kz
```

### Fonts
Download Nunito from https://fonts.google.com/specimen/Nunito and place in `mobile_app/assets/fonts/`:
- `Nunito-Regular.ttf`
- `Nunito-SemiBold.ttf`
- `Nunito-Bold.ttf`
- `Nunito-ExtraBold.ttf`

Or remove the font block from `pubspec.yaml` to use system fonts.

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/health` | No | Health check + model status |
| `GET` | `/model/version` | No | Model version info |
| `POST` | `/auth/register` | No | Register user |
| `POST` | `/auth/login` | No | Login → JWT tokens |
| `POST` | `/auth/refresh` | No | Refresh access token |
| `GET` | `/auth/me` | JWT | Current user |
| `POST` | `/inference` | JWT | **Image → top-3 disease predictions** |
| `GET/POST` | `/fields` | JWT | Field management |
| `GET` | `/diseases` | JWT | Disease knowledge base |
| `GET` | `/diseases/{id}` | JWT | Disease detail |
| `GET/POST` | `/scans` | JWT | Scan history |
| `GET` | `/suppliers` | JWT | Supplier list |
| `GET` | `/products` | JWT | Product catalog |
| `POST` | `/orders` | JWT | Create order |
| `POST` | `/orders/{id}/pay` | JWT | Mock Kaspi QR payment |
| `GET` | `/weather/{field_id}` | JWT | Weather data |
| `GET` | `/risk/{field_id}` | JWT | Risk assessment |
| `POST` | `/calculator/roi` | JWT | ROI calculation |
| `POST` | `/chat` | JWT | AI agronomist |

Full interactive docs: `http://localhost:8000/docs`

---

## Test Inference

### Via HTTP (curl)
```bash
# Health check
curl http://localhost:8000/health

# Login to get token
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@oskin.kz","password":"test1234"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# Run inference on a leaf image
curl -X POST http://localhost:8000/inference \
  -H "Authorization: Bearer $TOKEN" \
  -F "image=@/path/to/leaf.jpg" \
  | python3 -m json.tool
```

### Via ML CLI
```bash
cd ml
python inference/predict.py \
  --image /path/to/leaf.jpg \
  --model export/plant_disease.tflite \
  --top_k 3
```

### Expected response from POST /inference
```json
{
  "predictions": [
    {"rank": 1, "class_name": "Wheat___Yellow_Rust", "confidence": 0.874, "confidence_pct": "87.40%", "class_index": 34},
    {"rank": 2, "class_name": "Wheat___Brown_Rust",  "confidence": 0.091, "confidence_pct": "9.10%",  "class_index": 33},
    {"rank": 3, "class_name": "Wheat___healthy",      "confidence": 0.021, "confidence_pct": "2.10%",  "class_index": 35}
  ],
  "top_class": "Wheat___Yellow_Rust",
  "top_confidence": 0.874,
  "model_version": "1.0.0",
  "scan_id": 42
}
```

---

## Integration Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  mobile_app (Flutter)                                         │
│  ┌──────────────────┐    ┌─────────────────────────────────┐ │
│  │ TFLite (on-device)│    │ Dio HTTP client                 │ │
│  │ plant_disease.    │    │ JWT Bearer token                │ │
│  │   tflite         │    │ baseUrl = BACKEND_URL env var   │ │
│  └──────────────────┘    └───────────────┬─────────────────┘ │
└──────────────────────────────────────────┼─────────────────────┘
                                           │ HTTP/REST
┌──────────────────────────────────────────┼─────────────────────┐
│  backend (FastAPI)                        ▼                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ POST /inference ← PIL image → TFLite runtime → top-3   │   │
│  │ GET  /health    ← model_loaded status                   │   │
│  │ POST /auth/login, /register, /refresh → JWT tokens      │   │
│  │ CRUD /fields, /scans, /diseases, /products, /orders     │   │
│  │ POST /chat ← rule-based AI agronomist                   │   │
│  └──────────────────────────────┬──────────────────────────┘   │
│  ┌───────────────────────────────▼──────────────────────────┐  │
│  │ PostgreSQL 15  (SQLAlchemy 2.0 + Alembic migrations)     │  │
│  └─────────────────────────────────────────────────────────-┘  │
│  ┌─────────────────────────────────────────────────────────-┐  │
│  │ ml_models/                                               │  │
│  │   plant_disease.tflite  ← copied by ml/scripts/copy_model│  │
│  │   class_names.txt       ← extracted from export_meta.json│  │
│  └─────────────────────────────────────────────────────────-┘  │
└────────────────────────────────────────────────────────────────┘
                                           ▲
┌──────────────────────────────────────────┼─────────────────────┐
│  ml (PyTorch training pipeline)           │                     │
│  Stage 1: PlantVillage pretraining        │                     │
│  Stage 2: Kazakhstan field fine-tuning    │                     │
│  Export:  PyTorch → ONNX → TFLite        │                     │
│  Output:  export/plant_disease.tflite ───┘ copy_model.sh       │
└────────────────────────────────────────────────────────────────┘
```

---

## Production Notes

- Set a strong `SECRET_KEY` (run `openssl rand -hex 32`)
- Replace `allow_origins=["*"]` in `backend/app/main.py` with your domain
- Add your Google Maps API key before building mobile release
- Use `docker compose --profile tools up adminer` for DB admin UI
- The `model_storage` Docker volume persists the TFLite model across container restarts
