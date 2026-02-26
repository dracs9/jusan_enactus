# Оскін Mobile — Flutter App

Production-ready Flutter application for the Oskín AgTech platform.

## Requirements

- Flutter 3.19+ (stable channel)
- Dart 3.0+
- Android SDK 21+ / iOS 12+

## Setup

### 1. Install dependencies

```bash
flutter pub get
```

### 2. Add TFLite model

Copy the exported TFLite model to:
```
assets/models/plant_disease.tflite
```

This model comes from the `oskin_ml` pipeline. Run:
```bash
python convert_to_tflite.py --config configs/export.yaml
cp oskin_ml/export/plant_disease.tflite oskin_mobile/assets/models/
```

### 3. Google Maps API Key

1. Get a Google Maps API key from https://console.cloud.google.com
2. Enable Maps SDK for Android + iOS
3. Replace `YOUR_GOOGLE_MAPS_API_KEY` in:
   - `android/app/src/main/AndroidManifest.xml`
   - `ios/Runner/Info.plist`

### 4. Backend URL

The Flutter app reads its backend URL from a **compile-time env variable** `BACKEND_URL`, which you define in the root `.env` file:

```bash
# .env
BACKEND_URL=http://10.0.2.2:8000   # Android emulator
```

For iOS simulator you can use:

```bash
BACKEND_URL=http://localhost:8000
```

Then run the app from the repo root, loading `.env` and passing it to Flutter:

```bash
set -a; source .env; set +a
cd mobile_app
flutter run --dart-define=BACKEND_URL="${BACKEND_URL}"
```

### 5. Run

See the command above for local development, or pass `--dart-define=BACKEND_URL=...` directly when running/building.

## Architecture

Clean Architecture with feature-based modular structure:

```
lib/
├── core/
│   ├── constants/       # App constants, Hive keys
│   ├── network/         # Dio client, JWT interceptor, error handling
│   ├── router/          # GoRouter with auth guards
│   ├── theme/           # Colors, typography, theme
│   ├── utils/           # Extensions
│   └── widgets/         # Shared widgets (OskinButton, OskinCard, MainScaffold)
├── features/
│   ├── auth/            # Login, Register, JWT token management
│   ├── home/            # Dashboard with quick scan + risk summary
│   ├── detection/       # TFLite inference, camera/gallery picker
│   ├── disease/         # Disease knowledge base with offline cache
│   ├── marketplace/     # Products, suppliers, search filter
│   ├── cart/            # Cart state (Hive persistent), checkout
│   ├── field_map/       # Google Maps polygon field drawing + saving
│   ├── risk/            # Weather-based risk score visualization
│   ├── calculator/      # ROI economic calculator
│   ├── chat/            # AI agronomist chat with fallback responses
│   ├── history/         # Scan history list
│   └── profile/         # Language switch, logout
└── l10n/                # Russian + Kazakh localizations
```

## State Management

- **Riverpod** (`flutter_riverpod`) — all state management
- `FutureProvider` — async data fetching
- `StateNotifierProvider` — mutable state (auth, cart, chat, detection)
- `Provider` — services and computed values

## Key Features

### Disease Detection (TFLite)
- Camera or gallery image input
- MobileNetV3 model (224×224 input)
- ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
- Top-3 predictions with confidence bars

### Offline Support
- Diseases cached in Hive
- Products cached in Hive
- Cart persisted in Hive
- TFLite detection works without internet

### Auth
- JWT access + refresh tokens in FlutterSecureStorage
- Auto-refresh via Dio interceptor
- Auto-login on app start via `authStateProvider`
- GoRouter redirect guard

### Localization
- Russian (ru) — default
- Kazakh (kk)
- Switchable from Profile screen, persisted in Hive

## Fonts

The app uses Nunito font. Add font files to `assets/fonts/`:
- `Nunito-Regular.ttf`
- `Nunito-SemiBold.ttf`
- `Nunito-Bold.ttf`
- `Nunito-ExtraBold.ttf`

Download from: https://fonts.google.com/specimen/Nunito

Or remove font references from `pubspec.yaml` to use system fonts.
