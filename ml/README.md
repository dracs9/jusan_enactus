# Oskín ML — Plant Disease Classification Pipeline

Production ML pipeline for classifying plant diseases from field images. Optimized for TFLite mobile inference.

## Architecture

- **Model**: MobileNetV3 Large (pretrained on ImageNet)
- **Input**: 224×224 RGB
- **Training**: Two-stage domain adaptation
- **Export**: ONNX → TensorFlow → TFLite (float16 or int8)

## Directory Structure

```
oskin_ml/
├── configs/
│   ├── stage1_plantvillage.yaml     # Stage 1: pretrain on PlantVillage
│   ├── stage2_finetune.yaml         # Stage 2: fine-tune on field images
│   └── export.yaml                  # Export configuration
├── data/
│   ├── plantvillage/                # Stage 1 dataset (see below)
│   └── kazakhstan_fields/           # Stage 2 dataset (see below)
├── models/
│   ├── model.py                     # MobileNetV3 + custom head
│   ├── class_mapping.json           # Auto-generated during training
│   ├── best_model_stage1.pth
│   └── best_model_stage2.pth
├── training/
│   ├── dataset.py                   # Dataset loader + augmentation
│   ├── trainer.py                   # Training loop, metrics, checkpoints
│   └── utils.py                     # Seed, config, device utils
├── inference/
│   ├── predict.py                   # TFLite inference wrapper
│   └── benchmark.py                 # Latency benchmark
├── export/
│   ├── plant_disease.onnx
│   ├── plant_disease.tflite
│   ├── export_meta.json
│   └── tflite_meta.json
├── train.py
├── evaluate.py
├── export.py
├── convert_to_tflite.py
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset Structure

### PlantVillage (Stage 1)

Download from Kaggle: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

Structure the dataset as:
```
data/plantvillage/
├── Apple___Apple_scab/
│   ├── image1.jpg
│   └── ...
├── Apple___Black_rot/
├── Apple___Cedar_apple_rust/
├── Apple___healthy/
├── Corn___Cercospora_leaf_spot/
├── ...
└── Wheat___Yellow_Rust/
```

Each subdirectory is a class. The class name is the directory name.
PlantVillage has 38 classes across 14 crop species.

### Kazakhstan Field Images (Stage 2)

Collect or label real field photos structured as:
```
data/kazakhstan_fields/
├── healthy/
├── septoria_leaf_blotch/
├── wheat_yellow_rust/
├── fusarium_head_blight/
├── powdery_mildew/
└── root_rot/
```

Minimum recommended: 200+ images per class for fine-tuning.

## Training

### Stage 1 — Train on PlantVillage

```bash
python train.py --config configs/stage1_plantvillage.yaml
```

This will:
- Auto-split data into train/val/test
- Save class mapping to `models/class_mapping.json`
- Save best model to `models/best_model_stage1.pth`
- Save training curves to `models/checkpoints/training_curves.png`

### Stage 2 — Fine-tune on Kazakhstan Field Images

Requires Stage 1 checkpoint to exist.

```bash
python train.py --config configs/stage2_finetune.yaml
```

This will:
- Load Stage 1 weights
- Fine-tune on Kazakhstan field images
- Use smaller learning rate (0.0001 vs 0.001)
- Save best model to `models/best_model_stage2.pth`

### Configuration

Key hyperparameters in YAML configs:

| Parameter | Stage 1 | Stage 2 |
|-----------|---------|---------|
| learning_rate | 0.001 | 0.0001 |
| batch_size | 32 | 16 |
| epochs | 50 | 30 |
| freeze_backbone_epochs | 2 | 0 |
| early_stopping_patience | 10 | 8 |

## Evaluation

```bash
# Evaluate Stage 1 model on test set
python evaluate.py \
    --config configs/stage1_plantvillage.yaml \
    --checkpoint models/best_model_stage1.pth \
    --output_dir evaluation_results/stage1

# Evaluate Stage 2 model
python evaluate.py \
    --config configs/stage2_finetune.yaml \
    --checkpoint models/best_model_stage2.pth \
    --output_dir evaluation_results/stage2
```

Outputs:
- `classification_report.txt` — per-class precision/recall/F1
- `confusion_matrix.png` — visual confusion matrix
- `predictions.npy`, `labels.npy`, `probabilities.npy` — raw outputs

## Export to ONNX

```bash
python export.py --config configs/export.yaml
```

This produces:
- `export/plant_disease.onnx`
- `export/export_meta.json`

## Convert to TFLite

### Float16 Quantization (recommended, ~2x compression, minimal accuracy loss)

```bash
python convert_to_tflite.py --config configs/export.yaml --quantization float16
```

### Dynamic Range Quantization (faster, 4x compression)

```bash
python convert_to_tflite.py --config configs/export.yaml --quantization dynamic
```

### Int8 Quantization (maximum compression, requires representative dataset)

```bash
python convert_to_tflite.py \
    --config configs/export.yaml \
    --quantization int8 \
    --rep_data_dir data/plantvillage
```

Final output: `export/plant_disease.tflite`

## Inference

### Single Image Prediction

```bash
python inference/predict.py \
    --image path/to/leaf.jpg \
    --model export/plant_disease.tflite \
    --top_k 3
```

Output example:
```
Top-3 Predictions (inference: 12.3 ms):
  1. Wheat___Yellow_Rust                        87.42%  ██████████████████████████████████
  2. Wheat___Brown_Rust                          9.31%  ███
  3. Wheat___healthy                             2.14%  
  
Predicted: Wheat___Yellow_Rust (87.42%)
```

### Python API

```python
from inference.predict import PlantDiseasePredictor

predictor = PlantDiseasePredictor(
    model_path="export/plant_disease.tflite",
    meta_path="export/tflite_meta.json",
)

results, latency_ms = predictor.predict("path/to/image.jpg", top_k=3)
for pred in results:
    print(f"{pred['rank']}. {pred['class_name']}: {pred['confidence_pct']}")
```

### Benchmark

```bash
python inference/benchmark.py \
    --model export/plant_disease.tflite \
    --num_runs 100
```

Expected performance (CPU):
- MobileNetV3 float32: ~30-50 ms
- MobileNetV3 float16: ~20-35 ms
- MobileNetV3 dynamic: ~15-25 ms
- MobileNetV3 int8:    ~10-20 ms

## Reproducibility

All experiments use `seed=42` by default. The seed controls:
- Python `random` module
- NumPy random state
- PyTorch random state (CPU and GPU)
- CUDA deterministic mode
- Dataset split shuffling

Override seed in the config YAML: `experiment.seed: 123`

## Metrics

Training reports after each epoch:
- **Loss**: CrossEntropy with label smoothing (0.1)
- **Accuracy**: Top-1 accuracy
- **Macro F1**: Mean F1 across all classes
- **Macro Precision/Recall**

Evaluation additionally reports:
- Per-class F1
- Per-class accuracy
- Confusion matrix

## Data Augmentation

Augmentations simulating real field conditions:
- Random brightness/contrast (±30%)
- Gaussian blur (kernel 3–7)
- Random shadow simulation
- Perspective distortion
- Gaussian noise
- Horizontal flip
- Shift/scale/rotate
- Color jitter
- Coarse dropout (simulates occlusion)
