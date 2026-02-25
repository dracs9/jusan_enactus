#!/usr/bin/env python3
"""
Oskín ML — TFLite Inference Wrapper

Usage:
    python inference/predict.py --image path/to/image.jpg --model export/plant_disease.tflite
    python inference/predict.py --image path/to/image.jpg --model export/plant_disease.tflite --top_k 3
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Oskín ML - Plant Disease Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--model",
        type=str,
        default="export/plant_disease.tflite",
        help="Path to TFLite model",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default=None,
        help="Path to tflite_meta.json (auto-detected if not provided)",
    )
    parser.add_argument("--top_k", type=int, default=3, help="Number of top predictions")
    return parser.parse_args()


def load_tflite_model(model_path: str):
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=model_path)
    except ImportError:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=model_path)

    interpreter.allocate_tensors()
    return interpreter


def preprocess_image(
    image_path: str,
    image_size: int,
    mean: List[float],
    std: List[float],
    input_dtype,
    quantization: Tuple,
) -> np.ndarray:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.BILINEAR)

    img_array = np.array(image, dtype=np.float32) / 255.0
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    img_array = (img_array - mean_arr) / std_arr

    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)

    if input_dtype == np.int8:
        scale, zero_point = quantization
        img_array = (img_array / scale + zero_point).astype(np.int8)
    else:
        img_array = img_array.astype(np.float32)

    return img_array


def run_inference(
    interpreter,
    input_data: np.ndarray,
) -> np.ndarray:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    if output_details[0]["dtype"] == np.int8:
        scale, zero_point = output_details[0]["quantization"]
        output = (output.astype(np.float32) - zero_point) * scale

    return output[0]


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e_x = np.exp(x)
    return e_x / e_x.sum()


class PlantDiseasePredictor:
    def __init__(self, model_path: str, meta_path: str = None):
        self.model_path = model_path
        model_dir = Path(model_path).parent

        if meta_path is None:
            meta_path = str(model_dir / "tflite_meta.json")

        with open(meta_path) as f:
            self.meta = json.load(f)

        self.class_names = self.meta["class_names"]
        self.image_size = self.meta["image_size"]
        self.mean = self.meta.get("input_mean", [0.485, 0.456, 0.406])
        self.std = self.meta.get("input_std", [0.229, 0.224, 0.225])

        self.interpreter = load_tflite_model(model_path)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_dtype = self.input_details[0]["dtype"]
        self.input_quant = self.input_details[0].get("quantization", (1.0, 0))

        print(f"Model loaded: {model_path}")
        print(f"Classes: {len(self.class_names)}")
        print(f"Input: {self.input_details[0]['shape']} ({self.input_dtype.__name__})")

    def predict(self, image_path: str, top_k: int = 3) -> List[Dict]:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        t0 = time.perf_counter()

        input_data = preprocess_image(
            image_path=image_path,
            image_size=self.image_size,
            mean=self.mean,
            std=self.std,
            input_dtype=self.input_dtype,
            quantization=self.input_quant,
        )

        logits = run_inference(self.interpreter, input_data)
        probabilities = softmax(logits)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        top_indices = np.argsort(probabilities)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                "rank": rank + 1,
                "class_name": self.class_names[idx],
                "class_index": int(idx),
                "confidence": float(probabilities[idx]),
                "confidence_pct": f"{probabilities[idx] * 100:.2f}%",
            })

        return results, elapsed_ms

    def benchmark(self, image_path: str, num_runs: int = 50) -> Dict:
        latencies = []
        for i in range(num_runs + 5):
            t0 = time.perf_counter()
            input_data = preprocess_image(
                image_path, self.image_size, self.mean, self.std,
                self.input_dtype, self.input_quant,
            )
            run_inference(self.interpreter, input_data)
            elapsed = (time.perf_counter() - t0) * 1000
            if i >= 5:
                latencies.append(elapsed)

        return {
            "num_runs": num_runs,
            "mean_ms": float(np.mean(latencies)),
            "median_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
        }


def main():
    args = parse_args()

    predictor = PlantDiseasePredictor(
        model_path=args.model,
        meta_path=args.meta,
    )

    print(f"\nRunning inference on: {args.image}")
    results, latency_ms = predictor.predict(args.image, top_k=args.top_k)

    print(f"\nTop-{args.top_k} Predictions (inference: {latency_ms:.1f} ms):")
    print("-" * 60)
    for pred in results:
        bar = "█" * int(pred["confidence"] * 40)
        print(
            f"  {pred['rank']}. {pred['class_name']:<40} "
            f"{pred['confidence_pct']:>8}  {bar}"
        )

    print(f"\nPredicted: {results[0]['class_name']} ({results[0]['confidence_pct']})")


if __name__ == "__main__":
    main()
