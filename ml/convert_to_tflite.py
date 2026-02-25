#!/usr/bin/env python3
"""
Oskín ML — ONNX → TensorFlow → TFLite Conversion with Quantization

Usage:
    python convert_to_tflite.py --config configs/export.yaml
    python convert_to_tflite.py --config configs/export.yaml --quantization int8 --rep_data_dir data/plantvillage
"""

import argparse
import json
import sys
import os
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from training.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Oskín ML - TFLite Conversion")
    parser.add_argument("--config", type=str, default="configs/export.yaml")
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["float16", "int8", "dynamic", "none"],
        default=None,
        help="Override quantization type from config",
    )
    parser.add_argument(
        "--rep_data_dir",
        type=str,
        default=None,
        help="Representative dataset dir for int8 calibration (required for int8)",
    )
    return parser.parse_args()


def onnx_to_tensorflow(onnx_path: str, tf_saved_model_dir: str):
    import onnx
    from onnx_tf.backend import prepare

    print(f"Loading ONNX model: {onnx_path}")
    onnx_model = onnx.load(onnx_path)

    print("Converting ONNX → TensorFlow SavedModel...")
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_saved_model_dir)
    print(f"TF SavedModel saved: {tf_saved_model_dir}")


def get_representative_dataset(data_dir: str, image_size: int, num_samples: int = 100):
    import tensorflow as tf
    from pathlib import Path as P

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        image_paths.extend(P(data_dir).rglob(ext))
    image_paths = image_paths[:num_samples]

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def representative_dataset_gen():
        for path in image_paths:
            img = tf.io.read_file(str(path))
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, [image_size, image_size])
            img = tf.cast(img, tf.float32) / 255.0
            img = (img - mean) / std
            img = tf.expand_dims(img, axis=0)
            img = tf.transpose(img, perm=[0, 3, 1, 2])
            yield [img]

    return representative_dataset_gen


def tensorflow_to_tflite(
    tf_saved_model_dir: str,
    tflite_path: str,
    image_size: int,
    quantization: str = "float16",
    rep_data_dir: str = None,
):
    import tensorflow as tf

    Path(tflite_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Converting TF SavedModel → TFLite (quantization={quantization})...")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_dir)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if quantization == "float16":
        converter.target_spec.supported_types = [tf.float16]
        print("  Applying float16 quantization")

    elif quantization == "int8":
        if rep_data_dir is None:
            raise ValueError("int8 quantization requires --rep_data_dir")
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        rep_gen = get_representative_dataset(rep_data_dir, image_size)
        converter.representative_dataset = rep_gen
        print("  Applying int8 quantization with representative dataset")

    elif quantization == "dynamic":
        print("  Applying dynamic range quantization")

    elif quantization == "none":
        converter.optimizations = []
        print("  No quantization (float32)")

    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"TFLite model saved: {tflite_path} ({size_mb:.2f} MB)")
    return size_mb


def validate_tflite(tflite_path: str, image_size: int, num_classes: int):
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"\nTFLite model info:")
    print(f"  Input shape:  {input_details[0]['shape']}")
    print(f"  Input dtype:  {input_details[0]['dtype']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    print(f"  Output dtype: {output_details[0]['dtype']}")

    dummy_input = np.random.randn(1, 3, image_size, image_size).astype(np.float32)

    if input_details[0]["dtype"] == np.int8:
        scale, zero_point = input_details[0]["quantization"]
        dummy_input = (dummy_input / scale + zero_point).astype(np.int8)

    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])
    print(f"  Inference test passed. Output shape: {output.shape}")
    assert output.shape[-1] == num_classes or output.shape[-1] == 0, \
        f"Expected {num_classes} outputs, got {output.shape[-1]}"


def benchmark_tflite(tflite_path: str, image_size: int, num_runs: int = 50):
    import tensorflow as tf
    import time

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    dummy_input = np.random.randn(1, 3, image_size, image_size).astype(np.float32)

    if input_details[0]["dtype"] == np.int8:
        scale, zero_point = input_details[0]["quantization"]
        dummy_input = (dummy_input / scale + zero_point).astype(np.int8)

    latencies = []
    for i in range(num_runs + 5):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])
        elapsed = (time.perf_counter() - t0) * 1000
        if i >= 5:
            latencies.append(elapsed)

    print(f"\nTFLite Benchmark ({num_runs} runs):")
    print(f"  Mean latency:   {np.mean(latencies):.2f} ms")
    print(f"  Median latency: {np.median(latencies):.2f} ms")
    print(f"  P95 latency:    {np.percentile(latencies, 95):.2f} ms")
    print(f"  Min latency:    {np.min(latencies):.2f} ms")
    print(f"  Max latency:    {np.max(latencies):.2f} ms")


def main():
    args = parse_args()
    config = load_config(args.config)
    export_cfg = config["export"]

    onnx_path = export_cfg["onnx_path"]
    tflite_path = export_cfg["tflite_path"]
    image_size = export_cfg["image_size"]
    quantization = args.quantization or export_cfg.get("quantization", "float16")

    meta_path = str(Path(onnx_path).parent / "export_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    num_classes = meta["num_classes"]
    class_names = meta["class_names"]

    print("=" * 60)
    print("Oskín ML — TFLite Conversion")
    print("=" * 60)
    print(f"ONNX:         {onnx_path}")
    print(f"TFLite:       {tflite_path}")
    print(f"Quantization: {quantization}")
    print(f"Classes:      {num_classes}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tf_saved_model_dir = os.path.join(tmp_dir, "tf_saved_model")
        onnx_to_tensorflow(onnx_path, tf_saved_model_dir)

        tflite_size = tensorflow_to_tflite(
            tf_saved_model_dir=tf_saved_model_dir,
            tflite_path=tflite_path,
            image_size=image_size,
            quantization=quantization,
            rep_data_dir=args.rep_data_dir,
        )

    validate_tflite(tflite_path, image_size, num_classes)
    benchmark_tflite(tflite_path, image_size)

    tflite_meta = {
        "tflite_path": tflite_path,
        "quantization": quantization,
        "size_mb": tflite_size,
        "image_size": image_size,
        "num_classes": num_classes,
        "class_names": class_names,
        "input_mean": [0.485, 0.456, 0.406],
        "input_std": [0.229, 0.224, 0.225],
        "input_layout": "NCHW",
    }
    tflite_meta_path = str(Path(tflite_path).parent / "tflite_meta.json")
    with open(tflite_meta_path, "w") as f:
        json.dump(tflite_meta, f, indent=2)

    print(f"\nTFLite metadata saved: {tflite_meta_path}")
    print("\nConversion complete!")
    print(f"  Final model: {tflite_path} ({tflite_size:.2f} MB)")


if __name__ == "__main__":
    main()
