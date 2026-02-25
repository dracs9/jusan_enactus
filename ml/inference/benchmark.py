#!/usr/bin/env python3
"""
Oskín ML — Mobile Inference Benchmark

Usage:
    python inference/benchmark.py --model export/plant_disease.tflite --num_runs 100
"""

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="TFLite Inference Benchmark")
    parser.add_argument("--model", type=str, default="export/plant_disease.tflite")
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1])
    return parser.parse_args()


def benchmark_tflite(model_path: str, image_size: int, num_runs: int, batch_size: int = 1) -> dict:
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=model_path)
    except ImportError:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=model_path)

    interpreter.resize_tensor_input(
        interpreter.get_input_details()[0]["index"],
        [batch_size, 3, image_size, image_size],
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_dtype = input_details[0]["dtype"]

    dummy = np.random.randn(batch_size, 3, image_size, image_size).astype(np.float32)
    if input_dtype == np.int8:
        scale, zero_point = input_details[0]["quantization"]
        dummy = (dummy / scale + zero_point).astype(np.int8)

    latencies = []
    for i in range(num_runs + 10):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], dummy)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if i >= 10:
            latencies.append(elapsed_ms)

    latencies = np.array(latencies)
    return {
        "batch_size": batch_size,
        "num_runs": num_runs,
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "std_ms": float(np.std(latencies)),
        "p90_ms": float(np.percentile(latencies, 90)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "fps": float(batch_size * 1000.0 / np.mean(latencies)),
    }


def main():
    args = parse_args()
    model_path = args.model

    meta_path = str(Path(model_path).parent / "tflite_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    image_size = meta["image_size"]
    num_classes = meta["num_classes"]
    quantization = meta.get("quantization", "unknown")
    size_mb = meta.get("size_mb", Path(model_path).stat().st_size / 1024 / 1024)

    print("=" * 60)
    print("Oskín ML — TFLite Inference Benchmark")
    print("=" * 60)
    print(f"Model:        {model_path}")
    print(f"Size:         {size_mb:.2f} MB")
    print(f"Quantization: {quantization}")
    print(f"Classes:      {num_classes}")
    print(f"Input:        {image_size}x{image_size}")
    print(f"Runs:         {args.num_runs}")
    print()

    all_results = []
    for bs in args.batch_sizes:
        print(f"Benchmarking batch_size={bs}...")
        result = benchmark_tflite(model_path, image_size, args.num_runs, bs)
        all_results.append(result)

        print(f"  Mean:   {result['mean_ms']:.2f} ms")
        print(f"  Median: {result['median_ms']:.2f} ms")
        print(f"  P95:    {result['p95_ms']:.2f} ms")
        print(f"  P99:    {result['p99_ms']:.2f} ms")
        print(f"  FPS:    {result['fps']:.1f}")
        print()

    benchmark_output = {
        "model": model_path,
        "quantization": quantization,
        "size_mb": size_mb,
        "image_size": image_size,
        "results": all_results,
    }
    out_path = str(Path(model_path).parent / "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(benchmark_output, f, indent=2)
    print(f"Benchmark results saved: {out_path}")


if __name__ == "__main__":
    main()
