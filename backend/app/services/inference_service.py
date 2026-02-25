"""
Oskín — Backend TFLite Inference Service

Mirrors the preprocessing from ml/inference/predict.py exactly:
  - Resize to IMAGE_SIZE x IMAGE_SIZE
  - Normalize with ImageNet mean/std: mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]
  - Input layout: NCHW (1, 3, H, W) float32
  - Softmax output → top-K predictions
"""
import logging
import math
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from app.core.config import settings

logger = logging.getLogger("oskin.inference")

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class InferenceService:
    """Singleton TFLite inference service loaded at app startup."""

    def __init__(self):
        self._interpreter = None
        self._class_names: List[str] = []
        self._loaded = False
        self._model_path = settings.MODEL_PATH
        self._class_names_path = settings.MODEL_CLASS_NAMES_PATH

    def load(self) -> bool:
        if self._loaded:
            return True
        if not Path(self._model_path).exists():
            logger.warning(
                "TFLite model not found at %s — inference endpoint disabled. "
                "Run: cp ml/export/plant_disease.tflite backend/app/models/",
                self._model_path,
            )
            return False
        try:
            import tensorflow as tf
            self._interpreter = tf.lite.Interpreter(model_path=self._model_path)
        except ImportError:
            try:
                import tflite_runtime.interpreter as tflite
                self._interpreter = tflite.Interpreter(model_path=self._model_path)
            except ImportError:
                logger.error("Neither tensorflow nor tflite_runtime is installed.")
                return False

        self._interpreter.allocate_tensors()
        self._load_class_names()
        self._loaded = True
        logger.info(
            "TFLite model loaded: %s (%d classes)",
            self._model_path,
            len(self._class_names),
        )
        return True

    def _load_class_names(self):
        if Path(self._class_names_path).exists():
            with open(self._class_names_path) as f:
                self._class_names = [line.strip() for line in f if line.strip()]
        else:
            input_details = self._interpreter.get_input_details()
            output_details = self._interpreter.get_output_details()
            num_classes = output_details[0]["shape"][-1]
            self._class_names = [f"class_{i}" for i in range(num_classes)]
            logger.warning(
                "class_names.txt not found at %s — using generic class names",
                self._class_names_path,
            )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def num_classes(self) -> int:
        return len(self._class_names)

    def _preprocess(self, image_bytes: bytes) -> np.ndarray:
        from io import BytesIO
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img = img.resize(
            (settings.IMAGE_SIZE, settings.IMAGE_SIZE), Image.Resampling.BILINEAR
        )
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - _MEAN) / _STD
        # HWC → NCHW
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, axis=0)
        return arr

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exps = np.exp(shifted)
        return exps / exps.sum()

    def predict(self, image_bytes: bytes, top_k: int = 3) -> List[dict]:
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()

        input_data = self._preprocess(image_bytes)
        if input_details[0]["dtype"] == np.int8:
            scale, zero_point = input_details[0]["quantization"]
            input_data = (input_data / scale + zero_point).astype(np.int8)

        self._interpreter.set_tensor(input_details[0]["index"], input_data)
        self._interpreter.invoke()
        logits = self._interpreter.get_tensor(output_details[0]["index"])[0]

        if output_details[0]["dtype"] == np.int8:
            scale, zero_point = output_details[0]["quantization"]
            logits = (logits.astype(np.float32) - zero_point) * scale

        probs = self._softmax(logits.astype(np.float32))
        top_indices = np.argsort(probs)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            class_name = (
                self._class_names[idx]
                if idx < len(self._class_names)
                else f"class_{idx}"
            )
            results.append(
                {
                    "rank": rank + 1,
                    "class_index": int(idx),
                    "class_name": class_name,
                    "confidence": float(probs[idx]),
                    "confidence_pct": f"{probs[idx] * 100:.2f}%",
                }
            )
        return results


# Global singleton — initialized at app startup
inference_service = InferenceService()
