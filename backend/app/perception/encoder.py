"""Appearance vectors. A cosine is not a probability of identity.

DINOv2 ViT-S/14 is loaded only when that encoder is requested. Histogram
features are a separate, explicitly named descriptor and are never returned
in place of DINOv2.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import certifi
import cv2
import numpy as np

from app.perception.types import EncoderUnavailable

PREPROCESS_VERSION = "bgr-mask-224-imagenet-v1"
HISTOGRAM_PREPROCESS = "bgr-hist8-l2-v1"
DINOV2_ID = "dinov2_vits14"
DINOV2_VERSION = "facebookresearch/dinov2:dinov2_vits14"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def cache_key(crop: np.ndarray, mask: np.ndarray | None, encoder_id: str, encoder_version: str, preprocess: str) -> str:
    digest = hashlib.sha256()
    digest.update(encoder_id.encode())
    digest.update(b"\0")
    digest.update(encoder_version.encode())
    digest.update(b"\0")
    digest.update(preprocess.encode())
    digest.update(b"\0")
    digest.update(str(crop.shape).encode())
    digest.update(np.ascontiguousarray(crop).tobytes())
    if mask is None:
        digest.update(b"nomask")
    else:
        digest.update(np.ascontiguousarray(mask).tobytes())
    return digest.hexdigest()


def cached_encode(encoder, crops: list[np.ndarray], masks: list[np.ndarray | None], root: Path | None) -> np.ndarray:
    if not crops:
        return np.zeros((0, encoder.dimension), dtype=np.float32)
    keys = [
        cache_key(crop, mask, encoder.encoder_id, encoder.encoder_version, encoder.preprocess_version)
        for crop, mask in zip(crops, masks, strict=True)
    ]
    output = np.zeros((len(crops), encoder.dimension), dtype=np.float32)
    missing: list[int] = []
    for index, key in enumerate(keys):
        if root is None:
            missing.append(index)
            continue
        path = root / key[:2] / f"{key}.npy"
        if path.exists():
            loaded = np.load(path)
            if loaded.shape == (encoder.dimension,):
                output[index] = loaded
                continue
        missing.append(index)
    if missing:
        computed = encoder.encode([crops[index] for index in missing])
        if computed.shape != (len(missing), encoder.dimension):
            raise EncoderUnavailable(
                f"{encoder.encoder_id} returned shape {computed.shape}, "
                f"expected ({len(missing)}, {encoder.dimension})."
            )
        for row, index in enumerate(missing):
            output[index] = computed[row]
            if root is None:
                continue
            path = root / keys[index][:2] / f"{keys[index]}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, computed[row])
    return output


class HistogramEncoder:
    """24-D color histogram. This is not a pretrained recognition model."""

    encoder_id = "histogram_v1"
    encoder_version = "opencv-hist-8x3"
    preprocess_version = HISTOGRAM_PREPROCESS
    dimension = 24

    def encode(self, crops: list[np.ndarray]) -> np.ndarray:
        rows = []
        for crop in crops:
            if crop.size == 0:
                rows.append(np.zeros(self.dimension, dtype=np.float32))
                continue
            parts = []
            for channel in range(min(3, crop.shape[2])):
                hist = cv2.calcHist([crop], [channel], None, [8], [0, 256]).flatten()
                parts.append(hist)
            vector = np.concatenate(parts).astype(np.float32)
            if vector.shape[0] < self.dimension:
                vector = np.pad(vector, (0, self.dimension - vector.shape[0]))
            rows.append(vector[: self.dimension])
        return l2_normalize(np.stack(rows).astype(np.float32))


class Dinov2Encoder:
    """Facebook DINOv2 ViT-S/14. Weights download on the first encode call."""

    encoder_id = DINOV2_ID
    encoder_version = DINOV2_VERSION
    preprocess_version = PREPROCESS_VERSION
    dimension = 384

    def __init__(self):
        self._model = None
        self._torch = None
        self.device = "cpu"

    def encode(self, crops: list[np.ndarray]) -> np.ndarray:
        torch, model = self._load()
        batch = []
        for crop in crops:
            if crop.size == 0:
                batch.append(torch.zeros(3, 224, 224))
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
            array = resized.astype(np.float32) / 255.0
            array = (array - IMAGENET_MEAN) / IMAGENET_STD
            batch.append(torch.from_numpy(array.transpose(2, 0, 1)))
        tensor = torch.stack(batch).to(self.device)
        with torch.no_grad():
            features = model(tensor)
        if hasattr(features, "detach"):
            array = features.detach().float().cpu().numpy()
        else:
            array = np.asarray(features, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[1] != self.dimension:
            self.dimension = int(array.shape[1])
        return l2_normalize(array.astype(np.float32))

    def _load(self):
        if self._model is not None:
            return self._torch, self._model
        try:
            import torch
        except ImportError as exc:
            raise EncoderUnavailable(
                "DINOv2 was requested, but PyTorch is not installed. "
                "Histogram features were not substituted. Install the perception extra to enable it."
            ) from exc
        device = "cpu"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        # torch.hub downloads through urllib, which has no CA bundle on python.org
        # builds and reports the failed verification as "no internet connection".
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        try:
            # The repo is fixed here. The default trust check would wait on stdin.
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True)
        except Exception as exc:
            raise EncoderUnavailable(
                f"DINOv2 weights could not be loaded ({exc}). No other encoder was substituted."
            ) from exc
        model.eval()
        model.to(device)
        self._torch = torch
        self._model = model
        self.device = device
        return torch, model


def probe_encoder(encoder_id: str) -> dict:
    """Report availability without downloading weights."""
    if encoder_id in {None, ""}:
        return {"encoder_id": None, "available": False, "detail": "No encoder was requested."}
    if encoder_id == HistogramEncoder.encoder_id:
        return {
            "encoder_id": encoder_id,
            "available": True,
            "device": "cpu",
            "weights_loaded": False,
            "detail": "OpenCV color histogram. Not a pretrained recognition model.",
        }
    if encoder_id == DINOV2_ID:
        try:
            import torch
        except ImportError:
            return {
                "encoder_id": encoder_id,
                "available": False,
                "device": None,
                "weights_loaded": False,
                "detail": "PyTorch is not installed. DINOv2 was not loaded and no substitute encoder ran.",
            }
        device = "cpu"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        return {
            "encoder_id": encoder_id,
            "available": True,
            "device": device,
            "weights_loaded": False,
            "detail": (
                "DINOv2 ViT-S/14 can be loaded on first use. "
                "Weights are not downloaded until an appearance experiment encodes a crop."
            ),
        }
    return {
        "encoder_id": encoder_id,
        "available": False,
        "detail": f"Encoder '{encoder_id}' is not registered. No substitute was selected.",
    }


def load_encoder(encoder_id: str | None):
    if not encoder_id:
        return None
    if encoder_id == HistogramEncoder.encoder_id:
        return HistogramEncoder()
    if encoder_id == DINOV2_ID:
        status = probe_encoder(encoder_id)
        if not status["available"]:
            raise EncoderUnavailable(status["detail"])
        return Dinov2Encoder()
    raise EncoderUnavailable(f"Encoder '{encoder_id}' is not registered. No substitute was selected.")
