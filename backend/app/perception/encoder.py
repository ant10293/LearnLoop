"""Appearance vectors. A cosine is not a probability of identity.

DINOv2 ViT-S/14 is loaded only when that encoder is requested. Histogram
features are a separate, explicitly named descriptor and are never returned
in place of DINOv2.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path

import certifi
import cv2
import numpy as np

from app.perception.types import EncoderUnavailable

# v2: aspect-preserving letterbox, and a supplied mask replaces the background with the ImageNet mean.
PREPROCESS_VERSION = "rgb-letterbox-mask-224-imagenet-v2"
HISTOGRAM_PREPROCESS = "bgr-hist8-l2-v1"
DINOV2_ID = "dinov2_vits14"
DINOV2_VERSION = "facebookresearch/dinov2:dinov2_vits14"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Analyses can run side by side (Reliability Lab). One lock keeps a second copy of the weights from loading,
# and one keeps forward passes on the shared model from interleaving on an accelerator. Frozen weights make
# a serialized pass give the same vectors.
_LOAD_LOCK = threading.Lock()
_ENCODE_LOCK = threading.Lock()


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
        chosen = [crops[index] for index in missing]
        chosen_masks = [masks[index] for index in missing]
        # Encoders that take masks get them. A list of all None keeps the one-argument call.
        computed = encoder.encode(chosen, chosen_masks) if any(item is not None for item in chosen_masks) else encoder.encode(chosen)
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
            # Written aside and moved, so an analysis running beside this one never reads half a vector.
            tmp = path.with_name(f".{path.stem}.{threading.get_ident()}.tmp.npy")
            np.save(tmp, computed[row])
            tmp.replace(path)
    return output


def preprocess(crop: np.ndarray, mask: np.ndarray | None = None, side: int = 224) -> np.ndarray:
    """BGR crop to a normalized 3 x side x side float array.

    The longer side is scaled to `side` and the shorter one padded, so a narrow
    part is not stretched square. A mask the same size as the crop sets every
    pixel off the object, and the padding, to the ImageNet mean, which is zero
    after normalization. Without a mask the background pixels stay.
    """
    if crop.size == 0:
        return np.zeros((3, side, side), dtype=np.float32)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if mask is not None and mask.shape[:2] == crop.shape[:2] and mask.any():
        keep = (mask > 0)[..., None]
        rgb = np.where(keep, rgb, IMAGENET_MEAN[None, None, :])
    height, width = rgb.shape[:2]
    scale = side / max(height, width)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=interpolation)
    canvas = np.empty((side, side, 3), dtype=np.float32)
    canvas[:] = IMAGENET_MEAN
    top, left = (side - new_h) // 2, (side - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    normalized = (canvas - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1), dtype=np.float32)


class HistogramEncoder:
    """24-D color histogram. This is not a pretrained recognition model."""

    encoder_id = "histogram_v1"
    encoder_version = "opencv-hist-8x3"
    preprocess_version = HISTOGRAM_PREPROCESS
    dimension = 24

    def encode(self, crops: list[np.ndarray], masks: list[np.ndarray | None] | None = None) -> np.ndarray:
        rows = []
        for index, crop in enumerate(crops):
            if crop.size == 0:
                rows.append(np.zeros(self.dimension, dtype=np.float32))
                continue
            mask = masks[index] if masks else None
            if mask is not None and mask.shape[:2] == crop.shape[:2] and mask.any():
                mask = (mask > 0).astype(np.uint8)
            else:
                mask = None
            parts = []
            for channel in range(min(3, crop.shape[2])):
                hist = cv2.calcHist([crop], [channel], mask, [8], [0, 256]).flatten()
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

    def __init__(self, batch_size: int = 32):
        self._model = None
        self._torch = None
        self.device = "cpu"
        self.batch_size = max(1, batch_size)

    def encode(self, crops: list[np.ndarray], masks: list[np.ndarray | None] | None = None) -> np.ndarray:
        torch, model = self._load()
        if not crops:
            return np.zeros((0, self.dimension), dtype=np.float32)
        arrays = [preprocess(crop, masks[index] if masks else None) for index, crop in enumerate(crops)]
        rows = []
        for start in range(0, len(arrays), self.batch_size):
            tensor = torch.from_numpy(np.stack(arrays[start : start + self.batch_size])).to(self.device)
            with _ENCODE_LOCK, torch.no_grad():
                features = model(tensor)
            if hasattr(features, "detach"):
                array = features.detach().float().cpu().numpy()
            else:
                array = np.asarray(features, dtype=np.float32)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            rows.append(array)
        array = np.concatenate(rows, axis=0)
        if array.shape[1] != self.dimension:
            self.dimension = int(array.shape[1])
        return l2_normalize(array.astype(np.float32))

    def _load(self):
        if self._model is not None:
            return self._torch, self._model
        with _LOAD_LOCK:
            if self._model is not None:
                return self._torch, self._model
            return self._load_locked()

    def _load_locked(self):
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


_SHARED: dict[str, object] = {}


def shared_encoder(encoder_id: str = DINOV2_ID):
    """One loaded model per process, so a gallery build or a validation pass does not load weights per crop."""
    if encoder_id not in _SHARED:
        with _LOAD_LOCK:
            if encoder_id not in _SHARED:
                _SHARED[encoder_id] = load_encoder(encoder_id)
    return _SHARED[encoder_id]


if __name__ == "__main__":
    # python -m app.perception.encoder: load DINOv2 once so its weights are on disk before a demo.
    encoder = Dinov2Encoder()
    vectors = encoder.encode([np.zeros((32, 32, 3), dtype=np.uint8)])
    print(f"{DINOV2_ID} ready on {encoder.device}, {vectors.shape[1]}-D features, preprocessing {PREPROCESS_VERSION}.")
