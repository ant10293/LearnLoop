"""Local Whisper via faster-whisper. Audio is not sent to a cloud provider."""

from __future__ import annotations

from pathlib import Path

WHISPER_MODELS = ("tiny", "base", "small", "medium", "large-v3")
_REPO = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
}


class TranscriptUnavailable(RuntimeError):
    pass


class TranscriptCancelled(RuntimeError):
    pass


def acceleration() -> dict:
    cuda = 0
    try:
        import ctranslate2

        cuda = int(ctranslate2.get_cuda_device_count())
    except Exception:
        cuda = 0
    if cuda > 0:
        return {"device": "cuda", "compute_type": "float16", "backend": "cuda"}
    return {
        "device": "cpu",
        "compute_type": "int8",
        "backend": "cpu",
        "note": "CUDA was not detected. Apple Silicon does not use CUDA in this engine, so inference stays on CPU.",
    }


def model_cache_status() -> list[dict]:
    root = Path.home() / ".cache" / "huggingface" / "hub"
    rows = []
    for name in WHISPER_MODELS:
        folder = _REPO[name].replace("/", "--")
        present = root.exists() and any(root.glob(f"models--{folder}*"))
        rows.append(
            {
                "model": name,
                "downloaded": present,
                "repo": _REPO[name],
                "default": name == "base",
            }
        )
    return rows


def transcribe_file(
    path: Path,
    *,
    model_name: str = "base",
    language: str = "",
    vad: bool = True,
    word_timestamps: bool = False,
    should_stop=None,
) -> dict:
    if model_name not in WHISPER_MODELS:
        raise TranscriptUnavailable(f"Unknown Whisper model {model_name}.")
    if should_stop and should_stop():
        raise TranscriptCancelled("Transcription was canceled before the model loaded.")
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptUnavailable(
            "faster-whisper is not installed. Install it with "
            "backend/.venv/bin/pip install faster-whisper . "
            "Weights download on first use and stay in the Hugging Face cache."
        ) from exc
    accel = acceleration()
    model = WhisperModel(model_name, device=accel["device"], compute_type=accel["compute_type"])
    if should_stop and should_stop():
        raise TranscriptCancelled("Transcription was canceled before speech decoding started.")
    kwargs = {"vad_filter": vad, "word_timestamps": word_timestamps}
    if language:
        kwargs["language"] = language
    note = ""
    try:
        segments, info = model.transcribe(str(path), **kwargs)
        segment_list = list(segments)
    except Exception:
        if not vad:
            raise
        kwargs["vad_filter"] = False
        segments, info = model.transcribe(str(path), **kwargs)
        segment_list = list(segments)
        note = "Voice activity detection was unavailable, so silence was not filtered."
    rows = []
    for index, segment in enumerate(segment_list):
        if should_stop and should_stop():
            raise TranscriptCancelled("Transcription was canceled while segments were being read.")
        text = (segment.text or "").strip()
        if not text:
            continue
        words = []
        if word_timestamps and getattr(segment, "words", None):
            words = [
                {
                    "start": word.start,
                    "end": word.end,
                    "word": word.word,
                    "probability": word.probability,
                }
                for word in segment.words
            ]
        rows.append(
            {
                "local_id": f"Speech_{index + 1}",
                "start": float(segment.start),
                "end": float(segment.end),
                "text": text,
                "confidence": getattr(segment, "avg_logprob", None),
                "words": words,
            }
        )
    return {
        "language": getattr(info, "language", "") or "",
        "language_probability": getattr(info, "language_probability", None),
        "segments": rows,
        "device": accel["device"],
        "compute_type": accel["compute_type"],
        "model": model_name,
        "note": note,
        "vad": kwargs["vad_filter"],
    }
