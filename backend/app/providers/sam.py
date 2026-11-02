"""SAM 3.1 segmentation and video-tracking adapter.

Requests follow the Meta Model API Responses format documented at
https://dev.meta.ai/docs/sam/segmenting. Output follows the special-token
grammar at https://dev.meta.ai/docs/sam/reading-segmentation.

Image segmentation and video tracking are both implemented here. Mask rasters
are decoded with meta_sam_parser when that package is installed. A decode
failure keeps the raw payload and does not invent a mask.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import AsyncOpenAI

from app.providers.common import data_url, translate
from app.providers.meta_provider import META_BASE
from app.providers.types import ProviderError, provider_timeout

# Documented model limits. Chunk size is this lab's budget for a reliable short clip,
# not a second official limit. The API accepts up to 15,000 frames per request.
OFFICIAL_MAX_FRAMES = 15_000
DEFAULT_CHUNK_FRAMES = 300
MAX_PROMPTED_OBJECTS = 16
MAX_TRACKED_OBJECTS = 16
SUPPORTED_CODECS = frozenset({"h264", "avc", "hevc", "h265", "mpeg4"})
SUPPORTED_CONTAINERS = frozenset({"mp4", "mov", "qt"})
PROPAGATION_DIRECTIONS = frozenset({"both", "forward", "backward"})

BOX_RE = re.compile(
    r"(?P<id>\d+)<\|box;(?P<body>[^|]+)\|><\|mask;(?P<mask>[^|]+)\|>"
)
FRAME_RE = re.compile(r"^<(?P<frame>\d+)f>(?P<body>.*)$")
FRAME_MARK_RE = re.compile(r"<\d+f>")
TOKEN_RE = re.compile(r"<\|(?:box|point|mask);")


@dataclass
class SamImageResult:
    text: str
    records: list["SamRecord"]
    latency_ms: int
    request_id: str
    input_tokens: int
    output_tokens: int
    file_id: str = ""


@dataclass
class SamRecord:
    frame_index: int
    track_id: str
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    height: int
    confidence: float | None
    mask_payload: str
    mask_encoding: str
    mask_height: int = 0
    mask_width: int = 0
    mask_data: str = ""


@dataclass
class SpatialMark:
    """One box or point in source pixels. x2/y2 are inclusive, matching the API."""

    frame_index: int
    object_id: int
    kind: str
    width: int
    height: int
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    x: int = 0
    y: int = 0
    negative: bool = False


@dataclass
class VideoChunk:
    index: int
    start_frame: int
    end_frame: int
    time_start: float
    time_end: float
    session_id: str
    # Source presentation time of each frame in the chunk, in order. Empty when
    # only a nominal frame rate is known.
    frame_times: list[float] = field(default_factory=list)

    @property
    def frame_count(self) -> int:
        return max(self.end_frame - self.start_frame, 0)


@dataclass
class SamLineBuffer:
    """Join stream deltas that split a frame line, including mid-payload."""

    pending: str = ""
    records: list[SamRecord] = field(default_factory=list)

    def push(self, delta: str) -> list[SamRecord]:
        if not delta:
            return []
        self.pending += delta
        emitted: list[SamRecord] = []
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            found = parse_sam_output(line)
            self.records.extend(found)
            emitted.extend(found)
        return emitted

    def finish(self) -> list[SamRecord]:
        if not self.pending.strip():
            self.pending = ""
            return []
        found = parse_sam_output(self.pending)
        self.pending = ""
        self.records.extend(found)
        return found


def sam_video_accepted(codec: str, container: str) -> bool:
    """MP4 and MOV with H.264, H.265, or MPEG-4. VP9 and WebM are rejected."""
    return codec.lower() in SUPPORTED_CODECS and container.lower() in SUPPORTED_CONTAINERS


def sam_capabilities() -> dict:
    """Capabilities this adapter actually implements."""
    parser = _parser_available()
    return {
        "model_id": "sam-3.1",
        "provider_id": "meta",
        "image_segmentation": True,
        "image_concept_prompts": True,
        "box_prompts": True,
        "point_prompts": True,
        "video_tracking": True,
        "streaming": True,
        "file_upload": True,
        "propagation_directions": sorted(PROPAGATION_DIRECTIONS),
        "video_formats": ["mp4", "mov"],
        "video_codecs": ["h264", "h265", "mpeg4"],
        "max_frames_per_request": OFFICIAL_MAX_FRAMES,
        "chunk_frames": DEFAULT_CHUNK_FRAMES,
        "max_tracked_objects": MAX_TRACKED_OBJECTS,
        "max_prompted_objects": MAX_PROMPTED_OBJECTS,
        "mask_raster": parser,
        "video_tracking_detail": (
            "The adapter uploads or references a video, sends one concept or one set of "
            "box and point prompts, streams output_text, and parses one line per frame. "
            "Object ids are stable only inside that response. Live Meta API calls are not "
            "part of the default test suite. Mask rasters are decoded with meta-sam-parser "
            "when it is installed; otherwise the raw payload is stored and no mask is invented."
        ),
    }


def scoped_track_id(session_id: str, raw_track_id: str) -> str:
    """SAM ids restart on every request. Scope them to the invocation."""
    session = session_id.strip() or "session"
    raw = str(raw_track_id).strip()
    return f"{session}:{raw}"


def parse_sam_output(text: str) -> list[SamRecord]:
    """Parse complete frame lines. A partial line yields nothing."""
    records: list[SamRecord] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        frame_match = FRAME_RE.match(line)
        if not frame_match:
            continue
        frame_index = int(frame_match.group("frame"))
        for match in BOX_RE.finditer(frame_match.group("body")):
            fields = _fields(match.group("body"))
            if not {"x1", "y1", "x2", "y2", "w", "h"} <= set(fields):
                continue
            mask_fields = _fields(match.group("mask"))
            data = mask_fields.get("data", "")
            mask_height, mask_width, encoding, _payload = _mask_parts(data)
            confidence = _confidence(fields)
            records.append(
                SamRecord(
                    frame_index=frame_index,
                    track_id=str(match.group("id")),
                    x1=int(fields["x1"]),
                    y1=int(fields["y1"]),
                    x2=int(fields["x2"]),
                    y2=int(fields["y2"]),
                    width=int(fields["w"]),
                    height=int(fields["h"]),
                    confidence=confidence,
                    mask_payload=match.group("mask"),
                    mask_encoding=encoding,
                    mask_height=mask_height,
                    mask_width=mask_width,
                    mask_data=data,
                )
            )
    return records


def records_from_chunks(chunks: list[str]) -> list[SamRecord]:
    """Parse text that arrives split anywhere, including inside a mask payload."""
    buffer = SamLineBuffer()
    for chunk in chunks:
        buffer.push(chunk)
    buffer.finish()
    return buffer.records


def concept_text(label: str) -> str:
    """A short noun phrase. Commands and spatial tokens are rejected."""
    text = " ".join(label.split())
    if not text:
        raise ProviderError("SAM concept prompts must be a non-empty noun phrase.")
    if TOKEN_RE.search(text) or FRAME_MARK_RE.search(text):
        raise ProviderError("invalid prompt: object prompts cannot be combined with a phrase")
    return text[:80]


def render_spatial_prompt(marks: list[SpatialMark]) -> str:
    """Box and point prompt text. This never includes a noun phrase."""
    if not marks:
        raise ProviderError("SAM box and point prompts need at least one mark.")
    width, height = marks[0].width, marks[0].height
    if width <= 0 or height <= 0:
        raise ProviderError("SAM prompts need the media width and height in pixels.")
    if any(mark.width != width or mark.height != height for mark in marks):
        raise ProviderError(
            "invalid prompt: every prompt token must declare the same w/h; "
            "got mixed frame sizes"
        )
    by_frame: dict[int, list[SpatialMark]] = {}
    for mark in marks:
        _check_mark(mark, width, height)
        by_frame.setdefault(mark.frame_index, []).append(mark)
    object_ids = {mark.object_id for mark in marks}
    if len(object_ids) > MAX_PROMPTED_OBJECTS:
        raise ProviderError(
            f"invalid prompt: {len(object_ids)} objects exceed this model's limit of "
            f"{MAX_PROMPTED_OBJECTS}"
        )
    parts: list[str] = []
    for frame in by_frame:
        group = by_frame[frame]
        ordered: list[int] = []
        for mark in group:
            if mark.object_id not in ordered:
                ordered.append(mark.object_id)
        objects: list[str] = []
        for object_id in ordered:
            owned = [mark for mark in group if mark.object_id == object_id]
            objects.append(_object_tokens(owned))
        parts.append(f"<{frame}f>" + ",".join(objects))
    text = "".join(parts)
    if not FRAME_MARK_RE.search(text):
        raise ProviderError("Box and point prompts must name a frame.")
    return text


def video_request(
    prompt: str,
    *,
    video_url: str = "",
    file_id: str = "",
    prompt_kind: str = "concept",
    propagation_direction: str | None = None,
    stream: bool = True,
    model: str = "sam-3.1",
) -> dict:
    """Build the documented Responses body. Concept and spatial prompts stay separate."""
    if prompt_kind == "concept":
        text = concept_text(prompt)
    elif prompt_kind == "spatial":
        if TOKEN_RE.search(prompt) is None and FRAME_MARK_RE.search(prompt) is None:
            raise ProviderError("Box and point prompts must use the documented frame grammar.")
        if _looks_like_phrase(prompt):
            raise ProviderError("invalid prompt: object prompts cannot be combined with a phrase")
        text = prompt
    else:
        raise ProviderError(f"Unknown SAM prompt kind {prompt_kind}.")
    if bool(video_url) == bool(file_id):
        raise ProviderError("SAM video requires exactly one of video_url or file_id.")
    media: dict = {"type": "input_video"}
    if file_id:
        media["file_id"] = file_id
    else:
        media["video_url"] = video_url
    metadata: dict[str, str] = {
        "mask_encoding": "one_bit",
        "include_confidence": "true",
    }
    if propagation_direction is not None:
        if prompt_kind != "spatial":
            raise ProviderError(
                "propagation_direction applies to box and point prompts, not concept prompts."
            )
        if propagation_direction not in PROPAGATION_DIRECTIONS:
            raise ProviderError(
                "metadata['propagation_direction'] must be one of ['both', 'forward', 'backward']"
            )
        metadata["propagation_direction"] = propagation_direction
    return {
        "model": model,
        "stream": stream,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": text},
                    media,
                ],
            }
        ],
        "metadata": metadata,
    }


def image_request(prompt: str, image_url: str, *, prompt_kind: str = "concept", model: str = "sam-3.1") -> dict:
    if prompt_kind == "concept":
        text = concept_text(prompt)
    else:
        text = prompt
        if _looks_like_phrase(text):
            raise ProviderError("invalid prompt: object prompts cannot be combined with a phrase")
    return {
        "model": model,
        "stream": True,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": text},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ],
        "metadata": {"mask_encoding": "one_bit", "include_confidence": "true"},
    }


def plan_video_chunks(
    frame_count: int,
    fps: float,
    *,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
    duration_sec: float = 0,
    frame_times: list[float] | None = None,
) -> list[VideoChunk]:
    """Split a recording, or the frames of a requested interval, into tracked chunks.

    With `frame_times`, chunks cover exactly those source frames and map a SAM
    frame marker through them. Without it, times come from the nominal rate.
    SAM object ids do not continue from one chunk into the next. The boundary
    is unobserved identity, not evidence the object disappeared.
    """
    times = list(frame_times or [])
    count = len(times) if times else int(frame_count)
    rate = float(fps or 0)
    if not times and count <= 0 and duration_sec > 0 and rate > 0:
        count = int(round(duration_sec * rate))
    if count <= 0 or (rate <= 0 and not times):
        raise ProviderError(
            "SAM video tracking needs a positive frame count and frame rate to map "
            "frame indices back to timestamps."
        )
    size = max(1, min(int(chunk_frames or DEFAULT_CHUNK_FRAMES), OFFICIAL_MAX_FRAMES))
    chunks: list[VideoChunk] = []
    start = 0
    index = 0
    while start < count:
        end = min(count, start + size)
        if times:
            step = (times[end] - times[end - 1]) if end < count else (1.0 / rate if rate > 0 else 0.0)
            time_start = times[start]
            time_end = times[end] if end < count else times[end - 1] + step
            chunk_times = times[start:end]
        else:
            time_start, time_end, chunk_times = start / rate, end / rate, []
        chunks.append(
            VideoChunk(
                index=index,
                start_frame=start,
                end_frame=end,
                time_start=time_start,
                time_end=time_end,
                session_id=f"c{index}",
                frame_times=chunk_times,
            )
        )
        index += 1
        start = end
    return chunks


def frame_timestamp(frame_index: int, fps: float, chunk: VideoChunk) -> float | None:
    """Map a SAM frame marker onto the source clock.

    The marker is the index inside the submitted chunk, not the line number.
    Known frame times are used before the nominal rate. A missing rate and no
    known times do not invent a timestamp.
    """
    if frame_index < 0:
        return None
    if chunk.frame_times:
        if frame_index >= len(chunk.frame_times):
            return None
        return chunk.frame_times[frame_index]
    if fps <= 0:
        return None
    return chunk.time_start + (frame_index / fps)


def normalized_box(record: SamRecord) -> dict | None:
    if record.width <= 0 or record.height <= 0:
        return None
    width = max(record.x2 - record.x1, 1) / record.width
    height = max(record.y2 - record.y1, 1) / record.height
    return {
        "x": min(max(record.x1 / record.width, 0.0), 1.0),
        "y": min(max(record.y1 / record.height, 0.0), 1.0),
        "w": min(max(width, 1e-6), 1.0),
        "h": min(max(height, 1e-6), 1.0),
    }


def decode_mask_raster(record: SamRecord):
    """Return row-major bytes of 0 and 1, or None when decoding is unavailable."""
    height, width, encoding, payload = _mask_parts(record.mask_data or _data_from_payload(record))
    if height <= 0 or width <= 0 or encoding not in {"one_bit", "lossless"} or not payload:
        return None
    try:
        from meta_sam_parser import SegmentationMask, decode_mask_to_raster
    except ImportError:
        return None
    try:
        mask = SegmentationMask(encoding=encoding, payload=payload, width=width, height=height)
        return decode_mask_to_raster(mask)
    except Exception:
        return None


class SamAdapter:
    model_id = "sam-3.1"

    def __init__(self, secret: str, timeout_s: float = 180):
        self.secret = secret
        self.timeout_s = provider_timeout(timeout_s)

    def _client(self) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=self.secret, base_url=META_BASE, timeout=self.timeout_s)

    async def segment_image(self, image_path, concept: str) -> SamImageResult:
        body = image_request(concept, data_url(Path(image_path)), prompt_kind="concept", model=self.model_id)
        return await self._execute(body)

    async def segment_image_prompt(self, image_path, prompt: str, *, prompt_kind: str) -> SamImageResult:
        url = data_url(Path(image_path))
        body = image_request(prompt, url, prompt_kind=prompt_kind, model=self.model_id)
        return await self._execute(body)

    async def upload_video(self, path: Path) -> str:
        """Upload through the Files API. purpose is user_data, as documented."""
        try:
            async with self._client() as client:
                with Path(path).open("rb") as handle:
                    created = await client.files.create(file=handle, purpose="user_data")
        except Exception as exc:
            raise translate(exc) from exc
        file_id = str(getattr(created, "id", "") or "")
        if not file_id:
            raise ProviderError("Meta Files API did not return a file id.")
        return file_id

    async def segment_video(
        self,
        video_path: Path | None,
        prompt: str,
        *,
        prompt_kind: str = "concept",
        file_id: str = "",
        propagation_direction: str | None = None,
    ) -> SamImageResult:
        uploaded = file_id
        if not uploaded:
            if video_path is None:
                raise ProviderError("SAM video requires a file path or a file id.")
            uploaded = await self.upload_video(Path(video_path))
        body = video_request(
            prompt,
            file_id=uploaded,
            prompt_kind=prompt_kind,
            propagation_direction=propagation_direction,
            stream=True,
            model=self.model_id,
        )
        result = await self._execute(body)
        result.file_id = uploaded
        return result

    async def _execute(self, body: dict) -> SamImageResult:
        started = time.perf_counter()
        buffer = SamLineBuffer()
        text_parts: list[str] = []
        request_id = ""
        input_tokens = 0
        output_tokens = 0
        try:
            async with self._client() as client:
                stream = await client.responses.create(**body)
                async for event in stream:
                    kind = _event_type(event)
                    if kind == "response.output_text.delta":
                        delta = _event_field(event, "delta") or ""
                        if delta:
                            text_parts.append(delta)
                            buffer.push(delta)
                    elif kind in {"response.completed", "response.output_text.done"}:
                        response = _event_field(event, "response")
                        request_id = request_id or _response_id(response if response is not None else event)
                        usage = _usage(response if response is not None else event)
                        input_tokens = usage[0] or input_tokens
                        output_tokens = usage[1] or output_tokens
        except Exception as exc:
            raise translate(exc) from exc
        buffer.finish()
        text = "".join(text_parts)
        # Re-parse the joined text so a delta split cannot drop or duplicate a record.
        records = parse_sam_output(text)
        return SamImageResult(
            text=text,
            records=records,
            latency_ms=int((time.perf_counter() - started) * 1000),
            request_id=request_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


def _parser_available() -> bool:
    try:
        import meta_sam_parser  # noqa: F401
    except ImportError:
        return False
    return True


def _fields(body: str) -> dict[str, str]:
    found = {}
    for part in body.split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        found[key] = value
    return found


def _confidence(fields: dict[str, str]) -> float | None:
    raw = fields.get("c")
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if value < 0 or value > 1:
        return None
    return value


def _mask_parts(data: str) -> tuple[int, int, str, str]:
    """data is H,W,marker+payload. The decoder reads the marker as part of the payload."""
    parts = data.split(",", 2)
    if len(parts) < 3:
        return 0, 0, "unknown", ""
    try:
        height = int(parts[0])
        width = int(parts[1])
    except ValueError:
        return 0, 0, "unknown", ""
    rest = parts[2]
    if not rest:
        return height, width, "unknown", ""
    marker = rest[0]
    encoding = {"!": "one_bit", "~": "lossless"}.get(marker, "unknown")
    return height, width, encoding, rest


def _data_from_payload(record: SamRecord) -> str:
    if "data=" not in record.mask_payload:
        return ""
    return record.mask_payload.split("data=", 1)[-1]


def _check_mark(mark: SpatialMark, width: int, height: int) -> None:
    if mark.frame_index < 0:
        raise ProviderError("SAM prompt frame indices start at 0.")
    if mark.kind == "box":
        if mark.negative:
            raise ProviderError("invalid prompt: a box names the object; it cannot be negative")
        if mark.x1 < 0 or mark.y1 < 0 or mark.x2 >= width or mark.y2 >= height:
            raise ProviderError(
                f"invalid prompt: box ({mark.x1}, {mark.y1}, {mark.x2}, {mark.y2}) "
                f"lies outside the {width}x{height} frame"
            )
        if mark.x2 < mark.x1 or mark.y2 < mark.y1:
            raise ProviderError("invalid prompt: a box must have x2 >= x1 and y2 >= y1")
        return
    if mark.kind != "point":
        raise ProviderError(f"Unsupported SAM prompt token {mark.kind}.")
    if mark.x < 0 or mark.y < 0 or mark.x >= width or mark.y >= height:
        raise ProviderError(
            f"invalid prompt: point ({mark.x}, {mark.y}) lies outside the {width}x{height} frame"
        )


def _object_tokens(marks: list[SpatialMark]) -> str:
    boxes = [mark for mark in marks if mark.kind == "box"]
    if len(boxes) > 1:
        raise ProviderError(f"invalid prompt: object {marks[0].object_id} carries more than one box")
    body = str(marks[0].object_id)
    for mark in marks:
        if mark.kind == "box":
            body += (
                f"<|box;x1={mark.x1};y1={mark.y1};x2={mark.x2};y2={mark.y2};"
                f"w={mark.width};h={mark.height}|>"
            )
        else:
            token = f"<|point;x={mark.x};y={mark.y};w={mark.width};h={mark.height}|>"
            body += f"-{token}" if mark.negative else token
    return body


def _looks_like_phrase(text: str) -> bool:
    """True when a spatial prompt also contains a noun. Tokens themselves are not a phrase."""
    stripped = FRAME_MARK_RE.sub(" ", text)
    stripped = re.sub(r"<\|(?:box|point|mask);[^|]*\|>", " ", stripped)
    return bool(re.search(r"[A-Za-z]{2,}", stripped))


def _event_type(event) -> str:
    if isinstance(event, dict):
        return str(event.get("type") or "")
    return str(getattr(event, "type", "") or "")


def _event_field(event, name: str):
    if isinstance(event, dict):
        return event.get(name)
    return getattr(event, name, None)


def _response_id(response) -> str:
    if response is None:
        return ""
    if isinstance(response, dict):
        return str(response.get("id") or "")
    return str(getattr(response, "id", "") or "")


def _usage(response) -> tuple[int, int]:
    usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    if isinstance(usage, dict):
        return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
    return int(getattr(usage, "input_tokens", 0) or 0), int(getattr(usage, "output_tokens", 0) or 0)
