"""Reproducible perception configurations. Weights are uncalibrated."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class AssociationWeights:
    prediction: float = 0.34
    iou: float = 0.24
    mask: float = 0.08
    size: float = 0.10
    motion: float = 0.12
    appearance: float = 0.0
    time: float = 0.12

    def with_appearance(self, appearance: float = 0.22) -> AssociationWeights:
        scale = (1.0 - appearance) / max(self.prediction + self.iou + self.mask + self.size + self.motion + self.time, 1e-6)
        return AssociationWeights(
            prediction=self.prediction * scale,
            iou=self.iou * scale,
            mask=self.mask * scale,
            size=self.size * scale,
            motion=self.motion * scale,
            appearance=appearance,
            time=self.time * scale,
        )


@dataclass
class PerceptionConfig:
    strategy: str = "temporal"
    detector: str = "motion"
    encoder: str | None = None
    segment: bool = False
    target_fps: float = 8.0
    maximum_frames: int = 3000
    separate_stream: bool = True
    min_score: float = 0.34
    min_margin: float = 0.06
    reid_min_cosine: float = 0.75
    reid_min_margin: float = 0.12
    occlusion_seconds: float = 0.75
    lost_seconds: float = 2.5
    archive_seconds: float = 12.0
    tentative_hits: int = 2
    max_trusted_references: int = 8
    max_candidate_references: int = 4
    weights: AssociationWeights = field(default_factory=AssociationWeights)
    manual_regions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        return payload


PRESET_NAMES = (
    "A_legacy_motion",
    "B_temporal",
    "C_appearance",
    "D_appearance_segmented",
    "vlm_only",
)


def preset(name: str) -> PerceptionConfig:
    if name == "A_legacy_motion":
        return PerceptionConfig(
            strategy="legacy_motion",
            detector="motion",
            encoder=None,
            segment=False,
            separate_stream=False,
        )
    if name == "B_temporal":
        return PerceptionConfig(strategy="temporal", detector="proposals", encoder=None, segment=False)
    if name == "C_appearance":
        return PerceptionConfig(
            strategy="appearance",
            detector="proposals",
            encoder="dinov2_vits14",
            segment=False,
        )
    if name == "D_appearance_segmented":
        return PerceptionConfig(
            strategy="appearance_segmented",
            detector="proposals",
            encoder="dinov2_vits14",
            segment=True,
        )
    if name == "vlm_only":
        return PerceptionConfig(
            strategy="vlm_only",
            detector="motion",
            encoder=None,
            segment=False,
            separate_stream=False,
        )
    known = ", ".join(PRESET_NAMES)
    raise ValueError(f"Unknown perception preset {name}. Known presets: {known}.")


def config_from_mapping(data: dict | None) -> PerceptionConfig:
    """Build a config. A missing block is the legacy motion baseline.

    New runs write an explicit preset. Older runs that never stored one keep
    the motion baseline, including the scene-history repair, instead of being
    reinterpreted as an appearance experiment.
    """
    if not data:
        return preset("A_legacy_motion")
    name = str(data.get("preset") or "")
    if name and set(data) <= {"preset"}:
        return preset(name)
    base = preset(name) if name else PerceptionConfig()
    weights = data.get("weights") or {}
    if weights:
        current = asdict(base.weights)
        current.update({key: float(value) for key, value in weights.items() if key in current})
        base.weights = AssociationWeights(**current)
    for key in (
        "strategy",
        "detector",
        "encoder",
        "segment",
        "target_fps",
        "maximum_frames",
        "separate_stream",
        "min_score",
        "min_margin",
        "reid_min_cosine",
        "reid_min_margin",
        "occlusion_seconds",
        "lost_seconds",
        "archive_seconds",
        "tentative_hits",
    ):
        if key in data and data[key] is not None:
            setattr(base, key, data[key])
    if data.get("manual_regions"):
        base.manual_regions = list(data["manual_regions"])
    if base.strategy == "appearance" and not base.encoder:
        base.encoder = "dinov2_vits14"
    if base.strategy == "appearance_segmented":
        base.segment = True
        if not base.encoder:
            base.encoder = "dinov2_vits14"
    if base.strategy == "legacy_motion":
        base.encoder = None
        base.segment = False
        base.detector = "motion"
    if base.strategy == "vlm_only":
        base.encoder = None
    return base
