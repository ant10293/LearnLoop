"""Closed vocabularies shared by the API, database, and analysis documents."""

from enum import Enum


class EvidenceClass(str, Enum):
    DIRECTLY_OBSERVED = "DIRECTLY_OBSERVED"
    INFERRED = "INFERRED"
    UNKNOWN = "UNKNOWN"


class Provenance(str, Enum):
    AI_PREDICTION = "AI_PREDICTION"
    CV_DETECTION = "CV_DETECTION"
    DETERMINISTIC = "DETERMINISTIC"
    HUMAN_ANNOTATION = "HUMAN_ANNOTATION"
    HUMAN_CORRECTION = "HUMAN_CORRECTION"


class ResearchMode(str, Enum):
    FIXED_CAMERA = "fixed_camera"
    CONSTRAINED_HEAD_MOUNTED = "constrained_head_mounted"
    UNCONSTRAINED_HEAD_MOUNTED = "unconstrained_head_mounted"
    AUTO = "auto"


class Viewpoint(str, Enum):
    FIXED = "fixed"
    HEAD_MOUNTED = "head_mounted"
    UNKNOWN = "unknown"


class TaskConstraint(str, Enum):
    CONSTRAINED = "constrained"
    UNCONSTRAINED = "unconstrained"
    UNKNOWN = "unknown"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELED = "canceled"


class RevisionLayer(str, Enum):
    RAW = "raw"
    PARSED = "parsed"
    VALIDATED = "validated"
    HUMAN = "human"


class InputModality(str, Enum):
    SAMPLED_FRAMES = "sampled_frames"
    NATIVE_VIDEO = "native_video"


class SamplingPreset(str, Enum):
    SPARSE = "sparse"
    STANDARD = "standard"
    DETAILED = "detailed"
    HIGH_DETAIL = "high_detail"
    INTENSIVE = "intensive"
    CUSTOM = "custom"
    ADAPTIVE = "adaptive"


class AnalysisPass(str, Enum):
    SINGLE = "single"
    MULTI = "multi"
    MULTI_REINSPECT = "multi_reinspect"


class SegmentationMode(str, Enum):
    DISABLED = "disabled"
    BOXES_ONLY = "boxes_only"
    SAM = "sam"
    SAM_VIDEO = "sam_video"
    LOCAL_GRABCUT = "local_grabcut"


class ActionOutcome(str, Enum):
    COMPLETED = "completed"
    ATTEMPTED = "attempted"
    FAILED = "failed"
    UNKNOWN = "unknown"


class DependencyCategory(str, Enum):
    NECESSARY_PHYSICAL = "necessary_physical"
    OBSERVED_TEMPORAL = "observed_temporal"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class Visibility(str, Enum):
    VISIBLE = "visible"
    PARTIAL = "partial"
    OCCLUDED = "occluded"
    ABSENT = "absent"


class RelationKind(str, Enum):
    NEAR = "near"
    TOUCHING = "touching"
    INSIDE = "inside"
    AROUND = "around"
    ATTACHED = "attached"
    SUPPORTED_BY = "supported_by"
    HELD_BY = "held_by"
    ABOVE = "above"
    BELOW = "below"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    OCCLUDED_BY = "occluded_by"
    UNKNOWN = "unknown"


class Continuity(str, Enum):
    CONTINUOUS = "continuous"
    REASSOCIATED = "reassociated"
    UNCERTAIN = "uncertain"
    NEW = "new"


class TrackStatus(str, Enum):
    """Lifecycle of one physical-instance hypothesis. Archived tracks are retained."""

    TENTATIVE = "tentative"
    TRACKED = "tracked"
    OCCLUDED = "occluded"
    LOST = "lost"
    ARCHIVED = "archived"
    AMBIGUOUS = "ambiguous"


class AmbiguityKind(str, Enum):
    OCCLUSION = "occlusion"
    MISSING_EVIDENCE = "missing_evidence"
    UNCERTAIN_IDENTITY = "uncertain_identity"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"
    AMBIGUOUS_STATE = "ambiguous_state"
    UNVERIFIABLE_COMPLETION = "unverifiable_completion"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    ALTERNATIVE_INTERPRETATION = "alternative_interpretation"
    CAMERA_MOTION = "camera_motion"
    UNOBSERVED_INTERVAL = "unobserved_interval"


PRESET_FPS: dict[str, float] = {
    SamplingPreset.SPARSE.value: 0.5,
    SamplingPreset.STANDARD.value: 1.0,
    SamplingPreset.DETAILED.value: 2.0,
    SamplingPreset.HIGH_DETAIL.value: 5.0,
    SamplingPreset.INTENSIVE.value: 10.0,
}

RECOGNIZED_ACTIONS = {
    "pick_up",
    "put_down",
    "move",
    "rotate",
    "insert",
    "remove",
    "attach",
    "detach",
    "align",
    "adjust",
    "push",
    "pull",
    "twist",
    "release",
    "inspect",
    "motion",
    "hold",
    "approach",
    "open",
    "close",
    "place",
    "transfer",
    "unknown",
}

OBJECT_COLORS = [
    "#3DDC97",
    "#F0C05A",
    "#7EB6FF",
    "#F07178",
    "#C792EA",
    "#7AD7F0",
    "#FF9B71",
    "#B8E986",
    "#F7A1C4",
    "#D4C4A8",
    "#8FD3C4",
    "#E6A15C",
]

PIPELINE_STAGES = [
    "metadata",
    "preprocess",
    "sample_frames",
    "spatial_tracking",
    "scene_analysis",
    "object_extraction",
    "action_analysis",
    "state_tracking",
    "segmentation",
    "evidence_fusion",
    "workflow",
    "ambiguity",
    "validation",
    "reinspection",
    "transcription",
    "audio_context",
    "artifacts",
]
