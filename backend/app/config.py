"""Runtime configuration. Secrets are never given default values."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LEARNLOOP_",
        env_file=str(REPO_ROOT / ".env"),
        extra="ignore",
    )

    data_dir: Path = REPO_ROOT / "data"
    database_url: str = ""
    auto_seed: bool = False
    max_upload_bytes: int = 500 * 1024 * 1024
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    worker_poll_seconds: float = 0.4
    worker_enabled: bool = True
    ingest_inline: bool = False
    stale_job_seconds: int = 900
    # Meta bills SAM video per frame. No rate is published in this repository, so
    # cost stays unknown until this is set. Frame and request limits always apply.
    sam_price_per_frame_usd: float | None = None
    # A live Meta call spends credits. Nothing is sent unless this is set on purpose.
    live_inference: bool = False
    guided_max_output_tokens: int = 65_536
    # SAM thumbnails and timelines on demonstrations. Bills per frame, separately from the guided call.
    object_views: bool = True

    # Appearance references from DINOv2 ViT-S/14 (Meta, frozen weights). A cosine between two
    # embeddings is a similarity, not a probability, so these thresholds are tuning values.
    appearance_enabled: bool = True
    # Build a gallery of trusted crops from each demonstration's SAM tracks, with no person in the loop.
    auto_gallery: bool = True
    # Search near a rejected or lost track for the object and follow it again with SAM.
    appearance_recovery: bool = True
    # Ask the analysis model a narrow question about unresolved required steps, with a few frames.
    auto_reinspect: bool = True
    appearance_max_references: int = 6
    appearance_min_crop_px: int = 20
    appearance_sample_fps: float = 3.0
    appearance_batch_size: int = 32
    # A sighting matches its own references at or above accept, and beats every rival by margin.
    appearance_accept: float = 0.6
    appearance_margin: float = 0.05
    # Below this, a sighting does not look like its object at all.
    appearance_floor: float = 0.4
    recovery_max_intervals: int = 4
    recovery_max_candidates: int = 48
    # Extra SAM requests recovery may send in one object-views pass. Each bills frames.
    recovery_max_sam_requests: int = 1
    reinspect_max_calls: int = 3
    reinspect_max_frames: int = 6

    @property
    def resolved_data_dir(self) -> Path:
        path = self.data_dir
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        return path

    @property
    def sqlalchemy_url(self) -> str:
        if self.database_url:
            return self.database_url
        db = self.resolved_data_dir / "learnloop.db"
        return f"sqlite:///{db}"

    @property
    def origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


settings = Settings()
