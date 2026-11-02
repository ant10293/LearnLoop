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
