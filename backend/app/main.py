"""Application entrypoint."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.audio import router as audio_router
from app.api.recovery import router as recovery_router
from app.api.routes import router
from app.api.tasks import router as task_router
from app.api.verification import router as verification_router
from app.config import settings
from app.demo.seed import seed_if_empty
from app.jobs.worker import worker_loop
from app.services.experiments import seed_prompts
from app.services.ingest import repair_preview_paths, resume_processing
from app.services.recovery import recover_interrupted
from app.services.storage import ensure_data_dirs

log = logging.getLogger("learnloop")


def init_database() -> None:
    from app import db

    ensure_data_dirs()
    import app.models  # noqa: F401

    db.Base.metadata.create_all(db.engine)
    from app.services.schema_upgrade import ensure_task_columns

    ensure_task_columns(db.engine)
    session = db.SessionLocal()
    try:
        recover_interrupted(session)
        seed_prompts(session)
        if settings.auto_seed:
            seed_if_empty(session)
    finally:
        session.close()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_database()
    repair_preview_paths()
    resume_processing()
    stop = asyncio.Event()
    task = None
    if settings.worker_enabled:
        task = asyncio.create_task(worker_loop(stop))
    yield
    stop.set()
    if task is not None:
        await task


def create_app() -> FastAPI:
    app = FastAPI(title="LearnLoop Research Lab", version=__version__, lifespan=lifespan)

    @app.middleware("http")
    async def _api_json_not_cached(request, call_next):
        response = await call_next(request)
        content_type = response.headers.get("content-type", "")
        if request.url.path.startswith("/api/") and "json" in content_type:
            response.headers["Cache-Control"] = "no-store"
        return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    app.include_router(task_router)
    app.include_router(verification_router)
    app.include_router(audio_router)
    app.include_router(recovery_router)
    return app


app = create_app()
