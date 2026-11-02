"""API for the browser smoke test. The Meta client is a stub and every answer is authored.

The file name of the uploaded clip picks the answer. This is not part of the product.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
DATA = ROOT / "data" / "e2e"
DATA.mkdir(parents=True, exist_ok=True)
os.environ.update(
    LEARNLOOP_DATA_DIR=str(DATA),
    LEARNLOOP_DATABASE_URL=f"sqlite:///{DATA}/learnloop.db",
    LEARNLOOP_WORKER_ENABLED="1",
    LEARNLOOP_AUTO_SEED="0",
    LEARNLOOP_LIVE_INFERENCE="1",
    MODEL_API_KEY="stub-key",
)

import uvicorn  # noqa: E402
from app.main import app  # noqa: E402
from app.providers import model_list, registry  # noqa: E402
from app.providers.types import ProviderResponse, Usage  # noqa: E402

from tests.test_learnloop_flow import CORRECT, DEMONSTRATION, HIDDEN, MISPLACED, OMITTED  # noqa: E402

ANSWERS = {"demo": DEMONSTRATION, "correct": CORRECT, "omitted": OMITTED, "misplaced": MISPLACED, "hidden": HIDDEN}


class AuthoredMeta:
    async def generate(self, secret, request):
        from sqlalchemy import select

        from app.db import SessionLocal
        from app.models.media import VideoAsset

        session = SessionLocal()
        try:
            videos = session.scalars(select(VideoAsset)).all()
            match = next(v for v in videos if Path(v.proxy_path or v.stored_path).name == Path(request.video_path).name)
            name = Path(match.original_filename).stem
        finally:
            session.close()
        return ProviderResponse(text=ANSWERS[name], raw={"status": "completed"}, usage=Usage(1, 1), request_id=f"stub_{name}", model_id=request.model_id)


registry.get_provider = lambda provider_id, session=None: AuthoredMeta()
model_list.live_options = lambda *args, **kwargs: None

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8199)
