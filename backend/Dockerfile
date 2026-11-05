# LearnLoop API and worker. Python 3.12, ffmpeg, CPU PyTorch, and DINOv2 weights fetched at build time.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_HOME=/opt/torch

# Static ffmpeg and ffprobe (about 200 MB). The Debian package pulls in Mesa and LLVM for about 420 MB.
COPY --from=mwader/static-ffmpeg:7.1 /ffmpeg /ffprobe /usr/local/bin/

# The user and its writable directories exist before the weights are fetched, so no
# chown has to copy them into another layer.
RUN useradd --create-home --uid 1000 learnloop \
 && mkdir -p /app/data/app /opt/torch \
 && chown learnloop:learnloop /app/data /app/data/app /opt/torch

# CPU-only wheel, in its own layer: about 200 MB instead of the multi-GB CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# The repository layout is kept (/app/backend), so REPO_ROOT resolves to /app.
WORKDIR /app/backend
COPY pyproject.toml ./
COPY app ./app
COPY alembic.ini ./
COPY alembic ./alembic
RUN pip install './[perception]'

USER learnloop

# Load DINOv2 once so the first analysis does not wait on a download.
RUN python -m app.perception.encoder

ENV LEARNLOOP_DATA_DIR=/app/data/app \
    LEARNLOOP_DATABASE_URL=sqlite:////app/data/app/learnloop.db \
    LEARNLOOP_AUTO_SEED=0 \
    LEARNLOOP_WORKER_ENABLED=1

EXPOSE 8010
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/api/health', timeout=2)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010", "--timeout-graceful-shutdown", "5"]
