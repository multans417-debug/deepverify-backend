FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src PIP_NO_CACHE_DIR=1 \
    ARTIFACT_DIR=/app/artifacts UPLOAD_TMP_DIR=/tmp/uploads

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY artifacts ./artifacts
RUN useradd -m appuser && mkdir -p /app/artifacts /tmp/uploads && chown -R appuser /app /tmp/uploads
USER appuser

EXPOSE 10000
# One worker: each worker would load its own copy of the models. Scale vertically or add instances instead.
CMD ["sh", "-c", "uvicorn deepverify.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
