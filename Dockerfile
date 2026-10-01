# AI Research Assistant Platform
# Single-stage image; the stack is pure Python so no build toolchain is needed.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code.
COPY app ./app
COPY scripts ./scripts
COPY data/documents ./data/documents
COPY evaluation_examples ./evaluation_examples

# Runtime data volume (vector store, checkpoints, notes).
RUN mkdir -p /app/data/chroma /app/data/notes
VOLUME ["/app/data"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]