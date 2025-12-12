# backend/Dockerfile

# builder: pre-build wheels for faster final installs
FROM python:3.11-slim AS builder
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    poppler-utils \
    libtesseract-dev \
    libleptonica-dev \
    libjpeg-dev \
    zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /wheels

# copy only requirements first to leverage cache (assumes backend/requirements.txt)
COPY requirements.txt /wheels/requirements.txt

RUN pip install --upgrade pip setuptools wheel \
 && pip wheel --wheel-dir /wheels -r /wheels/requirements.txt

# runtime image
FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# install runtime system packages (tesseract + poppler)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
 && rm -rf /var/lib/apt/lists/*

# copy wheels and requirements and install from wheels to avoid recompiling
COPY --from=builder /wheels /wheels
COPY requirements.txt /wheels/requirements.txt

RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir --no-index --find-links /wheels -r /wheels/requirements.txt

# attempt to download spaCy model only if spaCy present (non-fatal)
RUN python - <<'PY'
import sys
try:
    import spacy
    spacy.cli.download("en_core_web_sm")
except Exception:
    pass
PY

# copy backend app code (copy only backend folder)
COPY . /app/

ENV PORT=8000
EXPOSE 8000

# default entry - adjust "app:app" to the actual WSGI ASGI entrypoint in your repo
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app", "--worker-class", "gthread", "--workers", "2"]
