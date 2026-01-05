# =======================
# Builder stage
# =======================
FROM python:3.11-slim AS builder
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    poppler-utils \
    libtesseract-dev \
    libleptonica-dev \
    libjpeg-dev \
    zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /wheels

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel \
 && pip wheel --wheel-dir /wheels -r requirements.txt


# =======================
# Runtime stage
# =======================
FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir --no-index --find-links /wheels -r requirements.txt

# OPTIONAL: spaCy model (safe)
RUN python - <<'PY'
try:
    import spacy
    spacy.cli.download("en_core_web_sm")
except Exception:
    pass
PY

COPY . .

# 🚨 Railway compatibility
ENV PORT=8080
EXPOSE 8080

# ✅ Safe default gunicorn config
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "4"]
