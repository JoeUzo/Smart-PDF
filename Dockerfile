# =====================
# Stage 1: Builder
# =====================
FROM python:3.12-slim AS builder

# prevent .pyc files, enable stdout/stderr buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# install build tools and ghostscript
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ghostscript \
    && rm -rf /var/lib/apt/lists/*

# copy and install Python deps into a venv
COPY requirements.txt setup.py /app/
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# =====================
# Stage 2: Runtime
# =====================
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# install ghostscript for PDF compression
RUN apt-get update && apt-get install -y --no-install-recommends \
    ghostscript \
    && rm -rf /var/lib/apt/lists/*

# copy venv and app code
COPY --from=builder /opt/venv /opt/venv
COPY . /app

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
