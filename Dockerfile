# =====================
# Stage 1: Frontend Builder
# =====================
FROM node:20-slim AS frontend

WORKDIR /app

# Copy package files and install dependencies
COPY package.json package-lock.json ./
RUN npm install

# Copy all frontend-related files
COPY tailwind.config.js postcss.config.js ./
COPY static/css/input.css ./static/css/input.css
COPY templates/ ./templates/

# Build the production CSS
RUN npm run build

# =====================
# Stage 2: Python Builder
# =====================
FROM python:3.12-slim AS builder

# Prevent .pyc files, enable stdout/stderr buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install build‑time tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ghostscript \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps into a venv
COPY requirements.txt setup.py /app/
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# =====================
# Stage 3: Runtime Image
# =====================
FROM python:3.12-slim

# Keep Python output unbuffered, no .pyc files, use our venv
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install runtime dependencies:
#  • ghostscript  for PDF compression
#  • poppler-utils for pdf2image (pdftoppm/pdfinfo)
#  • tesseract-ocr and eng data for OCR
#  • devlibs in case wheels need to compile
RUN apt-get update && apt-get install -y --no-install-recommends \
    ghostscript \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    libreoffice-writer \
    libreoffice-common \
    && rm -rf /var/lib/apt/lists/*

# Copy the venv from the builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY . /app

# Bring in the built CSS from the frontend stage
COPY --from=frontend /app/static/css/styles.css /app/static/css/styles.css

# Expose and launch
EXPOSE 80
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
