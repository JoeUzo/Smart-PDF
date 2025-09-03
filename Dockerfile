# =====================
# Stage 1, Frontend build
# =====================
FROM node:20-slim AS frontend
WORKDIR /app

# install node deps
COPY package.json package-lock.json ./
RUN npm ci

# bring in everything the build needs
COPY tailwind.config.js postcss.config.js ./
COPY static ./static
COPY templates ./templates

# build, this must write to static/css/styles.css
# ensure package.json contains: "build": "tailwindcss -i ./static/css/input.css -o ./static/css/styles.css --minify"
RUN npm run build


# =====================
# Stage 2, Python builder (venv)
# =====================
FROM python:3.12-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

# build tools only where needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libtesseract-dev \
    libleptonica-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt setup.py ./
RUN python -m venv /opt/venv \
  && /opt/venv/bin/pip install --upgrade pip \
  && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


# =====================
# Stage 3, Runtime image
# =====================
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"
WORKDIR /app

# runtime deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    ghostscript \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libreoffice-writer \
    libreoffice-common \
    && rm -rf /var/lib/apt/lists/*

# venv from builder
COPY --from=builder /opt/venv /opt/venv

# backend code
COPY . /app

# bring in the entire built static tree from the frontend stage
COPY --from=frontend /app/static /app/static

EXPOSE 80
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","80","--proxy-headers","--forwarded-allow-ips","*"]
