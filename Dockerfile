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
# Stage 3: Runtime
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

# Copy virtual environment from the python builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy the rest of the application code
COPY . /app

# Overwrite the static CSS with the built version from the frontend stage
COPY --from=frontend /app/static/css/styles.css /app/static/css/styles.css

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]