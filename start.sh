#!/usr/bin/env bash
set -euo pipefail

# One Celery worker that consumes from both queues
celery -A app.celery_worker.celery_app worker --loglevel=info -Q default,heavy &
CELERY_PID=$!

# Web server
PORT="${PORT:-10000}"
uvicorn app.main:app --host 0.0.0.0 --port "$PORT" &
UVICORN_PID=$!

# Keep both processes tied together
wait -n
kill "$CELERY_PID" "$UVICORN_PID" 2>/dev/null || true
wait
