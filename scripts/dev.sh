#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/backend${PYTHONPATH:+:$PYTHONPATH}"
if [[ ! -x .venv/bin/python ]]; then
  uv sync --python 3.12
fi
if [[ ! -d frontend/node_modules ]]; then
  (cd frontend && npm ci)
fi
.venv/bin/python -m uvicorn flylab.api:app --host 127.0.0.1 --port 8000 &
backend_pid=$!
cleanup() { kill "$backend_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
for attempt in {1..30}; do
  if ! kill -0 "$backend_pid" 2>/dev/null; then
    echo "Backend could not start. Check whether port 8000 is already in use." >&2
    exit 1
  fi
  if curl -fsS http://127.0.0.1:8000/api/health >/dev/null 2>&1; then break; fi
  sleep .2
done
cd frontend
npm run dev
