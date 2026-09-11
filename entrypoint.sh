#!/usr/bin/env bash
# entrypoint.sh – starts the FastAPI server with the configured worker count.
# Defaults to a single worker to avoid OOM caused by multiple model loads.

# Ensure the worker count is set (fallback to 1)
if [ -z "$UVIORN_WORKERS" ]; then
  UVIORN_WORKERS=1
fi

# Start uvicorn with the desired number of workers
exec uvicorn app:app --host 0.0.0.0 --port 8000 --workers ${UVIORN_WORKERS}
