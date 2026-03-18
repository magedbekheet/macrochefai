#!/usr/bin/env bash
set -e

# Download data from GCS (skips if already present)
python -c "from src.gcs_loader import ensure_data_downloaded; ensure_data_downloaded()"

# Start Streamlit in background (it also starts FastAPI as a thread)
streamlit run streamlit_app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true &

# Wait for Streamlit to be ready before starting nginx
echo "Waiting for Streamlit to start on :8501 ..."
for i in $(seq 1 60); do
  if curl -s http://127.0.0.1:8501/_stcore/health > /dev/null 2>&1; then
    echo "Streamlit is ready."
    break
  fi
  sleep 1
done

# Wait for FastAPI to be ready
echo "Waiting for FastAPI to start on :8000 ..."
for i in $(seq 1 30); do
  if curl -s http://127.0.0.1:8000/api/v1/health > /dev/null 2>&1; then
    echo "FastAPI is ready."
    break
  fi
  sleep 1
done

# Start nginx in foreground (keeps the container alive)
echo "Starting nginx on :8080 ..."
exec nginx -c /app/nginx.conf -g 'daemon off;'
