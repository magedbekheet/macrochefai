FROM python:3.12-slim

WORKDIR /app

# Install all dependencies (API + Streamlit)
COPY requirements-api.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt streamlit requests

# Copy application code + data
COPY api/ api/
COPY app/ app/
COPY data/processed_data/ data/processed_data/

# Expose ports (API=8000, Streamlit=8501)
EXPOSE 8000 8501

# Default: run the API (docker-compose overrides for streamlit)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
