FROM python:3.12-slim

WORKDIR /app

# Install nginx and curl
RUN apt-get update && apt-get install -y --no-install-recommends nginx curl && rm -rf /var/lib/apt/lists/*

# Install Python deps (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (data loaded from GCS at runtime)
COPY src/ src/
COPY api/ api/
COPY streamlit_app.py .
COPY .streamlit/ .streamlit/
COPY nginx.conf .
COPY entrypoint.sh .

RUN chmod +x entrypoint.sh

# Port 8080 = nginx (routes to FastAPI :8000 + Streamlit :8501)
EXPOSE 8080

ENTRYPOINT ["./entrypoint.sh"]
