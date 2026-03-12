FROM python:3.12-slim

WORKDIR /app

# Install system deps (libgomp for LightGBM, curl for downloads)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy source code
COPY src/ src/

# Download models
RUN curl -L -o /tmp/models.tar.gz https://github.com/divhit/metro-vancouver-pricing-engine/releases/download/v1.0-models/models.tar.gz \
    && tar xzf /tmp/models.tar.gz -C /app/ \
    && rm /tmp/models.tar.gz

# Download full data (parquet + SQLite DB for CMA)
RUN curl -L -o /tmp/data-api.tar.gz https://github.com/divhit/metro-vancouver-pricing-engine/releases/download/v1.0-models/data-api.tar.gz \
    && tar xzf /tmp/data-api.tar.gz -C /app/ \
    && rm /tmp/data-api.tar.gz

ENV LITE_MODE=false
ENV MAX_CACHED_MODELS=5

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
