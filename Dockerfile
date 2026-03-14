FROM python:3.12-slim

WORKDIR /app

# Install system deps (libgomp for LightGBM, curl for downloads)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl && rm -rf /var/lib/apt/lists/*

# Install Python deps — no pyarrow (we use CSV not parquet at runtime)
COPY requirements-api.txt .
RUN grep -v pyarrow requirements-api.txt > /tmp/req.txt && \
    pip install --no-cache-dir -r /tmp/req.txt && rm /tmp/req.txt

# Copy source code
COPY src/ src/

# Copy pre-computed trends data (15KB — correct multi-year medians)
COPY data/processed/trends_precomputed.csv data/processed/trends_precomputed.csv

# Download models
RUN curl -L -o /tmp/models.tar.gz https://github.com/divhit/metro-vancouver-pricing-engine/releases/download/v1.0-models/models.tar.gz \
    && tar xzf /tmp/models.tar.gz -C /app/ \
    && rm /tmp/models.tar.gz

# Download lite CSV data (46 columns, ~16MB) + SQLite DB for CMA
RUN curl -L -o /tmp/data-api-lite.tar.gz https://github.com/divhit/metro-vancouver-pricing-engine/releases/download/v1.0-models/data-api-lite.tar.gz \
    && tar xzf /tmp/data-api-lite.tar.gz -C /app/ \
    && rm /tmp/data-api-lite.tar.gz

ENV LITE_MODE=true
ENV MAX_CACHED_MODELS=3

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
