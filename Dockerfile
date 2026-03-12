##############################################
# Stage 1: Build — download data, convert parquet → CSV
##############################################
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install only what's needed for conversion
RUN pip install --no-cache-dir pandas pyarrow

# Download models and data
RUN curl -L -o /tmp/models.tar.gz https://github.com/divhit/metro-vancouver-pricing-engine/releases/download/v1.0-models/models.tar.gz \
    && tar xzf /tmp/models.tar.gz -C /app/ \
    && rm /tmp/models.tar.gz
RUN curl -L -o /tmp/data-api.tar.gz https://github.com/divhit/metro-vancouver-pricing-engine/releases/download/v1.0-models/data-api.tar.gz \
    && tar xzf /tmp/data-api.tar.gz -C /app/ \
    && rm /tmp/data-api.tar.gz

# Convert parquet → CSV (only needed columns), then delete parquet
RUN python3 -c "
import pandas as pd, pyarrow.parquet as pq
cols = [
    'pid','full_address','street_name','from_civic_number','to_civic_number',
    'latitude','longitude','property_type','zoning_district',
    'neighbourhood_code','total_assessed_value','current_land_value',
    'current_improvement_value','year_built','tax_assessment_year',
    'legal_type','estimated_living_area_sqft',
]
schema = pq.read_schema('data/processed/enriched_properties.parquet')
avail = [c for c in cols if c in schema.names]
df = pd.read_parquet('data/processed/enriched_properties.parquet', columns=avail)
df.to_csv('data/processed/enriched_properties.csv', index=False)
print(f'Converted {len(df)} rows, {len(df.columns)} cols')
" && rm -f data/processed/enriched_properties.parquet

##############################################
# Stage 2: Runtime — slim, no pyarrow
##############################################
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && rm -rf /var/lib/apt/lists/*

# Runtime requirements (no pyarrow!)
COPY requirements-api.txt .
RUN grep -v pyarrow requirements-api.txt > /tmp/req.txt && \
    pip install --no-cache-dir -r /tmp/req.txt && rm /tmp/req.txt

# Copy source code
COPY src/ src/

# Copy models and converted CSV data from builder
COPY --from=builder /app/models/ models/
COPY --from=builder /app/data/ data/

ENV LITE_MODE=true
ENV MAX_CACHED_MODELS=1

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
