FROM python:3.12-slim

WORKDIR /app

# Install system deps (libgomp for LightGBM, curl for downloads)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl && rm -rf /var/lib/apt/lists/*

# Install Python deps (includes pyarrow for build-time parquet→csv conversion)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy source code
COPY src/ src/
RUN curl -L -o /tmp/models.tar.gz https://github.com/divhit/metro-vancouver-pricing-engine/releases/download/v1.0-models/models.tar.gz \
    && tar xzf /tmp/models.tar.gz -C /app/ \
    && rm /tmp/models.tar.gz
RUN curl -L -o /tmp/data-api.tar.gz https://github.com/divhit/metro-vancouver-pricing-engine/releases/download/v1.0-models/data-api.tar.gz \
    && tar xzf /tmp/data-api.tar.gz -C /app/ \
    && rm /tmp/data-api.tar.gz

# Convert parquet → CSV at build time so we can drop pyarrow from runtime
# Only keep the columns we actually need for API endpoints
RUN python3 -c "
import pandas as pd
cols = [
    'pid', 'full_address', 'street_name', 'from_civic_number', 'to_civic_number',
    'latitude', 'longitude', 'property_type', 'zoning_district',
    'neighbourhood_code', 'total_assessed_value', 'current_land_value',
    'current_improvement_value', 'year_built', 'tax_assessment_year',
    'legal_type', 'estimated_living_area_sqft',
]
import pyarrow.parquet as pq
schema = pq.read_schema('data/processed/enriched_properties.parquet')
avail = [c for c in cols if c in schema.names]
df = pd.read_parquet('data/processed/enriched_properties.parquet', columns=avail)
df.to_csv('data/processed/enriched_properties.csv', index=False)
print(f'Converted {len(df)} rows, {len(df.columns)} columns to CSV')
" && rm -f data/processed/enriched_properties.parquet

# Uninstall pyarrow to free disk space (not needed at runtime anymore)
RUN pip uninstall -y pyarrow

ENV LITE_MODE=true
ENV MAX_CACHED_MODELS=1

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
