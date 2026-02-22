# Metro Vancouver Property Pricing Engine

A data-driven automated valuation model (AVM) for Metro Vancouver real estate, targeting sub-7% median error on off-market properties and sub-3% on listed ones.

## Architecture

**Three-tier hybrid system:**

1. **Tier 1: LightGBM ensemble** — Property-type-specific gradient-boosted models (condo, townhome, detached, development land)
2. **Tier 2: Rules-based adjustments** — Leasehold curves, assembly premiums, strata health, heritage, GST offsets
3. **Tier 3: Comparable sales reconciliation** — Top-5 comparable sales with appraisal-style adjustments

## Project Structure

```
metro-vancouver-pricing-engine/
├── BLUEPRINT.md              # Complete blueprint with all data sources
├── README.md                 # This file
├── config/                   # Configuration files
│   └── data_sources.yaml     # Data source registry with access details
├── src/                      # Source code
│   ├── ingestion/            # Data ingestion pipelines
│   ├── features/             # Feature engineering
│   ├── models/               # ML model training and inference
│   ├── adjustments/          # Tier 2 rules-based adjustments
│   ├── comparables/          # Tier 3 comparable sales engine
│   └── api/                  # FastAPI prediction service
├── notebooks/                # Jupyter notebooks for exploration
├── tests/                    # Test suite
└── docs/                     # Additional documentation
    └── data-sources/         # Detailed data source documentation
```

## Data Sources (90+)

See [BLUEPRINT.md](BLUEPRINT.md) for the complete data source catalog organized into tiers:

- **Tier 1 (Essential):** BC Assessment, MLS/BridgeAPI, Vancouver Open Data, TransLink GTFS, Walk Score, StatCan Census, CMHC, ParcelMap BC, Bank of Canada
- **Tier 2 (High Value):** Fraser Institute schools, VPD crime, Bill 44/47 data, municipal OCP/zoning, OSFI parameters, Teranet HPI, photo condition scoring, OpenStreetMap
- **Tier 3 (Enhanced):** Local Logic, Mapbox isochrones, seismic data, suburban municipal data, NRCan building footprints, CRT strata disputes, LiDAR
- **Tier 4 (Supplementary):** Houski, Buildify, satellite imagery, economic forecasts, energy benchmarking

## Tech Stack

- **Database:** PostgreSQL + PostGIS
- **ETL:** Python (pandas, geopandas), Prefect/Airflow
- **ML:** LightGBM, SHAP, scikit-learn, PyTorch (geographic embeddings)
- **API:** FastAPI + Redis cache
- **Frontend:** React + Mapbox GL
- **Infrastructure:** AWS (S3, EC2/SageMaker, RDS)

## Key Insights

1. **Transit-wealth interaction** — Transit proximity boosts East Side values but can depress West Side values
2. **Supply pipeline asymmetry** — Rental supply pressures condos, not detached homes
3. **Leasehold pricing cliff** — Non-linear discount accelerating below 25-year financing threshold
4. **Bill 44/47 density floor** — Every residential lot now has minimum 3-6 unit potential
5. **Photo condition scoring** — $0.01/image yields up to 18% MAE reduction

## Getting Started

```bash
# Clone
git clone https://github.com/divhit/metro-vancouver-pricing-engine.git
cd metro-vancouver-pricing-engine

# Setup Python environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure data source credentials
cp config/data_sources.example.yaml config/data_sources.yaml
# Edit with your API keys and credentials
```

## Phase Roadmap

| Phase | Timeline | Target MAPE | Focus |
|-------|----------|-------------|-------|
| 1 | Months 1-3 | <12% | Data foundation, baseline models |
| 2 | Months 4-6 | <9% | Feature enrichment, Tier 2 adjustments |
| 3 | Months 7-9 | <7% | Supply pipeline, market context, geo embeddings |
| 4 | Months 10-12 | <5% on-market | Production deployment, dashboard, CMA reports |
| 5 | Ongoing | Continuous | Monthly retraining, new data sources |

## License

Private — All rights reserved.
