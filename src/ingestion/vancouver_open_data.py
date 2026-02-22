"""
City of Vancouver Open Data ingestion pipeline.

Datasets:
- Property Tax Report (since 2006)
- Zoning Districts and Labels
- Issued Building Permits (since 2017)
- Property Parcel Polygons
- Heritage Sites
- Designated Floodplain
- School Catchments (elementary + secondary)
- Parks
"""

import logging
from typing import Optional

import pandas as pd
import geopandas as gpd
import requests
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

BASE_URL = "https://opendata.vancouver.ca/api/records/1.0/search/"
EXPORT_URL = "https://opendata.vancouver.ca/api/explore/v2.1/catalog/datasets/{dataset}/exports/geojson"

DATASETS = {
    "property_tax": {
        "id": "property-tax-report",
        "description": "Assessment values, zoning, tax levy per property",
        "update": "weekly",
        "spatial": False,
    },
    "zoning": {
        "id": "zoning-districts-and-labels",
        "description": "Zoning district polygons with codes",
        "update": "weekly",
        "spatial": True,
    },
    "building_permits": {
        "id": "issued-building-permits",
        "description": "Permits since 2017: type, value, address",
        "update": "ongoing",
        "spatial": False,
    },
    "parcel_polygons": {
        "id": "property-parcel-polygons",
        "description": "Assessment-based land polygons",
        "update": "periodic",
        "spatial": True,
    },
    "heritage": {
        "id": "heritage-sites",
        "description": "Vancouver Heritage Register",
        "update": "periodic",
        "spatial": True,
    },
    "floodplain": {
        "id": "designated-floodplain",
        "description": "Coastal and Still Creek floodplain",
        "update": "static",
        "spatial": True,
    },
    "school_catchments_elementary": {
        "id": "elementary-school-catchment-areas",
        "description": "Elementary school catchment boundaries",
        "update": "periodic",
        "spatial": True,
    },
    "school_catchments_secondary": {
        "id": "secondary-school-catchment-areas",
        "description": "Secondary school catchment boundaries",
        "update": "periodic",
        "spatial": True,
    },
    "parks": {
        "id": "parks-polygon-representation",
        "description": "Park boundaries",
        "update": "periodic",
        "spatial": True,
    },
    "local_areas": {
        "id": "local-area-boundary",
        "description": "22 neighbourhood boundaries",
        "update": "static",
        "spatial": True,
    },
}


class VancouverOpenDataIngestion:
    """Ingest City of Vancouver Open Data into PostgreSQL/PostGIS."""

    def __init__(self, db_url: str, api_key: Optional[str] = None):
        self.engine = create_engine(db_url)
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Apikey {api_key}"

    def fetch_dataset(self, dataset_key: str, limit: int = -1) -> pd.DataFrame | gpd.GeoDataFrame:
        """Fetch a dataset from the Vancouver Open Data API.

        Args:
            dataset_key: Key from DATASETS dict
            limit: Max records (-1 for all)

        Returns:
            DataFrame or GeoDataFrame depending on dataset type
        """
        config = DATASETS[dataset_key]
        dataset_id = config["id"]
        is_spatial = config["spatial"]

        if is_spatial:
            url = EXPORT_URL.format(dataset=dataset_id)
            logger.info(f"Fetching spatial dataset: {dataset_id}")
            response = self.session.get(url)
            response.raise_for_status()
            gdf = gpd.read_file(response.text, driver="GeoJSON")
            logger.info(f"Fetched {len(gdf)} features from {dataset_id}")
            return gdf
        else:
            # Paginate through non-spatial API
            all_records = []
            offset = 0
            page_size = 100
            total = limit if limit > 0 else float("inf")

            while len(all_records) < total:
                params = {
                    "dataset": dataset_id,
                    "rows": min(page_size, total - len(all_records)),
                    "start": offset,
                }
                response = self.session.get(BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()
                records = data.get("records", [])
                if not records:
                    break
                all_records.extend([r["fields"] for r in records])
                offset += len(records)
                if len(records) < page_size:
                    break

            df = pd.DataFrame(all_records)
            logger.info(f"Fetched {len(df)} records from {dataset_id}")
            return df

    def ingest_dataset(self, dataset_key: str) -> int:
        """Fetch and store a dataset in PostgreSQL."""
        config = DATASETS[dataset_key]
        table_name = f"cov_{dataset_key}"

        data = self.fetch_dataset(dataset_key)

        if isinstance(data, gpd.GeoDataFrame):
            data.to_postgis(
                table_name,
                self.engine,
                schema="raw",
                if_exists="replace",
                index=False,
            )
        else:
            data.to_sql(
                table_name,
                self.engine,
                schema="raw",
                if_exists="replace",
                index=False,
                chunksize=50_000,
            )

        row_count = len(data)
        logger.info(f"Ingested {row_count:,} rows into raw.{table_name}")
        return row_count

    def ingest_all(self) -> dict:
        """Ingest all configured Vancouver Open Data datasets."""
        results = {}
        for key in DATASETS:
            try:
                results[key] = self.ingest_dataset(key)
            except Exception as e:
                logger.error(f"Failed to ingest {key}: {e}")
                results[key] = 0
        return results
