"""
BC Assessment Data Advice ingestion pipeline.

Handles the six CSV tables (joined on FOLIO_ID):
- bca_folio_descriptions
- bca_folio_addresses
- bca_folio_legal_descriptions
- bca_folio_gnrl_prop_values
- bca_folio_sales
- ownership/jurisdictions

Plus Residential Inventory and Commercial Inventory extracts.
"""

import logging
from pathlib import Path

import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)


class BCAIngestion:
    """Ingest BC Assessment Data Advice CSV files into PostgreSQL."""

    # Six Data Advice tables keyed on FOLIO_ID
    DATA_ADVICE_TABLES = {
        "folio_descriptions": {
            "key_columns": ["FOLIO_ID"],
            "important_fields": [
                "ACTUAL_USE_CODE",
                "MANUAL_CLASS_CODE",
                "GEN_PROPERTY_CLASS_CODE",
                "JURISDICTION",
                "ROLL_NUMBER",
            ],
        },
        "folio_addresses": {
            "key_columns": ["FOLIO_ID"],
            "important_fields": [
                "UNIT_NUMBER",
                "STREET_NUMBER",
                "STREET_NAME",
                "STREET_TYPE",
                "CITY",
                "POSTAL_CODE",
                "PRIMARY_FLAG",
            ],
        },
        "folio_legal_descriptions": {
            "key_columns": ["FOLIO_ID"],
            "important_fields": [
                "LOT",
                "PLAN",
                "DISTRICT_LOT",
                "STRATA_PLAN",
                "PID",
            ],
        },
        "folio_gnrl_prop_values": {
            "key_columns": ["FOLIO_ID", "GEN_PROPERTY_CLASS_CODE"],
            "important_fields": [
                "LAND_VALUE",
                "IMPROVEMENT_VALUE",
                "TOTAL_VALUE",
            ],
        },
        "folio_sales": {
            "key_columns": ["FOLIO_ID"],
            "important_fields": [
                "SALE_PRICE",
                "SALE_DATE",
                "CONVEYANCE_TYPE",
                "CONVEYANCE_DATE",
            ],
        },
    }

    # Residential inventory fields for feature engineering
    RESIDENTIAL_FIELDS = [
        "BEDROOMS",
        "BATHROOMS",
        "TOTAL_FINISHED_AREA",
        "YEAR_BUILT",
        "EFFECTIVE_YEAR",
        "QUALITY",
        "CONDITION",
        "BASEMENT_FINISH",
        "NUMBER_OF_STOREYS",
    ]

    def __init__(self, db_url: str, data_dir: str):
        self.engine = create_engine(db_url)
        self.data_dir = Path(data_dir)

    def ingest_data_advice(self, table_name: str, file_path: str) -> int:
        """Load a single Data Advice CSV table into PostgreSQL.

        Returns number of rows ingested.
        """
        logger.info(f"Ingesting BCA table: {table_name} from {file_path}")

        df = pd.read_csv(file_path, low_memory=False)
        row_count = len(df)

        # Normalize column names
        df.columns = [c.upper().strip() for c in df.columns]

        # Write to PostgreSQL
        df.to_sql(
            f"bca_{table_name}",
            self.engine,
            schema="raw",
            if_exists="replace",
            index=False,
            chunksize=50_000,
        )

        logger.info(f"Ingested {row_count:,} rows into bca_{table_name}")
        return row_count

    def ingest_all_data_advice(self) -> dict:
        """Ingest all six Data Advice tables from the configured data directory."""
        results = {}
        for table_name in self.DATA_ADVICE_TABLES:
            # Look for CSV files matching the table name pattern
            pattern = f"*{table_name}*"
            matches = list(self.data_dir.glob(pattern))
            if matches:
                results[table_name] = self.ingest_data_advice(
                    table_name, str(matches[0])
                )
            else:
                logger.warning(f"No file found matching pattern: {pattern}")
        return results

    def build_unified_property_table(self) -> int:
        """Join all Data Advice tables into a unified property view.

        Creates a materialized view joining on FOLIO_ID with the most
        useful fields for the pricing engine.
        """
        sql = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS features.bca_properties AS
        SELECT
            d.FOLIO_ID,
            d.JURISDICTION,
            d.ROLL_NUMBER,
            d.ACTUAL_USE_CODE,
            d.GEN_PROPERTY_CLASS_CODE,
            a.UNIT_NUMBER,
            a.STREET_NUMBER,
            a.STREET_NAME,
            a.STREET_TYPE,
            a.CITY,
            a.POSTAL_CODE,
            l.PID,
            l.LOT,
            l.PLAN,
            l.DISTRICT_LOT,
            l.STRATA_PLAN,
            v.LAND_VALUE,
            v.IMPROVEMENT_VALUE,
            (v.LAND_VALUE + v.IMPROVEMENT_VALUE) AS TOTAL_ASSESSED_VALUE,
            CASE
                WHEN (v.LAND_VALUE + v.IMPROVEMENT_VALUE) > 0
                THEN v.LAND_VALUE::float / (v.LAND_VALUE + v.IMPROVEMENT_VALUE)
                ELSE NULL
            END AS LAND_TO_TOTAL_RATIO,
            s.SALE_PRICE AS LAST_SALE_PRICE,
            s.SALE_DATE AS LAST_SALE_DATE,
            s.CONVEYANCE_TYPE AS LAST_CONVEYANCE_TYPE
        FROM raw.bca_folio_descriptions d
        LEFT JOIN raw.bca_folio_addresses a
            ON d.FOLIO_ID = a.FOLIO_ID AND a.PRIMARY_FLAG = 'Y'
        LEFT JOIN raw.bca_folio_legal_descriptions l
            ON d.FOLIO_ID = l.FOLIO_ID
        LEFT JOIN raw.bca_folio_gnrl_prop_values v
            ON d.FOLIO_ID = v.FOLIO_ID
        LEFT JOIN (
            SELECT DISTINCT ON (FOLIO_ID)
                FOLIO_ID, SALE_PRICE, SALE_DATE, CONVEYANCE_TYPE
            FROM raw.bca_folio_sales
            ORDER BY FOLIO_ID, SALE_DATE DESC
        ) s ON d.FOLIO_ID = s.FOLIO_ID;
        """
        with self.engine.connect() as conn:
            conn.execute("CREATE SCHEMA IF NOT EXISTS features;")
            conn.execute(sql)
            result = conn.execute(
                "SELECT COUNT(*) FROM features.bca_properties;"
            )
            count = result.scalar()

        logger.info(f"Built unified property table with {count:,} properties")
        return count
