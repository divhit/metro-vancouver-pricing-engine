"""
Merge MLS sold listings with BC Assessment enriched properties.

Takes sold listings from the daily_intel SQLite database, matches them to
enriched_properties.parquet using address matching, and produces a training
dataset with sold_price as the target and all BC Assessment features.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.daily_intel.storage.database import get_connection
from src.daily_intel.analysis.market_vs_assessed import (
    _load_assessments,
    _build_lookups,
    _parse_mls_address,
    _find_match,
)

logger = logging.getLogger(__name__)


# MLS property_type → our canonical property_type
_MLS_TYPE_MAP = {
    "Apartment/Condo": "condo",
    "Townhouse": "townhome",
    "1/2 Duplex": "detached",
    "House/Single Family": "detached",
    "House with Acreage": "detached",
}


def load_sold_listings() -> pd.DataFrame:
    """Load all sold listings with valid prices from SQLite."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM sold_listings WHERE sold_price IS NOT NULL AND sold_price > 0"
    ).fetchall()
    cols = [d[0] for d in conn.execute("SELECT * FROM sold_listings LIMIT 1").description]
    conn.close()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame([dict(zip(cols, r)) for r in rows])


def build_market_training_data(
    enriched_path: str = "data/processed/enriched_properties.parquet",
) -> pd.DataFrame:
    """Build training dataset with sold_price as target.

    Matches MLS sold listings to BC Assessment enriched properties,
    merging sold_price and MLS features (bedrooms, floor_area, etc.)
    into the enriched property row.

    Returns:
        DataFrame with all enriched property features plus:
        - sold_price: actual sale price (training target)
        - mls_bedrooms, mls_floor_area, mls_bathrooms: MLS-sourced features
        - sold_date, dom: sale timing features
        - sale_to_assessment_ratio: sold_price / total_assessed_value
    """
    # Load enriched properties (BC Assessment + spatial features)
    assessments = pd.read_parquet(enriched_path)
    logger.info(f"Loaded {len(assessments)} enriched properties")

    # Load sold listings
    sold = load_sold_listings()
    if sold.empty:
        logger.warning("No sold listings found in database")
        return pd.DataFrame()
    logger.info(f"Loaded {len(sold)} sold listings")

    # Build lookups for address matching
    lookups = _build_lookups(assessments)

    # Match each sold listing to its enriched property row
    matched_rows = []
    unmatched = 0

    for _, mls_row in sold.iterrows():
        parsed = _parse_mls_address(mls_row["address"])
        match = _find_match(parsed, lookups)

        if match is None:
            unmatched += 1
            continue

        # Start with the full enriched property row (all features)
        row = match.to_dict()

        # Add sold price as the target
        row["sold_price"] = float(mls_row["sold_price"])

        # Add MLS-sourced features (more accurate than proxies)
        if pd.notna(mls_row.get("bedrooms")):
            row["bedrooms"] = int(mls_row["bedrooms"])
            row["bedrooms_imputed"] = False
        if pd.notna(mls_row.get("floor_area")) and mls_row["floor_area"] > 0:
            row["living_area_sqft"] = float(mls_row["floor_area"])
            row["living_area_imputed"] = False
        if pd.notna(mls_row.get("bathrooms")):
            row["bathrooms"] = float(mls_row["bathrooms"])
        if pd.notna(mls_row.get("year_built")):
            row["year_built"] = int(mls_row["year_built"])
        if pd.notna(mls_row.get("parking")):
            row["parking_spaces"] = int(mls_row["parking"])
        if pd.notna(mls_row.get("maint_fee")):
            row["maint_fee"] = float(mls_row["maint_fee"])
        if pd.notna(mls_row.get("dom")):
            row["days_on_market"] = int(mls_row["dom"])

        # Sale timing features
        row["sold_date"] = mls_row.get("sold_date")
        row["list_price"] = mls_row.get("list_price")
        row["mls_number"] = mls_row.get("mls_number")
        row["mls_property_type"] = mls_row.get("property_type")

        # Compute SAR
        if row.get("total_assessed_value", 0) > 0:
            row["sale_to_assessment_ratio"] = (
                row["sold_price"] / row["total_assessed_value"]
            )

        matched_rows.append(row)

    if not matched_rows:
        logger.error("No sold listings matched to assessments")
        return pd.DataFrame()

    result = pd.DataFrame(matched_rows)

    # Add derived MLS features
    if "maint_fee" in result.columns and "living_area_sqft" in result.columns:
        result["strata_fee_per_sqft"] = np.where(
            result["living_area_sqft"].notna() & (result["living_area_sqft"] > 0),
            result["maint_fee"] / result["living_area_sqft"],
            np.nan,
        )

    if "list_price" in result.columns:
        result["list_to_sold_ratio"] = np.where(
            result["list_price"].notna() & (result["list_price"] > 0),
            result["sold_price"] / result["list_price"],
            np.nan,
        )

    n_matched = len(result)
    total = len(sold)
    logger.info(
        f"Market training data built: {n_matched}/{total} matched "
        f"({n_matched/total*100:.1f}%), {unmatched} unmatched"
    )

    # Log breakdown by property type
    if "property_type" in result.columns:
        logger.info("By property_type:")
        for ptype, count in result["property_type"].value_counts().items():
            median_price = result[result["property_type"] == ptype]["sold_price"].median()
            logger.info(f"  {ptype}: {count} sales, median ${median_price:,.0f}")

    return result
