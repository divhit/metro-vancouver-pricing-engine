"""
BC Property Transfer Tax (PTT) data ingestion.

Source: BC Data Catalogue / Open Canada
URL: https://catalogue.data.gov.bc.ca/dataset/property-transfer-tax-data

Weekly CSV dumps containing:
- Transaction counts by municipality
- Fair market values
- Property types (residential, commercial, etc.)
- Foreign buyer flags (post-2016 foreign buyers tax)
- Transaction dates

This is the single best free source for actual transaction volumes and
price distributions across Metro Vancouver municipalities.
"""

import logging
import tempfile
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# DataBC Catalogue API endpoint for PTT dataset
CATALOGUE_URL = "https://catalogue.data.gov.bc.ca/api/3/action/package_show"
DATASET_ID = "property-transfer-tax-data"

# Direct CSV download URLs (updated periodically by BC government)
PTT_CSV_URL = (
    "https://catalogue.data.gov.bc.ca/dataset/"
    "property-transfer-tax-data/resource/{resource_id}/download"
)

# Key columns we care about
PTT_COLUMNS = {
    "TRANSACTION_DATE": "datetime64[ns]",
    "MUNICIPALITY": "str",
    "PROPERTY_TYPE": "str",
    "FAIR_MARKET_VALUE": "float64",
    "TRANSFER_TYPE": "str",
    "FOREIGN_INVOLVEMENT": "str",  # Y/N flag for foreign buyer
    "TRANSACTION_COUNT": "int64",
    "RESIDENTIAL_FLAG": "str",
}

# Metro Vancouver municipalities to filter for
METRO_VANCOUVER_MUNICIPALITIES = [
    "Vancouver",
    "Burnaby",
    "Surrey",
    "Richmond",
    "Coquitlam",
    "North Vancouver District",
    "North Vancouver City",
    "West Vancouver",
    "New Westminster",
    "Delta",
    "Langley Township",
    "Langley City",
    "Port Coquitlam",
    "Port Moody",
    "Maple Ridge",
    "Pitt Meadows",
    "White Rock",
    "Lions Bay",
    "Bowen Island",
    "Anmore",
    "Belcarra",
    "Tsawwassen First Nation",
]


class BCPropertyTransferTaxClient:
    """Client for BC Property Transfer Tax open data.

    Downloads and parses weekly PTT CSV dumps from the BC Data Catalogue.
    Provides methods for filtering by municipality, date range, property
    type, and foreign buyer involvement.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the PTT client.

        Args:
            cache_dir: Directory to cache downloaded CSVs. Uses temp dir if None.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "ptt_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "MetroVancouverPricingEngine/1.0"

    def _get_resource_urls(self) -> list[dict]:
        """Query the DataBC catalogue API to get current CSV resource URLs.

        Returns:
            List of dicts with 'id', 'name', 'url', 'last_modified' keys.
        """
        params = {"id": DATASET_ID}
        try:
            response = self.session.get(CATALOGUE_URL, params=params, timeout=30)
            response.raise_for_status()
            result = response.json()

            resources = []
            for r in result.get("result", {}).get("resources", []):
                if r.get("format", "").upper() == "CSV":
                    resources.append({
                        "id": r["id"],
                        "name": r.get("name", ""),
                        "url": r["url"],
                        "last_modified": r.get("last_modified"),
                    })

            logger.info(f"Found {len(resources)} CSV resources in PTT dataset")
            return resources

        except requests.RequestException as e:
            logger.error(f"Failed to query DataBC catalogue: {e}")
            return []

    def download_ptt_data(self, resource_url: Optional[str] = None) -> pd.DataFrame:
        """Download PTT CSV data.

        If no URL is provided, fetches the latest resource from the catalogue.

        Args:
            resource_url: Direct URL to a PTT CSV file. Auto-detects if None.

        Returns:
            Raw DataFrame of PTT records.
        """
        if resource_url is None:
            resources = self._get_resource_urls()
            if not resources:
                logger.error("No PTT CSV resources found in catalogue")
                return pd.DataFrame()
            # Use the most recently modified resource
            resources.sort(key=lambda r: r.get("last_modified", ""), reverse=True)
            resource_url = resources[0]["url"]
            logger.info(f"Using latest PTT resource: {resources[0]['name']}")

        logger.info(f"Downloading PTT data from {resource_url}")
        try:
            response = self.session.get(resource_url, timeout=120)
            response.raise_for_status()

            # Save to cache
            cache_file = self.cache_dir / "ptt_latest.csv"
            cache_file.write_bytes(response.content)

            df = pd.read_csv(cache_file, low_memory=False)
            logger.info(f"Downloaded {len(df):,} PTT records")
            return df

        except requests.RequestException as e:
            logger.error(f"Failed to download PTT data: {e}")
            return pd.DataFrame()

    def load_from_file(self, file_path: str) -> pd.DataFrame:
        """Load PTT data from a local CSV file.

        Args:
            file_path: Path to the PTT CSV file.

        Returns:
            Raw DataFrame of PTT records.
        """
        logger.info(f"Loading PTT data from {file_path}")
        df = pd.read_csv(file_path, low_memory=False)
        logger.info(f"Loaded {len(df):,} PTT records")
        return df

    def clean_ptt_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize PTT data.

        - Normalizes column names to uppercase
        - Parses dates
        - Converts FMV to float
        - Adds derived columns

        Args:
            df: Raw PTT DataFrame.

        Returns:
            Cleaned DataFrame.
        """
        if df.empty:
            return df

        # Normalize column names
        df.columns = [c.upper().strip().replace(" ", "_") for c in df.columns]

        # Parse transaction date if present
        date_col = None
        for candidate in ["TRANSACTION_DATE", "DATE", "TRANS_DATE"]:
            if candidate in df.columns:
                date_col = candidate
                break

        if date_col:
            df["TRANSACTION_DATE"] = pd.to_datetime(df[date_col], errors="coerce")

        # Parse fair market value
        fmv_col = None
        for candidate in ["FAIR_MARKET_VALUE", "FMV", "MARKET_VALUE"]:
            if candidate in df.columns:
                fmv_col = candidate
                break

        if fmv_col:
            df["FAIR_MARKET_VALUE"] = pd.to_numeric(
                df[fmv_col].astype(str).str.replace(",", "").str.replace("$", ""),
                errors="coerce",
            )

        # Derive year and month for aggregation
        if "TRANSACTION_DATE" in df.columns:
            df["TRANSACTION_YEAR"] = df["TRANSACTION_DATE"].dt.year
            df["TRANSACTION_MONTH"] = df["TRANSACTION_DATE"].dt.month
            df["TRANSACTION_QUARTER"] = df["TRANSACTION_DATE"].dt.quarter

        return df

    def filter_metro_vancouver(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to Metro Vancouver municipalities only.

        Args:
            df: PTT DataFrame with MUNICIPALITY column.

        Returns:
            Filtered DataFrame.
        """
        if "MUNICIPALITY" not in df.columns:
            logger.warning("No MUNICIPALITY column found; returning unfiltered data")
            return df

        # Case-insensitive match
        mv_lower = [m.lower() for m in METRO_VANCOUVER_MUNICIPALITIES]
        mask = df["MUNICIPALITY"].str.lower().str.strip().isin(mv_lower)
        filtered = df[mask].copy()
        logger.info(
            f"Filtered to {len(filtered):,} Metro Vancouver records "
            f"(from {len(df):,} total)"
        )
        return filtered

    def get_transaction_summary(
        self,
        df: pd.DataFrame,
        group_by: str = "MUNICIPALITY",
        period: str = "TRANSACTION_QUARTER",
    ) -> pd.DataFrame:
        """Aggregate transaction counts and values.

        Args:
            df: Cleaned PTT DataFrame.
            group_by: Column to group by (e.g., MUNICIPALITY, PROPERTY_TYPE).
            period: Time period column for aggregation.

        Returns:
            Summary DataFrame with counts, mean/median FMV, total volume.
        """
        agg_cols = [group_by]
        if period in df.columns:
            agg_cols.append(period)
            if "TRANSACTION_YEAR" in df.columns and period != "TRANSACTION_YEAR":
                agg_cols.append("TRANSACTION_YEAR")

        agg_dict = {}
        if "FAIR_MARKET_VALUE" in df.columns:
            agg_dict["FAIR_MARKET_VALUE"] = ["count", "mean", "median", "sum"]
        else:
            # Fall back to just counting rows
            agg_dict[group_by] = ["count"]

        summary = df.groupby(agg_cols).agg(agg_dict).reset_index()
        summary.columns = ["_".join(c).strip("_") for c in summary.columns]

        return summary

    def get_foreign_buyer_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute foreign buyer involvement statistics.

        Args:
            df: Cleaned PTT DataFrame with FOREIGN_INVOLVEMENT column.

        Returns:
            DataFrame with foreign buyer counts and percentages by municipality.
        """
        foreign_col = None
        for candidate in ["FOREIGN_INVOLVEMENT", "FOREIGN_BUYER", "FOREIGN_FLAG"]:
            if candidate in df.columns:
                foreign_col = candidate
                break

        if foreign_col is None:
            logger.warning("No foreign buyer column found in PTT data")
            return pd.DataFrame()

        df["IS_FOREIGN"] = df[foreign_col].str.upper().str.strip().isin(["Y", "YES", "TRUE", "1"])

        group_cols = ["MUNICIPALITY"]
        if "TRANSACTION_YEAR" in df.columns:
            group_cols.append("TRANSACTION_YEAR")

        stats = df.groupby(group_cols).agg(
            total_transactions=("IS_FOREIGN", "count"),
            foreign_transactions=("IS_FOREIGN", "sum"),
        ).reset_index()

        stats["foreign_pct"] = (
            stats["foreign_transactions"] / stats["total_transactions"] * 100
        ).round(2)

        return stats

    def get_price_distribution(
        self,
        df: pd.DataFrame,
        municipality: Optional[str] = None,
        bins: Optional[list[float]] = None,
    ) -> pd.DataFrame:
        """Get price distribution for a municipality.

        Args:
            df: Cleaned PTT DataFrame.
            municipality: Filter to specific municipality. None for all.
            bins: Custom price bins. Defaults to standard brackets.

        Returns:
            DataFrame with price bracket counts.
        """
        if bins is None:
            bins = [
                0, 250_000, 500_000, 750_000, 1_000_000,
                1_500_000, 2_000_000, 3_000_000, 5_000_000, float("inf"),
            ]

        subset = df.copy()
        if municipality:
            subset = subset[
                subset["MUNICIPALITY"].str.lower() == municipality.lower()
            ]

        if "FAIR_MARKET_VALUE" not in subset.columns:
            return pd.DataFrame()

        subset["PRICE_BRACKET"] = pd.cut(
            subset["FAIR_MARKET_VALUE"], bins=bins, right=False
        )

        dist = subset.groupby("PRICE_BRACKET", observed=True).size().reset_index(name="count")
        dist["pct"] = (dist["count"] / dist["count"].sum() * 100).round(2)

        return dist
