"""
Property Universe Builder — the spine of the pricing engine.

Fetches the Vancouver Property Tax Report from the City of Vancouver
Open Data portal, cleans and normalizes it, classifies property types
from zoning codes, and produces a canonical property table that every
other feature and model module joins against.

Data source:
    City of Vancouver Open Data — Property Tax Report
    https://opendata.vancouver.ca/explore/dataset/property-tax-report

Key fields:
    PID, FOLIO, LAND_COORDINATE, civic address fields,
    NEIGHBOURHOOD_CODE (22 local areas), CURRENT_LAND_VALUE,
    CURRENT_IMPROVEMENT_VALUE, PREVIOUS_LAND_VALUE,
    PREVIOUS_IMPROVEMENT_VALUE, TAX_ASSESSMENT_YEAR, YEAR_BUILT,
    ZONING_DISTRICT, ZONING_CLASSIFICATION, TAX_LEVY, LEGAL_TYPE
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.features.feature_registry import PropertyType
from src.models.types import IngestionResult

logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================

PROPERTY_TAX_EXPORT_URL = (
    "https://opendata.vancouver.ca/api/v2/catalog/datasets/"
    "property-tax-report/exports/csv"
)

# The 22 officially recognized Vancouver local areas
VANCOUVER_LOCAL_AREAS = {
    "ARBUTUS-RIDGE": "Arbutus Ridge",
    "DOWNTOWN": "Downtown",
    "DUNBAR-SOUTHLANDS": "Dunbar-Southlands",
    "FAIRVIEW": "Fairview",
    "GRANDVIEW-WOODLAND": "Grandview-Woodland",
    "HASTINGS-SUNRISE": "Hastings-Sunrise",
    "KENSINGTON-CEDAR COTTAGE": "Kensington-Cedar Cottage",
    "KERRISDALE": "Kerrisdale",
    "KILLARNEY": "Killarney",
    "KITSILANO": "Kitsilano",
    "MARPOLE": "Marpole",
    "MOUNT PLEASANT": "Mount Pleasant",
    "OAKRIDGE": "Oakridge",
    "RENFREW-COLLINGWOOD": "Renfrew-Collingwood",
    "RILEY PARK": "Riley Park",
    "SHAUGHNESSY": "Shaughnessy",
    "SOUTH CAMBIE": "South Cambie",
    "STRATHCONA": "Strathcona",
    "SUNSET": "Sunset",
    "VICTORIA-FRASERVIEW": "Victoria-Fraserview",
    "WEST END": "West End",
    "WEST POINT GREY": "West Point Grey",
}

# ============================================================
# ZONING CLASSIFICATION TABLES
# ============================================================

# Single-family / duplex residential zones → DETACHED
_DETACHED_PREFIXES = (
    "RS-1", "RS-1A", "RS-1B", "RS-2", "RS-3", "RS-3A",
    "RS-5", "RS-6", "RS-7",
    "RT-",   # Two-family dwelling zones (RT-1, RT-2, etc.)
    "RA-1",  # Limited agriculture
    "FSHCA",  # First Shaughnessy
)

# Low-density multi-family zones (row houses, duplexes) → TOWNHOME
_TOWNHOME_PREFIXES = (
    "RM-1", "RM-1N", "RM-2", "RM-3", "RM-3A",
)

# Medium- and high-density multi-family zones → CONDO
_CONDO_PREFIXES = (
    "RM-4", "RM-4N", "RM-5", "RM-5A", "RM-5B", "RM-5C", "RM-5D",
    "RM-6",
    "FM-1",   # Multiple dwelling (Fairview)
    "DD",     # Downtown District
    "HA-",    # Heritage Area (often high-density)
    "FC-",    # False Creek
    "CWD",    # Comprehensive development (downtown west)
)

# Commercial zones that may include residential above → CONDO
_COMMERCIAL_RESIDENTIAL_PREFIXES = (
    "C-1", "C-2", "C-2B", "C-2C", "C-2C1",
    "C-3", "C-3A", "C-5", "C-6",
    "MC-1", "MC-2",  # Mixed commercial
)

# Industrial / large commercial → DEVELOPMENT_LAND
_INDUSTRIAL_PREFIXES = (
    "I-1", "I-2", "I-3", "I-4",
    "M-1", "M-1A", "M-1B", "M-2",
    "IC-1", "IC-2", "IC-3",  # Industrial-commercial
)

# Zones to exclude entirely (no residential component)
_EXCLUDE_ZONES = (
    "I-1", "I-2", "I-3", "I-4",
    "M-1", "M-1A", "M-1B", "M-2",
)


# ============================================================
# PROPERTY UNIVERSE BUILDER
# ============================================================

class PropertyUniverseBuilder:
    """Builds the master property table from Vancouver Property Tax Report.

    The Property Tax Report is the spine of the entire pricing engine.
    Every property gets a canonical row, and all features join to it.

    Usage::

        builder = PropertyUniverseBuilder()
        universe_df = builder.build_universe(year=2024)
        stats = builder.get_universe_stats()

    The resulting DataFrame has one row per PID with standardized
    column names, classified property types, and derived fields.
    """

    def __init__(
        self,
        cache_dir: Optional[str | Path] = None,
        request_timeout: int = 120,
    ):
        """Initialize the universe builder.

        Args:
            cache_dir: Directory for caching downloaded CSVs.
                       If None, data is fetched fresh every time.
            request_timeout: HTTP request timeout in seconds.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.request_timeout = request_timeout
        self.session = requests.Session()
        self._universe: pd.DataFrame | None = None

    # --------------------------------------------------------
    # DATA FETCHING
    # --------------------------------------------------------

    def fetch_property_tax_data(
        self,
        year: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch property tax data from the Vancouver Open Data API.

        Tries a local CSV cache first (if cache_dir is set), then
        falls back to the API. The API returns the full historical
        dataset; if ``year`` is specified, the result is filtered.

        Args:
            year: Tax assessment year to filter on. If None, all
                  years are returned.

        Returns:
            Raw DataFrame with original column names.

        Raises:
            requests.HTTPError: If the API request fails.
            FileNotFoundError: If cache_dir is set but no cached
                               file exists and the API call fails.
        """
        # Try local cache first
        cached_df = self._load_from_cache(year)
        if cached_df is not None:
            return cached_df

        # Fetch from API
        logger.info("Fetching property tax data from Vancouver Open Data API")
        params: dict = {
            "delimiter": ",",
            "limit": -1,
        }
        if year is not None:
            params["where"] = f"tax_assessment_year={year}"

        try:
            response = self.session.get(
                PROPERTY_TAX_EXPORT_URL,
                params=params,
                timeout=self.request_timeout,
                stream=True,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error(f"Failed to fetch property tax data: {exc}")
            raise

        # Parse CSV from response body
        from io import StringIO

        df = pd.read_csv(StringIO(response.text), low_memory=False)
        logger.info(
            f"Fetched {len(df):,} rows from property tax API"
            + (f" (year={year})" if year else " (all years)")
        )

        # Save to cache
        self._save_to_cache(df, year)

        return df

    def _load_from_cache(self, year: Optional[int]) -> pd.DataFrame | None:
        """Attempt to load data from a local CSV cache."""
        if self.cache_dir is None:
            return None

        cache_file = self._cache_path(year)
        if not cache_file.exists():
            return None

        logger.info(f"Loading property tax data from cache: {cache_file}")
        df = pd.read_csv(cache_file, low_memory=False)
        logger.info(f"Loaded {len(df):,} rows from cache")
        return df

    def _save_to_cache(self, df: pd.DataFrame, year: Optional[int]) -> None:
        """Save fetched data to a local CSV cache."""
        if self.cache_dir is None:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self._cache_path(year)
        df.to_csv(cache_file, index=False)
        logger.info(f"Cached {len(df):,} rows to {cache_file}")

    def _cache_path(self, year: Optional[int]) -> Path:
        """Build the cache file path for a given year."""
        filename = f"property_tax_{year}.csv" if year else "property_tax_all.csv"
        return self.cache_dir / filename

    # --------------------------------------------------------
    # UNIVERSE BUILDING
    # --------------------------------------------------------

    def build_universe(self, year: Optional[int] = None) -> pd.DataFrame:
        """Build the canonical property universe.

        Full pipeline:
          1. Fetch raw data
          2. Normalize column names
          3. Deduplicate on PID (keep latest assessment year)
          4. Classify property type from zoning codes
          5. Compute derived fields
          6. Filter to residential properties
          7. Validate and return

        Args:
            year: Assessment year to build for. If None, uses the
                  latest year available in the data.

        Returns:
            Clean DataFrame with one row per PID.
        """
        logger.info("Building property universe")

        # 1. Fetch raw data
        raw_df = self.fetch_property_tax_data(year=year)

        # 2. Normalize column names to snake_case lowercase
        df = self._normalize_columns(raw_df)

        # 3. Deduplicate — keep most recent assessment year per PID
        df = self._deduplicate(df)

        # 4. Clean and cast types
        df = self._clean_types(df)

        # 5. Classify property type from zoning
        df["property_type"] = df.apply(
            lambda row: self.classify_property_type(
                row.get("zoning_district", ""),
                row.get("legal_type", ""),
            ),
            axis=1,
        )

        # 6. Compute derived fields
        df = self._compute_derived_fields(df)

        # 7. Filter to residential properties
        df = self._filter_residential(df)

        # 8. Final sort and index
        df = df.sort_values("pid").reset_index(drop=True)

        self._universe = df
        logger.info(
            f"Property universe built: {len(df):,} properties, "
            f"{df['property_type'].nunique()} property types"
        )
        return df

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to lowercase snake_case.

        The API returns upper-case names like CURRENT_LAND_VALUE;
        we standardize to current_land_value.
        """
        df = df.copy()
        df.columns = [col.strip().lower() for col in df.columns]
        return df

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate on PID, keeping the latest assessment year.

        Some PIDs appear in multiple years. We keep the row with the
        highest tax_assessment_year so the universe reflects current
        state.
        """
        if "pid" not in df.columns:
            logger.warning(
                "Column 'pid' not found in data — skipping deduplication"
            )
            return df

        before = len(df)
        df = df.sort_values("tax_assessment_year", ascending=False)
        df = df.drop_duplicates(subset=["pid"], keep="first")
        after = len(df)

        if before != after:
            logger.info(
                f"Deduplicated on PID: {before:,} → {after:,} "
                f"(removed {before - after:,} duplicate rows)"
            )
        return df

    def _clean_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast columns to correct types and handle missing values."""
        df = df.copy()

        # PID — zero-padded 9-digit string
        if "pid" in df.columns:
            df["pid"] = (
                df["pid"]
                .astype(str)
                .str.replace(r"[^0-9]", "", regex=True)
                .str.zfill(9)
            )

        # Numeric value columns
        value_cols = [
            "current_land_value",
            "current_improvement_value",
            "previous_land_value",
            "previous_improvement_value",
            "tax_levy",
        ]
        for col in value_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # Year columns
        year_cols = ["tax_assessment_year", "year_built"]
        for col in year_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # String columns — strip and upper-case for consistency
        str_cols = [
            "neighbourhood_code",
            "zoning_district",
            "zoning_classification",
            "legal_type",
            "street_name",
            "property_postal_code",
        ]
        for col in str_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.upper()
                # Replace 'NAN' strings from astype(str) on NaN values
                df[col] = df[col].replace("NAN", np.nan)

        # Civic number columns
        for col in ["from_civic_number", "to_civic_number"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    # --------------------------------------------------------
    # PROPERTY TYPE CLASSIFICATION
    # --------------------------------------------------------

    @staticmethod
    def classify_property_type(
        zoning_code: str | None,
        legal_type: str | None,
    ) -> str:
        """Classify a property into a PropertyType based on zoning code.

        Mapping logic (Vancouver-specific):
          - RS-*, RT-* → DETACHED (single-family, two-family)
          - RM-1, RM-2, RM-3, RM-3A → TOWNHOME (low-density multi)
          - RM-4+, FM-1, DD, C-* residential → CONDO (high-density)
          - LEGAL_TYPE == STRATA → override to CONDO or TOWNHOME
          - I-*, M-* → DEVELOPMENT_LAND (industrial)
          - CD-1 → context-dependent (use LEGAL_TYPE as tiebreaker)

        Args:
            zoning_code: The ZONING_DISTRICT value (e.g. "RS-1", "RM-5").
            legal_type: The LEGAL_TYPE value ("LAND", "STRATA", "OTHER").

        Returns:
            PropertyType value string (e.g. "condo", "detached").
        """
        zoning = (zoning_code or "").strip().upper()
        legal = (legal_type or "").strip().upper()

        # --- Industrial / heavy commercial → DEVELOPMENT_LAND ---
        for prefix in _INDUSTRIAL_PREFIXES:
            if zoning == prefix or zoning.startswith(prefix + " "):
                return PropertyType.DEVELOPMENT_LAND.value

        # --- Comprehensive Development (CD-1) — ambiguous ---
        if zoning.startswith("CD-1"):
            # Use LEGAL_TYPE as tiebreaker
            if legal == "STRATA":
                return PropertyType.CONDO.value
            # Default CD-1 to development land (large sites)
            return PropertyType.DEVELOPMENT_LAND.value

        # --- Detached residential ---
        for prefix in _DETACHED_PREFIXES:
            if zoning == prefix or zoning.startswith(prefix):
                # STRATA override: strata lot in RS zone → still detached
                # (bare-land strata subdivisions)
                return PropertyType.DETACHED.value

        # --- Townhome zones ---
        for prefix in _TOWNHOME_PREFIXES:
            if zoning == prefix or zoning.startswith(prefix):
                if legal == "STRATA":
                    return PropertyType.TOWNHOME.value
                # Non-strata in RM-1/2/3 → likely detached on multi-family
                # zoned land, but classify as townhome per zoning intent
                return PropertyType.TOWNHOME.value

        # --- Condo zones (high-density multi-family) ---
        for prefix in _CONDO_PREFIXES:
            if zoning == prefix or zoning.startswith(prefix):
                return PropertyType.CONDO.value

        # --- Commercial zones with residential above ---
        for prefix in _COMMERCIAL_RESIDENTIAL_PREFIXES:
            if zoning == prefix or zoning.startswith(prefix):
                if legal == "STRATA":
                    return PropertyType.CONDO.value
                return PropertyType.DEVELOPMENT_LAND.value

        # --- Fallback: use LEGAL_TYPE ---
        if legal == "STRATA":
            return PropertyType.CONDO.value

        return PropertyType.DETACHED.value

    # --------------------------------------------------------
    # DERIVED FIELDS
    # --------------------------------------------------------

    def _compute_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add computed columns to the universe.

        Derived fields:
          - total_assessed_value = land + improvement
          - land_ratio = land_value / total_assessed_value
          - yoy_change_pct = (current - previous) / previous
          - full_address = civic number + street name
          - effective_age = assessment_year - year_built
        """
        df = df.copy()

        # Total assessed value
        df["total_assessed_value"] = (
            df["current_land_value"] + df["current_improvement_value"]
        )

        # Land-to-total ratio (guard against division by zero)
        df["land_ratio"] = np.where(
            df["total_assessed_value"] > 0,
            df["current_land_value"] / df["total_assessed_value"],
            np.nan,
        )

        # Year-over-year change (current vs previous total)
        previous_total = (
            df["previous_land_value"] + df["previous_improvement_value"]
        )
        df["yoy_change_pct"] = np.where(
            previous_total > 0,
            (df["total_assessed_value"] - previous_total) / previous_total,
            np.nan,
        )

        # Full street address
        if "from_civic_number" in df.columns and "street_name" in df.columns:
            civic = df["from_civic_number"].fillna(0).astype(int).astype(str)
            civic = civic.replace("0", "")
            df["full_address"] = (civic + " " + df["street_name"].fillna("")).str.strip()

        # Effective age
        if "year_built" in df.columns and "tax_assessment_year" in df.columns:
            df["effective_age"] = np.where(
                df["year_built"].notna() & (df["year_built"] > 1800),
                df["tax_assessment_year"] - df["year_built"],
                np.nan,
            )

        return df

    # --------------------------------------------------------
    # RESIDENTIAL FILTER
    # --------------------------------------------------------

    def _filter_residential(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to residential properties only.

        Excludes:
          - Properties with zero total assessed value
          - Pure industrial / commercial with no residential component
          - Zoning codes in the exclusion list
        """
        before = len(df)

        # Exclude zero-value records (vacant / exempt)
        mask = df["total_assessed_value"] > 0

        # Exclude pure industrial zones
        if "zoning_district" in df.columns:
            for zone_prefix in _EXCLUDE_ZONES:
                mask &= ~df["zoning_district"].fillna("").str.startswith(zone_prefix)

        # Keep only residential property types
        residential_types = {
            PropertyType.DETACHED.value,
            PropertyType.TOWNHOME.value,
            PropertyType.CONDO.value,
        }
        mask &= df["property_type"].isin(residential_types)

        df = df[mask].copy()
        after = len(df)

        logger.info(
            f"Residential filter: {before:,} → {after:,} "
            f"(excluded {before - after:,} non-residential)"
        )
        return df

    # --------------------------------------------------------
    # MULTI-YEAR PANEL
    # --------------------------------------------------------

    def build_multi_year_panel(
        self,
        years: list[int],
    ) -> pd.DataFrame:
        """Build a panel dataset with one row per PID-year.

        Fetches data for each requested year, normalizes, and stacks
        into a single DataFrame. Computes year-over-year changes
        within the panel.

        Args:
            years: List of tax assessment years to include
                   (e.g. [2020, 2021, 2022, 2023, 2024]).

        Returns:
            Panel DataFrame indexed by (pid, tax_assessment_year).
        """
        logger.info(f"Building multi-year panel for years: {years}")

        panels = []
        for yr in sorted(years):
            try:
                raw = self.fetch_property_tax_data(year=yr)
                df = self._normalize_columns(raw)
                df = self._clean_types(df)

                # Classify property type
                df["property_type"] = df.apply(
                    lambda row: self.classify_property_type(
                        row.get("zoning_district", ""),
                        row.get("legal_type", ""),
                    ),
                    axis=1,
                )

                # Compute per-year derived fields
                df["total_assessed_value"] = (
                    df["current_land_value"]
                    + df["current_improvement_value"]
                )

                previous_total = (
                    df["previous_land_value"]
                    + df["previous_improvement_value"]
                )
                df["yoy_change_pct"] = np.where(
                    previous_total > 0,
                    (df["total_assessed_value"] - previous_total) / previous_total,
                    np.nan,
                )

                panels.append(df)
                logger.info(f"Year {yr}: {len(df):,} properties")
            except Exception as exc:
                logger.error(f"Failed to fetch year {yr}: {exc}")
                continue

        if not panels:
            logger.error("No data fetched for any requested year")
            return pd.DataFrame()

        panel_df = pd.concat(panels, ignore_index=True)

        # Sort by PID and year for clean panel structure
        sort_cols = []
        if "pid" in panel_df.columns:
            sort_cols.append("pid")
        if "tax_assessment_year" in panel_df.columns:
            sort_cols.append("tax_assessment_year")
        if sort_cols:
            panel_df = panel_df.sort_values(sort_cols).reset_index(drop=True)

        logger.info(
            f"Multi-year panel built: {len(panel_df):,} rows "
            f"across {len(panels)} years"
        )
        return panel_df

    # --------------------------------------------------------
    # UNIVERSE STATS
    # --------------------------------------------------------

    def get_universe_stats(self) -> dict:
        """Return summary statistics about the current universe.

        Must call ``build_universe()`` first.

        Returns:
            Dict with counts by local_area, property_type, and zoning.

        Raises:
            RuntimeError: If the universe has not been built yet.
        """
        if self._universe is None:
            raise RuntimeError(
                "Universe not built yet. Call build_universe() first."
            )

        df = self._universe

        stats: dict = {
            "total_properties": len(df),
            "assessment_year": (
                int(df["tax_assessment_year"].max())
                if "tax_assessment_year" in df.columns
                else None
            ),
            "by_property_type": (
                df["property_type"]
                .value_counts()
                .to_dict()
            ),
            "by_local_area": (
                df["neighbourhood_code"]
                .value_counts()
                .to_dict()
                if "neighbourhood_code" in df.columns
                else {}
            ),
            "by_zoning": (
                df["zoning_district"]
                .value_counts()
                .head(30)
                .to_dict()
                if "zoning_district" in df.columns
                else {}
            ),
            "value_stats": {
                "median_total": (
                    float(df["total_assessed_value"].median())
                    if "total_assessed_value" in df.columns
                    else None
                ),
                "mean_total": (
                    float(df["total_assessed_value"].mean())
                    if "total_assessed_value" in df.columns
                    else None
                ),
                "median_land_ratio": (
                    float(df["land_ratio"].median())
                    if "land_ratio" in df.columns
                    else None
                ),
            },
        }

        return stats
