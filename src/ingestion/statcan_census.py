"""
Statistics Canada Census data ingestion.

Sources:
- StatCan Web Data Service (WDS) REST API (free, no key required)
- Census boundary files (Shapefile/GeoJSON downloads)

Provides census profile data at Dissemination Area (DA) and Census Tract (CT)
levels for the Vancouver CMA (CMA code 933).

Key variables for real estate pricing:
- Population and population density
- Median household income
- Housing tenure (owned vs. rented)
- Dwelling types and counts
- Immigration and visible minority data
- Education levels
- Age distribution

Reference:
  https://www12.statcan.gc.ca/wds-sdw/cpr2016-eng.cfm
  https://www12.statcan.gc.ca/census-recensement/2021/dp-pd/prof/index.cfm
"""

import logging
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd
import geopandas as gpd
import requests

logger = logging.getLogger(__name__)

# StatCan Census Profile WDS endpoint
WDS_BASE = "https://www12.statcan.gc.ca/rest/census-recensement"

# Census Profile table product IDs
CENSUS_PRODUCTS = {
    "2021": "98-401-X2021006",
    "2016": "98-401-X2016044",
}

# Vancouver CMA code
VANCOUVER_CMA = "933"

# Key characteristic IDs for real estate pricing (2021 Census)
# These are the row IDs in the Census Profile data
KEY_CHARACTERISTICS = {
    # Population
    "population_2021": 1,
    "population_2016": 2,
    "population_pct_change": 3,
    "population_density_per_km2": 6,
    # Age
    "median_age": 39,
    "pct_65_plus": 42,
    # Dwellings
    "total_private_dwellings": 4,
    "dwellings_occupied": 5,
    "single_detached": 44,
    "semi_detached": 45,
    "row_house": 46,
    "apartment_duplex": 47,
    "apartment_5_plus": 48,
    "apartment_under_5": 49,
    # Income
    "median_household_income": 237,
    "median_after_tax_income": 239,
    "prevalence_low_income": 245,
    # Housing
    "avg_shelter_cost_owner": 283,
    "avg_shelter_cost_renter": 284,
    "pct_owner_occupied": 270,
    "pct_renter_occupied": 271,
    "pct_spending_30pct_plus": 278,
    # Immigration
    "pct_immigrants": 120,
    "pct_recent_immigrants_5yr": 125,
    # Education
    "pct_university_degree": 188,
}

# Census boundary file download URLs
BOUNDARY_URLS = {
    "da_2021": (
        "https://www12.statcan.gc.ca/census-recensement/2021/geo/"
        "sip-pis/boundary-limites/files-fichiers/lda_000a21a_e.zip"
    ),
    "ct_2021": (
        "https://www12.statcan.gc.ca/census-recensement/2021/geo/"
        "sip-pis/boundary-limites/files-fichiers/lct_000a21a_e.zip"
    ),
    "csd_2021": (
        "https://www12.statcan.gc.ca/census-recensement/2021/geo/"
        "sip-pis/boundary-limites/files-fichiers/lcsd000a21a_e.zip"
    ),
}

# Vancouver CMA province code for filtering boundaries
BC_PROVINCE_CODE = "59"


class StatCanCensusClient:
    """Client for Statistics Canada Census data and boundary files.

    Fetches census profile data at DA and CT geographic levels for the
    Vancouver CMA. Also downloads census boundary shapefiles for spatial
    joins with property data.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the StatCan Census client.

        Args:
            cache_dir: Directory to cache downloaded files.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "statcan_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "MetroVancouverPricingEngine/1.0"

    # ----------------------------------------------------------------
    # Census Profile data via WDS
    # ----------------------------------------------------------------

    def fetch_census_profile(
        self,
        geo_code: str,
        geo_level: str = "DA",
        census_year: str = "2021",
        characteristics: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """Fetch Census Profile data for a geographic unit.

        Uses the StatCan Census Profile CSV download endpoint which is
        freely available without authentication.

        Args:
            geo_code: Geographic code (DA ID, CT ID, or CMA code).
            geo_level: Geographic level - 'DA', 'CT', 'CSD', or 'CMA'.
            census_year: '2021' or '2016'.
            characteristics: List of characteristic IDs to filter. None for all.

        Returns:
            DataFrame with characteristic_id, characteristic_name, and value columns.
        """
        product_id = CENSUS_PRODUCTS.get(census_year)
        if not product_id:
            logger.error(f"Unsupported census year: {census_year}")
            return pd.DataFrame()

        # StatCan CSV profile download endpoint
        url = (
            f"https://www12.statcan.gc.ca/census-recensement/{census_year}/"
            f"dp-pd/prof/details/download-telecharger/comp/GetFile.cfm"
        )
        params = {
            "Lang": "E",
            "TYPE": "CSV",
            "GEO_LEVEL": geo_level,
            "GEO_CODE": geo_code,
        }

        try:
            response = self.session.get(url, params=params, timeout=60)
            response.raise_for_status()

            # Parse the CSV response
            df = pd.read_csv(BytesIO(response.content), encoding="latin-1", low_memory=False)

            if characteristics:
                id_col = [c for c in df.columns if "CHARACTERISTIC_ID" in c.upper()]
                if id_col:
                    df = df[df[id_col[0]].isin(characteristics)]

            logger.info(f"Fetched {len(df)} census characteristics for {geo_level} {geo_code}")
            return df

        except requests.RequestException as e:
            logger.error(f"Failed to fetch census profile for {geo_code}: {e}")
            return pd.DataFrame()

    def fetch_vancouver_cma_profile(
        self,
        census_year: str = "2021",
    ) -> pd.DataFrame:
        """Fetch full Census Profile for the Vancouver CMA.

        Args:
            census_year: '2021' or '2016'.

        Returns:
            DataFrame with all census characteristics for Vancouver CMA.
        """
        return self.fetch_census_profile(
            geo_code=VANCOUVER_CMA,
            geo_level="CMA",
            census_year=census_year,
        )

    def fetch_key_demographics(
        self,
        geo_code: str,
        geo_level: str = "CT",
        census_year: str = "2021",
    ) -> dict:
        """Fetch key demographic variables for a geographic unit.

        Returns a dict of human-readable variable names to values,
        filtered to the KEY_CHARACTERISTICS relevant for real estate.

        Args:
            geo_code: Geographic code.
            geo_level: 'DA', 'CT', 'CSD', or 'CMA'.
            census_year: Census year.

        Returns:
            Dict mapping variable name to numeric value.
        """
        char_ids = list(KEY_CHARACTERISTICS.values())
        df = self.fetch_census_profile(
            geo_code=geo_code,
            geo_level=geo_level,
            census_year=census_year,
            characteristics=char_ids,
        )

        if df.empty:
            return {}

        # Build reverse lookup: characteristic_id -> variable name
        id_to_name = {v: k for k, v in KEY_CHARACTERISTICS.items()}

        results = {}
        id_col = [c for c in df.columns if "CHARACTERISTIC_ID" in c.upper()]
        val_col = [c for c in df.columns if "C1_COUNT_TOTAL" in c.upper() or "TOTAL" in c.upper()]

        if id_col and val_col:
            for _, row in df.iterrows():
                char_id = row[id_col[0]]
                if char_id in id_to_name:
                    value = row[val_col[0]]
                    try:
                        results[id_to_name[char_id]] = float(str(value).replace(",", ""))
                    except (ValueError, TypeError):
                        results[id_to_name[char_id]] = None

        return results

    # ----------------------------------------------------------------
    # Census boundary file downloads
    # ----------------------------------------------------------------

    def download_boundaries(
        self,
        geo_level: str = "da",
        census_year: str = "2021",
        filter_bc: bool = True,
    ) -> gpd.GeoDataFrame:
        """Download census boundary shapefiles and return as GeoDataFrame.

        Args:
            geo_level: 'da' (dissemination area), 'ct' (census tract), or 'csd'.
            census_year: '2021' or '2016'.
            filter_bc: If True, filter to BC only (province code 59).

        Returns:
            GeoDataFrame with census boundary polygons.
        """
        key = f"{geo_level}_{census_year}"
        if key not in BOUNDARY_URLS:
            logger.error(f"No boundary URL for {key}")
            return gpd.GeoDataFrame()

        url = BOUNDARY_URLS[key]
        cache_file = self.cache_dir / f"{key}.zip"

        # Download if not cached
        if not cache_file.exists():
            logger.info(f"Downloading {geo_level.upper()} boundaries for {census_year}...")
            try:
                response = self.session.get(url, timeout=300)
                response.raise_for_status()
                cache_file.write_bytes(response.content)
                logger.info(f"Downloaded {len(response.content) / 1e6:.1f} MB")
            except requests.RequestException as e:
                logger.error(f"Failed to download boundary file: {e}")
                return gpd.GeoDataFrame()

        # Extract and read shapefile from ZIP
        try:
            extract_dir = self.cache_dir / key
            if not extract_dir.exists():
                with zipfile.ZipFile(cache_file, "r") as zf:
                    zf.extractall(extract_dir)

            # Find the .shp file in the extracted directory
            shp_files = list(extract_dir.rglob("*.shp"))
            if not shp_files:
                logger.error(f"No shapefile found in {cache_file}")
                return gpd.GeoDataFrame()

            gdf = gpd.read_file(shp_files[0])
            logger.info(f"Loaded {len(gdf)} boundary polygons")

            # Filter to BC
            if filter_bc:
                pr_col = [c for c in gdf.columns if "PRUID" in c.upper() or "PR" in c.upper()]
                if pr_col:
                    gdf = gdf[gdf[pr_col[0]].astype(str) == BC_PROVINCE_CODE]
                    logger.info(f"Filtered to {len(gdf)} BC boundaries")

            # Reproject to WGS84 if needed
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)

            return gdf

        except Exception as e:
            logger.error(f"Failed to read boundary shapefile: {e}")
            return gpd.GeoDataFrame()

    def get_vancouver_cma_boundaries(
        self,
        geo_level: str = "ct",
        census_year: str = "2021",
    ) -> gpd.GeoDataFrame:
        """Get census boundaries clipped to the Vancouver CMA.

        Downloads full BC boundaries and filters to CMA 933.

        Args:
            geo_level: 'da' or 'ct'.
            census_year: Census year.

        Returns:
            GeoDataFrame with Vancouver CMA boundaries.
        """
        gdf = self.download_boundaries(geo_level, census_year, filter_bc=True)

        if gdf.empty:
            return gdf

        # Filter to Vancouver CMA
        cma_col = [c for c in gdf.columns if "CMAUID" in c.upper() or "CMA" in c.upper()]
        if cma_col:
            gdf = gdf[gdf[cma_col[0]].astype(str) == VANCOUVER_CMA]
            logger.info(f"Filtered to {len(gdf)} Vancouver CMA {geo_level.upper()} boundaries")

        return gdf

    # ----------------------------------------------------------------
    # Batch operations
    # ----------------------------------------------------------------

    def build_demographic_layer(
        self,
        geo_level: str = "ct",
        census_year: str = "2021",
    ) -> gpd.GeoDataFrame:
        """Build a complete demographic GeoDataFrame for Vancouver CMA.

        Downloads boundaries and attempts to join with census profile data.
        This is a convenience method for creating the spatial demographic
        layer used in the pricing model.

        Args:
            geo_level: 'da' or 'ct'.
            census_year: Census year.

        Returns:
            GeoDataFrame with boundaries and key demographic attributes.
        """
        gdf = self.get_vancouver_cma_boundaries(geo_level, census_year)
        if gdf.empty:
            return gdf

        logger.info(
            f"Built demographic layer with {len(gdf)} "
            f"{geo_level.upper()} zones for Vancouver CMA"
        )
        return gdf
