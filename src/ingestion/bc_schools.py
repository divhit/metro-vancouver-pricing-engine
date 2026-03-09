"""
BC Schools data ingestion: FSA results and school locations.

Sources:
- BC Ministry of Education FSA results (free CSV)
  https://catalogue.data.gov.bc.ca/dataset/bc-schools-foundation-skills-assessment-fsa-
- BC Schools K-12 with Francophone Indicators (DataBC)
  https://catalogue.data.gov.bc.ca/dataset/bc-schools-k-12-with-francophone-indicators
  Contains school name, district, lat/lon, facility type, and French program indicators.

Foundation Skills Assessment (FSA) is administered to Grade 4 and Grade 7
students across BC. Results include reading, writing, and numeracy scores
at the school level. These serve as a proxy for school quality, which is
a major factor in Metro Vancouver real estate pricing.

Note: BC does not publish school catchment boundaries province-wide.
The City of Vancouver publishes elementary and secondary catchment areas
(ingested via vancouver_open_data.py). For areas outside Vancouver, this
module provides distance-based scoring as a fallback.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import Point

logger = logging.getLogger(__name__)

# DataBC catalogue endpoints
CATALOGUE_API = "https://catalogue.data.gov.bc.ca/api/3/action/package_show"

# Dataset IDs
FSA_DATASET_ID = "bc-schools-foundation-skills-assessment-fsa-"
SCHOOL_LOCATIONS_DATASET_ID = "bc-schools-k-12-with-francophone-indicators"

# Direct fallback CSV URL in case the CKAN API lookup changes
SCHOOL_LOCATIONS_DIRECT_CSV = (
    "https://catalogue.data.gov.bc.ca/dataset/"
    "95da1091-7e8c-4aa6-9c1b-5ab159ea7b42/resource/"
    "5832eff2-3380-435e-911b-5ada41c1d30b/download/bc_k12_schools_2024-10.csv"
)

# School districts in Metro Vancouver
METRO_VANCOUVER_DISTRICTS = {
    "039": "Vancouver",
    "040": "New Westminster",
    "041": "Burnaby",
    "043": "Coquitlam",
    "044": "North Vancouver",
    "045": "West Vancouver",
    "036": "Surrey",
    "038": "Richmond",
    "035": "Langley",
    "042": "Maple Ridge-Pitt Meadows",
    "037": "Delta",
}

# FSA score categories
FSA_CATEGORIES = {
    "reading": "Reading",
    "writing": "Writing",
    "numeracy": "Numeracy",
}

# FSA proficiency levels
FSA_PROFICIENCY = {
    "emerging": 1,
    "on_track": 2,
    "extending": 3,
}


class BCSchoolsClient:
    """Client for BC Schools data: FSA results and school locations.

    Provides school quality scores derived from FSA assessment results
    and spatial school location data for proximity-based scoring.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the BC Schools client.

        Args:
            cache_dir: Directory to cache downloaded files.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "bc_schools_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "MetroVancouverPricingEngine/1.0"

    # ----------------------------------------------------------------
    # Data download helpers
    # ----------------------------------------------------------------

    def _get_csv_resources(self, dataset_id: str) -> list[dict]:
        """Get CSV resource URLs from the DataBC catalogue.

        Args:
            dataset_id: DataBC catalogue dataset ID.

        Returns:
            List of resource dicts with 'url', 'name', 'last_modified'.
        """
        try:
            response = self.session.get(
                CATALOGUE_API, params={"id": dataset_id}, timeout=30
            )
            response.raise_for_status()
            result = response.json()

            resources = []
            for r in result.get("result", {}).get("resources", []):
                if r.get("format", "").upper() == "CSV":
                    resources.append({
                        "url": r["url"],
                        "name": r.get("name", ""),
                        "last_modified": r.get("last_modified"),
                    })
            return resources

        except requests.RequestException as e:
            logger.error(f"Failed to query DataBC catalogue for {dataset_id}: {e}")
            return []

    def _download_csv(self, url: str, filename: str) -> pd.DataFrame:
        """Download a CSV file and return as DataFrame.

        Args:
            url: URL to download.
            filename: Cache filename.

        Returns:
            DataFrame of CSV contents.
        """
        cache_file = self.cache_dir / filename
        try:
            response = self.session.get(url, timeout=120)
            response.raise_for_status()
            cache_file.write_bytes(response.content)
            df = pd.read_csv(cache_file, low_memory=False, encoding="latin-1")
            logger.info(f"Downloaded {len(df):,} rows from {filename}")
            return df
        except requests.RequestException as e:
            logger.error(f"Failed to download {url}: {e}")
            return pd.DataFrame()

    # ----------------------------------------------------------------
    # School locations
    # ----------------------------------------------------------------

    def fetch_school_locations(self) -> gpd.GeoDataFrame:
        """Download BC K-12 school locations.

        Tries the CKAN catalogue API first, then falls back to a known
        direct CSV download URL, and finally checks for a local cached file.

        Returns:
            GeoDataFrame with school_id, name, type, district, and geometry.
        """
        resources = self._get_csv_resources(SCHOOL_LOCATIONS_DATASET_ID)

        df = pd.DataFrame()
        if resources:
            df = self._download_csv(resources[0]["url"], "school_locations.csv")

        # Fallback: direct download URL
        if df.empty:
            logger.warning(
                "CKAN catalogue lookup returned no CSV resources; "
                "trying direct download URL"
            )
            df = self._download_csv(SCHOOL_LOCATIONS_DIRECT_CSV, "school_locations.csv")

        # Fallback: local cached file
        if df.empty:
            local_path = self.cache_dir / "school_locations.csv"
            if not local_path.exists():
                # Also check the project data/raw directory
                project_path = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "bc_k12_schools.csv"
                if project_path.exists():
                    local_path = project_path
            if local_path.exists():
                logger.info(f"Loading school locations from local cache: {local_path}")
                df = pd.read_csv(local_path, low_memory=False, encoding="latin-1")

        if df.empty:
            logger.error("No school locations data available from any source")
            return gpd.GeoDataFrame()

        # Normalize column names
        df.columns = [c.upper().strip().replace(" ", "_") for c in df.columns]

        # Find lat/lon columns (varies by dataset version)
        # Prefer columns named exactly LATITUDE / LONGITUDE or prefixed
        # with SCHOOL_ (e.g. SCHOOL_LATITUDE).  The generic "LAT" / "LON"
        # substring match is kept as a last resort but could collide with
        # unrelated columns like HAS_LATE_FRENCH_IMMERSION.
        lat_col = None
        lon_col = None
        for c in df.columns:
            c_up = c.upper()
            if c_up in ("LATITUDE", "SCHOOL_LATITUDE", "LAT"):
                lat_col = c
            if c_up in ("LONGITUDE", "SCHOOL_LONGITUDE", "LON", "LONG"):
                lon_col = c

        # Fallback: loose substring match (last match wins)
        if lat_col is None or lon_col is None:
            for c in df.columns:
                if lat_col is None and "LATITUD" in c:
                    lat_col = c
                if lon_col is None and "LONGITUD" in c:
                    lon_col = c

        if lat_col is None or lon_col is None:
            logger.error("Could not find lat/lon columns in school locations data")
            return gpd.GeoDataFrame()

        # Drop rows without coordinates
        df = df.dropna(subset=[lat_col, lon_col])

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs="EPSG:4326",
        )

        logger.info(f"Loaded {len(gdf):,} school locations")
        return gdf

    def filter_metro_vancouver_schools(
        self, gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Filter schools to Metro Vancouver districts only.

        Args:
            gdf: Full BC schools GeoDataFrame.

        Returns:
            Filtered GeoDataFrame.
        """
        district_col = None
        for c in gdf.columns:
            if "DISTRICT" in c and "NUM" in c:
                district_col = c
                break

        if district_col is None:
            # Try matching on district name
            for c in gdf.columns:
                if "DISTRICT" in c and "NAME" not in c:
                    district_col = c
                    break

        if district_col is None:
            logger.warning("Could not identify district column; returning all schools")
            return gdf

        district_codes = set(METRO_VANCOUVER_DISTRICTS.keys())
        mask = gdf[district_col].astype(str).str.zfill(3).isin(district_codes)
        filtered = gdf[mask].copy()
        logger.info(f"Filtered to {len(filtered):,} Metro Vancouver schools")
        return filtered

    # ----------------------------------------------------------------
    # FSA results
    # ----------------------------------------------------------------

    def fetch_fsa_results(self) -> pd.DataFrame:
        """Download BC FSA (Foundation Skills Assessment) results.

        Returns:
            DataFrame with school-level FSA results across years.
        """
        resources = self._get_csv_resources(FSA_DATASET_ID)
        if not resources:
            logger.error("No CSV resources found for FSA data")
            return pd.DataFrame()

        # Download all available FSA CSV files (may be multiple years)
        all_dfs = []
        for r in resources:
            name = r.get("name", "")
            df = self._download_csv(r["url"], f"fsa_{name}.csv")
            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)
        combined.columns = [c.upper().strip().replace(" ", "_") for c in combined.columns]

        logger.info(f"Loaded {len(combined):,} total FSA records")
        return combined

    def compute_school_quality_scores(
        self,
        fsa_df: pd.DataFrame,
        year: Optional[int] = None,
    ) -> pd.DataFrame:
        """Compute school quality scores from FSA results.

        Creates a composite score (0-100) based on the percentage of
        students meeting or exceeding expectations in reading, writing,
        and numeracy.

        Args:
            fsa_df: Raw FSA DataFrame.
            year: Filter to specific year. Uses most recent if None.

        Returns:
            DataFrame with school_id, school_name, quality_score, and
            individual subject scores.
        """
        if fsa_df.empty:
            return pd.DataFrame()

        df = fsa_df.copy()

        # Identify year column
        year_col = None
        for c in df.columns:
            if "YEAR" in c or "SCHOOL_YEAR" in c:
                year_col = c
                break

        if year and year_col:
            df = df[df[year_col].astype(str).str.contains(str(year))]

        # Identify school ID column
        school_id_col = None
        for c in df.columns:
            if "SCHOOL" in c and ("CODE" in c or "ID" in c or "NUM" in c):
                school_id_col = c
                break

        school_name_col = None
        for c in df.columns:
            if "SCHOOL" in c and "NAME" in c:
                school_name_col = c
                break

        if school_id_col is None:
            logger.warning("Could not identify school ID column in FSA data")
            return pd.DataFrame()

        # Look for proficiency/score columns
        # FSA data typically has columns like PCT_MEETING_EXPECTATIONS or similar
        score_cols = []
        for c in df.columns:
            c_lower = c.lower()
            if any(k in c_lower for k in ["meeting", "exceeding", "on_track", "extending", "proficien"]):
                score_cols.append(c)

        if not score_cols:
            # Try numeric columns that might represent scores
            for c in df.columns:
                c_lower = c.lower()
                if any(k in c_lower for k in ["reading", "writing", "numeracy", "literacy"]):
                    score_cols.append(c)

        if not score_cols:
            logger.warning("Could not identify score columns in FSA data")
            return pd.DataFrame()

        # Convert score columns to numeric
        for col in score_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Compute composite score per school (average of all score columns)
        df["composite_raw"] = df[score_cols].mean(axis=1)

        # Aggregate to school level
        agg_dict = {"composite_raw": "mean"}
        if school_name_col:
            agg_dict[school_name_col] = "first"
        for sc in score_cols:
            agg_dict[sc] = "mean"

        school_scores = df.groupby(school_id_col).agg(agg_dict).reset_index()

        # Normalize to 0-100 scale
        raw_min = school_scores["composite_raw"].min()
        raw_max = school_scores["composite_raw"].max()
        if raw_max > raw_min:
            school_scores["quality_score"] = (
                (school_scores["composite_raw"] - raw_min) / (raw_max - raw_min) * 100
            ).round(1)
        else:
            school_scores["quality_score"] = 50.0

        return school_scores

    # ----------------------------------------------------------------
    # Proximity-based school scoring
    # ----------------------------------------------------------------

    def nearest_schools(
        self,
        lat: float,
        lon: float,
        schools_gdf: gpd.GeoDataFrame,
        n: int = 5,
        max_distance_m: float = 3000,
    ) -> gpd.GeoDataFrame:
        """Find the nearest schools to a location.

        Args:
            lat: Latitude.
            lon: Longitude.
            schools_gdf: GeoDataFrame of school locations.
            n: Maximum number of schools to return.
            max_distance_m: Maximum search distance in meters.

        Returns:
            GeoDataFrame of nearest schools with distance_m column.
        """
        point = gpd.GeoDataFrame(
            geometry=[Point(lon, lat)], crs="EPSG:4326"
        )

        # Project to UTM 10N for accurate distance
        schools_proj = schools_gdf.to_crs(epsg=32610)
        point_proj = point.to_crs(epsg=32610)

        schools_proj["distance_m"] = schools_proj.geometry.distance(
            point_proj.geometry.iloc[0]
        )

        nearby = schools_proj[schools_proj["distance_m"] <= max_distance_m].copy()
        nearby = nearby.nsmallest(n, "distance_m")

        return nearby.to_crs(epsg=4326)

    def compute_school_proximity_score(
        self,
        lat: float,
        lon: float,
        schools_gdf: gpd.GeoDataFrame,
        quality_scores: Optional[pd.DataFrame] = None,
    ) -> dict:
        """Compute a school proximity score for a location.

        Combines distance to nearest schools with quality scores (if available)
        to produce a composite education accessibility metric.

        Args:
            lat: Latitude.
            lon: Longitude.
            schools_gdf: GeoDataFrame of school locations.
            quality_scores: Optional DataFrame with school quality scores.

        Returns:
            Dict with proximity_score (0-100), nearest_school_m,
            schools_within_1km, and schools_within_2km.
        """
        nearby = self.nearest_schools(lat, lon, schools_gdf, n=10, max_distance_m=3000)

        if nearby.empty:
            return {
                "proximity_score": 0,
                "nearest_school_m": None,
                "schools_within_1km": 0,
                "schools_within_2km": 0,
                "avg_quality_nearby": None,
            }

        nearest_dist = nearby["distance_m"].min()
        within_1km = len(nearby[nearby["distance_m"] <= 1000])
        within_2km = len(nearby[nearby["distance_m"] <= 2000])

        # Distance-based score (exponential decay)
        # 0m = 100, 1000m = ~37, 2000m = ~14, 3000m = ~5
        distance_score = np.exp(-nearest_dist / 1000) * 100

        # Blend with quality if available
        avg_quality = None
        if quality_scores is not None and not quality_scores.empty:
            # Try to join on school ID
            school_id_col = None
            for c in nearby.columns:
                if "SCHOOL" in c.upper() and ("CODE" in c.upper() or "ID" in c.upper()):
                    school_id_col = c
                    break

            quality_id_col = None
            if school_id_col:
                for c in quality_scores.columns:
                    if "SCHOOL" in c.upper() and ("CODE" in c.upper() or "ID" in c.upper()):
                        quality_id_col = c
                        break

            if school_id_col and quality_id_col:
                merged = nearby.merge(
                    quality_scores[[quality_id_col, "quality_score"]],
                    left_on=school_id_col,
                    right_on=quality_id_col,
                    how="left",
                )
                if "quality_score" in merged.columns:
                    avg_quality = merged["quality_score"].mean()

        # Composite: 60% distance, 40% quality (if available)
        if avg_quality is not None:
            composite = 0.6 * distance_score + 0.4 * avg_quality
        else:
            composite = distance_score

        return {
            "proximity_score": round(min(composite, 100), 1),
            "nearest_school_m": round(nearest_dist, 0),
            "schools_within_1km": within_1km,
            "schools_within_2km": within_2km,
            "avg_quality_nearby": round(avg_quality, 1) if avg_quality else None,
        }
