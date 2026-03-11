"""
Building footprint-based living area estimation.

When MLS data is unavailable, the single biggest missing feature is living area (sqft).
Microsoft Building Footprints provides 11.8M Canadian building outlines generated from
satellite imagery. By matching these to property parcels and estimating story count,
we derive a living area proxy with ~30% error — noisy but far better than nothing.

Sources:
- Microsoft Canadian Building Footprints: https://github.com/microsoft/CanadianBuildingFootprints
- Format: GeoJSON with polygon geometries
- License: ODbL (Open Database License)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import box
except ImportError:
    gpd = None  # type: ignore[assignment]
    box = None  # type: ignore[assignment,misc]

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

UTM10N = "EPSG:32610"
WGS84 = "EPSG:4326"

# Metro Vancouver bounding box (WGS84)
METRO_VAN_BBOX = {
    "min_lat": 49.0,
    "max_lat": 49.4,
    "min_lon": -123.3,
    "max_lon": -122.5,
}

# Microsoft Building Footprints GeoJSON URL for British Columbia
# Source: https://github.com/microsoft/CanadianBuildingFootprints
MS_FOOTPRINTS_URL = (
    "https://minedbuildings.z5.web.core.windows.net/legacy/"
    "canadian-buildings-v2/BritishColumbia.zip"
)

# Efficiency factors: ratio of usable floor area to gross floor area
# These account for walls, stairs, hallways, mechanical rooms
EFFICIENCY_FACTORS = {
    "wood_frame": 0.80,    # Older wood-frame buildings
    "concrete": 0.85,      # Concrete high-rises (more efficient)
    "townhome": 0.85,      # Townhomes (less common area)
    "detached": 0.90,      # Detached homes (minimal common area)
    "default": 0.82,       # Fallback
}

# Story estimation from zoning when not directly available
# Based on typical Vancouver development patterns
ZONING_STORY_ESTIMATES = {
    # Residential single/two-family (detached)
    "RS": 2.0,
    "RT": 2.5,     # Two-family, often with basement suite
    # Low-rise multi-family
    "RM-1": 3.0,
    "RM-2": 3.0,
    "RM-3": 4.0,
    "RM-3A": 4.0,
    # Mid-rise multi-family
    "RM-4": 6.0,
    "RM-5": 6.0,
    "RM-6": 6.0,
    # High-rise
    "FM-1": 12.0,
    # Commercial with residential
    "C-1": 4.0,
    "C-2": 6.0,
    "C-3A": 10.0,
    # Comprehensive Development (varies widely)
    "CD-1": 6.0,
    # Default for unmapped zones
    "default": 2.0,
}

SQM_TO_SQFT = 10.7639

# Maximum search radius (meters) for matching footprints to properties
MAX_MATCH_DISTANCE_M = 50.0

# Reasonable living area bounds (sqft)
MIN_LIVING_AREA_SQFT = 200.0
MAX_LIVING_AREA_SQFT = 15_000.0

# Reasonable story count bounds
MIN_STORIES = 1
MAX_STORIES = 60

# Garage size estimates (sqft)
SINGLE_GARAGE_SQFT = 400.0
DOUBLE_GARAGE_SQFT = 600.0
DOUBLE_GARAGE_LOT_THRESHOLD_SQFT = 4_000.0

# Basement utilization: typical Vancouver homes have 60-80% of footprint
# as usable basement space
BASEMENT_UTILIZATION_LOW = 0.60
BASEMENT_UTILIZATION_HIGH = 0.80

# Typical unit value for estimating condo unit counts
# Based on Metro Vancouver median condo assessment ~$500K-$700K
TYPICAL_CONDO_UNIT_VALUE = 600_000.0

# Improvement value per sqm thresholds for story estimation
# Higher improvement_value per sqm of footprint implies more stories
IMPROVEMENT_PER_SQM_STORY_BREAKPOINTS = [
    (2_000, 1.5),    # < $2K/sqm footprint → 1-2 stories
    (5_000, 3.0),    # $2K-$5K → ~3 stories
    (10_000, 6.0),   # $5K-$10K → ~6 stories
    (20_000, 12.0),  # $10K-$20K → ~12 stories
    (50_000, 20.0),  # $20K-$50K → ~20 stories
    (float("inf"), 35.0),  # > $50K → ~35 stories
]


class BuildingFootprintEstimator:
    """Estimate living area from satellite-derived building footprints.

    This module bridges the gap when MLS square footage is unavailable by
    using Microsoft Building Footprints (satellite-derived polygons) to
    estimate usable living area. The approach:

    1. Match each property to its building footprint polygon
    2. Estimate story count from zoning, assessment values, and year built
    3. Compute gross floor area = footprint * stories
    4. Apply efficiency factors and adjustments (garage, basement, units)

    Accuracy is roughly +/- 30% for detached homes and +/- 40% for condos
    (where unit count estimation introduces additional noise). Despite
    the noise, this is a high-value feature — living area has the single
    largest feature importance in pricing models.

    Usage:
        estimator = BuildingFootprintEstimator()
        enriched_gdf = estimator.enrich_properties(properties_gdf)
    """

    def __init__(self, cache_dir: str = "data/footprints") -> None:
        """Initialize the building footprint estimator.

        Args:
            cache_dir: Directory for caching downloaded footprint data.
                       Created automatically if it does not exist.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._footprints_cache: Optional[gpd.GeoDataFrame] = None
        logger.info(
            "BuildingFootprintEstimator initialized with cache_dir=%s",
            self.cache_dir,
        )

    # ------------------------------------------------------------------
    # 1. LOAD FOOTPRINTS
    # ------------------------------------------------------------------

    def load_footprints(
        self, filepath: str | None = None
    ) -> gpd.GeoDataFrame:
        """Load Microsoft Building Footprints for the Metro Vancouver area.

        If a local file is provided, reads it directly. Otherwise, attempts
        to download the BC GeoJSON from Microsoft's CDN and caches it
        locally. The full BC file is large (~500MB uncompressed), so we
        filter to the Metro Vancouver bounding box immediately after load.

        Args:
            filepath: Optional path to a local GeoJSON or GeoJSON.zip file
                      containing building footprints. If None, downloads
                      from Microsoft's CDN.

        Returns:
            GeoDataFrame projected to UTM10N with columns:
                - geometry: building footprint polygon (UTM10N)
                - footprint_area_sqm: polygon area in square meters

        Raises:
            FileNotFoundError: If the specified filepath does not exist.
            requests.HTTPError: If download from Microsoft CDN fails.
        """
        # Return cached footprints if already loaded
        if self._footprints_cache is not None:
            logger.debug("Returning cached footprints (%d polygons)", len(self._footprints_cache))
            return self._footprints_cache

        if filepath is not None:
            fp = Path(filepath)
            if not fp.exists():
                raise FileNotFoundError(
                    f"Footprint file not found: {filepath}"
                )
            logger.info("Loading footprints from local file: %s", filepath)
            gdf = gpd.read_file(filepath)
        else:
            cached_path = self.cache_dir / "BritishColumbia.zip"
            if cached_path.exists():
                logger.info(
                    "Loading footprints from cached file: %s", cached_path
                )
                gdf = gpd.read_file(cached_path)
            else:
                logger.info(
                    "Downloading BC building footprints from Microsoft CDN "
                    "(this may take several minutes)..."
                )
                gdf = self._download_footprints(cached_path)

        # Ensure WGS84 before bounding box filter
        if gdf.crs is None:
            logger.warning("Footprints have no CRS; assuming WGS84")
            gdf = gdf.set_crs(WGS84)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(WGS84)

        # Filter to Metro Vancouver bounding box
        initial_count = len(gdf)
        bbox_geom = box(
            METRO_VAN_BBOX["min_lon"],
            METRO_VAN_BBOX["min_lat"],
            METRO_VAN_BBOX["max_lon"],
            METRO_VAN_BBOX["max_lat"],
        )
        gdf = gdf[gdf.geometry.intersects(bbox_geom)].copy()
        logger.info(
            "Filtered footprints to Metro Vancouver bounding box: "
            "%d -> %d polygons",
            initial_count,
            len(gdf),
        )

        if len(gdf) == 0:
            logger.warning(
                "No footprints found within Metro Vancouver bounding box. "
                "Check that the source file covers this region."
            )

        # Project to UTM Zone 10N for accurate area calculations
        gdf = gdf.to_crs(UTM10N)

        # Compute footprint area in square meters from the polygon geometry
        gdf["footprint_area_sqm"] = gdf.geometry.area

        # Drop degenerate polygons (area < 1 sqm is noise)
        degenerate_count = (gdf["footprint_area_sqm"] < 1.0).sum()
        if degenerate_count > 0:
            logger.info(
                "Dropping %d degenerate footprints with area < 1 sqm",
                degenerate_count,
            )
            gdf = gdf[gdf["footprint_area_sqm"] >= 1.0].copy()

        # Reset index for clean spatial joins downstream
        gdf = gdf.reset_index(drop=True)

        logger.info(
            "Loaded %d building footprints. Median area: %.1f sqm, "
            "Mean area: %.1f sqm",
            len(gdf),
            gdf["footprint_area_sqm"].median(),
            gdf["footprint_area_sqm"].mean(),
        )

        self._footprints_cache = gdf
        return gdf

    def _download_footprints(self, dest_path: Path) -> gpd.GeoDataFrame:
        """Download building footprints from Microsoft CDN.

        Streams the download to avoid loading the entire file into memory
        at once, then reads the cached file with GeoPandas.

        Args:
            dest_path: Local path to save the downloaded file.

        Returns:
            GeoDataFrame of building footprint polygons.

        Raises:
            requests.HTTPError: If the download request fails.
        """
        response = requests.get(MS_FOOTPRINTS_URL, stream=True, timeout=300)
        response.raise_for_status()

        total_bytes = 0
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                total_bytes += len(chunk)

        logger.info(
            "Downloaded %.1f MB to %s",
            total_bytes / (1024 * 1024),
            dest_path,
        )

        return gpd.read_file(dest_path)

    # ------------------------------------------------------------------
    # 2. MATCH FOOTPRINTS TO PROPERTIES
    # ------------------------------------------------------------------

    def match_footprints_to_properties(
        self,
        properties_gdf: gpd.GeoDataFrame,
        footprints_gdf: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """Spatial join building footprints to property locations.

        For each property point, finds the nearest building footprint
        polygon within MAX_MATCH_DISTANCE_M (50m). When multiple
        footprints are near a single property (e.g., a detached home
        with a separate garage structure), the largest footprint is
        selected as the primary building.

        For strata/condo properties, the matched footprint represents
        the entire building, not an individual unit. The per-unit
        calculation happens downstream in estimate_living_area().

        Args:
            properties_gdf: GeoDataFrame of property locations with point
                            geometries. Must have a geometry column.
                            Optionally includes 'property_type' column.
            footprints_gdf: GeoDataFrame of building footprint polygons,
                            as returned by load_footprints(). Must be
                            projected to UTM10N.

        Returns:
            A copy of properties_gdf with an additional column:
                - building_footprint_sqm (float): Area of the matched
                  building footprint in square meters, or NaN if no
                  footprint was found within the search radius.

        Note:
            Both GeoDataFrames must be in the same CRS (UTM10N).
            If properties_gdf is in WGS84, it will be reprojected
            automatically.
        """
        props = properties_gdf.copy()
        n_props = len(props)
        logger.info(
            "Matching %d properties to %d footprints (max distance: %.0fm)",
            n_props,
            len(footprints_gdf),
            MAX_MATCH_DISTANCE_M,
        )

        # Ensure both are in UTM10N
        if props.crs is None:
            logger.warning("Properties have no CRS; assuming WGS84")
            props = props.set_crs(WGS84)
        if props.crs.to_epsg() != 32610:
            props = props.to_crs(UTM10N)

        if footprints_gdf.crs is None or footprints_gdf.crs.to_epsg() != 32610:
            raise ValueError(
                "Footprints GeoDataFrame must be projected to UTM10N "
                "(EPSG:32610). Call load_footprints() first."
            )

        # Buffer property points to create search radius circles
        props_buffered = props.copy()
        props_buffered["_original_geometry"] = props_buffered.geometry
        props_buffered.geometry = props_buffered.geometry.buffer(MAX_MATCH_DISTANCE_M)

        # Spatial join: find all footprints that intersect each property buffer
        joined = gpd.sjoin(
            props_buffered,
            footprints_gdf[["geometry", "footprint_area_sqm"]],
            how="left",
            predicate="intersects",
        )

        if joined.empty:
            logger.warning("Spatial join produced no matches")
            props["building_footprint_sqm"] = np.nan
            return props

        # For properties with multiple footprint matches, keep the largest
        # This handles cases like detached homes with separate garage structures
        # — we want the main building, not the garage
        joined["_prop_idx"] = joined.index

        # Calculate distance from property point to footprint centroid
        # to break ties and prefer closer buildings
        footprint_centroids = footprints_gdf.geometry.centroid
        joined["_distance_to_footprint"] = np.nan

        # For each match, compute distance from the property point
        # to the matched footprint centroid
        for idx in joined.index.unique():
            rows = joined.loc[[idx]]
            if rows["index_right"].isna().all():
                continue
            prop_point = rows["_original_geometry"].iloc[0]
            for row_idx, row in rows.iterrows():
                fp_idx = row["index_right"]
                if pd.notna(fp_idx):
                    fp_idx = int(fp_idx)
                    fp_centroid = footprint_centroids.iloc[fp_idx]
                    joined.loc[row_idx, "_distance_to_footprint"] = (
                        prop_point.distance(fp_centroid)
                    )

        # Sort by footprint area descending (largest first), then by distance
        # ascending (closest first as tiebreaker)
        joined = joined.sort_values(
            ["_prop_idx", "footprint_area_sqm", "_distance_to_footprint"],
            ascending=[True, False, True],
        )

        # Keep only the best match per property (largest footprint)
        best_matches = joined.drop_duplicates(subset=["_prop_idx"], keep="first")

        # Map the matched footprint area back to the original properties
        props["building_footprint_sqm"] = best_matches.set_index("_prop_idx")[
            "footprint_area_sqm"
        ]

        matched_count = props["building_footprint_sqm"].notna().sum()
        unmatched_count = props["building_footprint_sqm"].isna().sum()
        logger.info(
            "Footprint matching complete: %d matched (%.1f%%), "
            "%d unmatched (%.1f%%)",
            matched_count,
            100 * matched_count / max(n_props, 1),
            unmatched_count,
            100 * unmatched_count / max(n_props, 1),
        )

        if matched_count > 0:
            logger.info(
                "Matched footprint stats — Median: %.1f sqm, "
                "Mean: %.1f sqm, Min: %.1f sqm, Max: %.1f sqm",
                props["building_footprint_sqm"].median(),
                props["building_footprint_sqm"].mean(),
                props["building_footprint_sqm"].min(),
                props["building_footprint_sqm"].max(),
            )

        return props

    # ------------------------------------------------------------------
    # 3. ESTIMATE STORIES
    # ------------------------------------------------------------------

    def estimate_stories(self, properties_df: pd.DataFrame) -> pd.DataFrame:
        """Estimate the number of stories for each property.

        Uses a cascading strategy, applying the most informative signal
        available for each property:

        1. Improvement value / footprint area ratio (most direct signal
           for how much vertical development exists on the footprint)
        2. Zoning code lookup (reliable for standard zones)
        3. Year built heuristic (weaker signal, used as fallback)

        For strata/condo properties, a specialized estimator uses the
        improvement value per square meter of footprint to infer the
        likely number of stories in the building.

        Args:
            properties_df: DataFrame with optional columns:
                - building_footprint_sqm (float)
                - improvement_value (float): BC Assessment improvement value
                - land_value (float): BC Assessment land value
                - zoning_code (str): e.g., 'RS-1', 'RM-3A', 'C-2'
                - year_built (int)
                - property_type (str): 'DETACHED', 'CONDO', 'TOWNHOME'

        Returns:
            A copy of properties_df with additional columns:
                - estimated_stories (float): Estimated number of stories,
                  clamped to [1, 60].
                - stories_source (str): Method used for estimation, one of:
                  'improvement_ratio', 'zoning', 'year_built', 'default'.
        """
        df = properties_df.copy()
        n_rows = len(df)
        logger.info("Estimating stories for %d properties", n_rows)

        # Initialize output columns
        df["estimated_stories"] = np.nan
        df["stories_source"] = "default"

        # Normalize column availability
        has_footprint = "building_footprint_sqm" in df.columns
        has_improvement = "improvement_value" in df.columns
        has_land = "land_value" in df.columns
        has_zoning = "zoning_code" in df.columns
        has_year = "year_built" in df.columns
        has_ptype = "property_type" in df.columns

        # ----- Strategy 1: Improvement value per sqm of footprint -----
        # This is the best proxy: more $ invested above the footprint = more stories
        if has_footprint and has_improvement:
            mask = (
                df["building_footprint_sqm"].notna()
                & (df["building_footprint_sqm"] > 0)
                & df["improvement_value"].notna()
                & (df["improvement_value"] > 0)
                & df["estimated_stories"].isna()
            )
            if mask.any():
                improvement_per_sqm = (
                    df.loc[mask, "improvement_value"]
                    / df.loc[mask, "building_footprint_sqm"]
                )
                stories = self._improvement_ratio_to_stories(improvement_per_sqm)
                df.loc[mask, "estimated_stories"] = stories
                df.loc[mask, "stories_source"] = "improvement_ratio"
                logger.info(
                    "Strategy 1 (improvement ratio): estimated stories for "
                    "%d properties",
                    mask.sum(),
                )

        # ----- Strategy 2: Zoning code lookup -----
        if has_zoning:
            mask = df["estimated_stories"].isna() & df["zoning_code"].notna()
            if mask.any():
                df.loc[mask, "estimated_stories"] = df.loc[mask, "zoning_code"].apply(
                    self._zoning_to_stories
                )
                df.loc[mask, "stories_source"] = "zoning"
                logger.info(
                    "Strategy 2 (zoning): estimated stories for %d properties",
                    mask.sum(),
                )

        # ----- Strategy 3: Year built heuristic -----
        if has_year:
            mask = df["estimated_stories"].isna() & df["year_built"].notna()
            if mask.any():
                df.loc[mask, "estimated_stories"] = df.loc[mask, "year_built"].apply(
                    self._year_built_to_stories
                )
                df.loc[mask, "stories_source"] = "year_built"
                logger.info(
                    "Strategy 3 (year built): estimated stories for "
                    "%d properties",
                    mask.sum(),
                )

        # ----- Fallback: default -----
        still_missing = df["estimated_stories"].isna()
        if still_missing.any():
            df.loc[still_missing, "estimated_stories"] = ZONING_STORY_ESTIMATES[
                "default"
            ]
            df.loc[still_missing, "stories_source"] = "default"
            logger.info(
                "Fallback (default=%.1f): assigned stories for %d properties",
                ZONING_STORY_ESTIMATES["default"],
                still_missing.sum(),
            )

        # ----- Property-type-specific adjustments -----
        if has_ptype:
            # Detached homes rarely exceed 3 stories above grade
            detached_mask = (
                df["property_type"].str.upper() == "DETACHED"
            ) & (df["estimated_stories"] > 3.5)
            if detached_mask.any():
                logger.debug(
                    "Clamping %d detached properties from >3.5 to 3.0 stories",
                    detached_mask.sum(),
                )
                df.loc[detached_mask, "estimated_stories"] = 3.0

            # Townhomes rarely exceed 4 stories
            townhome_mask = (
                df["property_type"].str.upper() == "TOWNHOME"
            ) & (df["estimated_stories"] > 4.5)
            if townhome_mask.any():
                logger.debug(
                    "Clamping %d townhome properties from >4.5 to 4.0 stories",
                    townhome_mask.sum(),
                )
                df.loc[townhome_mask, "estimated_stories"] = 4.0

        # Final clamp to absolute bounds
        df["estimated_stories"] = df["estimated_stories"].clip(
            lower=MIN_STORIES, upper=MAX_STORIES
        )

        # Log summary
        for source in df["stories_source"].unique():
            source_mask = df["stories_source"] == source
            logger.info(
                "Stories source '%s': %d properties, median=%.1f, mean=%.1f",
                source,
                source_mask.sum(),
                df.loc[source_mask, "estimated_stories"].median(),
                df.loc[source_mask, "estimated_stories"].mean(),
            )

        return df

    @staticmethod
    def _improvement_ratio_to_stories(
        improvement_per_sqm: pd.Series,
    ) -> pd.Series:
        """Convert improvement value per sqm of footprint to story estimate.

        Uses piecewise linear interpolation between breakpoints derived
        from Metro Vancouver assessment data patterns.

        Args:
            improvement_per_sqm: Series of $ improvement value per sqm
                                 of building footprint area.

        Returns:
            Series of estimated story counts.
        """
        stories = pd.Series(index=improvement_per_sqm.index, dtype=float)

        for i, (threshold, est_stories) in enumerate(
            IMPROVEMENT_PER_SQM_STORY_BREAKPOINTS
        ):
            if i == 0:
                mask = improvement_per_sqm < threshold
                stories[mask] = est_stories
            else:
                prev_threshold = IMPROVEMENT_PER_SQM_STORY_BREAKPOINTS[i - 1][0]
                prev_stories = IMPROVEMENT_PER_SQM_STORY_BREAKPOINTS[i - 1][1]
                mask = (improvement_per_sqm >= prev_threshold) & (
                    improvement_per_sqm < threshold
                )
                if mask.any():
                    # Linear interpolation within bracket
                    if threshold == float("inf"):
                        stories[mask] = est_stories
                    else:
                        frac = (
                            improvement_per_sqm[mask] - prev_threshold
                        ) / (threshold - prev_threshold)
                        stories[mask] = prev_stories + frac * (
                            est_stories - prev_stories
                        )

        return stories

    @staticmethod
    def _zoning_to_stories(zoning_code: str) -> float:
        """Look up estimated stories from zoning code.

        Handles both exact matches (e.g., 'RM-3A') and prefix matches
        (e.g., 'RS-1' matches the 'RS' entry). Falls back to the
        default value for unrecognized codes.

        Args:
            zoning_code: Vancouver zoning designation string.

        Returns:
            Estimated number of stories.
        """
        if not isinstance(zoning_code, str):
            return ZONING_STORY_ESTIMATES["default"]

        code = zoning_code.strip().upper()

        # Try exact match first
        if code in ZONING_STORY_ESTIMATES:
            return ZONING_STORY_ESTIMATES[code]

        # Try prefix match (e.g., 'RS-1' -> 'RS', 'RM-3A' -> 'RM-3A' then 'RM-3')
        # Walk from the full code down to just the first two characters
        for end_idx in range(len(code), 1, -1):
            prefix = code[:end_idx]
            if prefix in ZONING_STORY_ESTIMATES:
                return ZONING_STORY_ESTIMATES[prefix]

        return ZONING_STORY_ESTIMATES["default"]

    @staticmethod
    def _year_built_to_stories(year_built: int) -> float:
        """Heuristic story estimate from year of construction.

        Pre-1960 Vancouver homes were predominantly 1-2 story bungalows
        and small houses. The 1960-1990 era saw more varied forms. Post-
        1990 construction follows modern zoning more closely (handled
        by the zoning lookup strategy instead).

        This is the weakest estimation signal and should only be used
        when improvement ratio and zoning are both unavailable.

        Args:
            year_built: Four-digit year of construction.

        Returns:
            Estimated number of stories.
        """
        if not isinstance(year_built, (int, float)) or np.isnan(year_built):
            return ZONING_STORY_ESTIMATES["default"]

        year = int(year_built)
        if year < 1940:
            return 1.5  # Character homes, often 1.5-story
        elif year < 1960:
            return 2.0  # Post-war bungalows and split-levels
        elif year < 1980:
            return 2.0  # Mix of forms, conservative estimate
        elif year < 2000:
            return 2.5  # Trend toward larger homes
        else:
            return 2.5  # Modern builds, but if we're here we lack zoning info

    # ------------------------------------------------------------------
    # 4. ESTIMATE LIVING AREA
    # ------------------------------------------------------------------

    def estimate_living_area(self, properties_df: pd.DataFrame) -> pd.DataFrame:
        """Compute estimated living area from footprint and story estimates.

        Applies property-type-specific logic:

        **Detached homes:**
        - Base area = footprint * stories * efficiency factor
        - Subtract estimated garage (400 sqft single, 600 sqft double)
        - Add estimated basement (60-80% of footprint as usable space)

        **Townhomes:**
        - Base area = footprint * stories * efficiency factor
        - Divide by estimated unit count if the footprint covers
          multiple attached units

        **Condos (most complex):**
        - Building area = footprint * stories * efficiency factor
        - Divide by estimated unit count
        - Unit count is the hardest part — proxied from
          total_building_improvement_value / typical_unit_value

        Args:
            properties_df: DataFrame with required columns:
                - building_footprint_sqm (float)
                - estimated_stories (float)
              And optional columns for refinement:
                - property_type (str)
                - lot_size_sqft (float)
                - improvement_value (float)
                - building_total_units (int)
                - year_built (int)

        Returns:
            A copy of properties_df with additional columns:
                - estimated_living_area_sqft (float): Living area estimate,
                  clamped to [200, 15000].
                - living_area_source (str): Always 'footprint_estimate'.
                - living_area_confidence (float): Confidence score 0-1,
                  based on estimation method quality.
        """
        df = properties_df.copy()
        n_rows = len(df)
        logger.info("Estimating living area for %d properties", n_rows)

        # Initialize output columns
        df["estimated_living_area_sqft"] = np.nan
        df["living_area_source"] = "footprint_estimate"
        df["living_area_confidence"] = 0.0

        # Can only estimate where we have a footprint
        has_footprint = (
            df["building_footprint_sqm"].notna()
            & (df["building_footprint_sqm"] > 0)
        )
        has_stories = (
            df["estimated_stories"].notna() & (df["estimated_stories"] > 0)
        )
        estimable = has_footprint & has_stories

        if not estimable.any():
            logger.warning(
                "No properties have both footprint and story estimates. "
                "Cannot estimate living area."
            )
            return df

        logger.info(
            "%d of %d properties have footprint + story data for estimation",
            estimable.sum(),
            n_rows,
        )

        # Detect property type (normalize to uppercase for comparison)
        has_ptype = "property_type" in df.columns
        if has_ptype:
            ptype = df["property_type"].fillna("").str.strip().str.upper()
        else:
            ptype = pd.Series("", index=df.index)

        # ----- DETACHED HOMES -----
        detached_mask = estimable & (ptype == "DETACHED")
        if detached_mask.any():
            self._estimate_detached(df, detached_mask)

        # ----- TOWNHOMES -----
        townhome_mask = estimable & (ptype == "TOWNHOME")
        if townhome_mask.any():
            self._estimate_townhome(df, townhome_mask)

        # ----- CONDOS -----
        condo_mask = estimable & (ptype == "CONDO")
        if condo_mask.any():
            self._estimate_condo(df, condo_mask)

        # ----- UNKNOWN / OTHER PROPERTY TYPES -----
        # Treat as detached if type is unknown (conservative)
        other_mask = estimable & ~detached_mask & ~townhome_mask & ~condo_mask
        if other_mask.any():
            logger.info(
                "Estimating %d properties with unknown type as detached",
                other_mask.sum(),
            )
            self._estimate_detached(df, other_mask, confidence_penalty=0.15)

        # Final clamp to reasonable bounds
        df["estimated_living_area_sqft"] = df["estimated_living_area_sqft"].clip(
            lower=MIN_LIVING_AREA_SQFT, upper=MAX_LIVING_AREA_SQFT
        )

        # Clamp confidence to [0, 1]
        df["living_area_confidence"] = df["living_area_confidence"].clip(
            lower=0.0, upper=1.0
        )

        # Log summary
        estimated_mask = df["estimated_living_area_sqft"].notna()
        if estimated_mask.any():
            logger.info(
                "Living area estimation complete: %d properties estimated. "
                "Median: %.0f sqft, Mean: %.0f sqft, "
                "Median confidence: %.2f",
                estimated_mask.sum(),
                df.loc[estimated_mask, "estimated_living_area_sqft"].median(),
                df.loc[estimated_mask, "estimated_living_area_sqft"].mean(),
                df.loc[estimated_mask, "living_area_confidence"].median(),
            )

        return df

    def _estimate_detached(
        self,
        df: pd.DataFrame,
        mask: pd.Series,
        confidence_penalty: float = 0.0,
    ) -> None:
        """Estimate living area for detached homes (in-place).

        Detached homes are the most straightforward:
        - Main living area = footprint * stories * efficiency
        - Subtract garage area
        - Add usable basement area (Vancouver homes commonly have basements)

        Args:
            df: DataFrame to modify in-place.
            mask: Boolean mask selecting detached properties.
            confidence_penalty: Reduction to confidence score (used when
                                property type is uncertain).
        """
        footprint_sqft = (
            df.loc[mask, "building_footprint_sqm"] * SQM_TO_SQFT
        )
        stories = df.loc[mask, "estimated_stories"]
        efficiency = EFFICIENCY_FACTORS["detached"]

        # Base above-grade living area
        base_area = footprint_sqft * stories * efficiency

        # Subtract estimated garage
        garage_sqft = pd.Series(
            SINGLE_GARAGE_SQFT, index=df.loc[mask].index
        )
        if "lot_size_sqft" in df.columns:
            large_lot_mask = (
                df.loc[mask, "lot_size_sqft"].fillna(0)
                > DOUBLE_GARAGE_LOT_THRESHOLD_SQFT
            )
            garage_sqft[large_lot_mask] = DOUBLE_GARAGE_SQFT

        # Only subtract garage from first floor
        base_area = base_area - garage_sqft

        # Add usable basement space
        # Most Vancouver detached homes (especially pre-2000) have basements.
        # Newer homes tend to have more finished basement area.
        has_year = "year_built" in df.columns
        basement_ratio = pd.Series(
            BASEMENT_UTILIZATION_LOW, index=df.loc[mask].index
        )
        if has_year:
            newer_mask = df.loc[mask, "year_built"].fillna(0) >= 1990
            basement_ratio[newer_mask] = BASEMENT_UTILIZATION_HIGH

        basement_sqft = footprint_sqft * basement_ratio
        base_area = base_area + basement_sqft

        df.loc[mask, "estimated_living_area_sqft"] = base_area

        # Confidence: detached is our most reliable estimate
        base_confidence = 0.55 - confidence_penalty
        # Boost confidence if we had improvement_ratio for stories
        if "stories_source" in df.columns:
            ratio_boost = (
                df.loc[mask, "stories_source"] == "improvement_ratio"
            ).astype(float) * 0.10
            df.loc[mask, "living_area_confidence"] = base_confidence + ratio_boost
        else:
            df.loc[mask, "living_area_confidence"] = base_confidence

        logger.info(
            "Detached estimation: %d properties, median=%.0f sqft",
            mask.sum(),
            df.loc[mask, "estimated_living_area_sqft"].median(),
        )

    def _estimate_townhome(self, df: pd.DataFrame, mask: pd.Series) -> None:
        """Estimate living area for townhome properties (in-place).

        Townhome footprints may cover the entire row of attached units.
        If the building footprint is substantially larger than a typical
        single townhome (~1200-2000 sqft footprint), we divide by an
        estimated unit count.

        Args:
            df: DataFrame to modify in-place.
            mask: Boolean mask selecting townhome properties.
        """
        footprint_sqft = (
            df.loc[mask, "building_footprint_sqm"] * SQM_TO_SQFT
        )
        stories = df.loc[mask, "estimated_stories"]
        efficiency = EFFICIENCY_FACTORS["townhome"]

        # Total building area
        building_area = footprint_sqft * stories * efficiency

        # Estimate units per building
        units = pd.Series(1.0, index=df.loc[mask].index)

        # If we have the actual unit count from MLS data, use it
        if "building_total_units" in df.columns:
            known_units = df.loc[mask, "building_total_units"]
            has_known = known_units.notna() & (known_units > 0)
            units[has_known] = known_units[has_known]
        else:
            # Heuristic: typical townhome unit has ~150-200 sqm footprint
            # If the building footprint is much larger, it likely covers
            # multiple attached units
            typical_unit_footprint_sqm = 175.0  # ~1,884 sqft
            estimated_units = (
                df.loc[mask, "building_footprint_sqm"] / typical_unit_footprint_sqm
            ).clip(lower=1.0)
            # Round to nearest integer (townhomes come in whole units)
            units = estimated_units.round().clip(lower=1.0)

        per_unit_area = building_area / units
        df.loc[mask, "estimated_living_area_sqft"] = per_unit_area

        # Confidence: moderate — unit count estimation adds uncertainty
        base_confidence = 0.45
        if "building_total_units" in df.columns:
            known_boost = (
                df.loc[mask, "building_total_units"].notna()
                & (df.loc[mask, "building_total_units"] > 0)
            ).astype(float) * 0.10
            df.loc[mask, "living_area_confidence"] = base_confidence + known_boost
        else:
            df.loc[mask, "living_area_confidence"] = base_confidence

        logger.info(
            "Townhome estimation: %d properties, median=%.0f sqft",
            mask.sum(),
            df.loc[mask, "estimated_living_area_sqft"].median(),
        )

    def _estimate_condo(self, df: pd.DataFrame, mask: pd.Series) -> None:
        """Estimate living area for condo/strata properties (in-place).

        Condos are the most challenging case because a single building
        footprint represents dozens to hundreds of units. We need to
        estimate:
        1. Total building floor area (footprint * stories * efficiency)
        2. Number of units in the building

        Unit count estimation strategies (in order of preference):
        - Actual unit count from MLS data (building_total_units)
        - Improvement value / typical unit value (from assessment data)
        - Heuristic from building size (weakest)

        Args:
            df: DataFrame to modify in-place.
            mask: Boolean mask selecting condo properties.
        """
        footprint_sqft = (
            df.loc[mask, "building_footprint_sqm"] * SQM_TO_SQFT
        )
        stories = df.loc[mask, "estimated_stories"]
        efficiency = EFFICIENCY_FACTORS["concrete"]

        # Total building livable area
        total_building_area = footprint_sqft * stories * efficiency

        # ----- Estimate unit count -----
        units = pd.Series(np.nan, index=df.loc[mask].index)

        # Strategy A: Known unit count from MLS data
        if "building_total_units" in df.columns:
            known = df.loc[mask, "building_total_units"]
            has_known = known.notna() & (known > 0)
            units[has_known] = known[has_known]
            logger.debug(
                "Condo units — %d known from MLS data", has_known.sum()
            )

        # Strategy B: Improvement value / typical unit value
        if "improvement_value" in df.columns:
            still_missing = units.isna()
            imp_val = df.loc[mask, "improvement_value"]
            has_imp = still_missing & imp_val.notna() & (imp_val > 0)
            if has_imp.any():
                # For condos, improvement_value is typically for the whole
                # building. Divide by typical unit value to get unit count.
                estimated = (imp_val[has_imp] / TYPICAL_CONDO_UNIT_VALUE).clip(
                    lower=1.0
                )
                units[has_imp] = estimated.round().clip(lower=1.0)
                logger.debug(
                    "Condo units — %d estimated from improvement value",
                    has_imp.sum(),
                )

        # Strategy C: Heuristic from total building area
        # Average Metro Vancouver condo unit is ~700-900 sqft
        still_missing = units.isna()
        if still_missing.any():
            avg_unit_sqft = 800.0
            heuristic_units = (
                total_building_area[still_missing] / avg_unit_sqft
            ).clip(lower=1.0)
            units[still_missing] = heuristic_units.round().clip(lower=1.0)
            logger.debug(
                "Condo units — %d estimated from building size heuristic",
                still_missing.sum(),
            )

        # Sanity check: units should be at least 1
        units = units.clip(lower=1.0)

        # Per-unit living area
        per_unit_area = total_building_area / units
        df.loc[mask, "estimated_living_area_sqft"] = per_unit_area

        # Confidence: lowest of all types — unit count estimation is noisy
        base_confidence = 0.30
        confidence = pd.Series(base_confidence, index=df.loc[mask].index)

        if "building_total_units" in df.columns:
            known_mask = (
                df.loc[mask, "building_total_units"].notna()
                & (df.loc[mask, "building_total_units"] > 0)
            )
            # Known unit count significantly improves confidence
            confidence[known_mask] = 0.50

        if "improvement_value" in df.columns:
            imp_mask = (
                df.loc[mask, "improvement_value"].notna()
                & (df.loc[mask, "improvement_value"] > 0)
                & ~(
                    df.loc[mask, "building_total_units"].notna()
                    if "building_total_units" in df.columns
                    else pd.Series(False, index=df.loc[mask].index)
                )
            )
            confidence[imp_mask] = 0.38

        df.loc[mask, "living_area_confidence"] = confidence

        logger.info(
            "Condo estimation: %d properties, median=%.0f sqft, "
            "median units/building=%.0f",
            mask.sum(),
            df.loc[mask, "estimated_living_area_sqft"].median(),
            units.median(),
        )

    # ------------------------------------------------------------------
    # 5. ENRICH PROPERTIES (master method)
    # ------------------------------------------------------------------

    def enrich_properties(
        self,
        properties_gdf: gpd.GeoDataFrame,
        footprints_filepath: str | None = None,
    ) -> gpd.GeoDataFrame:
        """Master method: add footprint-based living area estimates to properties.

        Orchestrates the full pipeline:
        1. Load building footprints (from cache or download)
        2. Match footprints to property locations via spatial join
        3. Estimate story count from assessment data, zoning, and year built
        4. Estimate living area with property-type-specific logic
        5. Compute building coverage ratio (footprint / lot)

        This method is idempotent — calling it multiple times on the same
        GeoDataFrame will overwrite previous estimates rather than
        duplicating columns.

        Args:
            properties_gdf: GeoDataFrame of properties with point geometries.
                            Expected columns vary by available data quality;
                            at minimum, a geometry column is required.
                            Beneficial columns:
                                - property_type (str)
                                - improvement_value (float)
                                - land_value (float)
                                - zoning_code (str)
                                - year_built (int)
                                - lot_size_sqft (float)
                                - building_total_units (int)
            footprints_filepath: Optional path to local building footprint
                                 file. If None, downloads from Microsoft CDN.

        Returns:
            GeoDataFrame with additional columns:
                - building_footprint_sqm (float): Matched footprint area
                - estimated_stories (float): Estimated story count
                - stories_source (str): Estimation method used
                - estimated_living_area_sqft (float): Living area estimate
                - living_area_source (str): Always 'footprint_estimate'
                - living_area_confidence (float): Confidence score 0-1
                - footprint_to_lot_ratio (float): Building coverage ratio,
                  or NaN if lot size is unavailable
        """
        logger.info(
            "=== Building footprint enrichment pipeline starting "
            "(%d properties) ===",
            len(properties_gdf),
        )

        # Step 1: Load footprints
        logger.info("Step 1/5: Loading building footprints...")
        footprints_gdf = self.load_footprints(filepath=footprints_filepath)

        # Step 2: Match footprints to properties
        logger.info("Step 2/5: Matching footprints to properties...")
        enriched = self.match_footprints_to_properties(
            properties_gdf, footprints_gdf
        )

        # Step 3: Estimate stories
        logger.info("Step 3/5: Estimating stories...")
        enriched = self.estimate_stories(enriched)

        # Step 4: Estimate living area
        logger.info("Step 4/5: Estimating living area...")
        enriched = self.estimate_living_area(enriched)

        # Step 5: Compute building coverage (footprint-to-lot ratio)
        logger.info("Step 5/5: Computing building coverage ratio...")
        enriched = self._compute_footprint_to_lot_ratio(enriched)

        # Summary statistics
        self._log_enrichment_summary(enriched)

        logger.info(
            "=== Building footprint enrichment pipeline complete ==="
        )
        return enriched

    @staticmethod
    def _compute_footprint_to_lot_ratio(df: pd.DataFrame) -> pd.DataFrame:
        """Compute the building footprint to lot area ratio.

        This ratio (also called building coverage) indicates how much of
        the lot is occupied by the building footprint. Typical values:
        - Detached Vancouver: 0.25-0.45
        - Townhomes: 0.40-0.65
        - Low-rise condo: 0.45-0.70
        - High-rise: 0.30-0.50 (smaller footprint, more height)

        Values > 0.80 or < 0.05 likely indicate data quality issues.

        Args:
            df: DataFrame with building_footprint_sqm and optionally
                lot_size_sqft columns.

        Returns:
            DataFrame with added footprint_to_lot_ratio column.
        """
        df = df.copy()
        df["footprint_to_lot_ratio"] = np.nan

        if "lot_size_sqft" not in df.columns:
            logger.info(
                "lot_size_sqft not available; skipping footprint-to-lot ratio"
            )
            return df

        has_both = (
            df["building_footprint_sqm"].notna()
            & (df["building_footprint_sqm"] > 0)
            & df["lot_size_sqft"].notna()
            & (df["lot_size_sqft"] > 0)
        )

        if has_both.any():
            footprint_sqft = df.loc[has_both, "building_footprint_sqm"] * SQM_TO_SQFT
            lot_sqft = df.loc[has_both, "lot_size_sqft"]
            df.loc[has_both, "footprint_to_lot_ratio"] = footprint_sqft / lot_sqft

            # Flag suspicious ratios
            suspicious = df["footprint_to_lot_ratio"].notna() & (
                (df["footprint_to_lot_ratio"] > 0.95)
                | (df["footprint_to_lot_ratio"] < 0.02)
            )
            if suspicious.any():
                logger.warning(
                    "%d properties have suspicious footprint-to-lot ratios "
                    "(outside 0.02-0.95 range). This may indicate matching "
                    "errors or incorrect lot sizes.",
                    suspicious.sum(),
                )

            logger.info(
                "Footprint-to-lot ratio: %d computed, "
                "median=%.2f, mean=%.2f",
                has_both.sum(),
                df.loc[has_both, "footprint_to_lot_ratio"].median(),
                df.loc[has_both, "footprint_to_lot_ratio"].mean(),
            )

        return df

    @staticmethod
    def _log_enrichment_summary(df: pd.DataFrame) -> None:
        """Log a comprehensive summary of the enrichment results.

        Args:
            df: The fully enriched DataFrame.
        """
        total = len(df)
        if total == 0:
            logger.info("No properties to summarize.")
            return

        has_footprint = df["building_footprint_sqm"].notna().sum()
        has_area = df["estimated_living_area_sqft"].notna().sum()

        logger.info("--- Enrichment Summary ---")
        logger.info("  Total properties:        %d", total)
        logger.info(
            "  Footprint matched:       %d (%.1f%%)",
            has_footprint,
            100 * has_footprint / total,
        )
        logger.info(
            "  Living area estimated:   %d (%.1f%%)",
            has_area,
            100 * has_area / total,
        )

        if has_area > 0:
            area = df.loc[
                df["estimated_living_area_sqft"].notna(),
                "estimated_living_area_sqft",
            ]
            logger.info(
                "  Living area (sqft):      median=%.0f, mean=%.0f, "
                "p10=%.0f, p90=%.0f",
                area.median(),
                area.mean(),
                area.quantile(0.10),
                area.quantile(0.90),
            )

            conf = df.loc[
                df["living_area_confidence"].notna(),
                "living_area_confidence",
            ]
            logger.info(
                "  Confidence:              median=%.2f, mean=%.2f",
                conf.median(),
                conf.mean(),
            )

        # Breakdown by property type if available
        if "property_type" in df.columns:
            for ptype in df["property_type"].dropna().unique():
                ptype_mask = df["property_type"] == ptype
                ptype_area = df.loc[
                    ptype_mask & df["estimated_living_area_sqft"].notna(),
                    "estimated_living_area_sqft",
                ]
                if len(ptype_area) > 0:
                    logger.info(
                        "  %s: n=%d, median=%.0f sqft, confidence=%.2f",
                        ptype,
                        len(ptype_area),
                        ptype_area.median(),
                        df.loc[
                            ptype_mask & df["living_area_confidence"].notna(),
                            "living_area_confidence",
                        ].median(),
                    )

        # Breakdown by stories source
        if "stories_source" in df.columns:
            logger.info("  Stories estimation sources:")
            for source, count in (
                df["stories_source"].value_counts().items()
            ):
                logger.info(
                    "    %s: %d (%.1f%%)",
                    source,
                    count,
                    100 * count / total,
                )
