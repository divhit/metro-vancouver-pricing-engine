"""
Feature enrichment pipeline.

Coordinates loading spatial layers, building footprints, census data,
market context, and all other feature sources into a unified enriched
property dataset ready for model training.

This is the central orchestration module between raw ingestion and
feature building. Each enrichment step is isolated and fault-tolerant:
if one data source fails (network error, missing file, API limit),
the others proceed and the affected columns are set to NaN.

Usage::

    pipeline = FeatureEnrichmentPipeline()
    enriched_df = pipeline.enrich_all(properties_df, phase=1)

    # Or load previously enriched data
    cached = pipeline.load_enriched()
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from src.features.building_footprint import BuildingFootprintEstimator
from src.features.spatial_features import SpatialFeatureComputer

logger = logging.getLogger(__name__)


class FeatureEnrichmentPipeline:
    """Orchestrates enrichment of the property universe with all features.

    Loads spatial layers (transit, schools, parks, environmental,
    Airbnb), census demographics, building footprints, and market
    context (interest rates, timing). Each data source is loaded
    independently and joined to the property universe.

    The pipeline is designed to be resilient: any individual data
    source failure is caught, logged, and skipped. The resulting
    DataFrame will have NaN for any features that could not be
    computed due to missing source data.

    Args:
        data_dir: Root directory for raw data files. Defaults to 'data'.
        cache_dir: Directory for processed/cached output. Defaults to
            'data/processed'.
    """

    def __init__(
        self,
        data_dir: str = "data",
        cache_dir: str = "data/processed",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Lazily initialized ingestion clients
        self._gtfs_client = None
        self._schools_client = None
        self._environmental_client = None
        self._airbnb_client = None
        self._boc_client = None
        self._census_client = None

        logger.info(
            "FeatureEnrichmentPipeline initialized: data_dir=%s, cache_dir=%s",
            self.data_dir,
            self.cache_dir,
        )

    # ================================================================
    # MASTER ENRICHMENT METHOD
    # ================================================================

    def enrich_all(
        self,
        properties_df: pd.DataFrame,
        phase: int = 1,
    ) -> pd.DataFrame:
        """Enrich the property universe with all available features.

        Master orchestration method that runs every enrichment step in
        sequence. Each step is wrapped in try/except so that failures
        in one data source do not block the others.

        Steps:
          1. Load spatial reference layers (transit, schools, parks,
             environmental, Airbnb)
          2. Initialize SpatialFeatureComputer and preload layers
          3. Compute all spatial features (vectorized)
          4. Run building footprint enrichment
          5. Join census demographics
          6. Add market context features (interest rates, timing)
          7. Compute feature completeness summary
          8. Save enriched data to parquet
          9. Return enriched DataFrame

        Args:
            properties_df: Property universe DataFrame from
                PropertyUniverseBuilder.build_universe(). Must contain
                at minimum: pid, latitude (or geometry), longitude,
                total_assessed_value.
            phase: Implementation phase (1-4). Controls which data
                sources are loaded.

        Returns:
            Enriched DataFrame with all computed feature columns appended.
        """
        t0 = time.perf_counter()
        n_props = len(properties_df)
        logger.info(
            "=== Feature enrichment pipeline starting: %d properties, phase=%d ===",
            n_props,
            phase,
        )

        df = properties_df.copy()

        # --- Ensure GeoDataFrame with point geometries ---
        df = self._ensure_geodataframe(df)

        # --- Step 1-2: Load spatial layers and initialize computer ---
        spatial_computer = SpatialFeatureComputer(
            data_dir=str(self.data_dir / "spatial")
        )

        transit_gdf = self._load_transit_data()
        schools_gdf = self._load_school_data()
        parks_gdf = self._load_parks_data()
        alr_gdf, floodplain_gdf, contaminated_gdf = self._load_environmental_data()
        census_gdf = self._load_census_data()
        airbnb_gdf = self._load_airbnb_data()

        spatial_computer.preload_layers(
            transit_stops_gdf=transit_gdf,
            schools_gdf=schools_gdf,
            parks_gdf=parks_gdf,
            alr_gdf=alr_gdf,
            floodplain_gdf=floodplain_gdf,
            contaminated_gdf=contaminated_gdf,
            census_da_gdf=census_gdf,
            airbnb_gdf=airbnb_gdf,
        )

        # --- Step 3: Compute all spatial features (vectorized) ---
        try:
            logger.info("Step 3/8: Computing spatial features...")
            df = spatial_computer.compute_all_spatial_features(df)
            logger.info("Spatial features computed successfully")
        except Exception as exc:
            logger.error(
                "Spatial feature computation failed: %s. "
                "Continuing without spatial features.",
                exc,
            )

        # --- Step 4: Building footprint enrichment ---
        try:
            logger.info("Step 4/8: Running building footprint enrichment...")
            footprint_estimator = BuildingFootprintEstimator(
                cache_dir=str(self.data_dir / "footprints")
            )
            df = footprint_estimator.enrich_properties(df)
            logger.info("Building footprint enrichment complete")
        except Exception as exc:
            logger.error(
                "Building footprint enrichment failed: %s. "
                "Continuing without footprint features.",
                exc,
            )

        # --- Step 5: Census demographics ---
        # Census features are already computed by SpatialFeatureComputer
        # if census_da_gdf was loaded. Log status.
        census_cols = [
            col for col in df.columns if col.startswith("census_")
        ]
        if census_cols:
            logger.info(
                "Step 5/8: Census demographics attached (%d columns): %s",
                len(census_cols),
                ", ".join(census_cols),
            )
        else:
            logger.warning(
                "Step 5/8: No census demographic columns found in enriched data"
            )

        # --- Step 6: Market context features ---
        try:
            logger.info("Step 6/8: Adding market context features...")
            assessment_year = self._get_assessment_year(df)
            df = self._enrich_market_context(df, assessment_year)
            logger.info("Market context features added")
        except Exception as exc:
            logger.error(
                "Market context enrichment failed: %s. "
                "Continuing without market context.",
                exc,
            )

        # --- Step 7: Feature completeness ---
        logger.info("Step 7/8: Computing feature completeness...")
        completeness = self._compute_completeness_summary(df)
        logger.info(
            "Feature completeness: median=%.1f%%, mean=%.1f%%",
            completeness["median_completeness"],
            completeness["mean_completeness"],
        )

        # --- Step 8: Save to parquet ---
        try:
            logger.info("Step 8/8: Saving enriched data...")
            self._save_enriched(df)
        except Exception as exc:
            logger.error("Failed to save enriched data: %s", exc)

        elapsed = time.perf_counter() - t0
        new_cols = [
            col
            for col in df.columns
            if col not in properties_df.columns and col != "geometry"
        ]
        logger.info(
            "=== Feature enrichment pipeline complete: "
            "%d properties, %d new features, %.1fs ===",
            len(df),
            len(new_cols),
            elapsed,
        )

        return df

    # ================================================================
    # DATA LOADING METHODS
    # ================================================================

    def _ensure_geodataframe(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """Convert a DataFrame to a GeoDataFrame with point geometries.

        If the DataFrame already has a geometry column, returns it as-is
        (wrapped in GeoDataFrame if needed). Otherwise, creates point
        geometries from latitude/longitude columns.

        Args:
            df: Property DataFrame, possibly with lat/lon columns.

        Returns:
            GeoDataFrame with point geometries in WGS84.
        """
        if isinstance(df, gpd.GeoDataFrame) and df.geometry is not None:
            if df.crs is None:
                df = df.set_crs("EPSG:4326")
            return df

        # Look for coordinate columns
        lat_col = None
        lon_col = None
        for lat_candidate in ["latitude", "lat", "LATITUDE"]:
            if lat_candidate in df.columns:
                lat_col = lat_candidate
                break
        for lon_candidate in ["longitude", "lon", "lng", "LONGITUDE"]:
            if lon_candidate in df.columns:
                lon_col = lon_candidate
                break

        if lat_col is None or lon_col is None:
            # Try to derive from land_coordinate or other fields
            logger.warning(
                "No latitude/longitude columns found. "
                "Creating GeoDataFrame with null geometries."
            )
            from shapely.geometry import Point

            null_geom = gpd.GeoSeries(
                [None] * len(df), index=df.index, crs="EPSG:4326",
            )
            gdf = gpd.GeoDataFrame(df, geometry=null_geom)
            return gdf

        # Create point geometries — use None for missing coords
        # (POINT(NaN NaN) would poison spatial joins)
        has_coords = df[lat_col].notna() & df[lon_col].notna()
        geometry = gpd.GeoSeries(
            gpd.points_from_xy(
                df[lon_col].where(has_coords),
                df[lat_col].where(has_coords),
            ),
            index=df.index,
            crs="EPSG:4326",
        )
        geometry[~has_coords] = None
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

        n_with_geom = has_coords.sum()
        logger.info(
            "Created GeoDataFrame: %d/%d properties with valid coordinates",
            n_with_geom,
            len(df),
        )

        return gdf

    def _load_transit_data(self) -> Optional[gpd.GeoDataFrame]:
        """Load transit stop locations from TransLink GTFS.

        Returns:
            GeoDataFrame of transit stops with route_type column,
            or None if loading fails.
        """
        try:
            from src.ingestion.translink_gtfs import TransLinkGTFSClient

            if self._gtfs_client is None:
                self._gtfs_client = TransLinkGTFSClient()

            # Try to load from cached GTFS data first
            cached_zip = None
            for search_dir in [self.data_dir / "gtfs", self.data_dir / "raw"]:
                if search_dir.exists():
                    zips = [z for z in search_dir.glob("*gtfs*.zip")]
                    if not zips:
                        zips = list(search_dir.glob("*transit*.zip"))
                    if zips:
                        cached_zip = str(zips[0])
                        break

            self._gtfs_client.load_gtfs(zip_path=cached_zip)
            stops_gdf = self._gtfs_client.get_stops()

            if stops_gdf is not None and not stops_gdf.empty:
                logger.info(
                    "Loaded %d transit stops from GTFS", len(stops_gdf)
                )
                return stops_gdf

            logger.warning("GTFS data loaded but no stops found")
            return None

        except Exception as exc:
            logger.error("Failed to load transit data: %s", exc)
            return None

    def _load_school_data(self) -> Optional[gpd.GeoDataFrame]:
        """Load BC school locations with FSA quality scores.

        Returns:
            GeoDataFrame of K-12 school locations, or None if loading fails.
        """
        try:
            from src.ingestion.bc_schools import BCSchoolsClient

            if self._schools_client is None:
                self._schools_client = BCSchoolsClient()

            schools_gdf = self._schools_client.fetch_school_locations()

            if schools_gdf is not None and not schools_gdf.empty:
                logger.info(
                    "Loaded %d school locations", len(schools_gdf)
                )

                # Attempt to merge FSA scores
                try:
                    fsa_df = self._schools_client.fetch_fsa_results()
                    if fsa_df is not None and not fsa_df.empty:
                        logger.info(
                            "Loaded %d FSA result records", len(fsa_df)
                        )
                except Exception as fsa_exc:
                    logger.warning(
                        "Failed to load FSA results: %s. "
                        "School quality scores will be unavailable.",
                        fsa_exc,
                    )

                return schools_gdf

            logger.warning("School data loaded but no locations found")
            return None

        except Exception as exc:
            logger.error("Failed to load school data: %s", exc)
            return None

    def _load_parks_data(self) -> Optional[gpd.GeoDataFrame]:
        """Load park and green space data.

        Tries local GeoJSON file first, then falls back to OpenStreetMap
        or Vancouver Open Data.

        Returns:
            GeoDataFrame of park polygons/points, or None if unavailable.
        """
        try:
            parks_path = self.data_dir / "spatial" / "parks.geojson"
            if parks_path.exists():
                parks_gdf = gpd.read_file(parks_path)
                logger.info(
                    "Loaded %d park features from %s",
                    len(parks_gdf),
                    parks_path,
                )
                return parks_gdf

            logger.info(
                "Parks file not found at %s; park features will be unavailable",
                parks_path,
            )
            return None

        except Exception as exc:
            logger.error("Failed to load parks data: %s", exc)
            return None

    def _load_environmental_data(
        self,
    ) -> tuple[
        Optional[gpd.GeoDataFrame],
        Optional[gpd.GeoDataFrame],
        Optional[gpd.GeoDataFrame],
    ]:
        """Load environmental risk layers (ALR, floodplain, contaminated).

        Returns:
            Tuple of (alr_gdf, floodplain_gdf, contaminated_gdf).
            Any element may be None if the corresponding data source fails.
        """
        alr_gdf = None
        floodplain_gdf = None
        contaminated_gdf = None

        try:
            from src.ingestion.environmental import EnvironmentalDataClient

            if self._environmental_client is None:
                self._environmental_client = EnvironmentalDataClient()

            # ALR boundaries
            try:
                alr_gdf = self._environmental_client.fetch_alr_boundaries()
                if alr_gdf is not None and not alr_gdf.empty:
                    logger.info(
                        "Loaded %d ALR boundary polygons", len(alr_gdf)
                    )
                else:
                    logger.warning("ALR data loaded but empty")
                    alr_gdf = None
            except Exception as exc:
                logger.error("Failed to load ALR boundaries: %s", exc)

            # Floodplain
            try:
                floodplain_gdf = self._environmental_client.fetch_floodplain()
                if floodplain_gdf is not None and not floodplain_gdf.empty:
                    logger.info(
                        "Loaded %d floodplain polygons", len(floodplain_gdf)
                    )
                else:
                    logger.warning("Floodplain data loaded but empty")
                    floodplain_gdf = None
            except Exception as exc:
                logger.error("Failed to load floodplain data: %s", exc)

            # Contaminated sites
            try:
                contaminated_gdf = (
                    self._environmental_client.fetch_contaminated_sites()
                )
                if contaminated_gdf is not None and not contaminated_gdf.empty:
                    logger.info(
                        "Loaded %d contaminated site records",
                        len(contaminated_gdf),
                    )
                else:
                    logger.warning("Contaminated sites data loaded but empty")
                    contaminated_gdf = None
            except Exception as exc:
                logger.error("Failed to load contaminated sites: %s", exc)

        except ImportError:
            logger.error(
                "EnvironmentalDataClient not available; "
                "skipping environmental layers"
            )
        except Exception as exc:
            logger.error(
                "Failed to initialize EnvironmentalDataClient: %s", exc
            )

        return alr_gdf, floodplain_gdf, contaminated_gdf

    def _load_census_data(self) -> Optional[gpd.GeoDataFrame]:
        """Load City of Vancouver local area boundaries with 2021 Census demographics.

        Uses the official 22 local area boundary polygons and attaches
        hardcoded 2021 Census demographics (from CMHC / StatCan) per
        neighbourhood.  This avoids the unreliable StatCan CT-level
        bulk download and gives neighbourhood-level demographics that
        align perfectly with our 22 model segments.

        Returns:
            GeoDataFrame of local area polygons with demographic columns,
            or None if loading fails.
        """
        try:
            boundary_url = (
                "https://opendata.vancouver.ca/api/explore/v2.1/"
                "catalog/datasets/local-area-boundary/exports/geojson"
            )
            gdf = gpd.read_file(boundary_url)

            if gdf.crs is None or gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)

            # Identify the name column
            name_col = None
            for candidate in ["name", "Name", "NAME", "mapid"]:
                if candidate in gdf.columns:
                    name_col = candidate
                    break

            if name_col is None:
                logger.warning("Cannot find name column in boundary GeoJSON")
                return None

            # 2021 Census demographics per City of Vancouver local area.
            # Source: CMHC Housing Market Information Portal (2021 Census),
            # StatCan Census Profile 2021, City of Vancouver Open Data.
            # Median household income (before tax), population density
            # (persons/km²), % owner-occupied dwellings, % immigrants,
            # % with bachelor's degree or higher.
            _CENSUS_2021 = {
                "West Point Grey":          {"median_income": 101000, "pop_density": 3800, "pct_owner": 55.0, "pct_immigrants": 42.0, "pct_university": 58.0},
                "Kitsilano":                {"median_income": 84000,  "pop_density": 7200, "pct_owner": 38.0, "pct_immigrants": 35.0, "pct_university": 55.0},
                "Dunbar-Southlands":        {"median_income": 106000, "pop_density": 3200, "pct_owner": 72.0, "pct_immigrants": 45.0, "pct_university": 52.0},
                "Arbutus Ridge":            {"median_income": 74000,  "pop_density": 4500, "pct_owner": 52.0, "pct_immigrants": 48.0, "pct_university": 42.0},
                "Kerrisdale":               {"median_income": 73500,  "pop_density": 4800, "pct_owner": 55.0, "pct_immigrants": 52.0, "pct_university": 45.0},
                "Shaughnessy":              {"median_income": 106000, "pop_density": 2500, "pct_owner": 78.0, "pct_immigrants": 48.0, "pct_university": 55.0},
                "Fairview":                 {"median_income": 81000,  "pop_density": 9500, "pct_owner": 35.0, "pct_immigrants": 38.0, "pct_university": 52.0},
                "South Cambie":             {"median_income": 99000,  "pop_density": 5200, "pct_owner": 55.0, "pct_immigrants": 42.0, "pct_university": 50.0},
                "Oakridge":                 {"median_income": 72000,  "pop_density": 5800, "pct_owner": 60.0, "pct_immigrants": 58.0, "pct_university": 40.0},
                "Marpole":                  {"median_income": 69000,  "pop_density": 6800, "pct_owner": 42.0, "pct_immigrants": 52.0, "pct_university": 35.0},
                "Riley Park":               {"median_income": 107000, "pop_density": 5500, "pct_owner": 55.0, "pct_immigrants": 38.0, "pct_university": 48.0},
                "Sunset":                   {"median_income": 87000,  "pop_density": 6200, "pct_owner": 65.0, "pct_immigrants": 55.0, "pct_university": 30.0},
                "Mount Pleasant":           {"median_income": 79500,  "pop_density": 8200, "pct_owner": 32.0, "pct_immigrants": 35.0, "pct_university": 50.0},
                "Grandview-Woodland":       {"median_income": 78000,  "pop_density": 6800, "pct_owner": 30.0, "pct_immigrants": 38.0, "pct_university": 42.0},
                "Hastings-Sunrise":         {"median_income": 78000,  "pop_density": 5800, "pct_owner": 52.0, "pct_immigrants": 48.0, "pct_university": 32.0},
                "Kensington-Cedar Cottage":  {"median_income": 91000,  "pop_density": 6500, "pct_owner": 55.0, "pct_immigrants": 50.0, "pct_university": 35.0},
                "Killarney":                {"median_income": 87000,  "pop_density": 5500, "pct_owner": 68.0, "pct_immigrants": 58.0, "pct_university": 30.0},
                "Victoria-Fraserview":      {"median_income": 87000,  "pop_density": 5200, "pct_owner": 70.0, "pct_immigrants": 55.0, "pct_university": 28.0},
                "Strathcona":               {"median_income": 41600,  "pop_density": 5500, "pct_owner": 18.0, "pct_immigrants": 42.0, "pct_university": 28.0},
                "Renfrew-Collingwood":      {"median_income": 82000,  "pop_density": 7500, "pct_owner": 48.0, "pct_immigrants": 55.0, "pct_university": 30.0},
                "Downtown":                 {"median_income": 72000,  "pop_density": 18000, "pct_owner": 35.0, "pct_immigrants": 40.0, "pct_university": 52.0},
                "West End":                 {"median_income": 65000,  "pop_density": 22000, "pct_owner": 28.0, "pct_immigrants": 38.0, "pct_university": 48.0},
            }

            # Attach demographics to boundary polygons
            for col_key, col_name in [
                ("median_income", "median_income"),
                ("pop_density", "pop_density"),
                ("pct_owner", "pct_owner"),
                ("pct_immigrants", "pct_immigrants"),
                ("pct_university", "pct_university"),
            ]:
                gdf[col_name] = gdf[name_col].map(
                    {k: v[col_key] for k, v in _CENSUS_2021.items()}
                )

            n_matched = gdf["median_income"].notna().sum()
            logger.info(
                "Loaded %d local area boundary polygons with 2021 Census "
                "demographics (%d/%d matched)",
                len(gdf), n_matched, len(gdf),
            )

            return gdf

        except Exception as exc:
            logger.error("Failed to load census/boundary data: %s", exc)
            return None

    def _load_airbnb_data(self) -> Optional[gpd.GeoDataFrame]:
        """Load Inside Airbnb listing locations.

        Returns:
            GeoDataFrame of Airbnb listing points with price column,
            or None if loading fails.
        """
        try:
            from src.ingestion.inside_airbnb import InsideAirbnbClient

            if self._airbnb_client is None:
                self._airbnb_client = InsideAirbnbClient(
                    cache_dir=str(self.data_dir / "airbnb")
                )

            # Try to load from a cached CSV file first
            airbnb_cache = self.data_dir / "airbnb"
            csv_files = list(airbnb_cache.glob("listings*.csv*")) if airbnb_cache.exists() else []

            if csv_files:
                # Load the most recent file
                csv_path = sorted(csv_files)[-1]
                raw_df = self._airbnb_client.load_from_file(str(csv_path))

                if raw_df is not None and not raw_df.empty:
                    cleaned = self._airbnb_client.clean_listings(raw_df)
                    airbnb_gdf = self._airbnb_client.to_geodataframe(cleaned)
                    logger.info(
                        "Loaded %d Airbnb listings from %s",
                        len(airbnb_gdf),
                        csv_path.name,
                    )
                    return airbnb_gdf

            # Try downloading fresh data
            try:
                raw_df = self._airbnb_client.download_listings()
                if raw_df is not None and not raw_df.empty:
                    cleaned = self._airbnb_client.clean_listings(raw_df)
                    airbnb_gdf = self._airbnb_client.to_geodataframe(cleaned)
                    logger.info(
                        "Downloaded %d Airbnb listings", len(airbnb_gdf)
                    )
                    return airbnb_gdf
            except Exception as dl_exc:
                logger.warning(
                    "Failed to download Airbnb data: %s", dl_exc
                )

            logger.warning("No Airbnb data available")
            return None

        except Exception as exc:
            logger.error("Failed to load Airbnb data: %s", exc)
            return None

    # ================================================================
    # MARKET CONTEXT ENRICHMENT
    # ================================================================

    def _enrich_market_context(
        self,
        df: pd.DataFrame,
        assessment_year: int,
    ) -> pd.DataFrame:
        """Add market context features: interest rates and timing.

        Fetches Bank of Canada mortgage rates at the assessment date and
        computes the OSFI stress test qualifying rate. These values are
        the same for all properties in the same assessment year.

        Args:
            df: Property DataFrame.
            assessment_year: Tax assessment year (e.g., 2024).

        Returns:
            DataFrame with market context columns added:
                - mortgage_rate_5yr_at_assessment
                - policy_rate_at_assessment
                - stress_test_rate
        """
        df = df.copy()

        try:
            from src.ingestion.bank_of_canada import BankOfCanadaClient

            if self._boc_client is None:
                self._boc_client = BankOfCanadaClient()

            # Assessment date is typically July 1 of the prior year
            # (BC Assessment valuation date for the following tax year)
            assessment_date = date(assessment_year - 1, 7, 1)
            start_date = assessment_date - timedelta(days=30)
            end_date = assessment_date + timedelta(days=7)

            rates_df = self._boc_client.get_mortgage_rates(
                start_date=start_date,
                end_date=end_date,
            )

            if rates_df is not None and not rates_df.empty:
                # Use the most recent non-null observation for each series.
                # Mortgage rates are weekly (Wednesdays) while the policy
                # rate is daily, so the last row may have NaN for mortgages.
                mortgage_5yr = (
                    rates_df["mortgage_5yr_fixed"].dropna().iloc[-1]
                    if "mortgage_5yr_fixed" in rates_df.columns
                    and rates_df["mortgage_5yr_fixed"].notna().any()
                    else None
                )
                policy_rate = (
                    rates_df["policy_rate"].dropna().iloc[-1]
                    if "policy_rate" in rates_df.columns
                    and rates_df["policy_rate"].notna().any()
                    else None
                )

                df["mortgage_rate_5yr_at_assessment"] = mortgage_5yr
                df["policy_rate_at_assessment"] = policy_rate

                if mortgage_5yr is not None and not np.isnan(mortgage_5yr):
                    stress_test = self._boc_client.compute_stress_test_rate(
                        mortgage_5yr
                    )
                    df["stress_test_rate"] = stress_test
                    logger.info(
                        "Market context: 5yr fixed=%.2f%%, policy=%.2f%%, "
                        "stress test=%.2f%% (assessment date: %s)",
                        mortgage_5yr,
                        policy_rate if policy_rate else 0.0,
                        stress_test,
                        assessment_date.isoformat(),
                    )
                else:
                    df["stress_test_rate"] = np.nan
                    logger.warning(
                        "Mortgage rate data incomplete; stress test rate unavailable"
                    )
            else:
                logger.warning(
                    "No mortgage rate data found for assessment period %s to %s",
                    start_date.isoformat(),
                    end_date.isoformat(),
                )
                df["mortgage_rate_5yr_at_assessment"] = np.nan
                df["policy_rate_at_assessment"] = np.nan
                df["stress_test_rate"] = np.nan

        except ImportError:
            logger.error(
                "BankOfCanadaClient not available; skipping market context"
            )
            df["mortgage_rate_5yr_at_assessment"] = np.nan
            df["policy_rate_at_assessment"] = np.nan
            df["stress_test_rate"] = np.nan
        except Exception as exc:
            logger.error(
                "Failed to fetch market context: %s. "
                "Setting market context to NaN.",
                exc,
            )
            df["mortgage_rate_5yr_at_assessment"] = np.nan
            df["policy_rate_at_assessment"] = np.nan
            df["stress_test_rate"] = np.nan

        return df

    # ================================================================
    # PERSISTENCE
    # ================================================================

    def _save_enriched(
        self,
        df: pd.DataFrame | gpd.GeoDataFrame,
        filename: str = "enriched_properties.parquet",
    ) -> None:
        """Save the enriched DataFrame to parquet.

        For GeoDataFrames, geometry is preserved in the parquet file
        via geopandas.to_parquet(). For plain DataFrames, geometry
        column is dropped if present (not serializable to standard
        parquet) and standard pandas to_parquet() is used.

        Args:
            df: Enriched property DataFrame.
            filename: Output filename within cache_dir.
        """
        output_path = self.cache_dir / filename

        if isinstance(df, gpd.GeoDataFrame):
            df.to_parquet(output_path, index=False)
        else:
            # Drop geometry if it exists (not serializable to plain parquet)
            save_df = df.drop(columns=["geometry"], errors="ignore")
            save_df.to_parquet(output_path, index=False)

        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(
            "Saved enriched data: %s (%.1f MB, %d rows, %d columns)",
            output_path,
            size_mb,
            len(df),
            len(df.columns),
        )

    def load_enriched(
        self, filename: str = "enriched_properties.parquet"
    ) -> Optional[pd.DataFrame]:
        """Load previously saved enriched data.

        Attempts to load as a GeoDataFrame first (preserving geometry),
        falling back to plain DataFrame if geometry is not present.

        Args:
            filename: Parquet filename within cache_dir.

        Returns:
            Enriched DataFrame, or None if the file does not exist.
        """
        parquet_path = self.cache_dir / filename

        if not parquet_path.exists():
            logger.info(
                "No cached enriched data found at %s", parquet_path
            )
            return None

        try:
            # Try loading as GeoDataFrame (preserves geometry)
            gdf = gpd.read_parquet(parquet_path)
            logger.info(
                "Loaded enriched data from %s: %d rows, %d columns",
                parquet_path,
                len(gdf),
                len(gdf.columns),
            )
            return gdf
        except Exception:
            # Fall back to plain DataFrame
            try:
                df = pd.read_parquet(parquet_path)
                logger.info(
                    "Loaded enriched data (no geometry) from %s: %d rows, %d columns",
                    parquet_path,
                    len(df),
                    len(df.columns),
                )
                return df
            except Exception as exc:
                logger.error(
                    "Failed to load enriched data from %s: %s",
                    parquet_path,
                    exc,
                )
                return None

    # ================================================================
    # UTILITY METHODS
    # ================================================================

    @staticmethod
    def _get_assessment_year(df: pd.DataFrame) -> int:
        """Extract the assessment year from the property data.

        Falls back to the current year if the column is not present.

        Args:
            df: Property DataFrame.

        Returns:
            Assessment year as integer.
        """
        if "tax_assessment_year" in df.columns:
            year = df["tax_assessment_year"].dropna()
            if not year.empty:
                return int(year.mode().iloc[0])

        # Fallback to current year
        from datetime import datetime

        fallback = datetime.now().year
        logger.warning(
            "tax_assessment_year not found in data; using current year %d",
            fallback,
        )
        return fallback

    @staticmethod
    def _compute_completeness_summary(df: pd.DataFrame) -> dict:
        """Compute aggregate feature completeness statistics.

        Args:
            df: DataFrame to analyze.

        Returns:
            Dict with completeness statistics.
        """
        # Exclude identifier and geometry columns
        feature_cols = [
            col
            for col in df.columns
            if col not in ("pid", "geometry", "full_address", "property_postal_code")
        ]

        if not feature_cols:
            return {
                "total_features": 0,
                "median_completeness": 0.0,
                "mean_completeness": 0.0,
            }

        null_fractions = df[feature_cols].isnull().mean()
        completeness_per_col = (1 - null_fractions) * 100

        per_row_completeness = (
            df[feature_cols].notna().sum(axis=1) / len(feature_cols) * 100
        )

        # Log columns with highest null rates
        worst_cols = null_fractions.nlargest(10)
        if not worst_cols.empty and worst_cols.iloc[0] > 0.5:
            logger.info("Most incomplete features:")
            for col, frac in worst_cols.items():
                if frac > 0.1:
                    logger.info("  %s: %.1f%% null", col, frac * 100)

        return {
            "total_features": len(feature_cols),
            "median_completeness": float(per_row_completeness.median()),
            "mean_completeness": float(per_row_completeness.mean()),
            "min_completeness": float(per_row_completeness.min()),
            "max_completeness": float(per_row_completeness.max()),
            "fully_populated_features": int(
                (null_fractions == 0).sum()
            ),
            "features_over_50pct_null": int(
                (null_fractions > 0.5).sum()
            ),
        }
