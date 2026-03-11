"""
Vectorized spatial feature computation for the property universe.

Pre-loads all spatial reference layers into memory and uses
geopandas spatial operations for efficient batch computation.
All distance calculations use EPSG:32610 (UTM 10N) for accuracy.

This module replaces per-property loops (e.g., compute_transit_score(lat, lon))
with O(n log n) spatial joins via geopandas.sjoin_nearest and buffer-based
counting, enabling feature computation across 200K+ properties in minutes
rather than days.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import Point
except ImportError:
    gpd = None  # type: ignore[assignment]
    Point = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# Metro Vancouver key reference points (lat, lon)
DOWNTOWN_VANCOUVER = (49.2827, -123.1207)
METROTOWN = (49.2276, -123.0038)
LOUGHEED = (49.2488, -122.8971)
UBC = (49.2606, -123.2460)

# CRS constants
WGS84 = "EPSG:4326"
UTM10N = "EPSG:32610"  # NAD83 / UTM zone 10N -- accurate for Metro Vancouver


class SpatialFeatureComputer:
    """Vectorized spatial feature computation engine.

    Pre-loads spatial reference layers (transit stops, schools, parks,
    environmental polygons, census areas, Airbnb listings) and computes
    all spatial features for a property universe in batch using geopandas
    spatial joins and distance operations.

    All internal computations use EPSG:32610 (UTM zone 10N) for meter-
    accurate distances in the Metro Vancouver region.
    """

    def __init__(self, data_dir: str = "data/spatial") -> None:
        """Initialize the spatial feature computer.

        Args:
            data_dir: Path to directory containing cached spatial data.
                      Used for any auxiliary datasets loaded from disk.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Spatial reference layers -- populated via preload_layers()
        self._transit_stops: Optional[gpd.GeoDataFrame] = None
        self._schools: Optional[gpd.GeoDataFrame] = None
        self._parks: Optional[gpd.GeoDataFrame] = None
        self._local_areas: Optional[gpd.GeoDataFrame] = None
        self._alr: Optional[gpd.GeoDataFrame] = None
        self._floodplain: Optional[gpd.GeoDataFrame] = None
        self._contaminated: Optional[gpd.GeoDataFrame] = None
        self._census_da: Optional[gpd.GeoDataFrame] = None
        self._airbnb: Optional[gpd.GeoDataFrame] = None

        # Pre-computed reference points in UTM10N
        self._downtown_utm: Optional[Point] = None

    # ----------------------------------------------------------------
    # Layer loading
    # ----------------------------------------------------------------

    def preload_layers(
        self,
        transit_stops_gdf: Optional[gpd.GeoDataFrame] = None,
        schools_gdf: Optional[gpd.GeoDataFrame] = None,
        parks_gdf: Optional[gpd.GeoDataFrame] = None,
        local_areas_gdf: Optional[gpd.GeoDataFrame] = None,
        alr_gdf: Optional[gpd.GeoDataFrame] = None,
        floodplain_gdf: Optional[gpd.GeoDataFrame] = None,
        contaminated_gdf: Optional[gpd.GeoDataFrame] = None,
        census_da_gdf: Optional[gpd.GeoDataFrame] = None,
        airbnb_gdf: Optional[gpd.GeoDataFrame] = None,
    ) -> None:
        """Accept pre-loaded GeoDataFrames and project them all to UTM10N.

        Each parameter is optional.  Features that require a missing layer
        will be silently skipped (with a logged warning).

        Args:
            transit_stops_gdf: Transit stops with route_type column.
            schools_gdf: K-12 school locations with type/FSA columns.
            parks_gdf: Park polygons or centroids.
            local_areas_gdf: Local area / neighbourhood polygons.
            alr_gdf: Agricultural Land Reserve polygons.
            floodplain_gdf: Designated floodplain polygons.
            contaminated_gdf: Contaminated site points from BC Site Registry.
            census_da_gdf: Census dissemination area polygons with demographics.
            airbnb_gdf: Inside Airbnb listing points with price column.
        """
        layer_map = {
            "transit_stops": (transit_stops_gdf, "_transit_stops"),
            "schools": (schools_gdf, "_schools"),
            "parks": (parks_gdf, "_parks"),
            "local_areas": (local_areas_gdf, "_local_areas"),
            "alr": (alr_gdf, "_alr"),
            "floodplain": (floodplain_gdf, "_floodplain"),
            "contaminated": (contaminated_gdf, "_contaminated"),
            "census_da": (census_da_gdf, "_census_da"),
            "airbnb": (airbnb_gdf, "_airbnb"),
        }

        loaded = []
        skipped = []

        for name, (gdf, attr) in layer_map.items():
            if gdf is not None and not gdf.empty:
                projected = self._ensure_utm(gdf, name)
                setattr(self, attr, projected)
                loaded.append(f"{name} ({len(projected):,} features)")
            else:
                skipped.append(name)

        # Pre-compute reference points in UTM10N
        downtown_gdf = gpd.GeoDataFrame(
            geometry=[Point(DOWNTOWN_VANCOUVER[1], DOWNTOWN_VANCOUVER[0])],
            crs=WGS84,
        ).to_crs(UTM10N)
        self._downtown_utm = downtown_gdf.geometry.iloc[0]

        logger.info(f"Preloaded {len(loaded)} spatial layers: {', '.join(loaded)}")
        if skipped:
            logger.warning(f"Skipped {len(skipped)} layers (not provided): {', '.join(skipped)}")

    def _ensure_utm(self, gdf: gpd.GeoDataFrame, layer_name: str) -> gpd.GeoDataFrame:
        """Project a GeoDataFrame to UTM10N if needed.

        Args:
            gdf: Input GeoDataFrame in any CRS.
            layer_name: Name for logging.

        Returns:
            GeoDataFrame in EPSG:32610.
        """
        if gdf.crs is None:
            logger.warning(f"Layer '{layer_name}' has no CRS; assuming WGS84")
            gdf = gdf.set_crs(WGS84)

        if gdf.crs.to_epsg() != 32610:
            gdf = gdf.to_crs(UTM10N)

        return gdf

    # ----------------------------------------------------------------
    # Master computation method
    # ----------------------------------------------------------------

    def compute_all_spatial_features(
        self, properties_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Compute ALL spatial features for the property universe.

        Takes property points in any CRS (typically WGS84), projects to
        UTM10N, runs every spatial computation, and returns the
        GeoDataFrame with all new feature columns added.

        Args:
            properties_gdf: Property points with ``geometry`` (Point)
                and ``pid`` columns.

        Returns:
            The same GeoDataFrame with spatial feature columns appended.
            The geometry is returned in the original CRS.
        """
        t0 = time.perf_counter()
        n = len(properties_gdf)
        logger.info(f"Computing spatial features for {n:,} properties")

        if "pid" not in properties_gdf.columns:
            raise ValueError("properties_gdf must contain a 'pid' column")

        # Preserve original CRS for return
        original_crs = properties_gdf.crs or WGS84

        # Filter out properties without valid geometry before projecting
        # (POINT(NaN NaN) or None geometries poison spatial joins)
        has_geom = properties_gdf.geometry.notna() & ~properties_gdf.geometry.is_empty
        props_valid = properties_gdf.loc[has_geom].copy()
        props_no_geom = properties_gdf.loc[~has_geom].copy()
        n_no_geom = len(props_no_geom)
        if n_no_geom > 0:
            logger.info(
                "Excluding %d properties without valid geometry from spatial computation",
                n_no_geom,
            )

        # Project to UTM10N
        props = self._ensure_utm(props_valid, "properties")

        # Store original index for safe column assignment
        original_idx = props.index.copy()
        props = props.reset_index(drop=True)

        computed = []
        skipped = []

        # --- Transit features ---
        if self._transit_stops is not None:
            try:
                props = self._compute_transit_features(props, self._transit_stops)
                computed.append("transit")
            except Exception as exc:
                logger.warning("Transit feature computation failed: %s", exc)
                skipped.append("transit")
        else:
            skipped.append("transit")

        # --- School features ---
        if self._schools is not None:
            try:
                props = self._compute_school_features(props, self._schools)
                computed.append("schools")
            except Exception as exc:
                logger.warning("School feature computation failed: %s", exc)
                skipped.append("schools")
        else:
            skipped.append("schools")

        # --- Park features ---
        if self._parks is not None:
            try:
                props = self._compute_park_features(props, self._parks)
                computed.append("parks")
            except Exception as exc:
                logger.warning("Park feature computation failed: %s", exc)
                skipped.append("parks")
        else:
            skipped.append("parks")

        # --- Environmental risk features ---
        if any(layer is not None for layer in [self._alr, self._floodplain, self._contaminated]):
            try:
                props = self._compute_environmental_features(
                    props, self._alr, self._floodplain, self._contaminated
                )
                computed.append("environmental")
            except Exception as exc:
                logger.warning("Environmental feature computation failed: %s", exc)
                skipped.append("environmental")
        else:
            skipped.append("environmental")

        # --- Census / demographic features ---
        if self._census_da is not None:
            try:
                props = self._compute_census_features(props, self._census_da)
                computed.append("census")
            except Exception as exc:
                logger.warning("Census feature computation failed: %s", exc)
                skipped.append("census")
        else:
            skipped.append("census")

        # --- Location reference features ---
        try:
            props = self._compute_location_features(props)
            computed.append("location")
        except Exception as exc:
            logger.warning("Location feature computation failed: %s", exc)
            skipped.append("location")

        # --- STR (short-term rental) features ---
        if self._airbnb is not None:
            try:
                props = self._compute_str_features(props, self._airbnb)
                computed.append("str")
            except Exception as exc:
                logger.warning("STR feature computation failed: %s", exc)
                skipped.append("str")
        else:
            skipped.append("str")

        # Project back to original CRS
        props = props.to_crs(original_crs)

        # Restore original index
        props.index = original_idx

        # Reassemble: add back properties without geometry
        if n_no_geom > 0:
            new_cols_list = [
                c for c in props.columns
                if c not in props_no_geom.columns and c != "geometry"
            ]
            for col in new_cols_list:
                props_no_geom[col] = np.nan
            props = pd.concat([props, props_no_geom]).sort_index()

        # Ensure boolean columns are numeric (concat with NaN turns bool→object)
        bool_cols = ["has_skytrain_800m", "in_alr", "in_floodplain", "is_tod_area"]
        for col in bool_cols:
            if col in props.columns:
                props[col] = props[col].astype(float).fillna(0).astype(int)

        elapsed = time.perf_counter() - t0
        new_cols = [
            c for c in props.columns
            if c not in properties_gdf.columns and c != "geometry"
        ]
        logger.info(
            f"Spatial feature computation complete: {len(new_cols)} features "
            f"for {n:,} properties in {elapsed:.1f}s"
        )
        logger.info(f"  Computed groups: {', '.join(computed)}")
        if skipped:
            logger.warning(f"  Skipped groups (layer not loaded): {', '.join(skipped)}")

        return props

    # ================================================================
    # HELPER METHODS — vectorized spatial primitives
    # ================================================================

    def _nearest_distance(
        self,
        from_gdf: gpd.GeoDataFrame,
        to_gdf: gpd.GeoDataFrame,
        column_name: str,
        max_distance: float = 10_000,
    ) -> gpd.GeoDataFrame:
        """Compute distance from each point in *from_gdf* to nearest in *to_gdf*.

        Uses ``geopandas.sjoin_nearest`` which leverages R-tree spatial
        indexing for O(n log n) performance.

        Both GeoDataFrames must be in projected CRS (EPSG:32610).

        Args:
            from_gdf: Source points (properties).
            to_gdf: Target points/polygons to find nearest of.
            column_name: Name for the resulting distance column.
            max_distance: Maximum search radius in meters.  Matches beyond
                this distance are capped to ``max_distance``.

        Returns:
            *from_gdf* with ``column_name`` column added.
        """
        if to_gdf is None or to_gdf.empty:
            from_gdf[column_name] = np.nan
            return from_gdf

        # Reset index on to_gdf to avoid duplicate-label issues in the join.
        to_clean = to_gdf[["geometry"]].reset_index(drop=True)

        # sjoin_nearest returns one row per match; keep only nearest
        joined = gpd.sjoin_nearest(
            from_gdf[["geometry"]],
            to_clean,
            how="left",
            max_distance=max_distance,
            distance_col="_dist",
        )

        # Handle duplicate rows from multiple equidistant matches.
        # Sort by distance then drop duplicates on the left-side index
        # to keep exactly one (nearest) row per property.
        joined = (
            joined
            .sort_values("_dist")
            .loc[~joined.index.duplicated(keep="first")]
        )

        # Align back to from_gdf index
        from_gdf[column_name] = joined["_dist"].reindex(from_gdf.index).values

        return from_gdf

    def _count_within_radius(
        self,
        from_gdf: gpd.GeoDataFrame,
        to_gdf: gpd.GeoDataFrame,
        radius_m: float,
        column_name: str,
    ) -> gpd.GeoDataFrame:
        """Count points in *to_gdf* within *radius_m* of each point in *from_gdf*.

        Uses a buffer + spatial-join approach:
        1. Buffer each property point by *radius_m*.
        2. Spatial join the buffered polygons with *to_gdf* points.
        3. Group by original property index and count matches.

        Args:
            from_gdf: Source points (properties).
            to_gdf: Target points to count.
            radius_m: Search radius in meters.
            column_name: Name for the resulting count column.

        Returns:
            *from_gdf* with ``column_name`` column added (int, 0 where none).
        """
        if to_gdf is None or to_gdf.empty:
            from_gdf[column_name] = 0
            return from_gdf

        # Reset index on to_gdf to avoid "cannot reindex on an axis with
        # duplicate labels" errors from non-unique indices.
        to_clean = to_gdf[["geometry"]].reset_index(drop=True)

        # Create buffered geometry
        buffers = from_gdf.copy()
        buffers["geometry"] = from_gdf.geometry.buffer(radius_m)

        # Spatial join: which to_gdf points fall within each buffer?
        joined = gpd.sjoin(
            buffers[["geometry"]],
            to_clean,
            how="left",
            predicate="contains",
        )

        # Count matches per property (original index)
        filtered = joined.dropna(subset=["index_right"])
        counts = (
            filtered
            .groupby(filtered.index)
            .size()
        )

        from_gdf[column_name] = counts.reindex(from_gdf.index, fill_value=0).astype(int)

        return from_gdf

    def _point_in_polygon(
        self,
        points_gdf: gpd.GeoDataFrame,
        polygons_gdf: gpd.GeoDataFrame,
        column_name: str,
    ) -> gpd.GeoDataFrame:
        """Check which points fall within any polygon.

        Uses ``sjoin`` with the ``within`` predicate.

        Args:
            points_gdf: Property points.
            polygons_gdf: Polygon layer (ALR, floodplain, etc.).
            column_name: Name for the resulting boolean column.

        Returns:
            *points_gdf* with ``column_name`` boolean column added.
        """
        if polygons_gdf is None or polygons_gdf.empty:
            points_gdf[column_name] = False
            return points_gdf

        # Reset index on polygons to avoid duplicate-label reindex errors
        polys_clean = polygons_gdf[["geometry"]].reset_index(drop=True)

        joined = gpd.sjoin(
            points_gdf[["geometry"]],
            polys_clean,
            how="left",
            predicate="within",
        )

        # A point is "within" if it matched at least one polygon
        matched_indices = joined.dropna(subset=["index_right"]).index.unique()
        points_gdf[column_name] = points_gdf.index.isin(matched_indices)

        return points_gdf

    def _nearest_distance_to_polygon_boundary(
        self,
        from_gdf: gpd.GeoDataFrame,
        polygons_gdf: gpd.GeoDataFrame,
        column_name: str,
        in_column: Optional[str] = None,
    ) -> gpd.GeoDataFrame:
        """Compute distance from each point to the nearest polygon boundary.

        For points inside a polygon the distance is set to 0.

        Args:
            from_gdf: Source points (properties).
            polygons_gdf: Polygon layer.
            column_name: Name for the resulting distance column.
            in_column: If provided, name of an existing boolean column
                indicating containment (avoids recomputation).

        Returns:
            *from_gdf* with ``column_name`` column added.
        """
        if polygons_gdf is None or polygons_gdf.empty:
            from_gdf[column_name] = np.nan
            return from_gdf

        # Convert polygon boundaries to linestrings for distance calc
        boundaries = polygons_gdf.copy()
        boundaries["geometry"] = boundaries.geometry.boundary

        # Remove empty geometries that can result from invalid polygons
        boundaries = boundaries[~boundaries.geometry.is_empty].copy()

        if boundaries.empty:
            from_gdf[column_name] = np.nan
            return from_gdf

        from_gdf = self._nearest_distance(
            from_gdf, boundaries, column_name, max_distance=50_000
        )

        # Set distance to 0 for points inside any polygon
        if in_column is not None and in_column in from_gdf.columns:
            from_gdf.loc[from_gdf[in_column], column_name] = 0.0

        return from_gdf

    def _aggregate_within_radius(
        self,
        from_gdf: gpd.GeoDataFrame,
        to_gdf: gpd.GeoDataFrame,
        radius_m: float,
        value_column: str,
        agg_func: str,
        output_column: str,
    ) -> gpd.GeoDataFrame:
        """Aggregate a numeric column from *to_gdf* within *radius_m* of each property.

        Args:
            from_gdf: Source points (properties).
            to_gdf: Target points with a numeric column to aggregate.
            radius_m: Search radius in meters.
            value_column: Column in *to_gdf* to aggregate.
            agg_func: Pandas aggregation function name (``mean``, ``sum``, ``median``).
            output_column: Name for the resulting column.

        Returns:
            *from_gdf* with ``output_column`` column added.
        """
        if to_gdf is None or to_gdf.empty or value_column not in to_gdf.columns:
            from_gdf[output_column] = np.nan
            return from_gdf

        # Reset index on to_gdf to avoid duplicate-label reindex errors
        to_clean = to_gdf[["geometry", value_column]].reset_index(drop=True)

        # Buffer properties
        buffers = from_gdf.copy()
        buffers["geometry"] = from_gdf.geometry.buffer(radius_m)

        # Spatial join
        joined = gpd.sjoin(
            buffers[["geometry"]],
            to_clean,
            how="left",
            predicate="contains",
        )

        # Aggregate per property
        filtered = joined.dropna(subset=[value_column])
        agg_result = (
            filtered
            .groupby(filtered.index)[value_column]
            .agg(agg_func)
        )

        from_gdf[output_column] = agg_result.reindex(from_gdf.index).values

        return from_gdf

    # ================================================================
    # FEATURE GROUP METHODS
    # ================================================================

    def _compute_transit_features(
        self,
        properties: gpd.GeoDataFrame,
        transit_stops: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """Compute transit accessibility features.

        Features produced:
            - dist_nearest_transit_m
            - dist_nearest_skytrain_m
            - transit_stops_400m
            - transit_stops_800m
            - unique_routes_400m
            - has_skytrain_800m
            - is_tod_area

        Args:
            properties: Property points in UTM10N.
            transit_stops: Transit stop points in UTM10N.

        Returns:
            *properties* with transit feature columns added.
        """
        t0 = time.perf_counter()
        logger.info("Computing transit features...")

        # --- Distance to nearest transit stop (any type) ---
        properties = self._nearest_distance(
            properties, transit_stops, "dist_nearest_transit_m"
        )

        # --- SkyTrain-specific features ---
        # Filter to SkyTrain stations (route_type == 1 in GTFS = Metro/Subway)
        skytrain_mask = None
        for col in ["route_type", "ROUTE_TYPE"]:
            if col in transit_stops.columns:
                skytrain_mask = transit_stops[col] == 1
                break

        # Fallback: detect SkyTrain stations by name pattern if route_type
        # is unavailable or produced no matches. TransLink SkyTrain stations
        # contain "Station" in their stop_name.
        if skytrain_mask is None or not skytrain_mask.any():
            name_col = None
            for col in ["stop_name", "STOP_NAME"]:
                if col in transit_stops.columns:
                    name_col = col
                    break

            if name_col is not None:
                # SkyTrain stations: name contains "Station" (excludes
                # bus stops like "SkyTrain Stn" bus bays by also checking
                # location_type == 1 for parent stations if available)
                name_mask = transit_stops[name_col].str.contains(
                    "Station", case=False, na=False
                )
                if "location_type" in transit_stops.columns:
                    # GTFS location_type 1 = parent station
                    parent_mask = transit_stops["location_type"] == 1
                    skytrain_mask = name_mask & parent_mask
                    if not skytrain_mask.any():
                        # Fall back to name-only match
                        skytrain_mask = name_mask
                else:
                    skytrain_mask = name_mask

                if skytrain_mask is not None and skytrain_mask.any():
                    logger.info(
                        "Detected %d SkyTrain stations via name pattern fallback",
                        skytrain_mask.sum(),
                    )

        if skytrain_mask is not None and skytrain_mask.any():
            skytrain_stops = transit_stops[skytrain_mask].copy()
            properties = self._nearest_distance(
                properties, skytrain_stops, "dist_nearest_skytrain_m"
            )
        else:
            logger.warning(
                "No route_type column and name-based detection found no "
                "SkyTrain stops; skipping SkyTrain-specific features"
            )
            properties["dist_nearest_skytrain_m"] = np.nan

        # --- Count of transit stops within radii ---
        properties = self._count_within_radius(
            properties, transit_stops, 400, "transit_stops_400m"
        )
        properties = self._count_within_radius(
            properties, transit_stops, 800, "transit_stops_800m"
        )

        # --- Unique routes within 400m ---
        # If the transit_stops GDF has a route_id or route_short_name column
        route_col = None
        for col in ["route_id", "route_short_name", "ROUTE_ID", "ROUTE_SHORT_NAME"]:
            if col in transit_stops.columns:
                route_col = col
                break

        if route_col is not None:
            buffers = properties.copy()
            buffers["geometry"] = properties.geometry.buffer(400)

            # Reset index to avoid duplicate-label errors
            transit_route_clean = transit_stops[["geometry", route_col]].reset_index(drop=True)

            joined = gpd.sjoin(
                buffers[["geometry"]],
                transit_route_clean,
                how="left",
                predicate="contains",
            )

            filtered = joined.dropna(subset=[route_col])
            unique_routes = (
                filtered
                .groupby(filtered.index)[route_col]
                .nunique()
            )

            properties["unique_routes_400m"] = (
                unique_routes.reindex(properties.index, fill_value=0).astype(int)
            )
        else:
            logger.warning(
                "No route identifier column found in transit_stops; "
                "setting unique_routes_400m to NaN"
            )
            properties["unique_routes_400m"] = np.nan

        # --- Boolean: SkyTrain within 800m ---
        properties["has_skytrain_800m"] = (
            properties["dist_nearest_skytrain_m"].notna()
            & (properties["dist_nearest_skytrain_m"] <= 800)
        )

        # --- TOD area (within 800m of SkyTrain) ---
        properties["is_tod_area"] = properties["has_skytrain_800m"]

        elapsed = time.perf_counter() - t0
        logger.info(f"Transit features computed in {elapsed:.1f}s")

        return properties

    def _compute_school_features(
        self,
        properties: gpd.GeoDataFrame,
        schools: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """Compute school proximity and quality features.

        Features produced:
            - dist_nearest_school_m
            - dist_nearest_elementary_m
            - dist_nearest_secondary_m
            - school_fsa_score_nearest
            - school_fsa_score_secondary_nearest
            - schools_within_1km

        Args:
            properties: Property points in UTM10N.
            schools: School location points in UTM10N.

        Returns:
            *properties* with school feature columns added.
        """
        t0 = time.perf_counter()
        logger.info("Computing school features...")

        # --- Distance to nearest school (any K-12) ---
        properties = self._nearest_distance(
            properties, schools, "dist_nearest_school_m"
        )

        # --- Elementary schools ---
        # Detect school type column
        type_col = None
        for col in schools.columns:
            col_upper = col.upper()
            if "TYPE" in col_upper and ("SCHOOL" in col_upper or "FACILITY" in col_upper):
                type_col = col
                break
            if col_upper in ("SCHOOL_TYPE", "FACILITY_TYPE", "TYPE"):
                type_col = col
                break

        if type_col is not None:
            type_values = schools[type_col].astype(str).str.upper()

            elementary_mask = type_values.str.contains(
                "ELEM|PRIMARY|ELEMENTARY", na=False
            )
            if elementary_mask.any():
                elementary = schools[elementary_mask].copy()
                properties = self._nearest_distance(
                    properties, elementary, "dist_nearest_elementary_m"
                )
            else:
                properties["dist_nearest_elementary_m"] = np.nan

            secondary_mask = type_values.str.contains(
                "SECOND|HIGH|SENIOR", na=False
            )
            if secondary_mask.any():
                secondary = schools[secondary_mask].copy()
                properties = self._nearest_distance(
                    properties, secondary, "dist_nearest_secondary_m"
                )
            else:
                properties["dist_nearest_secondary_m"] = np.nan
        else:
            logger.warning(
                "No school type column found; cannot compute type-specific distances"
            )
            properties["dist_nearest_elementary_m"] = np.nan
            properties["dist_nearest_secondary_m"] = np.nan

        # --- FSA quality score of nearest school ---
        fsa_col = None
        for col in schools.columns:
            col_upper = col.upper()
            if "QUALITY" in col_upper or "FSA" in col_upper or "SCORE" in col_upper:
                fsa_col = col
                break

        if fsa_col is not None:
            # Reset index on schools to prevent duplicate-label errors
            schools_fsa = schools[["geometry", fsa_col]].reset_index(drop=True)

            # Join nearest school and pick up its quality score
            joined = gpd.sjoin_nearest(
                properties[["geometry"]],
                schools_fsa,
                how="left",
                max_distance=10_000,
                distance_col="_dist_school",
            )
            joined = (
                joined
                .sort_values("_dist_school")
                .loc[~joined.index.duplicated(keep="first")]
            )
            properties["school_fsa_score_nearest"] = (
                joined[fsa_col].reindex(properties.index).values
            )

            # FSA score of nearest secondary school
            if type_col is not None and secondary_mask.any():
                secondary_with_fsa = schools[secondary_mask & schools[fsa_col].notna()].copy()
                if not secondary_with_fsa.empty:
                    # Reset index to prevent duplicate-label errors
                    sec_fsa = secondary_with_fsa[["geometry", fsa_col]].reset_index(drop=True)
                    joined_sec = gpd.sjoin_nearest(
                        properties[["geometry"]],
                        sec_fsa,
                        how="left",
                        max_distance=10_000,
                        distance_col="_dist_sec",
                    )
                    joined_sec = (
                        joined_sec
                        .sort_values("_dist_sec")
                        .loc[~joined_sec.index.duplicated(keep="first")]
                    )
                    properties["school_fsa_score_secondary_nearest"] = (
                        joined_sec[fsa_col].reindex(properties.index).values
                    )
                else:
                    properties["school_fsa_score_secondary_nearest"] = np.nan
            else:
                properties["school_fsa_score_secondary_nearest"] = np.nan
        else:
            logger.warning(
                "No FSA / quality score column in schools layer; "
                "setting school_fsa_score columns to NaN"
            )
            properties["school_fsa_score_nearest"] = np.nan
            properties["school_fsa_score_secondary_nearest"] = np.nan

        # --- Count of schools within 1km ---
        properties = self._count_within_radius(
            properties, schools, 1000, "schools_within_1km"
        )

        elapsed = time.perf_counter() - t0
        logger.info(f"School features computed in {elapsed:.1f}s")

        return properties

    def _compute_park_features(
        self,
        properties: gpd.GeoDataFrame,
        parks: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """Compute park and green-space features.

        Features produced:
            - dist_nearest_park_m
            - parks_within_500m
            - park_area_within_1km_sqm  (if parks are polygons)

        Args:
            properties: Property points in UTM10N.
            parks: Park features (polygons or points) in UTM10N.

        Returns:
            *properties* with park feature columns added.
        """
        t0 = time.perf_counter()
        logger.info("Computing park features...")

        # For distance and counting, use park centroids if polygons
        geom_types = parks.geometry.geom_type.unique()
        has_polygons = any(
            gt in ("Polygon", "MultiPolygon") for gt in geom_types
        )

        if has_polygons:
            park_centroids = parks.copy()
            park_centroids["geometry"] = parks.geometry.centroid
        else:
            park_centroids = parks

        # --- Distance to nearest park ---
        properties = self._nearest_distance(
            properties, park_centroids, "dist_nearest_park_m"
        )

        # --- Count of parks within 500m ---
        properties = self._count_within_radius(
            properties, park_centroids, 500, "parks_within_500m"
        )

        # --- Total park area within 1km (polygon parks only) ---
        if has_polygons:
            # Compute area of each park polygon in sq metres
            parks_with_area = parks.copy()
            parks_with_area["_park_area_sqm"] = parks_with_area.geometry.area

            # Buffer properties by 1km
            buffers = properties.copy()
            buffers["geometry"] = properties.geometry.buffer(1000)

            # Spatial join: which parks intersect each buffer?
            joined = gpd.sjoin(
                buffers[["geometry"]],
                parks_with_area[["geometry", "_park_area_sqm"]],
                how="left",
                predicate="intersects",
            )

            # Sum park area per property
            area_sum = (
                joined
                .dropna(subset=["_park_area_sqm"])
                .groupby(joined.index)["_park_area_sqm"]
                .sum()
            )

            properties["park_area_within_1km_sqm"] = (
                area_sum.reindex(properties.index, fill_value=0.0).values
            )
        else:
            logger.info("Parks layer contains points, not polygons; skipping area computation")
            properties["park_area_within_1km_sqm"] = np.nan

        elapsed = time.perf_counter() - t0
        logger.info(f"Park features computed in {elapsed:.1f}s")

        return properties

    def _compute_environmental_features(
        self,
        properties: gpd.GeoDataFrame,
        alr: Optional[gpd.GeoDataFrame],
        floodplain: Optional[gpd.GeoDataFrame],
        contaminated: Optional[gpd.GeoDataFrame],
    ) -> gpd.GeoDataFrame:
        """Compute environmental risk features.

        Features produced:
            - in_alr
            - dist_alr_m
            - in_floodplain
            - dist_floodplain_m
            - contaminated_sites_500m
            - dist_nearest_contaminated_m

        Args:
            properties: Property points in UTM10N.
            alr: ALR polygon boundaries (or None).
            floodplain: Floodplain polygon boundaries (or None).
            contaminated: Contaminated site points (or None).

        Returns:
            *properties* with environmental feature columns added.
        """
        t0 = time.perf_counter()
        logger.info("Computing environmental risk features...")

        # --- Agricultural Land Reserve ---
        if alr is not None and not alr.empty:
            properties = self._point_in_polygon(properties, alr, "in_alr")
            properties = self._nearest_distance_to_polygon_boundary(
                properties, alr, "dist_alr_m", in_column="in_alr"
            )
        else:
            logger.warning("ALR layer not loaded; skipping ALR features")
            properties["in_alr"] = False
            properties["dist_alr_m"] = np.nan

        # --- Floodplain ---
        if floodplain is not None and not floodplain.empty:
            properties = self._point_in_polygon(properties, floodplain, "in_floodplain")
            properties = self._nearest_distance_to_polygon_boundary(
                properties, floodplain, "dist_floodplain_m", in_column="in_floodplain"
            )
        else:
            logger.warning("Floodplain layer not loaded; skipping floodplain features")
            properties["in_floodplain"] = False
            properties["dist_floodplain_m"] = np.nan

        # --- Contaminated sites ---
        if contaminated is not None and not contaminated.empty:
            properties = self._count_within_radius(
                properties, contaminated, 500, "contaminated_sites_500m"
            )
            properties = self._nearest_distance(
                properties, contaminated, "dist_nearest_contaminated_m"
            )
        else:
            logger.warning("Contaminated sites layer not loaded; skipping contamination features")
            properties["contaminated_sites_500m"] = 0
            properties["dist_nearest_contaminated_m"] = np.nan

        elapsed = time.perf_counter() - t0
        logger.info(f"Environmental features computed in {elapsed:.1f}s")

        return properties

    def _compute_census_features(
        self,
        properties: gpd.GeoDataFrame,
        census_da: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """Attach census / demographic features via spatial join to DA polygons.

        Features produced (if present in census_da):
            - census_median_income
            - census_pop_density
            - census_pct_owner_occupied
            - census_pct_immigrants
            - census_pct_university

        The census DA GeoDataFrame is expected to have demographic columns
        already attached from the Statistics Canada ingestion step.

        Args:
            properties: Property points in UTM10N.
            census_da: Census dissemination area polygons in UTM10N
                with demographic columns.

        Returns:
            *properties* with census feature columns added.
        """
        t0 = time.perf_counter()
        logger.info("Computing census / demographic features...")

        # Define the column mapping: target name -> possible source columns
        census_columns = {
            "census_median_income": [
                "median_income", "median_household_income", "MEDIAN_INCOME",
                "median_total_income", "v_CA21_906",
            ],
            "census_pop_density": [
                "pop_density", "population_density", "POP_DENSITY",
                "pop_density_per_sqkm",
            ],
            "census_pct_owner_occupied": [
                "pct_owner", "pct_owner_occupied", "PCT_OWNER_OCCUPIED",
                "owner_pct",
            ],
            "census_pct_immigrants": [
                "pct_immigrants", "pct_immigrant", "PCT_IMMIGRANTS",
                "immigrant_pct",
            ],
            "census_pct_university": [
                "pct_university", "pct_bachelors_or_higher", "PCT_UNIVERSITY",
                "university_pct",
            ],
        }

        # Resolve which source columns actually exist
        available_cols = set(census_da.columns)
        resolved = {}
        for target, candidates in census_columns.items():
            for c in candidates:
                if c in available_cols:
                    resolved[target] = c
                    break

        if not resolved:
            logger.warning(
                "No recognised demographic columns found in census_da; "
                "skipping census feature attachment"
            )
            for target in census_columns:
                properties[target] = np.nan
            return properties

        # Keep only geometry + resolved source columns for the join
        join_cols = ["geometry"] + list(resolved.values())
        census_subset = census_da[join_cols].copy()

        # Spatial join: assign each property to its DA polygon
        joined = gpd.sjoin(
            properties[["geometry"]],
            census_subset,
            how="left",
            predicate="within",
        )

        # Handle properties that fall in multiple overlapping DAs (keep first)
        joined = joined[~joined.index.duplicated(keep="first")]

        # Assign resolved columns with target names
        for target, source in resolved.items():
            properties[target] = joined[source].reindex(properties.index).values

        # Fill missing targets with NaN
        for target in census_columns:
            if target not in properties.columns:
                properties[target] = np.nan

        attached = list(resolved.keys())
        elapsed = time.perf_counter() - t0
        logger.info(
            f"Census features computed in {elapsed:.1f}s; "
            f"attached: {', '.join(attached)}"
        )

        return properties

    def _compute_location_features(
        self,
        properties: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """Compute location reference features.

        Features produced:
            - dist_downtown_m
            - dist_waterfront_m  (only if coastline data is available on disk)

        Args:
            properties: Property points in UTM10N.

        Returns:
            *properties* with location feature columns added.
        """
        t0 = time.perf_counter()
        logger.info("Computing location reference features...")

        # --- Distance to Downtown Vancouver ---
        if self._downtown_utm is not None:
            properties["dist_downtown_m"] = properties.geometry.distance(
                self._downtown_utm
            )
        else:
            properties["dist_downtown_m"] = np.nan

        # --- Distance to waterfront / coastline ---
        coastline_path = self.data_dir / "coastline.geojson"
        if coastline_path.exists():
            try:
                coastline = gpd.read_file(coastline_path).to_crs(UTM10N)
                properties = self._nearest_distance(
                    properties, coastline, "dist_waterfront_m", max_distance=50_000
                )
                logger.info("Computed waterfront distances from coastline data")
            except Exception as e:
                logger.warning(f"Failed to load coastline data: {e}")
                properties["dist_waterfront_m"] = np.nan
        else:
            logger.info(
                f"Coastline file not found at {coastline_path}; "
                "skipping dist_waterfront_m"
            )
            properties["dist_waterfront_m"] = np.nan

        elapsed = time.perf_counter() - t0
        logger.info(f"Location features computed in {elapsed:.1f}s")

        return properties

    def _compute_str_features(
        self,
        properties: gpd.GeoDataFrame,
        airbnb: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """Compute short-term rental (STR / Airbnb) density features.

        Features produced:
            - str_count_500m
            - str_density_per_km2
            - str_avg_price_500m

        Args:
            properties: Property points in UTM10N.
            airbnb: Airbnb listing points in UTM10N.

        Returns:
            *properties* with STR feature columns added.
        """
        t0 = time.perf_counter()
        logger.info("Computing short-term rental features...")

        radius_m = 500

        # --- Count within 500m ---
        properties = self._count_within_radius(
            properties, airbnb, radius_m, "str_count_500m"
        )

        # --- Density per km2 (area of 500m circle = pi * 0.5^2 km2) ---
        circle_area_km2 = np.pi * (radius_m / 1000) ** 2
        properties["str_density_per_km2"] = (
            properties["str_count_500m"] / circle_area_km2
        ).round(1)

        # --- Average nightly price within 500m ---
        price_col = None
        for col in ["price", "PRICE", "nightly_price"]:
            if col in airbnb.columns:
                price_col = col
                break

        if price_col is not None:
            properties = self._aggregate_within_radius(
                properties,
                airbnb,
                radius_m,
                price_col,
                "mean",
                "str_avg_price_500m",
            )
        else:
            logger.warning(
                "No price column found in Airbnb layer; "
                "setting str_avg_price_500m to NaN"
            )
            properties["str_avg_price_500m"] = np.nan

        elapsed = time.perf_counter() - t0
        logger.info(f"STR features computed in {elapsed:.1f}s")

        return properties
