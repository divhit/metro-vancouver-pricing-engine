"""
Sub-region segmentation engine.

Vancouver has 22 official local areas with dramatically different pricing dynamics.
This module segments the property universe into model-ready segments:
1. Macro: 22 local areas
2. Micro: K-Means clusters within each local area
3. Cross-cut with property type for segment-specific models

Fallback hierarchy ensures every property gets a prediction even
when its specific segment has too few training samples.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
except ImportError:
    KMeans = None  # type: ignore[assignment,misc]
    StandardScaler = None  # type: ignore[assignment,misc]

from src.features.feature_registry import PropertyType
from src.pipeline.property_universe import VANCOUVER_LOCAL_AREAS

logger = logging.getLogger(__name__)

# Downtown Vancouver coordinates for distance gradient computation
_DOWNTOWN_LAT = 49.2827
_DOWNTOWN_LON = -123.1207

# Earth radius in meters for Haversine
_EARTH_RADIUS_M = 6_371_000


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute Haversine distance in meters between two points."""
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def _haversine_m_vectorized(
    lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float
) -> np.ndarray:
    """Vectorized Haversine distance computation in meters."""
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


class SubRegionEngine:
    """Geographic and property-type segmentation engine.

    Segments properties into model-ready groups using a hierarchy:
      - Macro segments: 22 Vancouver local areas x property type
      - Micro segments: K-Means clusters within each local area x property type
      - Fallback hierarchy: micro -> area -> citywide (by property type) -> citywide (all)

    A segment needs at least ``min_segment_size`` properties to
    warrant its own model; otherwise predictions fall back to the
    next level up in the hierarchy.

    Usage::

        engine = SubRegionEngine(min_segment_size=200)
        df = engine.define_micro_neighborhoods(properties_df)
        segment = engine.assign_segment(property_data, properties_df)

    Args:
        min_segment_size: Minimum number of training samples required
            for a segment to receive its own model.
    """

    def __init__(self, min_segment_size: int = 200) -> None:
        self.min_segment_size = min_segment_size
        # Cluster centers per local area for assigning new properties
        self._cluster_centers: dict[str, tuple[np.ndarray, StandardScaler]] = {}
        # Mapping of area -> number of clusters actually used
        self._n_clusters: dict[str, int] = {}

        logger.info(
            "SubRegionEngine initialized: min_segment_size=%d",
            min_segment_size,
        )

    # ================================================================
    # MICRO NEIGHBORHOODS
    # ================================================================

    def define_micro_neighborhoods(
        self,
        properties_df: pd.DataFrame,
        n_clusters_per_area: int = 3,
        min_cluster_size: int = 50,
    ) -> pd.DataFrame:
        """Create micro-neighborhoods via K-Means within each local area.

        For each of Vancouver's 22 local areas, runs K-Means on a
        standardized feature set to identify pricing sub-clusters.
        Areas with fewer properties automatically get fewer clusters.

        Clustering features:
          - log(total_assessed_value)
          - land_to_total_ratio
          - dist_nearest_skytrain_m (if available)
          - year_built

        Args:
            properties_df: Property universe DataFrame with at least
                ``neighbourhood_code``, ``total_assessed_value``, and
                ``year_built`` columns.
            n_clusters_per_area: Target number of clusters per local area.
                Reduced automatically if the area is too small.
            min_cluster_size: Minimum properties per cluster. If an area
                has fewer than ``min_cluster_size * n_clusters_per_area``
                properties, the cluster count is reduced.

        Returns:
            DataFrame with ``micro_neighborhood`` column appended
            (e.g., 'Kitsilano_0', 'Kitsilano_1').
        """
        df = properties_df.copy()
        df["micro_neighborhood"] = np.nan

        if "neighbourhood_code" not in df.columns:
            logger.warning(
                "No neighbourhood_code column; skipping micro-neighborhood clustering"
            )
            return df

        # Determine clustering feature columns
        cluster_features = []
        if "total_assessed_value" in df.columns:
            df["_log_value"] = np.where(
                df["total_assessed_value"] > 0,
                np.log(df["total_assessed_value"]),
                np.nan,
            )
            cluster_features.append("_log_value")

        if "land_to_total_ratio" in df.columns:
            cluster_features.append("land_to_total_ratio")
        elif "land_ratio" in df.columns:
            cluster_features.append("land_ratio")

        if "dist_nearest_skytrain_m" in df.columns:
            cluster_features.append("dist_nearest_skytrain_m")

        if "year_built" in df.columns:
            cluster_features.append("year_built")

        if not cluster_features:
            logger.warning(
                "No clustering features available; skipping micro-neighborhoods"
            )
            return df

        logger.info(
            "Defining micro-neighborhoods using features: %s",
            ", ".join(cluster_features),
        )

        # Pre-initialize the column as object dtype so string labels can be assigned
        df["micro_neighborhood"] = pd.Series(
            [None] * len(df), index=df.index, dtype="object",
        )

        areas_processed = 0
        areas_skipped = 0

        for area_code in df["neighbourhood_code"].dropna().unique():
            area_mask = df["neighbourhood_code"] == area_code
            area_df = df.loc[area_mask, cluster_features].dropna()

            n_area_properties = len(area_df)

            if n_area_properties < min_cluster_size:
                # Too few properties for any clustering -- assign all to cluster 0
                df.loc[area_mask, "micro_neighborhood"] = f"{area_code}_0"
                self._n_clusters[area_code] = 1
                areas_skipped += 1
                continue

            # Adaptively reduce cluster count for small areas
            n_clusters = min(
                n_clusters_per_area,
                max(1, n_area_properties // min_cluster_size),
            )

            # Standardize features for K-Means
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(area_df.values)

            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
            )
            labels = kmeans.fit_predict(X_scaled)

            # Store cluster centers and scaler for later assignment
            self._cluster_centers[area_code] = (kmeans.cluster_centers_, scaler)
            self._n_clusters[area_code] = n_clusters

            # Assign labels to the properties that had valid features
            label_series = pd.Series(
                [f"{area_code}_{lbl}" for lbl in labels],
                index=area_df.index,
            )
            df.loc[label_series.index, "micro_neighborhood"] = label_series
            areas_processed += 1

        # Fill any remaining NaN micro_neighborhoods with area-level default
        remaining_mask = df["micro_neighborhood"].isna() & df["neighbourhood_code"].notna()
        if remaining_mask.any():
            df.loc[remaining_mask, "micro_neighborhood"] = (
                df.loc[remaining_mask, "neighbourhood_code"] + "_0"
            )

        # Clean up temporary column
        if "_log_value" in df.columns:
            df = df.drop(columns=["_log_value"])

        logger.info(
            "Micro-neighborhoods defined: %d areas clustered, %d areas too small "
            "(assigned single cluster), %d total micro-neighborhoods",
            areas_processed,
            areas_skipped,
            df["micro_neighborhood"].nunique(),
        )

        return df

    # ================================================================
    # SEGMENT KEY GENERATION
    # ================================================================

    @staticmethod
    def get_segment_key(
        neighbourhood_code: str,
        property_type: str,
        use_micro: bool = False,
        micro_id: Optional[str] = None,
    ) -> str:
        """Generate a canonical segment key.

        Segment key format:
          - Standard: 'Kitsilano__condo'
          - With micro: 'Kitsilano_0__condo'
          - Citywide: 'citywide__condo'

        Args:
            neighbourhood_code: Local area name or code.
            property_type: Property type string (e.g. 'condo', 'detached').
            use_micro: Whether to use micro-neighborhood granularity.
            micro_id: Full micro-neighborhood ID (e.g. 'Kitsilano_0').
                Required when ``use_micro=True``.

        Returns:
            Canonical segment key string.
        """
        if use_micro and micro_id is not None:
            return f"{micro_id}__{property_type}"
        return f"{neighbourhood_code}__{property_type}"

    # ================================================================
    # SEGMENT STATISTICS
    # ================================================================

    def get_segment_stats(self, properties_df: pd.DataFrame) -> pd.DataFrame:
        """Compute statistics for each segment.

        Produces a summary table used to decide which segments get
        their own model versus which fall back to a broader segment.

        Args:
            properties_df: Property universe DataFrame with
                ``neighbourhood_code``, ``property_type``, and
                ``total_assessed_value`` columns.

        Returns:
            DataFrame with columns: segment_key, count, median_value,
            mean_value, std_value, min_year, max_year.
        """
        if "neighbourhood_code" not in properties_df.columns:
            logger.warning("No neighbourhood_code column; returning empty stats")
            return pd.DataFrame()

        records = []
        for (area, ptype), group in properties_df.groupby(
            ["neighbourhood_code", "property_type"], observed=True
        ):
            segment_key = self.get_segment_key(area, ptype)
            values = group["total_assessed_value"] if "total_assessed_value" in group.columns else pd.Series(dtype=float)

            record = {
                "segment_key": segment_key,
                "count": len(group),
                "median_value": values.median() if not values.empty else np.nan,
                "mean_value": values.mean() if not values.empty else np.nan,
                "std_value": values.std() if not values.empty else np.nan,
                "min_year": (
                    int(group["year_built"].min())
                    if "year_built" in group.columns and group["year_built"].notna().any()
                    else np.nan
                ),
                "max_year": (
                    int(group["year_built"].max())
                    if "year_built" in group.columns and group["year_built"].notna().any()
                    else np.nan
                ),
            }
            records.append(record)

        stats_df = pd.DataFrame(records)

        if not stats_df.empty:
            stats_df = stats_df.sort_values("count", ascending=False).reset_index(drop=True)
            logger.info(
                "Segment stats: %d segments, median count=%d, "
                "largest=%d (%s), smallest=%d (%s)",
                len(stats_df),
                int(stats_df["count"].median()),
                int(stats_df["count"].max()),
                stats_df.loc[stats_df["count"].idxmax(), "segment_key"],
                int(stats_df["count"].min()),
                stats_df.loc[stats_df["count"].idxmin(), "segment_key"],
            )

        return stats_df

    # ================================================================
    # SEGMENT MODEL DECISIONS
    # ================================================================

    def should_use_segment_model(
        self, segment_key: str, segment_stats: pd.DataFrame
    ) -> bool:
        """Determine whether a segment has enough data for its own model.

        Args:
            segment_key: Canonical segment key string.
            segment_stats: Output of ``get_segment_stats()``.

        Returns:
            True if the segment has >= ``min_segment_size`` properties.
        """
        match = segment_stats[segment_stats["segment_key"] == segment_key]
        if match.empty:
            return False
        return int(match.iloc[0]["count"]) >= self.min_segment_size

    @staticmethod
    def get_fallback_segment(segment_key: str) -> str:
        """Get the next level up in the fallback hierarchy.

        Fallback chain:
          micro__type  ->  area__type  ->  citywide__type  ->  citywide__all

        Examples:
          'Kitsilano_0__condo'  ->  'Kitsilano__condo'
          'Kitsilano__condo'    ->  'citywide__condo'
          'citywide__condo'     ->  'citywide__all'
          'citywide__all'       ->  'citywide__all' (terminal)

        Args:
            segment_key: Current segment key string.

        Returns:
            Segment key for the next broader level.
        """
        if segment_key == "citywide__all":
            return "citywide__all"

        parts = segment_key.split("__")
        if len(parts) != 2:
            return "citywide__all"

        area_part, type_part = parts

        # Check if this is a micro-neighborhood (has an underscore-digit suffix)
        # Pattern: 'AreaName_0' or 'Area-Name_2'
        if "_" in area_part:
            # Check if the last segment after underscore is a digit (micro ID)
            last_underscore_idx = area_part.rfind("_")
            potential_micro_id = area_part[last_underscore_idx + 1:]
            if potential_micro_id.isdigit():
                # Strip the micro cluster suffix to get the area
                area_name = area_part[:last_underscore_idx]
                return f"{area_name}__{type_part}"

        # Area-level segment -> citywide by type
        if area_part != "citywide":
            return f"citywide__{type_part}"

        # citywide__type -> citywide__all
        if type_part != "all":
            return "citywide__all"

        # Already at the broadest level
        return "citywide__all"

    # ================================================================
    # SEGMENT ASSIGNMENT
    # ================================================================

    def assign_segment(
        self,
        property_data: dict,
        properties_df: Optional[pd.DataFrame] = None,
    ) -> str:
        """Determine the segment key for a single property.

        If micro-neighborhoods have been defined and cluster centers
        are available, assigns the property to the nearest cluster
        center within its local area.

        Args:
            property_data: Dict with at least ``neighbourhood_code`` and
                ``property_type``. For micro-neighborhood assignment also
                needs ``total_assessed_value`` and ``year_built``.
            properties_df: Optional property universe. Not currently used
                for assignment but reserved for future enhancements.

        Returns:
            Canonical segment key string.
        """
        area = property_data.get("neighbourhood_code", "")
        ptype = property_data.get("property_type", "")

        if not area or not ptype:
            logger.warning(
                "Missing neighbourhood_code or property_type; assigning to citywide"
            )
            return f"citywide__{ptype or 'all'}"

        # Try micro-neighborhood assignment if cluster centers exist
        if area in self._cluster_centers:
            cluster_features = []
            feature_values = []

            if "total_assessed_value" in property_data:
                val = property_data["total_assessed_value"]
                if val and val > 0:
                    feature_values.append(np.log(val))
                    cluster_features.append("_log_value")

            for feat_name in ["land_to_total_ratio", "land_ratio"]:
                if feat_name in property_data and property_data[feat_name] is not None:
                    feature_values.append(property_data[feat_name])
                    cluster_features.append(feat_name)
                    break

            if "dist_nearest_skytrain_m" in property_data:
                val = property_data.get("dist_nearest_skytrain_m")
                if val is not None:
                    feature_values.append(val)
                    cluster_features.append("dist_nearest_skytrain_m")

            if "year_built" in property_data:
                val = property_data.get("year_built")
                if val is not None:
                    feature_values.append(val)
                    cluster_features.append("year_built")

            centers, scaler = self._cluster_centers[area]

            # Only assign if we have the right number of features
            if len(feature_values) == centers.shape[1]:
                X = np.array(feature_values).reshape(1, -1)
                X_scaled = scaler.transform(X)

                # Find nearest cluster center
                distances = np.linalg.norm(centers - X_scaled, axis=1)
                nearest_cluster = int(np.argmin(distances))
                micro_id = f"{area}_{nearest_cluster}"

                return self.get_segment_key(
                    area, ptype, use_micro=True, micro_id=micro_id
                )

        # Fall back to area-level segment
        return self.get_segment_key(area, ptype)

    # ================================================================
    # PRICE GRADIENT ANALYSIS
    # ================================================================

    def compute_price_gradient(self, properties_df: pd.DataFrame) -> pd.DataFrame:
        """Compute downtown distance-based price gradients.

        For each property, computes distance to downtown Vancouver
        and the residual log-value after controlling for distance.
        This reveals the spatial pricing structure per property type:

        - Condos: steep price decay with distance from downtown
        - Detached: can be inverted on the West Side (Shaughnessy,
          Point Grey, Dunbar are further but more expensive)

        Args:
            properties_df: Property universe DataFrame with ``latitude``,
                ``longitude``, ``total_assessed_value``, and
                ``property_type`` columns.

        Returns:
            DataFrame with added columns:
              - dist_downtown_m: Haversine distance to downtown in meters
              - log_value_residual: residual after linear distance control
              - price_gradient_group: property type used for the gradient
        """
        df = properties_df.copy()

        required = ["latitude", "longitude", "total_assessed_value", "property_type"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            logger.warning(
                "Cannot compute price gradient; missing columns: %s",
                ", ".join(missing),
            )
            return df

        # Compute distance to downtown
        valid_coords = df["latitude"].notna() & df["longitude"].notna()
        df["dist_downtown_m"] = np.nan
        if valid_coords.any():
            df.loc[valid_coords, "dist_downtown_m"] = _haversine_m_vectorized(
                df.loc[valid_coords, "latitude"].values,
                df.loc[valid_coords, "longitude"].values,
                _DOWNTOWN_LAT,
                _DOWNTOWN_LON,
            )

        # Compute log-value residuals per property type
        df["log_value_residual"] = np.nan
        df["price_gradient_group"] = df["property_type"]

        for ptype in df["property_type"].dropna().unique():
            mask = (
                (df["property_type"] == ptype)
                & df["dist_downtown_m"].notna()
                & (df["total_assessed_value"] > 0)
            )

            if mask.sum() < 10:
                continue

            log_values = np.log(df.loc[mask, "total_assessed_value"].values)
            distances = df.loc[mask, "dist_downtown_m"].values

            # Simple linear regression: log_value = a + b * distance
            X_dist = np.column_stack([np.ones(mask.sum()), distances])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X_dist, log_values, rcond=None)
                predicted = X_dist @ coeffs
                residuals = log_values - predicted
                df.loc[mask, "log_value_residual"] = residuals
            except np.linalg.LinAlgError:
                logger.warning(
                    "Linear regression failed for property type '%s'", ptype
                )

        n_computed = df["log_value_residual"].notna().sum()
        logger.info(
            "Price gradient computed: %d properties with residuals, "
            "%d property types analyzed",
            n_computed,
            df["property_type"].nunique(),
        )

        return df
