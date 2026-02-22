"""
Central feature computation engine.

Computes all features defined in the feature registry for each property.
Operates in two modes:
1. Batch mode: Full property universe for training (vectorized)
2. Single mode: One property for API inference

Handles MLS absence gracefully by computing proxy features from
building footprints, zoning codes, and assessment values.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd

from src.features.feature_registry import (
    ALL_FEATURES,
    FeatureDefinition,
    PropertyType,
    get_features_by_phase,
    get_features_by_property_type,
)
from src.features.spatial_features import SpatialFeatureComputer
from src.features.building_footprint import BuildingFootprintEstimator

logger = logging.getLogger(__name__)

# Categorical columns handled natively by LightGBM via pandas Categorical dtype.
# No one-hot encoding needed -- LightGBM finds optimal split points on categoricals.
CATEGORICAL_COLUMNS = [
    "property_type",
    "zoning_district",
    "neighbourhood_code",
    "construction_type",
    "building_age_bucket",
    "tod_tier",
    "municipality",
    "parking_type",
    "view_type",
    "view_quality",
    "basement_type",
    "heating_system",
    "ownership_structure",
]

# Features that cannot be proxied when MLS data is absent.
# These are skipped entirely and logged.
MLS_ONLY_FEATURES = [
    "view_type",
    "view_quality",
    "parking_type",
    "strata_fee_per_sqft",
    "floor_level",
    "unit_exposure",
    "has_concierge",
    "has_pool",
    "has_gym",
    "pet_restriction",
    "age_restriction",
    "rainscreen_status",
]

# Maximum null fraction allowed before a feature column is dropped.
MAX_NULL_FRACTION = 0.80


class FeatureBuilder:
    """Central feature computation engine for the pricing model.

    Takes raw property data (the enriched property universe) and computes
    every feature needed for model training or single-property inference.
    Feature selection is controlled by ``phase`` (1-4) and
    ``property_type``, following the definitions in ``feature_registry``.

    When MLS data is unavailable (``mls_available=False``), proxy features
    are computed from building footprints, zoning codes, and assessment
    values. Imputation flags are added so the model can learn to
    discount imputed features.

    Usage::

        builder = FeatureBuilder(
            spatial_computer=spatial_computer,
            footprint_estimator=footprint_estimator,
            phase=1,
            mls_available=False,
        )
        X, y = builder.build_features_batch(enriched_df, PropertyType.CONDO)

    Args:
        spatial_computer: Pre-initialized SpatialFeatureComputer instance
            with spatial layers already preloaded.
        footprint_estimator: Pre-initialized BuildingFootprintEstimator
            for satellite-based living area estimates.
        phase: Implementation phase (1-4). Controls which features from
            the registry are included.
        mls_available: Whether MLS listing data is available. When False,
            proxy features are computed for living area, bedrooms, and
            construction type.
    """

    def __init__(
        self,
        spatial_computer: SpatialFeatureComputer,
        footprint_estimator: BuildingFootprintEstimator,
        phase: int = 1,
        mls_available: bool = False,
    ) -> None:
        self.spatial_computer = spatial_computer
        self.footprint_estimator = footprint_estimator
        self.phase = phase
        self.mls_available = mls_available

        logger.info(
            "FeatureBuilder initialized: phase=%d, mls_available=%s",
            phase,
            mls_available,
        )

    # ================================================================
    # BATCH MODE — full property universe for training
    # ================================================================

    def build_features_batch(
        self,
        properties_df: pd.DataFrame,
        property_type: Optional[PropertyType] = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Compute all features for the full property universe.

        Pipeline:
          1. Select features by property_type and phase
          2. Compute derived features
          3. Handle MLS absence (proxy features)
          4. Encode categoricals for LightGBM
          5. Create target: y = log(total_assessed_value)
          6. Drop features with >80% null

        Args:
            properties_df: Enriched property universe DataFrame. Expected
                to already have spatial features, footprint estimates,
                and census demographics attached.
            property_type: Filter features to those applicable for a
                given property type. If None, uses all features.

        Returns:
            Tuple of (X, y) where X is the feature DataFrame and y is
            the log-transformed target series. Both are aligned by index.

        Raises:
            ValueError: If total_assessed_value column is missing or
                all target values are invalid.
        """
        t0 = time.perf_counter()
        n_rows = len(properties_df)
        logger.info(
            "Building features (batch): %d properties, phase=%d, "
            "property_type=%s, mls_available=%s",
            n_rows,
            self.phase,
            property_type.value if property_type else "all",
            self.mls_available,
        )

        df = properties_df.copy()

        # --- 1. Compute derived features ---
        df = self._compute_derived_features(df)

        # --- 2. Handle MLS absence ---
        if not self.mls_available:
            df = self._handle_mls_absence(df)

        # --- 3. Encode categoricals ---
        df = self._encode_categoricals(df)

        # --- 4. Select features by phase and property type ---
        df = self._select_features(df, property_type)

        # --- 5. Create target variable ---
        if "total_assessed_value" not in properties_df.columns:
            raise ValueError(
                "Cannot create target: 'total_assessed_value' column not found. "
                "Run PropertyUniverseBuilder.build_universe() first."
            )

        # Use log-transformed assessed value as target
        raw_target = properties_df["total_assessed_value"].reindex(df.index)
        valid_mask = raw_target.notna() & (raw_target > 0)

        if not valid_mask.any():
            raise ValueError(
                "No valid target values found. All total_assessed_value entries "
                "are null or non-positive."
            )

        y = np.log(raw_target[valid_mask])
        df = df.loc[valid_mask]

        # --- 6. Drop high-null features ---
        df = self._drop_high_null_features(df)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Feature building complete: %d samples, %d features in %.1fs",
            len(df),
            len(df.columns),
            elapsed,
        )
        self._log_feature_summary(df)

        return df, y

    # ================================================================
    # SINGLE MODE — one property for API inference
    # ================================================================

    def build_features_single(
        self,
        property_data: dict,
        properties_df: Optional[pd.DataFrame] = None,
    ) -> dict:
        """Compute features for a single property (API inference).

        If ``properties_df`` is provided and the property's PID is found
        within it, pre-computed features are looked up directly. Otherwise,
        features are computed on-the-fly from the property attributes.

        Args:
            property_data: Dict with property attributes. Expected keys:
                pid, latitude, longitude, total_assessed_value,
                current_land_value, current_improvement_value,
                year_built, zoning_district, property_type.
            properties_df: Optional enriched property universe. If
                provided, the property is looked up by PID and
                pre-computed features are returned.

        Returns:
            Dict mapping feature_name to feature value.
        """
        pid = property_data.get("pid")
        logger.info("Building features (single): pid=%s", pid)

        # --- Try lookup from pre-computed universe ---
        if properties_df is not None and pid is not None:
            match = properties_df[properties_df["pid"] == str(pid)]
            if not match.empty:
                logger.info(
                    "Found property %s in pre-computed universe; "
                    "using pre-computed features",
                    pid,
                )
                row = match.iloc[0]
                # Apply derived features to a single-row DataFrame
                single_df = pd.DataFrame([row])
                single_df = self._compute_derived_features(single_df)
                if not self.mls_available:
                    single_df = self._handle_mls_absence(single_df)
                return single_df.iloc[0].to_dict()

        # --- Compute features on-the-fly ---
        logger.info(
            "Property %s not found in universe; computing features on-the-fly",
            pid,
        )
        single_df = pd.DataFrame([property_data])

        # Compute derived features
        single_df = self._compute_derived_features(single_df)

        # Handle MLS absence
        if not self.mls_available:
            single_df = self._handle_mls_absence(single_df)

        return single_df.iloc[0].to_dict()

    # ================================================================
    # DERIVED FEATURES
    # ================================================================

    def _compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all derived features from raw columns.

        These features combine or transform existing columns to capture
        non-linear relationships and domain-specific signals. The
        transit-wealth interaction term is particularly important: it
        captures how transit proximity has a *reversed sign* in wealthy
        areas (e.g., Shaughnessy), where proximity to transit lowers
        rather than raises values.

        Derived features:
            - effective_age
            - land_to_total_ratio
            - improvement_to_land_ratio
            - log_total_value, log_land_value, log_improvement_value
            - is_tod_area, tod_tier
            - transit_wealth_interaction (critical interaction)
            - building_age_bucket
            - value_per_lot_sqft
            - improvement_intensity

        Args:
            df: DataFrame with raw property columns.

        Returns:
            DataFrame with derived feature columns appended.
        """
        df = df.copy()

        # --- Effective age ---
        if "tax_assessment_year" in df.columns and "year_built" in df.columns:
            df["effective_age"] = np.where(
                df["year_built"].notna() & (df["year_built"] > 1800),
                df["tax_assessment_year"] - df["year_built"],
                np.nan,
            )
            # Cap at 0 (future year_built should not yield negative age)
            df["effective_age"] = df["effective_age"].clip(lower=0)

        # --- Value ratios ---
        if "current_land_value" in df.columns and "total_assessed_value" in df.columns:
            df["land_to_total_ratio"] = np.where(
                df["total_assessed_value"] > 0,
                df["current_land_value"] / df["total_assessed_value"],
                np.nan,
            )

        if "current_improvement_value" in df.columns and "current_land_value" in df.columns:
            df["improvement_to_land_ratio"] = np.where(
                df["current_land_value"] > 0,
                df["current_improvement_value"] / df["current_land_value"],
                np.nan,
            )

        # --- Log-transformed values ---
        if "total_assessed_value" in df.columns:
            df["log_total_value"] = np.where(
                df["total_assessed_value"] > 0,
                np.log(df["total_assessed_value"]),
                np.nan,
            )

        if "current_land_value" in df.columns:
            df["log_land_value"] = np.where(
                df["current_land_value"] > 0,
                np.log(df["current_land_value"]),
                np.nan,
            )

        if "current_improvement_value" in df.columns:
            # +1 to handle land-only properties with $0 improvement
            df["log_improvement_value"] = np.log(
                df["current_improvement_value"].clip(lower=0) + 1
            )

        # --- Transit-Oriented Development (TOD) tiers ---
        if "dist_nearest_skytrain_m" in df.columns:
            skytrain_dist = df["dist_nearest_skytrain_m"]

            # Binary TOD flag (within 800m of SkyTrain)
            df["is_tod_area"] = np.where(
                skytrain_dist.notna(),
                skytrain_dist <= 800,
                False,
            )

            # Tiered TOD classification (Bill 47 alignment)
            conditions = [
                skytrain_dist.notna() & (skytrain_dist <= 200),
                skytrain_dist.notna() & (skytrain_dist <= 400),
                skytrain_dist.notna() & (skytrain_dist <= 800),
            ]
            choices = ["tier1", "tier2", "tier3"]
            df["tod_tier"] = np.select(conditions, choices, default="none")

        # --- Transit-wealth interaction ---
        # CRITICAL: This captures the reversed sign of transit proximity
        # in wealthy areas like Shaughnessy, where SkyTrain proximity
        # actually *decreases* property values (noise, density concerns).
        if (
            "dist_nearest_skytrain_m" in df.columns
            and "census_median_income" in df.columns
        ):
            df["transit_wealth_interaction"] = (
                df["dist_nearest_skytrain_m"] * df["census_median_income"]
            )
            logger.debug(
                "Computed transit_wealth_interaction for %d rows",
                df["transit_wealth_interaction"].notna().sum(),
            )

        # --- Building age bucket ---
        if "effective_age" in df.columns:
            conditions = [
                df["effective_age"].notna() & (df["effective_age"] <= 5),
                df["effective_age"].notna() & (df["effective_age"] <= 15),
                df["effective_age"].notna() & (df["effective_age"] <= 30),
                df["effective_age"].notna() & (df["effective_age"] <= 60),
                df["effective_age"].notna() & (df["effective_age"] > 60),
            ]
            choices = ["New", "Recent", "Mature", "Older", "Heritage"]
            df["building_age_bucket"] = np.select(
                conditions, choices, default="Unknown"
            )
            df.loc[df["effective_age"].isna(), "building_age_bucket"] = np.nan

        # --- Value per lot sqft ---
        if "total_assessed_value" in df.columns and "lot_size_sqft" in df.columns:
            df["value_per_lot_sqft"] = np.where(
                df["lot_size_sqft"].notna() & (df["lot_size_sqft"] > 0),
                df["total_assessed_value"] / df["lot_size_sqft"],
                np.nan,
            )

        # --- Improvement intensity ---
        if (
            "current_improvement_value" in df.columns
            and "building_footprint_sqm" in df.columns
        ):
            df["improvement_intensity"] = np.where(
                df["building_footprint_sqm"].notna()
                & (df["building_footprint_sqm"] > 0),
                df["current_improvement_value"] / df["building_footprint_sqm"],
                np.nan,
            )

        n_derived = sum(
            1
            for col in [
                "effective_age",
                "land_to_total_ratio",
                "improvement_to_land_ratio",
                "log_total_value",
                "log_land_value",
                "log_improvement_value",
                "is_tod_area",
                "tod_tier",
                "transit_wealth_interaction",
                "building_age_bucket",
                "value_per_lot_sqft",
                "improvement_intensity",
            ]
            if col in df.columns
        )
        logger.info("Computed %d derived features", n_derived)

        return df

    # ================================================================
    # MLS ABSENCE HANDLING
    # ================================================================

    def _handle_mls_absence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create proxy features when MLS data is unavailable.

        When MLS listing data is not available, several critical features
        (living area, bedrooms, construction type) are missing. This
        method creates proxies from building footprints, zoning codes,
        and assessment values.

        Each proxy gets a companion ``_imputed`` flag column so the model
        can learn to weight imputed features differently.

        Proxied features:
            - living_area_sqft: from estimated_living_area_sqft
            - construction_type: inferred from year_built and zoning
            - bedrooms: estimated from living area

        Skipped features (cannot be proxied):
            - view_type, view_quality, parking_type
            - strata_fee_per_sqft, floor_level, unit_exposure

        Args:
            df: DataFrame potentially missing MLS-sourced columns.

        Returns:
            DataFrame with proxy features added and imputation flags set.
        """
        df = df.copy()
        proxied = []
        skipped = []

        # --- Living area proxy ---
        if "living_area_sqft" not in df.columns or df["living_area_sqft"].isna().all():
            if "estimated_living_area_sqft" in df.columns:
                df["living_area_sqft"] = df["estimated_living_area_sqft"]
                df["living_area_imputed"] = True
                proxied.append("living_area_sqft <- estimated_living_area_sqft")
            else:
                df["living_area_sqft"] = np.nan
                df["living_area_imputed"] = True
                skipped.append("living_area_sqft (no footprint estimate available)")

        # --- Construction type proxy ---
        if (
            "construction_type" not in df.columns
            or df["construction_type"].isna().all()
        ):
            df["construction_type"] = self._infer_construction_type(df)
            df["construction_type_imputed"] = True
            proxied.append("construction_type <- inferred from year_built + zoning")

        # --- Bedrooms proxy ---
        if "bedrooms" not in df.columns or df["bedrooms"].isna().all():
            df["bedrooms"] = self._estimate_bedrooms(df)
            df["bedrooms_imputed"] = True
            proxied.append("bedrooms <- estimated from living_area_sqft")

        # --- Log skipped MLS-only features ---
        for feature in MLS_ONLY_FEATURES:
            if feature not in df.columns:
                skipped.append(feature)

        if proxied:
            logger.info(
                "MLS absence: proxied %d features: %s",
                len(proxied),
                "; ".join(proxied),
            )
        if skipped:
            logger.info(
                "MLS absence: skipped %d features (cannot proxy): %s",
                len(skipped),
                ", ".join(skipped),
            )

        return df

    @staticmethod
    def _infer_construction_type(df: pd.DataFrame) -> pd.Series:
        """Infer construction type from year built and zoning code.

        Heuristic:
            - year_built < 1970: wood_frame (pre-concrete era)
            - Zoning RM-4+ and post-2000: concrete (modern high-rise)
            - Everything else: wood_frame (conservative default)

        Args:
            df: DataFrame with optional year_built and zoning_district.

        Returns:
            Series of inferred construction type strings.
        """
        result = pd.Series("wood_frame", index=df.index)

        has_year = "year_built" in df.columns
        has_zoning = "zoning_district" in df.columns

        if has_year and has_zoning:
            # Modern high-rise zones (RM-4+, FM-1, DD, C-3A) built after 2000
            high_rise_prefixes = ("RM-4", "RM-5", "RM-6", "FM-1", "DD", "C-3")
            zoning = df["zoning_district"].fillna("").str.upper()
            is_high_rise_zone = pd.Series(False, index=df.index)
            for prefix in high_rise_prefixes:
                is_high_rise_zone |= zoning.str.startswith(prefix)

            modern_concrete = (
                is_high_rise_zone
                & df["year_built"].notna()
                & (df["year_built"] >= 2000)
            )
            result[modern_concrete] = "concrete"

        return result

    @staticmethod
    def _estimate_bedrooms(df: pd.DataFrame) -> pd.Series:
        """Estimate bedroom count from living area.

        Based on typical Metro Vancouver unit sizes:
            - < 600 sqft: studio (0 bedrooms)
            - 600-899 sqft: 1 bedroom
            - 900-1299 sqft: 2 bedrooms
            - 1300-1799 sqft: 3 bedrooms
            - 1800-2499 sqft: 4 bedrooms
            - 2500+ sqft: 5 bedrooms

        Args:
            df: DataFrame with living_area_sqft column.

        Returns:
            Series of estimated bedroom counts (int).
        """
        if "living_area_sqft" not in df.columns:
            return pd.Series(np.nan, index=df.index)

        area = df["living_area_sqft"]
        conditions = [
            area < 600,
            (area >= 600) & (area < 900),
            (area >= 900) & (area < 1300),
            (area >= 1300) & (area < 1800),
            (area >= 1800) & (area < 2500),
            area >= 2500,
        ]
        choices = [0, 1, 2, 3, 4, 5]
        result = np.select(conditions, choices, default=np.nan)
        return pd.Series(result, index=df.index).astype("Int64")

    # ================================================================
    # CATEGORICAL ENCODING
    # ================================================================

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert string categoricals to pandas Categorical dtype.

        LightGBM handles pandas Categorical dtype natively, finding
        optimal split points without one-hot encoding. This preserves
        ordinality where it exists and reduces memory usage.

        Args:
            df: DataFrame with string categorical columns.

        Returns:
            DataFrame with categoricals converted to pandas Categorical.
        """
        df = df.copy()
        encoded_count = 0

        for col in CATEGORICAL_COLUMNS:
            if col in df.columns:
                # Convert to string first (handles mixed types), then to Categorical
                df[col] = df[col].astype(str).replace("nan", np.nan)
                df[col] = pd.Categorical(df[col])
                encoded_count += 1

        if encoded_count > 0:
            logger.debug(
                "Encoded %d categorical columns for LightGBM", encoded_count
            )

        return df

    # ================================================================
    # FEATURE SELECTION
    # ================================================================

    def _select_features(
        self,
        df: pd.DataFrame,
        property_type: Optional[PropertyType] = None,
    ) -> pd.DataFrame:
        """Select features based on phase and property type.

        Uses the feature registry to determine which features should be
        included, then intersects with columns actually present in the
        DataFrame. Features defined in the registry but absent from the
        data are silently skipped (they may not have been computed yet
        due to data source availability).

        Args:
            df: DataFrame with all computed feature columns.
            property_type: Property type to filter features for. If None,
                all features for the current phase are selected.

        Returns:
            DataFrame with only the selected feature columns.
        """
        # Get registry-defined features for this phase
        phase_features: list[FeatureDefinition] = get_features_by_phase(self.phase)
        phase_names = {f.name for f in phase_features}

        # Optionally filter by property type
        if property_type is not None:
            type_features = get_features_by_property_type(property_type)
            type_names = {f.name for f in type_features}
            eligible_names = phase_names & type_names
        else:
            eligible_names = phase_names

        # Also include derived features that are not in the registry
        # but are computed by _compute_derived_features
        derived_features = {
            "effective_age",
            "land_to_total_ratio",
            "improvement_to_land_ratio",
            "log_total_value",
            "log_land_value",
            "log_improvement_value",
            "is_tod_area",
            "tod_tier",
            "transit_wealth_interaction",
            "building_age_bucket",
            "value_per_lot_sqft",
            "improvement_intensity",
            # MLS proxy flags
            "living_area_imputed",
            "construction_type_imputed",
            "bedrooms_imputed",
            # Building footprint estimates
            "estimated_living_area_sqft",
            "estimated_stories",
            "building_footprint_sqm",
            "living_area_confidence",
            "footprint_to_lot_ratio",
            # Spatial features not in registry by exact name
            "dist_nearest_transit_m",
            "dist_nearest_skytrain_m",
            "transit_stops_400m",
            "transit_stops_800m",
            "unique_routes_400m",
            "has_skytrain_800m",
            "dist_nearest_school_m",
            "dist_nearest_elementary_m",
            "dist_nearest_secondary_m",
            "schools_within_1km",
            "dist_nearest_park_m",
            "parks_within_500m",
            "park_area_within_1km_sqm",
            "in_alr",
            "in_floodplain",
            "contaminated_sites_500m",
            "dist_nearest_contaminated_m",
            "dist_downtown_m",
            "dist_waterfront_m",
            "str_count_500m",
            "str_density_per_km2",
            "str_avg_price_500m",
            "census_median_income",
            "census_pop_density",
            "census_pct_owner_occupied",
            "census_pct_immigrants",
            "census_pct_university",
        }

        all_eligible = eligible_names | derived_features

        # Intersect with columns actually present
        present_columns = [col for col in df.columns if col in all_eligible]

        # Exclude the target variable and identifiers from features
        exclude = {
            "total_assessed_value",
            "pid",
            "geometry",
            "full_address",
            "property_postal_code",
            "street_name",
            "log_total_value",  # This is the target, not a feature
        }
        selected = [col for col in present_columns if col not in exclude]

        not_found = eligible_names - set(df.columns) - exclude
        if not_found:
            logger.debug(
                "Registry features not found in data (%d): %s",
                len(not_found),
                ", ".join(sorted(not_found)[:20]),
            )

        logger.info(
            "Feature selection: %d eligible, %d present, %d selected",
            len(all_eligible),
            len(present_columns),
            len(selected),
        )

        return df[selected]

    # ================================================================
    # HIGH-NULL FEATURE DROPPING
    # ================================================================

    def _drop_high_null_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop feature columns with more than 80% null values.

        Features with excessive nulls provide little signal and can
        degrade model performance. The threshold is set at 80% to be
        conservative -- features with moderate missingness are retained
        because LightGBM handles NaN natively.

        Args:
            df: Feature DataFrame.

        Returns:
            DataFrame with high-null columns removed.
        """
        null_fractions = df.isnull().mean()
        high_null = null_fractions[null_fractions > MAX_NULL_FRACTION]

        if not high_null.empty:
            logger.warning(
                "Dropping %d features with >%.0f%% null: %s",
                len(high_null),
                MAX_NULL_FRACTION * 100,
                ", ".join(high_null.index.tolist()),
            )
            df = df.drop(columns=high_null.index)

        return df

    # ================================================================
    # UTILITIES
    # ================================================================

    def get_feature_names(
        self, property_type: Optional[PropertyType] = None
    ) -> list[str]:
        """Return the list of feature names for a given property type.

        Useful for inspecting which features would be used without
        actually computing them.

        Args:
            property_type: Property type to get features for. If None,
                returns all features for the current phase.

        Returns:
            Sorted list of feature name strings.
        """
        phase_features = get_features_by_phase(self.phase)
        phase_names = {f.name for f in phase_features}

        if property_type is not None:
            type_features = get_features_by_property_type(property_type)
            type_names = {f.name for f in type_features}
            return sorted(phase_names & type_names)

        return sorted(phase_names)

    def compute_feature_completeness(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute per-property feature completeness statistics.

        For each property (row), counts how many of the expected features
        are populated (non-null). This is useful for identifying
        properties that may produce unreliable predictions due to
        missing data.

        Args:
            df: Feature DataFrame (output of build_features_batch X).

        Returns:
            DataFrame with columns:
                - pid (if present in original data)
                - total_features: number of expected features
                - populated_features: number of non-null features
                - completeness_pct: populated / total as percentage
                - missing_features: list of missing feature names
        """
        total_features = len(df.columns)

        populated = df.notna().sum(axis=1)
        completeness_pct = (populated / total_features * 100).round(1)

        missing_features = df.apply(
            lambda row: row.index[row.isna()].tolist(), axis=1
        )

        result = pd.DataFrame(
            {
                "total_features": total_features,
                "populated_features": populated,
                "completeness_pct": completeness_pct,
                "missing_features": missing_features,
            },
            index=df.index,
        )

        logger.info(
            "Feature completeness: median=%.1f%%, mean=%.1f%%, "
            "min=%.1f%%, max=%.1f%%",
            completeness_pct.median(),
            completeness_pct.mean(),
            completeness_pct.min(),
            completeness_pct.max(),
        )

        return result

    def _log_feature_summary(self, df: pd.DataFrame) -> None:
        """Log a summary of the feature DataFrame.

        Args:
            df: Feature DataFrame.
        """
        n_numeric = df.select_dtypes(include=[np.number]).shape[1]
        n_categorical = df.select_dtypes(include=["category"]).shape[1]
        n_bool = df.select_dtypes(include=["bool"]).shape[1]
        n_other = len(df.columns) - n_numeric - n_categorical - n_bool

        null_pct = df.isnull().mean().mean() * 100

        logger.info("--- Feature Summary ---")
        logger.info("  Total features:    %d", len(df.columns))
        logger.info("  Numeric:           %d", n_numeric)
        logger.info("  Categorical:       %d", n_categorical)
        logger.info("  Boolean:           %d", n_bool)
        logger.info("  Other:             %d", n_other)
        logger.info("  Avg null %%:        %.1f%%", null_pct)
        logger.info("  Total samples:     %d", len(df))
