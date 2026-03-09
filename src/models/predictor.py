"""
Property value predictor.

Orchestrates the full prediction pipeline:
1. Resolve property identity
2. Build features
3. Select and run ML model
4. Apply Tier 2 adjustments
5. Find comparables and reconcile
6. Package results with confidence and explanation
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import shap

from src.adjustments.adjustment_engine import AdjustmentEngine
from src.comparables.comparable_engine import ComparableEngine
from src.features.building_footprint import BuildingFootprintEstimator
from src.features.feature_builder import FeatureBuilder
from src.features.feature_registry import PropertyType
from src.features.spatial_features import SpatialFeatureComputer
from src.models.quantile_models import QuantileModelTrainer
from src.models.subregions import SubRegionEngine
from src.models.trainer import ModelTrainer
from src.models.types import AdjustmentResult, ComparableProperty, PredictionResult

logger = logging.getLogger(__name__)

# Version string for prediction metadata
_MODEL_VERSION = "0.1.0"

# Neighbourhood code to name mapping — 22 official City of Vancouver local areas.
# These match the codes assigned by NeighbourhoodAssigner via spatial join
# against the City's local-area-boundary GeoJSON polygons.
_NEIGHBOURHOOD_NAMES: dict[str, str] = {
    "1": "West Point Grey", "2": "Kitsilano", "3": "Dunbar-Southlands",
    "4": "Arbutus Ridge", "5": "Kerrisdale", "6": "Shaughnessy",
    "7": "Fairview", "8": "South Cambie", "9": "Oakridge",
    "10": "Marpole", "11": "Riley Park", "12": "Sunset",
    "13": "Mount Pleasant", "14": "Grandview-Woodland",
    "15": "Hastings-Sunrise", "16": "Kensington-Cedar Cottage",
    "17": "Killarney", "18": "Victoria-Fraserview",
    "19": "Strathcona", "20": "Renfrew-Collingwood",
    "21": "Downtown", "22": "West End",
}


class PropertyPredictor:
    """Orchestrates the full prediction pipeline for property valuation.

    Lazy-loads models, feature builder, subregion engine, adjustment engine,
    and comparable engine. Caches loaded models in memory to avoid repeated
    disk I/O across predictions.

    Usage::

        predictor = PropertyPredictor(model_dir="models")
        result = predictor.predict(
            pid="012-345-678",
            properties_df=enriched_df,
        )
        print(f"Estimate: ${result.point_estimate:,.0f}")
        print(f"Grade: {result.confidence_grade}")

    Args:
        model_dir: Directory containing trained model files.
        mls_available: Whether MLS listing data is available for
            enhanced feature computation and similarity scoring.
    """

    def __init__(
        self,
        model_dir: str = "models",
        mls_available: bool = False,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.market_model_dir = Path(model_dir) / "market"
        self.mls_available = mls_available

        # Lazy-loaded components
        self._feature_builder: Optional[FeatureBuilder] = None
        self._subregion_engine: Optional[SubRegionEngine] = None
        self._adjustment_engine: Optional[AdjustmentEngine] = None
        self._comparable_engine: Optional[ComparableEngine] = None
        self._quantile_trainer: Optional[QuantileModelTrainer] = None
        self._model_trainer: Optional[ModelTrainer] = None

        # In-memory model cache: segment_key -> (model, metadata)
        self._model_cache: dict[str, tuple[Any, dict]] = {}
        # Quantile model cache: segment_key -> dict[float, model]
        self._quantile_cache: dict[str, dict[float, Any]] = {}
        # Market model cache: property_type -> (model, metadata)
        self._market_model_cache: dict[str, tuple[Any, dict]] = {}

        logger.info(
            "PropertyPredictor initialized: model_dir=%s, mls_available=%s",
            self.model_dir,
            mls_available,
        )

    # ================================================================
    # LAZY COMPONENT INITIALIZATION
    # ================================================================

    @property
    def feature_builder(self) -> FeatureBuilder:
        """Lazy-load the feature builder."""
        if self._feature_builder is None:
            spatial = SpatialFeatureComputer()
            footprint = BuildingFootprintEstimator()
            self._feature_builder = FeatureBuilder(
                spatial_computer=spatial,
                footprint_estimator=footprint,
                phase=1,
                mls_available=self.mls_available,
            )
        return self._feature_builder

    @property
    def subregion_engine(self) -> SubRegionEngine:
        """Lazy-load the subregion engine."""
        if self._subregion_engine is None:
            self._subregion_engine = SubRegionEngine(min_segment_size=200)
        return self._subregion_engine

    @property
    def adjustment_engine(self) -> AdjustmentEngine:
        """Lazy-load the adjustment engine."""
        if self._adjustment_engine is None:
            self._adjustment_engine = AdjustmentEngine()
        return self._adjustment_engine

    @property
    def comparable_engine(self) -> ComparableEngine:
        """Lazy-load the comparable engine."""
        if self._comparable_engine is None:
            self._comparable_engine = ComparableEngine(
                mls_available=self.mls_available,
            )
        return self._comparable_engine

    @property
    def quantile_trainer(self) -> QuantileModelTrainer:
        """Lazy-load the quantile model trainer (used for interval prediction)."""
        if self._quantile_trainer is None:
            self._quantile_trainer = QuantileModelTrainer()
        return self._quantile_trainer

    @property
    def model_trainer(self) -> ModelTrainer:
        """Lazy-load the model trainer (used for model loading)."""
        if self._model_trainer is None:
            self._model_trainer = ModelTrainer(model_dir=str(self.model_dir))
        return self._model_trainer

    # ================================================================
    # MAIN PREDICTION
    # ================================================================

    def predict(
        self,
        pid: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        address: Optional[str] = None,
        property_type: Optional[str] = None,
        override_features: Optional[dict] = None,
        properties_df: Optional[pd.DataFrame] = None,
        boundary_gdf=None,
    ) -> PredictionResult:
        """Generate a full property valuation.

        Orchestrates the complete prediction pipeline:
          1. Resolve property identity (PID lookup or lat/lon)
          2. Determine segment (neighbourhood_code x property_type)
          3. Build feature vector
          4. Apply override features if provided
          5. Select model via segment key + fallback hierarchy
          6. Predict: exp(model.predict(features)) for dollar amount
          7. Compute confidence interval from quantile models
          8. Apply Tier 2 adjustments
          9. Find top 5 comparables
          10. Reconcile ML estimate with comparable range
          11. Compute SHAP values
          12. Assign confidence grade
          13. Identify risk flags
          14. Package into PredictionResult

        Args:
            pid: BC Assessment Property Identifier (PID). Used to look
                up the property in properties_df.
            lat: Latitude in decimal degrees (alternative to PID).
            lon: Longitude in decimal degrees (alternative to PID).
            address: Street address (for display; not used for lookup).
            property_type: Property type override (e.g. 'condo').
            override_features: Dict of feature overrides provided by the
                user (e.g. sqft, bedrooms). These replace computed values.
            properties_df: Enriched property universe DataFrame. Required
                for PID lookup, comparable search, and market context.

        Returns:
            PredictionResult with point estimate, confidence interval,
            comparables, SHAP values, adjustments, and risk flags.
        """
        t0 = time.perf_counter()

        # ------------------------------------------------------------------
        # 1. Resolve property identity
        # ------------------------------------------------------------------
        property_data = self._resolve_property(
            pid=pid, lat=lat, lon=lon, address=address,
            property_type=property_type, properties_df=properties_df,
        )
        resolved_pid = property_data.get("pid", pid or "unknown")

        # Override neighbourhood using City of Vancouver boundaries ONLY for
        # synthetic records (not in BC Assessment) or properties without a
        # neighbourhood_code.  BC Assessment-matched properties already have
        # correct spatial neighbourhood codes from training (assigned by
        # NeighbourhoodAssigner during data pipeline).
        is_synthetic = property_data.get("_synthetic", False)
        has_hood = bool(property_data.get("neighbourhood_code"))
        prop_lat = property_data.get("latitude", lat or 0.0)
        prop_lon = property_data.get("longitude", lon or 0.0)
        if boundary_gdf is not None and prop_lat and prop_lon and (is_synthetic or not has_hood):
            city_code, city_name = self.assign_neighbourhood_from_latlon(
                float(prop_lat), float(prop_lon), boundary_gdf,
            )
            if city_code is not None:
                old_code = property_data.get("neighbourhood_code", "?")
                old_name = _NEIGHBOURHOOD_NAMES.get(str(old_code), str(old_code))
                property_data["neighbourhood_code"] = city_code
                if old_code != city_code:
                    logger.info(
                        "Neighbourhood override (synthetic/unmatched): %s (%s) -> %s (%s) "
                        "via City of Vancouver boundaries",
                        old_code, old_name, city_code, city_name,
                    )

        logger.info(
            "Predicting for PID=%s (lat=%.5f, lon=%.5f, type=%s)",
            resolved_pid,
            property_data.get("latitude", 0.0),
            property_data.get("longitude", 0.0),
            property_data.get("property_type", "unknown"),
        )

        # ------------------------------------------------------------------
        # 2. Determine segment
        # ------------------------------------------------------------------
        segment_key = self.subregion_engine.assign_segment(property_data)
        logger.info("Assigned segment: %s", segment_key)

        # ------------------------------------------------------------------
        # 3. Build feature vector
        # ------------------------------------------------------------------
        features_dict = self.feature_builder.build_features_single(
            property_data, properties_df=properties_df,
        )

        # ------------------------------------------------------------------
        # 4. Apply override features
        # ------------------------------------------------------------------
        if override_features:
            for key, value in override_features.items():
                features_dict[key] = value
            logger.info(
                "Applied %d feature overrides: %s",
                len(override_features),
                list(override_features.keys()),
            )

        # Compute feature completeness for confidence grading
        total_features = len(features_dict)
        populated_features = sum(
            1 for v in features_dict.values()
            if v is not None and (not isinstance(v, float) or not np.isnan(v))
        )
        feature_completeness = (
            populated_features / total_features * 100 if total_features > 0 else 0.0
        )

        # ------------------------------------------------------------------
        # 5. Select model using segment key + fallback hierarchy
        # ------------------------------------------------------------------
        is_synthetic = property_data.get("_synthetic", False)
        if is_synthetic:
            # Synthetic record (new build not in BC Assessment).
            # Skip the ML model — features are mostly NaN so the
            # prediction would be meaningless.  Rely on comparables.
            model = None
            actual_segment = segment_key + " (synthetic)"
            logger.info(
                "Synthetic property — skipping ML model, will use "
                "comparable-based valuation only",
            )
        else:
            model, actual_segment = self._select_model(segment_key)

        # ------------------------------------------------------------------
        # 6-7. Predict point estimate and confidence interval
        # ------------------------------------------------------------------
        # Try market price model first (trained on actual sold prices)
        market_result = self.predict_market_price(property_data, features_dict)
        market_estimate = None
        market_model_info = None
        if market_result is not None:
            market_estimate, market_model_info = market_result
            logger.info(
                "Market price estimate: $%,.0f (%s)",
                market_estimate, market_model_info,
            )

        if model is not None:
            # Build a single-row DataFrame from features dict
            features_df = pd.DataFrame([features_dict])

            # Align columns with what the model expects
            model_feature_names = model.feature_name()
            for feat in model_feature_names:
                if feat not in features_df.columns:
                    features_df[feat] = np.nan
            features_df = features_df[model_feature_names]

            # Ensure categorical columns match training dtype exactly.
            # LightGBM stores the category lists used during training in
            # model.pandas_categorical. We must reconstruct the same
            # Categorical dtype for prediction.
            pandas_cats = model.pandas_categorical
            if pandas_cats:
                cat_col_idx = 0
                for col in features_df.columns:
                    if pd.api.types.is_string_dtype(features_df[col]) or pd.api.types.is_categorical_dtype(features_df[col]):
                        if cat_col_idx < len(pandas_cats):
                            cat_type = pd.CategoricalDtype(
                                categories=pandas_cats[cat_col_idx],
                            )
                            features_df[col] = features_df[col].astype(cat_type)
                            cat_col_idx += 1
                        else:
                            features_df[col] = features_df[col].astype("category")

            # Convert to numpy to avoid categorical/string issues with
            # single-row prediction (e.g. tod_tier="none", building_age_bucket).
            for col in features_df.columns:
                col_dtype = features_df[col].dtype
                if isinstance(col_dtype, pd.CategoricalDtype):
                    features_df[col] = features_df[col].cat.codes.astype(float)
                elif col_dtype == object or pd.api.types.is_string_dtype(col_dtype):
                    features_df[col] = pd.to_numeric(features_df[col], errors="coerce")
                elif col_dtype == bool or col_dtype == "boolean":
                    features_df[col] = features_df[col].astype(float)

            # Point estimate: exp(model.predict(features))
            log_pred = model.predict(features_df.values)
            ml_estimate = float(np.expm1(log_pred[0]))

            # Confidence interval from quantile models
            ci_lower, ci_upper = self._compute_confidence_interval(
                actual_segment, features_df,
            )
        else:
            # No model available -- fall back to comparable median
            logger.warning(
                "No model found for segment '%s' (or fallbacks); "
                "using comparable median only",
                segment_key,
            )
            ml_estimate = 0.0
            ci_lower, ci_upper = 0.0, 0.0

        # ------------------------------------------------------------------
        # 8. Apply Tier 2 adjustments
        # ------------------------------------------------------------------
        if ml_estimate > 0:
            adj_result: AdjustmentResult = self.adjustment_engine.apply_all_adjustments(
                ml_estimate=ml_estimate,
                property_data=property_data,
            )
            adjusted_estimate = adj_result.adjusted_value
            adjustments = adj_result.adjustments

            # Scale CI by the same adjustment ratio
            if ml_estimate > 0:
                adj_ratio = adjusted_estimate / ml_estimate
                ci_lower *= adj_ratio
                ci_upper *= adj_ratio
        else:
            adjusted_estimate = 0.0
            adjustments = []

        # ------------------------------------------------------------------
        # 9. Find top 5 comparables
        # ------------------------------------------------------------------
        comparables: list[ComparableProperty] = []
        if properties_df is not None and not properties_df.empty:
            comparables = self.comparable_engine.find_comparables(
                subject=property_data,
                candidates_df=properties_df,
                k=5,
            )

        # ------------------------------------------------------------------
        # 10. Reconcile ML estimate with comparable range
        # ------------------------------------------------------------------
        if comparables and adjusted_estimate > 0:
            comp_range = self.comparable_engine.compute_comparable_range(
                subject=property_data,
                comparables=comparables,
            )
            final_estimate, divergence_pct, reconciliation_note = (
                self.comparable_engine.reconcile_with_ml(
                    ml_estimate=adjusted_estimate,
                    comparable_range=comp_range,
                    comparable_count=len(comparables),
                )
            )

            # Also adjust CI using reconciliation ratio
            if adjusted_estimate > 0:
                recon_ratio = final_estimate / adjusted_estimate
                ci_lower *= recon_ratio
                ci_upper *= recon_ratio
        elif comparables and adjusted_estimate <= 0:
            # Use comparable median as the estimate when no model is available
            comp_range = self.comparable_engine.compute_comparable_range(
                subject=property_data,
                comparables=comparables,
            )
            final_estimate = comp_range[1]  # median
            ci_lower = comp_range[0]
            ci_upper = comp_range[2]
        else:
            final_estimate = adjusted_estimate

        # Ensure CI is sensible and brackets the point estimate
        ci_width = abs(ci_upper - ci_lower)
        needs_reset = (
            ci_lower <= 0
            or ci_upper <= 0
            or ci_lower > final_estimate
            or ci_upper < final_estimate
            or (final_estimate > 0 and ci_width > final_estimate * 5)
        )
        if needs_reset:
            # Fallback: ±15% of the point estimate
            ci_lower = final_estimate * 0.85
            ci_upper = final_estimate * 1.15

        # ------------------------------------------------------------------
        # 11. Compute SHAP values
        # ------------------------------------------------------------------
        shap_values: dict[str, float] = {}
        if model is not None:
            shap_values = self._compute_shap_for_prediction(model, features_dict)

        # ------------------------------------------------------------------
        # 12. Assign confidence grade
        # ------------------------------------------------------------------
        interval_width_pct = (
            (ci_upper - ci_lower) / final_estimate * 100
            if final_estimate > 0
            else 100.0
        )
        confidence_grade = self._assign_confidence_grade(
            feature_completeness=feature_completeness,
            comparable_count=len(comparables),
            interval_width_pct=interval_width_pct,
        )

        # ------------------------------------------------------------------
        # 13. Identify risk flags
        # ------------------------------------------------------------------
        risk_flags = self._identify_risk_flags(property_data)

        # Add data quality risk if feature completeness is low
        if feature_completeness < 50:
            risk_flags.append({
                "category": "data_quality",
                "severity": "medium",
                "description": (
                    f"Feature completeness is {feature_completeness:.0f}%% "
                    f"({populated_features}/{total_features} features populated). "
                    f"Prediction reliability may be reduced."
                ),
            })

        # ------------------------------------------------------------------
        # 14. Build market context
        # ------------------------------------------------------------------
        market_context = self._build_market_context(property_data, properties_df)

        # ------------------------------------------------------------------
        # Package result
        # ------------------------------------------------------------------
        elapsed = time.perf_counter() - t0

        result = PredictionResult(
            pid=str(resolved_pid),
            point_estimate=round(final_estimate, 2),
            confidence_interval=(round(ci_lower, 2), round(ci_upper, 2)),
            confidence_grade=confidence_grade,
            comparables=comparables,
            shap_values=shap_values,
            adjustments=adjustments,
            market_context=market_context,
            risk_flags=risk_flags,
            model_segment=actual_segment,
            model_version=_MODEL_VERSION,
            market_estimate=round(market_estimate, 2) if market_estimate else None,
            market_model_info=market_model_info,
        )

        logger.info(
            "Prediction complete for PID=%s: $%,.0f [%s] "
            "(CI: $%,.0f - $%,.0f, %d comps, %d adjustments) in %.2fs",
            resolved_pid,
            final_estimate,
            confidence_grade,
            ci_lower,
            ci_upper,
            len(comparables),
            len(adjustments),
            elapsed,
        )

        return result

    # ================================================================
    # BATCH PREDICTION
    # ================================================================

    def predict_batch(
        self,
        pids: list[str],
        properties_df: pd.DataFrame,
    ) -> list[PredictionResult]:
        """Generate predictions for a batch of properties.

        Iterates over all PIDs, calling predict() for each. Logs
        progress every 100 properties.

        Args:
            pids: List of PID strings to predict.
            properties_df: Enriched property universe DataFrame.

        Returns:
            List of PredictionResult objects, one per PID. Failed
            predictions are logged and skipped.
        """
        results: list[PredictionResult] = []
        n_total = len(pids)
        n_failed = 0

        logger.info("Starting batch prediction for %d properties", n_total)
        t0 = time.perf_counter()

        for i, pid in enumerate(pids):
            if (i + 1) % 100 == 0 or i == 0:
                logger.info(
                    "Batch progress: %d/%d (%.1f%%)",
                    i + 1,
                    n_total,
                    (i + 1) / n_total * 100,
                )

            try:
                result = self.predict(
                    pid=pid,
                    properties_df=properties_df,
                )
                results.append(result)
            except Exception as exc:
                logger.error(
                    "Prediction failed for PID=%s: %s", pid, exc, exc_info=True,
                )
                n_failed += 1

        elapsed = time.perf_counter() - t0
        logger.info(
            "Batch prediction complete: %d succeeded, %d failed out of %d "
            "in %.1fs (%.1f predictions/sec)",
            len(results),
            n_failed,
            n_total,
            elapsed,
            len(results) / max(elapsed, 0.001),
        )

        return results

    # ================================================================
    # MODEL SELECTION
    # ================================================================

    # Cross-type fallback order: townhome ↔ condo are similar (both strata),
    # detached is a last resort.
    _CROSS_TYPE_FALLBACKS: dict[str, list[str]] = {
        "townhome": ["condo", "detached"],
        "condo": ["townhome", "detached"],
        "detached": ["condo", "townhome"],
    }

    def _scan_available_models(self) -> set[str]:
        """Scan the models directory and return available segment keys.

        Caches the result so the directory is only scanned once.
        """
        if not hasattr(self, "_available_segments"):
            self._available_segments: set[str] = set()
            for f in self.model_dir.glob("*.pkl"):
                name = f.stem
                # Skip quantile model variants
                if "_q0" in name:
                    continue
                # Convert filename back to segment key: 6-detached -> 6__detached
                segment = name.replace("-", "__", 1)
                self._available_segments.add(segment)
            logger.info(
                "Scanned %d available model segments", len(self._available_segments),
            )
        return self._available_segments

    def _load_market_model(self, property_type: str) -> tuple[Any, dict] | None:
        """Try to load a market price model for the given property type.

        Market models are trained on actual MLS sold prices and stored
        in models/market/. Returns (model, metadata) or None.
        """
        if property_type in self._market_model_cache:
            return self._market_model_cache[property_type]

        model_path = self.market_model_dir / f"market_{property_type}.pkl"
        meta_path = self.market_model_dir / f"market_{property_type}_metadata.json"

        if not model_path.exists():
            self._market_model_cache[property_type] = None
            return None

        import joblib
        import json

        model = joblib.load(model_path)
        metadata = {}
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)

        self._market_model_cache[property_type] = (model, metadata)
        logger.info(
            "Loaded market model for '%s' (MAPE=%.2f%%, %d samples)",
            property_type,
            metadata.get("cv_mape", 0),
            metadata.get("n_samples", 0),
        )
        return (model, metadata)

    def predict_market_price(
        self,
        property_data: dict,
        features_dict: dict,
    ) -> tuple[float, str] | None:
        """Predict market price using neighbourhood SAR × assessed value.

        Strategy:
        1. Look up neighbourhood + property_type median SAR from actual sales
        2. If ML model available, blend it in (30% weight) for property-level
           adjustment — but the neighbourhood SAR is the anchor (70% weight)
        3. Multiply assessed_value × blended_SAR

        Returns (estimate, model_info_string) or None if unavailable.
        """
        ptype = property_data.get("property_type", "")
        assessed_value = property_data.get("total_assessed_value", 0)
        hood = str(property_data.get("neighbourhood_code", ""))

        if not assessed_value or assessed_value <= 0:
            logger.warning("No assessed value for market prediction")
            return None

        # Load neighbourhood SAR lookup (cached)
        hood_sar, hood_n = self._get_neighbourhood_sar(hood, ptype)
        citywide_sar, citywide_n = self._get_neighbourhood_sar("_all", ptype)

        if hood_sar is None and citywide_sar is None:
            return None

        # Determine neighbourhood SAR with fallback to city-wide
        if hood_sar is not None and hood_n >= 5:
            local_sar = hood_sar
            local_n = hood_n
            sar_source = f"hood_{hood}"
        elif hood_sar is not None and hood_n >= 2:
            # Few local sales — blend with city-wide
            local_sar = 0.5 * hood_sar + 0.5 * (citywide_sar or 1.0)
            local_n = hood_n
            sar_source = f"hood_{hood}_blended"
        else:
            local_sar = citywide_sar or 1.0
            local_n = citywide_n or 0
            sar_source = "citywide"

        # Try ML model for property-level adjustment
        ml_sar = None
        result = self._load_market_model(ptype)
        if result is not None:
            model, metadata = result

            if metadata.get("fallback"):
                # No trained model — just use neighbourhood SAR
                ml_sar = None
            else:
                features_df = pd.DataFrame([{**property_data, **features_dict}])
                model_feature_names = model.feature_name()
                for feat in model_feature_names:
                    if feat not in features_df.columns:
                        features_df[feat] = np.nan
                features_df = features_df[model_feature_names]

                for col in features_df.columns:
                    if not pd.api.types.is_numeric_dtype(features_df[col]):
                        features_df[col] = pd.to_numeric(
                            features_df[col], errors="coerce",
                        )
                features_arr = features_df.to_numpy(
                    dtype=np.float64, na_value=np.nan,
                )

                try:
                    raw_ml_sar = float(model.predict(features_arr)[0])
                    # Clamp to reasonable range
                    ml_sar = max(0.6, min(1.8, raw_ml_sar))
                except Exception as e:
                    logger.warning(
                        "ML SAR prediction failed for %s: %s", ptype, e,
                    )

        # Blend: neighbourhood SAR is anchor (70%), ML is adjustment (30%)
        if ml_sar is not None:
            blended_sar = 0.70 * local_sar + 0.30 * ml_sar
            blend_note = f"blend(hood={local_sar:.3f}×70% + ml={ml_sar:.3f}×30%)"
        else:
            blended_sar = local_sar
            blend_note = f"{sar_source}={local_sar:.3f}"

        market_estimate = float(assessed_value * blended_sar)

        n_samples = local_n
        if result and not metadata.get("fallback"):
            n_samples = metadata.get("n_samples", local_n)

        info = (
            f"market_{ptype} (SAR={blended_sar:.3f}, {blend_note}, "
            f"n_local={local_n}, n_model={n_samples})"
        )
        logger.info(
            "SAR prediction: assessed=$%,.0f × SAR=%.3f = market=$%,.0f [%s]",
            assessed_value, blended_sar, market_estimate, blend_note,
        )
        return market_estimate, info

    def _get_neighbourhood_sar(
        self, hood: str, ptype: str,
    ) -> tuple[float | None, int]:
        """Get median SAR for a neighbourhood + property type from sales data.

        Returns (median_sar, n_sales) or (None, 0) if no data.
        Caches results so DB is only queried once per session.
        """
        cache_key = f"{hood}_{ptype}"
        if not hasattr(self, "_sar_cache"):
            self._sar_cache: dict[str, tuple[float | None, int]] = {}
            self._sar_cache_loaded = False

        if cache_key in self._sar_cache:
            return self._sar_cache[cache_key]

        # Build the full SAR cache on first call
        if not self._sar_cache_loaded:
            try:
                from src.pipeline.sold_price_enrichment import (
                    build_market_training_data,
                )

                mkt = build_market_training_data()
                if not mkt.empty:
                    mkt = mkt.copy()
                    mkt["sar"] = mkt["sold_price"] / mkt["total_assessed_value"]
                    # Remove extreme outliers
                    mkt = mkt[(mkt["sar"] >= 0.5) & (mkt["sar"] <= 2.0)]

                    # Remove IQR outliers within each neighbourhood+type
                    keep_mask = pd.Series(True, index=mkt.index)
                    for (h, p), grp in mkt.groupby(
                        ["neighbourhood_code", "property_type"]
                    ):
                        if len(grp) >= 5:
                            q1 = grp["sar"].quantile(0.25)
                            q3 = grp["sar"].quantile(0.75)
                            iqr = q3 - q1
                            lo = q1 - 1.5 * iqr
                            hi = q3 + 1.5 * iqr
                            outlier_idx = grp[
                                (grp["sar"] < lo) | (grp["sar"] > hi)
                            ].index
                            keep_mask.loc[outlier_idx] = False

                    n_removed = (~keep_mask).sum()
                    mkt = mkt[keep_mask]
                    logger.info(
                        "SAR cache: removed %d IQR outliers, %d clean sales remain",
                        n_removed, len(mkt),
                    )

                    # By neighbourhood + type
                    for (h, p), grp in mkt.groupby(
                        ["neighbourhood_code", "property_type"]
                    ):
                        key = f"{h}_{p}"
                        self._sar_cache[key] = (
                            float(grp["sar"].median()),
                            len(grp),
                        )

                    # City-wide by type
                    for p, grp in mkt.groupby("property_type"):
                        key = f"_all_{p}"
                        self._sar_cache[key] = (
                            float(grp["sar"].median()),
                            len(grp),
                        )

                    logger.info(
                        "Built SAR cache: %d neighbourhood-type combos from %d sales",
                        len(self._sar_cache),
                        len(mkt),
                    )
            except Exception as e:
                logger.warning("Failed to build SAR cache: %s", e)

            self._sar_cache_loaded = True

        return self._sar_cache.get(cache_key, (None, 0))

    def _select_model(self, segment_key: str) -> tuple[Any, str]:
        """Select the best available model for a segment.

        Fallback priority:
          1. Standard chain: exact segment -> citywide by type -> citywide all
          2. Same type, any neighbourhood (prefer numerically close codes)
          3. Cross-type same neighbourhood (condo for townhome, etc.)
          4. Cross-type any neighbourhood

        Args:
            segment_key: Canonical segment key.

        Returns:
            Tuple of (model, actual_segment_used). If no model is found
            at any level, returns (None, 'none').
        """
        # 1. Standard fallback chain for the original type
        result = self._try_fallback_chain(segment_key, segment_key)
        if result is not None:
            return result

        parts = segment_key.split("__")
        if len(parts) != 2:
            logger.warning(
                "No model found at any fallback level for segment '%s'",
                segment_key,
            )
            return None, "none"

        area_part, type_part = parts
        available = self._scan_available_models()

        # 2. Same type, any neighbourhood — find closest available
        same_type_segments = sorted(
            (s for s in available if s.endswith(f"__{type_part}")),
            key=lambda s: self._neighbourhood_distance(area_part, s.split("__")[0]),
        )
        for alt_seg in same_type_segments:
            result = self._try_fallback_chain(alt_seg, segment_key)
            if result is not None:
                logger.info(
                    "Same-type neighbourhood fallback: '%s' -> '%s'",
                    segment_key, result[1],
                )
                return result

        # 3. Cross-type: same neighbourhood first, then any neighbourhood
        alt_types = self._CROSS_TYPE_FALLBACKS.get(type_part, [])
        for alt_type in alt_types:
            # Try same neighbourhood
            alt_key = f"{area_part}__{alt_type}"
            result = self._try_fallback_chain(alt_key, segment_key)
            if result is not None:
                logger.info(
                    "Cross-type fallback: '%s' -> '%s' (used '%s')",
                    segment_key, alt_type, result[1],
                )
                return result

            # Try other neighbourhoods for this alt type
            alt_segments = sorted(
                (s for s in available if s.endswith(f"__{alt_type}")),
                key=lambda s: self._neighbourhood_distance(
                    area_part, s.split("__")[0],
                ),
            )
            for alt_seg in alt_segments:
                result = self._try_fallback_chain(alt_seg, segment_key)
                if result is not None:
                    logger.info(
                        "Cross-type neighbourhood fallback: '%s' -> '%s'",
                        segment_key, result[1],
                    )
                    return result

        logger.warning(
            "No model found at any fallback level for segment '%s'",
            segment_key,
        )
        return None, "none"

    @staticmethod
    def _neighbourhood_distance(code_a: str, code_b: str) -> float:
        """Rough proximity score between two neighbourhood codes.

        Uses numeric distance as a simple heuristic — adjacent codes
        tend to be geographically close in BC Assessment's numbering.
        """
        try:
            return abs(int(code_a) - int(code_b))
        except (ValueError, TypeError):
            return 999

    def _try_fallback_chain(
        self, start_key: str, original_key: str,
    ) -> tuple[Any, str] | None:
        """Walk the standard fallback hierarchy for a single segment key.

        Returns (model, actual_segment) on success, or None if the
        entire chain is exhausted without finding a model.
        """
        current_key = start_key

        while True:
            # Check cache first
            if current_key in self._model_cache:
                model, _ = self._model_cache[current_key]
                logger.info(
                    "Model cache hit for segment '%s' (requested '%s')",
                    current_key,
                    original_key,
                )
                return model, current_key

            # Try loading from disk
            try:
                model, metadata = self.model_trainer.load_model(current_key)
                self._model_cache[current_key] = (model, metadata)
                logger.info(
                    "Loaded model for segment '%s' from disk (requested '%s')",
                    current_key,
                    original_key,
                )
                return model, current_key
            except FileNotFoundError:
                pass

            # Walk up the fallback hierarchy
            fallback_key = SubRegionEngine.get_fallback_segment(current_key)
            if fallback_key == current_key:
                # Reached the top of the hierarchy with no model
                return None

            logger.info(
                "No model for segment '%s'; falling back to '%s'",
                current_key,
                fallback_key,
            )
            current_key = fallback_key

    # ================================================================
    # CONFIDENCE INTERVAL
    # ================================================================

    def _compute_confidence_interval(
        self,
        segment_key: str,
        features_df: pd.DataFrame,
        confidence_level: float = 0.80,
    ) -> tuple[float, float]:
        """Compute prediction interval from quantile models.

        Attempts to load quantile models for the segment. If unavailable,
        falls back to a heuristic interval based on the point estimate.

        Args:
            segment_key: Segment key for which quantile models may exist.
            features_df: Single-row feature DataFrame (model-aligned).
            confidence_level: Desired confidence level (default 0.80).

        Returns:
            Tuple of (lower_bound, upper_bound) in dollar values.
        """
        # Try loading quantile models
        if segment_key not in self._quantile_cache:
            try:
                q_models = self.quantile_trainer.load_quantile_models(
                    segment_key, str(self.model_dir),
                )
                self._quantile_cache[segment_key] = q_models
            except FileNotFoundError:
                logger.info(
                    "No quantile models for segment '%s'; using heuristic CI",
                    segment_key,
                )
                return 0.0, 0.0  # Caller will apply fallback

        if segment_key in self._quantile_cache:
            q_models = self._quantile_cache[segment_key]
            try:
                lower_arr, upper_arr = self.quantile_trainer.predict_intervals(
                    q_models, features_df, confidence_level=confidence_level,
                )
                return float(lower_arr[0]), float(upper_arr[0])
            except (ValueError, KeyError) as exc:
                if "categorical_feature" in str(exc):
                    # Same categorical mismatch — convert to numeric
                    logger.warning(
                        "Quantile categorical mismatch for '%s' — "
                        "falling back to numeric encoding",
                        segment_key,
                    )
                    numeric_df = features_df.copy()
                    for col in numeric_df.columns:
                        col_dtype = numeric_df[col].dtype
                        if isinstance(col_dtype, pd.CategoricalDtype):
                            numeric_df[col] = numeric_df[col].cat.codes.astype(float)
                        elif col_dtype == object or pd.api.types.is_string_dtype(col_dtype):
                            numeric_df[col] = pd.Categorical(numeric_df[col]).codes.astype(float)
                        elif col_dtype == bool or col_dtype == "boolean":
                            numeric_df[col] = numeric_df[col].astype(float)
                    try:
                        lower_arr, upper_arr = self.quantile_trainer.predict_intervals(
                            q_models, pd.DataFrame(numeric_df.values, columns=numeric_df.columns),
                            confidence_level=confidence_level,
                        )
                        return float(lower_arr[0]), float(upper_arr[0])
                    except Exception as exc2:
                        logger.warning(
                            "Quantile prediction still failed for '%s': %s",
                            segment_key, exc2,
                        )
                else:
                    logger.warning(
                        "Quantile prediction failed for '%s': %s", segment_key, exc,
                    )

        return 0.0, 0.0

    # ================================================================
    # SHAP EXPLANATION
    # ================================================================

    def _compute_shap_for_prediction(
        self,
        model: Any,
        features_dict: dict,
    ) -> dict[str, float]:
        """Compute SHAP values for a single prediction.

        Uses SHAP's TreeExplainer to attribute the prediction to
        individual features. Returns the top 10 features by absolute
        SHAP contribution.

        Args:
            model: Trained LightGBM Booster.
            features_dict: Dict of feature name -> value for this property.

        Returns:
            Dict mapping feature_name to SHAP value, sorted by absolute
            value descending. Limited to top 10 features.
        """
        try:
            model_features = model.feature_name()
            features_row = {}
            for feat in model_features:
                features_row[feat] = features_dict.get(feat, np.nan)

            features_df = pd.DataFrame([features_row])

            # Convert categoricals to numeric codes to avoid LightGBM
            # categorical mismatch errors during SHAP computation
            for col in features_df.columns:
                col_dtype = features_df[col].dtype
                if isinstance(col_dtype, pd.CategoricalDtype):
                    features_df[col] = features_df[col].cat.codes.astype(float)
                elif col_dtype == object or pd.api.types.is_string_dtype(col_dtype):
                    features_df[col] = pd.Categorical(features_df[col]).codes.astype(float)
                elif col_dtype == bool or col_dtype == "boolean":
                    features_df[col] = features_df[col].astype(float)

            # Use numpy array to fully bypass pandas categorical validation
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features_df.values)

            # shap_values is (1, n_features) array
            shap_dict = dict(zip(model_features, shap_values[0]))

            # Sort by absolute value and take top 10
            sorted_shap = sorted(
                shap_dict.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:10]

            return {k: round(float(v), 6) for k, v in sorted_shap}

        except Exception as exc:
            logger.warning("SHAP computation failed: %s", exc)
            return {}

    # ================================================================
    # RISK FLAGS
    # ================================================================

    def _identify_risk_flags(self, property_data: dict) -> list[dict]:
        """Identify risk conditions for a property.

        Checks for conditions that may affect value reliability or
        represent material risks to the property holder.

        Checks:
          - Leasehold with < 40 years remaining (high)
          - In floodplain (medium)
          - Near contaminated site < 500m (medium)
          - Pre-2000 wood-frame without rainscreen (medium)
          - In ALR (high)

        Args:
            property_data: Dict of property attributes.

        Returns:
            List of risk flag dicts with category, severity, description.
        """
        flags: list[dict] = []

        # --- Leasehold with short remaining term ---
        lease_remaining = property_data.get("lease_remaining_years")
        lease_type = property_data.get("lease_type", "freehold")
        if (
            lease_remaining is not None
            and lease_type != "freehold"
            and lease_remaining < 40
        ):
            flags.append({
                "category": "leasehold",
                "severity": "high",
                "description": (
                    f"Leasehold property with only {lease_remaining:.0f} years "
                    f"remaining. Lease type: {lease_type}. Properties with "
                    f"< 40 years remaining face significant financing challenges "
                    f"and accelerating value discount."
                ),
            })

        # --- Floodplain ---
        in_floodplain = property_data.get("in_floodplain", False)
        if in_floodplain:
            flags.append({
                "category": "environmental",
                "severity": "medium",
                "description": (
                    "Property is located in a designated floodplain area. "
                    "May face insurance surcharges and climate-related risk."
                ),
            })

        # --- Near contaminated site ---
        dist_contaminated = property_data.get("dist_nearest_contaminated_m")
        contaminated_count = property_data.get("contaminated_sites_500m", 0)
        if dist_contaminated is not None and dist_contaminated < 500:
            flags.append({
                "category": "environmental",
                "severity": "medium",
                "description": (
                    f"Property is {dist_contaminated:.0f}m from a contaminated "
                    f"site ({contaminated_count} contaminated sites within 500m). "
                    f"May affect value and require environmental review."
                ),
            })

        # --- Pre-2000 wood-frame without rainscreen ---
        year_built = property_data.get("year_built")
        construction_type = property_data.get("construction_type", "")
        rainscreen_status = property_data.get("rainscreen_status")
        if (
            year_built is not None
            and 1982 <= year_built <= 2000
            and "wood" in str(construction_type).lower()
            and rainscreen_status not in ("completed", "not_required")
        ):
            flags.append({
                "category": "building_envelope",
                "severity": "medium",
                "description": (
                    f"Wood-frame building constructed in {year_built} "
                    f"(leaky condo era 1982-2000) without confirmed "
                    f"rainscreen remediation. May face significant "
                    f"envelope repair costs."
                ),
            })

        # --- Agricultural Land Reserve ---
        in_alr = property_data.get("in_alr", False)
        if in_alr:
            flags.append({
                "category": "land_use",
                "severity": "high",
                "description": (
                    "Property is within the Agricultural Land Reserve (ALR). "
                    "Development is heavily restricted. Assessed value may "
                    "not reflect fee-simple development potential."
                ),
            })

        return flags

    # ================================================================
    # CONFIDENCE GRADING
    # ================================================================

    def _assign_confidence_grade(
        self,
        feature_completeness: float,
        comparable_count: int,
        interval_width_pct: float,
    ) -> str:
        """Assign a letter grade reflecting prediction confidence.

        Grade criteria:
          - A: completeness > 80%, 5+ comps, interval < 14%
          - B: completeness > 60%, 3+ comps, interval < 24%
          - C: everything else

        Args:
            feature_completeness: Percentage of non-null features (0-100).
            comparable_count: Number of comparables found.
            interval_width_pct: Confidence interval width as percentage
                of the point estimate.

        Returns:
            Letter grade string: 'A', 'B', or 'C'.
        """
        if (
            feature_completeness > 80
            and comparable_count >= 5
            and interval_width_pct < 14
        ):
            return "A"

        if (
            feature_completeness > 60
            and comparable_count >= 3
            and interval_width_pct < 24
        ):
            return "B"

        return "C"

    # ================================================================
    # PRIVATE HELPERS
    # ================================================================

    def _resolve_property(
        self,
        pid: Optional[str],
        lat: Optional[float],
        lon: Optional[float],
        address: Optional[str],
        property_type: Optional[str],
        properties_df: Optional[pd.DataFrame],
    ) -> dict:
        """Resolve property identity from PID, lat/lon, or address.

        Looks up the property in properties_df by PID if available.
        Otherwise constructs a minimal property dict from the provided
        lat/lon and property_type.

        Args:
            pid: PID string for lookup.
            lat: Latitude in decimal degrees.
            lon: Longitude in decimal degrees.
            address: Street address (for display).
            property_type: Property type override.
            properties_df: Enriched property universe.

        Returns:
            Dict of property attributes.
        """
        # Try PID lookup in the property universe
        if pid is not None and properties_df is not None:
            match = properties_df[properties_df["pid"].astype(str) == str(pid)]
            if not match.empty:
                property_data = match.iloc[0].to_dict()
                # Apply overrides
                if property_type:
                    property_data["property_type"] = property_type
                if lat is not None:
                    property_data["latitude"] = lat
                if lon is not None:
                    property_data["longitude"] = lon
                if address:
                    property_data["address"] = address
                logger.info("Resolved PID=%s from property universe", pid)
                return property_data

            logger.warning(
                "PID=%s not found in property universe (%d properties)",
                pid,
                len(properties_df),
            )

        # Try address-based matching before falling back to empty data.
        # Parse the address to extract civic number and street name, then
        # find the closest property on that street in our database.
        if (
            address
            and properties_df is not None
            and not properties_df.empty
            and "street_name" in properties_df.columns
        ):
            import re

            addr_upper = address.upper().strip()
            # Extract civic number, optional unit, and street.
            # Handles multiple formats:
            #   "1858 West 5th Avenue #302, Vancouver, BC"  (civic street #unit)
            #   "6149 Fremlin Street, Vancouver, BC"         (civic street)
            #   "608 - 2228 W Broadway, Vancouver, BC"       (unit - civic street)
            #   "PH5 - 2088 W 11th Avenue, Vancouver, BC"   (unit - civic street)

            # Format 1: "UNIT - CIVIC STREET" (common on listing sites)
            m_unit_first = re.match(
                r"^(?:PH)?(\d+)\s*[-–]\s*(\d+)\s+(.+?)(?:,|\s+VANCOUVER)",
                addr_upper,
            )
            # Format 2: "CIVIC STREET #UNIT" (standard)
            m = re.match(
                r"^(\d+)\s+(.+?)(?:\s*#(\d+))?(?:,|\s+VANCOUVER)",
                addr_upper,
            )

            if m_unit_first:
                target_unit = int(m_unit_first.group(1))
                target_civic = int(m_unit_first.group(2))
                raw_street = m_unit_first.group(3).strip()
            elif m:
                target_civic = int(m.group(1))
                raw_street = m.group(2).strip()
                target_unit = int(m.group(3)) if m.group(3) else None
            else:
                m = None  # will skip the matching block

            if m_unit_first or m:
                # Normalize common suffixes and cardinal directions
                _SUFFIX_MAP = {
                    "STREET": "ST", "AVENUE": "AVE", "DRIVE": "DR",
                    "ROAD": "RD", "PLACE": "PL", "CRESCENT": "CRES",
                    "BOULEVARD": "BLVD", "COURT": "CT", "WAY": "WAY",
                    "LANE": "LANE", "TERRACE": "TERR", "CIRCLE": "CIR",
                    # Cardinal directions (Google → BC Assessment)
                    "WEST": "W", "EAST": "E", "NORTH": "N", "SOUTH": "S",
                }
                raw_street = re.sub(
                    r"\b(" + "|".join(_SUFFIX_MAP.keys()) + r")\b",
                    lambda x: _SUFFIX_MAP.get(x.group(0), x.group(0)),
                    raw_street,
                )
                # Split into words and try matching street_name
                street_words = raw_street.split()
                street_col = properties_df["street_name"].fillna("")
                street_mask = pd.Series(True, index=properties_df.index)
                for word in street_words:
                    pattern = r"(?:^|[\s\-])" + re.escape(word)
                    street_mask = street_mask & street_col.str.contains(
                        pattern, na=False, regex=True,
                    )

                street_matches = properties_df[street_mask]
                if not street_matches.empty:
                    # ---------------------------------------------------------
                    # Strata unit matching: address has "#302" → match via
                    # to_civic_number (street address) + from_civic_number (unit)
                    # ---------------------------------------------------------
                    if target_unit is not None and "to_civic_number" in street_matches.columns:
                        to_c = street_matches["to_civic_number"].fillna(0).astype(int)
                        from_c = street_matches["from_civic_number"].fillna(0).astype(int)
                        unit_mask = (to_c == target_civic) & (from_c == target_unit)
                        if unit_mask.any():
                            unit_matches = street_matches[unit_mask]
                            if len(unit_matches) > 1 and "tax_assessment_year" in unit_matches.columns:
                                nearest_idx = unit_matches["tax_assessment_year"].fillna(0).idxmax()
                            else:
                                nearest_idx = unit_matches.index[0]
                            property_data = street_matches.loc[nearest_idx].to_dict()
                            if property_type:
                                property_data["property_type"] = property_type
                            if lat is not None:
                                property_data["latitude"] = lat
                            if lon is not None:
                                property_data["longitude"] = lon
                            property_data["address"] = address
                            logger.info(
                                "Resolved unit address '%s' to PID=%s",
                                address, property_data.get("pid"),
                            )
                            return property_data

                    # ---------------------------------------------------------
                    # If a unit number was specified but no strata match was
                    # found, the property likely isn't in BC Assessment data
                    # yet (new build).  Do NOT fall back to a nearby detached
                    # house — instead skip to the lat/lon stub with condo type.
                    # ---------------------------------------------------------
                    if target_unit is not None:
                        logger.info(
                            "Unit #%d at %d %s not found in data — "
                            "new build; using synthetic condo record",
                            target_unit, target_civic, raw_street,
                        )
                        inferred_type = property_type or "condo"
                        # Build a synthetic property record.  Use the
                        # condo property type (unit number implies strata)
                        # and pull neighbourhood + median value from nearby
                        # condos for comparable matching context.
                        synth: dict = {
                            "pid": "unknown",
                            "latitude": lat or 0.0,
                            "longitude": lon or 0.0,
                            "address": address or "",
                            "property_type": inferred_type,
                            "_synthetic": True,
                        }
                        # Borrow neighbourhood from nearest street match
                        hood_code = None
                        if not street_matches.empty:
                            ref = street_matches.iloc[0]
                            for field in [
                                "neighbourhood_code", "zoning_district",
                            ]:
                                if field in ref.index and pd.notna(ref.get(field)):
                                    synth[field] = ref[field]
                            hood_code = ref.get("neighbourhood_code")

                        # Populate total_assessed_value with the median of
                        # same-type same-neighbourhood properties so the
                        # similarity scorer can find price-appropriate comps.
                        if hood_code is not None and properties_df is not None:
                            peers = properties_df[
                                (properties_df["neighbourhood_code"].astype(str) == str(hood_code))
                                & (properties_df["property_type"].astype(str) == str(inferred_type))
                            ]
                            if not peers.empty and "total_assessed_value" in peers.columns:
                                median_val = peers["total_assessed_value"].median()
                                synth["total_assessed_value"] = median_val
                                synth["current_land_value"] = median_val * 0.4
                                synth["current_improvement_value"] = median_val * 0.6
                                logger.info(
                                    "Synthetic record: using median assessed value "
                                    "$%,.0f from %d %s peers in neighbourhood %s",
                                    median_val, len(peers), inferred_type, hood_code,
                                )
                        return synth
                    else:
                        # ---------------------------------------------------------
                        # Standard matching (no unit number): match by
                        # to_civic_number first, then civic_number fallback
                        # ---------------------------------------------------------
                        to_c_col = street_matches.get(
                            "to_civic_number",
                            pd.Series(0, index=street_matches.index),
                        ).fillna(0).astype(int)
                        to_match = street_matches[to_c_col == target_civic]
                        if not to_match.empty:
                            if len(to_match) > 1 and "tax_assessment_year" in to_match.columns:
                                nearest_idx = to_match["tax_assessment_year"].fillna(0).idxmax()
                            else:
                                nearest_idx = to_match.index[0]
                            property_data = street_matches.loc[nearest_idx].to_dict()
                            if property_type:
                                property_data["property_type"] = property_type
                            if lat is not None:
                                property_data["latitude"] = lat
                            if lon is not None:
                                property_data["longitude"] = lon
                            property_data["address"] = address
                            logger.info(
                                "Resolved address '%s' via to_civic_number to PID=%s",
                                address, property_data.get("pid"),
                            )
                            return property_data

                        # Fallback: civic_number column (from_civic or to_civic)
                        if "civic_number" in street_matches.columns:
                            civic_col = street_matches["civic_number"].fillna(0).astype(int)
                        else:
                            from_c = street_matches["from_civic_number"].fillna(0).astype(int)
                            to_c = street_matches.get("to_civic_number", pd.Series(0, index=street_matches.index)).fillna(0).astype(int)
                            civic_col = from_c.where(from_c > 0, to_c)

                        has_civic = civic_col > 0
                        if has_civic.any():
                            with_civic = street_matches[has_civic]
                            dists = (civic_col[with_civic.index] - target_civic).abs()
                            min_dist = dists.min()

                            # If civic number doesn't match exactly and we have
                            # lat/lon, skip to lat/lon fallback — Google often
                            # renumbers addresses differently from BC Assessment
                            if min_dist > 0 and lat is not None and lon is not None:
                                logger.info(
                                    "Civic %d not found on %s (nearest: %d, off by %d) "
                                    "— falling through to lat/lon match",
                                    target_civic, raw_street,
                                    int(civic_col[dists.idxmin()]), int(min_dist),
                                )
                                pass  # fall through to lat/lon fallback below
                            else:
                                exact_matches = with_civic[dists == min_dist]
                                if len(exact_matches) > 1 and "tax_assessment_year" in exact_matches.columns:
                                    nearest_idx = exact_matches["tax_assessment_year"].fillna(0).idxmax()
                                else:
                                    nearest_idx = dists.idxmin()

                                property_data = street_matches.loc[nearest_idx].to_dict()
                                if property_type:
                                    property_data["property_type"] = property_type
                                if lat is not None:
                                    property_data["latitude"] = lat
                                if lon is not None:
                                    property_data["longitude"] = lon
                                property_data["address"] = address
                                logger.info(
                                    "Resolved address '%s' to nearest PID=%s on same street",
                                    address, property_data.get("pid"),
                                )
                                return property_data
                        else:
                            nearest_idx = street_matches.index[0]

                            property_data = street_matches.loc[nearest_idx].to_dict()
                            if property_type:
                                property_data["property_type"] = property_type
                            if lat is not None:
                                property_data["latitude"] = lat
                            if lon is not None:
                                property_data["longitude"] = lon
                            property_data["address"] = address
                            logger.info(
                                "Resolved address '%s' to PID=%s (first on street)",
                                address, property_data.get("pid"),
                            )
                            return property_data

        # If we have lat/lon and properties have lat/lon, find nearest
        if (
            lat is not None
            and lon is not None
            and properties_df is not None
            and not properties_df.empty
            and "latitude" in properties_df.columns
            and "longitude" in properties_df.columns
        ):
            lat_col = properties_df["latitude"].fillna(0)
            lon_col = properties_df["longitude"].fillna(0)
            has_coords = (lat_col != 0) & (lon_col != 0)
            if has_coords.any():
                subset = properties_df[has_coords]

                # If property_type is specified, prefer same-type matches
                # within a tight radius (~100m ≈ 0.001 degrees)
                if property_type and "property_type" in subset.columns:
                    typed = subset[subset["property_type"] == property_type]
                    if not typed.empty:
                        typed_dists = np.sqrt(
                            (typed["latitude"] - lat) ** 2
                            + (typed["longitude"] - lon) ** 2
                        )
                        if typed_dists.min() < 0.002:  # ~200m
                            subset = typed

                dists = np.sqrt(
                    (subset["latitude"] - lat) ** 2
                    + (subset["longitude"] - lon) ** 2
                )
                nearest_idx = dists.idxmin()
                property_data = subset.loc[nearest_idx].to_dict()
                if property_type:
                    property_data["property_type"] = property_type
                property_data["latitude"] = lat
                property_data["longitude"] = lon
                if address:
                    property_data["address"] = address
                logger.info(
                    "Resolved lat/lon to nearest PID=%s (type=%s, dist=%.5f)",
                    property_data.get("pid"),
                    property_data.get("property_type"),
                    float(dists.loc[nearest_idx]),
                )
                return property_data

        # Last resort: minimal stub with neighbourhood from address heuristic
        # Infer "condo" if the address contains a unit number (e.g. "#205")
        inferred_type = property_type or "detached"
        if not property_type and address and "#" in address:
            inferred_type = "condo"

        logger.warning(
            "No property match found; using minimal stub (type=%s)",
            inferred_type,
        )
        property_data: dict = {
            "pid": pid or "unknown",
            "latitude": lat or 0.0,
            "longitude": lon or 0.0,
            "address": address or "",
            "property_type": inferred_type,
        }
        return property_data

    def _build_market_context(
        self,
        property_data: dict,
        properties_df: Optional[pd.DataFrame],
    ) -> dict:
        """Build market context dict for the prediction response.

        Computes neighbourhood-level statistics for context.

        Args:
            property_data: Resolved property attributes.
            properties_df: Enriched property universe.

        Returns:
            Dict with neighbourhood stats, YoY change, and interest rate.
        """
        neighbourhood_code = property_data.get("neighbourhood_code", "")
        ptype = property_data.get("property_type", "")

        context = {
            "neighbourhood_code": neighbourhood_code,
            "neighbourhood_name": _NEIGHBOURHOOD_NAMES.get(
                str(neighbourhood_code),
                str(neighbourhood_code).replace("-", " ").title(),
            ),
            "median_assessed_value": 0.0,
            "yoy_change_pct": None,
            "interest_rate_5yr": None,
            "property_count": 0,
            "assessment_year": datetime.utcnow().year,
        }

        if properties_df is None or properties_df.empty:
            return context

        # Filter to same neighbourhood (cast to string for type safety)
        if neighbourhood_code and "neighbourhood_code" in properties_df.columns:
            hood_df = properties_df[
                properties_df["neighbourhood_code"].astype(str) == str(neighbourhood_code)
            ]
        else:
            hood_df = properties_df

        if hood_df.empty:
            return context

        # Compute stats
        if "total_assessed_value" in hood_df.columns:
            context["median_assessed_value"] = round(
                float(hood_df["total_assessed_value"].median()), 2,
            )

        context["property_count"] = len(hood_df)

        # Assessment year
        if "tax_assessment_year" in hood_df.columns:
            context["assessment_year"] = int(
                hood_df["tax_assessment_year"].max()
            )

        # YoY change if previous values are available
        if (
            "total_assessed_value" in hood_df.columns
            and "previous_land_value" in hood_df.columns
            and "previous_improvement_value" in hood_df.columns
        ):
            current_total = hood_df["total_assessed_value"].median()
            previous_total = (
                hood_df["previous_land_value"].fillna(0)
                + hood_df["previous_improvement_value"].fillna(0)
            ).median()

            if previous_total > 0:
                yoy = (current_total - previous_total) / previous_total * 100
                context["yoy_change_pct"] = round(float(yoy), 2)

        return context

    @staticmethod
    def assign_neighbourhood_from_latlon(
        lat: float,
        lon: float,
        boundary_gdf,
    ) -> tuple[Optional[str], Optional[str]]:
        """Determine the correct neighbourhood using City of Vancouver boundaries.

        Does a point-in-polygon test against the official local area
        boundary GeoJSON to override BC Assessment's neighbourhood_code,
        which is inconsistent with City boundaries for some areas.

        Args:
            lat: Latitude in decimal degrees.
            lon: Longitude in decimal degrees.
            boundary_gdf: GeoDataFrame of local area boundary polygons.

        Returns:
            Tuple of (neighbourhood_code, neighbourhood_name) if a match
            is found, or (None, None) if no match.
        """
        if boundary_gdf is None or lat == 0.0 or lon == 0.0:
            return None, None

        try:
            from shapely.geometry import Point

            point = Point(lon, lat)  # shapely uses (x=lon, y=lat)

            for _, row in boundary_gdf.iterrows():
                geom = row.get("geometry") or row.get("geom")
                if geom is not None and geom.contains(point):
                    # Extract the neighbourhood name from the GeoJSON
                    area_name = (
                        row.get("name")
                        or row.get("Name")
                        or row.get("NAME")
                        or row.get("mapid")
                        or ""
                    )
                    if not area_name:
                        continue

                    # Reverse-lookup: find the matching code in our mapping
                    area_name_upper = area_name.strip().upper()
                    for code, name in _NEIGHBOURHOOD_NAMES.items():
                        if name.upper() == area_name_upper:
                            return code, name
                        # Fuzzy: check if the boundary name contains our name
                        if area_name_upper in name.upper() or name.upper() in area_name_upper:
                            return code, name

                    # No code match — return name only
                    return None, area_name.strip()

        except Exception as exc:
            logger.warning("Spatial neighbourhood assignment failed: %s", exc)

        return None, None

    def get_loaded_model_count(self) -> int:
        """Return the number of models currently cached in memory.

        Returns:
            Count of cached segment models.
        """
        return len(self._model_cache)

    def get_available_model_count(self) -> int:
        """Count the number of model files on disk.

        Returns:
            Count of .pkl model files in model_dir (excluding quantile models).
        """
        if not self.model_dir.exists():
            return 0
        # Count .pkl files that are not quantile models (contain _q)
        return sum(
            1
            for f in self.model_dir.glob("*.pkl")
            if "_q" not in f.stem
        )
