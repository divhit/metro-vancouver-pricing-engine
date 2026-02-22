"""
Comparable sales reconciliation engine (Tier 3).

Finds the K most similar properties to a subject and computes an
adjusted comparable value range. Reconciles the ML model estimate
with the comparable range to produce the final blended prediction.

This serves as a safety net -- when the ML model produces implausible
values for unusual properties, comparable reconciliation catches them.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.comparables.similarity import SimilarityScorer
from src.models.types import ComparableProperty

logger = logging.getLogger(__name__)

# Downtown Vancouver CBD reference point (Waterfront Station)
_CBD_LAT = 49.2856
_CBD_LON = -123.1115

# Earth radius in metres
_EARTH_RADIUS_M = 6_371_000.0


class ComparableEngine:
    """Find, adjust, and reconcile comparable properties.

    Orchestrates the full Tier 3 pipeline:
    1. Filter the candidate universe by geography, time, and type.
    2. Score all remaining candidates for similarity to the subject.
    3. Select the top K comparables.
    4. Adjust comparable values toward the subject.
    5. Reconcile the ML model estimate with the comparable range.

    Args:
        similarity_scorer: A pre-configured SimilarityScorer instance.
            If None, a default scorer is created based on *mls_available*.
        mls_available: Whether MLS-enhanced data (sqft, bedrooms) is
            available. Ignored if *similarity_scorer* is provided.
    """

    def __init__(
        self,
        similarity_scorer: Optional[SimilarityScorer] = None,
        mls_available: bool = False,
    ) -> None:
        self.mls_available = mls_available
        self.scorer = similarity_scorer or SimilarityScorer(
            mls_available=mls_available
        )

    # ================================================================== #
    # 1. FIND COMPARABLES
    # ================================================================== #

    def find_comparables(
        self,
        subject: dict,
        candidates_df: pd.DataFrame,
        k: int = 5,
        max_distance_m: float = 2000,
        max_age_months: int = 12,
        same_property_type: bool = True,
    ) -> list[ComparableProperty]:
        """Find the K most similar properties to the subject.

        Applies progressive filtering (geography, time, type), scores
        all survivors, and returns the top K. If fewer than K candidates
        pass the initial filters, the search is automatically widened.

        Args:
            subject: Dict of property features for the subject.
            candidates_df: Full property universe as a DataFrame.
            k: Number of comparables to return.
            max_distance_m: Maximum geographic radius in metres.
            max_age_months: Maximum assessment/sale age in months.
            same_property_type: Require the same property type.

        Returns:
            List of up to *k* ComparableProperty objects, sorted by
            similarity (best first).
        """
        if candidates_df.empty:
            logger.warning("Empty candidate DataFrame — no comparables available")
            return []

        filtered = self._apply_filters(
            subject, candidates_df, max_distance_m, max_age_months, same_property_type
        )

        # Widen search if too few candidates survive filtering
        if len(filtered) < k:
            logger.info(
                "Only %d candidates after initial filters (need %d) — "
                "widening search to 5 km and relaxing property type",
                len(filtered),
                k,
            )
            filtered = self._apply_filters(
                subject,
                candidates_df,
                max_distance_m=5000,
                max_age_months=max(max_age_months, 24),
                same_property_type=False,
            )

        if filtered.empty:
            logger.warning(
                "No candidates found even after widened search for subject PID=%s",
                subject.get("pid", "unknown"),
            )
            return []

        # Score all surviving candidates
        similarity_scores = self.scorer.score_batch(subject, filtered)
        filtered = filtered.copy()
        filtered["_similarity_score"] = similarity_scores

        # Sort by similarity (ascending — lower = more similar)
        filtered = filtered.sort_values("_similarity_score").head(k)

        # Build ComparableProperty objects
        comparables: list[ComparableProperty] = []
        for _, row in filtered.iterrows():
            # Compute distance for the result object
            dist = SimilarityScorer._haversine_distance(
                subject.get("latitude", 0.0),
                subject.get("longitude", 0.0),
                row.get("latitude", 0.0),
                row.get("longitude", 0.0),
            )

            # Per-dimension breakdown for explainability
            _, breakdown = self.scorer.score_with_breakdown(
                subject, row.to_dict()
            )

            comp = ComparableProperty(
                pid=str(row.get("pid", "")),
                address=str(row.get("address", "")),
                assessed_value=float(row.get("assessed_value", 0)),
                year_built=int(row["year_built"]) if pd.notna(row.get("year_built")) else None,
                zoning=str(row.get("zoning", "")) or None,
                neighbourhood_code=str(row.get("neighbourhood_code", "")),
                latitude=float(row.get("latitude", 0.0)),
                longitude=float(row.get("longitude", 0.0)),
                distance_m=round(dist, 1),
                similarity_score=round(float(row["_similarity_score"]), 6),
                similarity_breakdown=breakdown,
            )
            comparables.append(comp)

        logger.info(
            "Found %d comparables for subject PID=%s (best score=%.4f)",
            len(comparables),
            subject.get("pid", "unknown"),
            comparables[0].similarity_score if comparables else float("nan"),
        )
        return comparables

    # ================================================================== #
    # 2. COMPUTE COMPARABLE RANGE
    # ================================================================== #

    def compute_comparable_range(
        self,
        subject: dict,
        comparables: list[ComparableProperty],
    ) -> tuple[float, float, float]:
        """Compute an adjusted value range from comparable properties.

        For each comparable, adjusts its assessed value toward the
        subject based on age difference, size difference, and location
        (distance to CBD as a proxy for price gradient).

        Args:
            subject: Dict of property features for the subject.
            comparables: List of ComparableProperty objects.

        Returns:
            Tuple of (low, median, high) representing the 25th
            percentile, median, and 75th percentile of adjusted
            comparable values.
        """
        if not comparables:
            logger.warning("No comparables provided for range computation")
            return (0.0, 0.0, 0.0)

        adjusted_values: list[float] = []

        for comp in comparables:
            adj_value = comp.assessed_value

            # --- Age adjustment: +/- 0.5% per year difference ---
            subject_year = subject.get("year_built")
            if subject_year is not None and comp.year_built is not None:
                year_diff = subject_year - comp.year_built
                # Positive year_diff means subject is newer => comp should
                # be adjusted upward to match the subject's newness
                age_adj = 1.0 + (year_diff * 0.005)
                age_adj = max(0.80, min(1.20, age_adj))  # cap at +/- 20%
                adj_value *= age_adj

            # --- Size adjustment (if living area available) ---
            subject_sqft = (
                subject.get("living_area_sqft")
                or subject.get("estimated_living_area")
            )
            comp_sqft = (
                subject.get("_comp_living_area_sqft")  # passed through
                or None
            )
            # Use the comp's assessed value as a proxy for $/sqft
            # if MLS data provides living area for both
            if self.mls_available and subject_sqft and comp_sqft:
                if comp_sqft > 0:
                    ppsf = comp.assessed_value / comp_sqft
                    size_diff = subject_sqft - comp_sqft
                    adj_value += ppsf * size_diff

            # --- Location adjustment (CBD distance gradient) ---
            subject_cbd_dist = self._cbd_distance(
                subject.get("latitude", 0.0),
                subject.get("longitude", 0.0),
            )
            comp_cbd_dist = self._cbd_distance(
                comp.latitude, comp.longitude
            )

            if subject_cbd_dist > 0 and comp_cbd_dist > 0:
                # Properties closer to CBD command a premium.
                # Approximate gradient: 2% per km closer to CBD.
                dist_diff_km = (comp_cbd_dist - subject_cbd_dist) / 1000.0
                location_adj = 1.0 + (dist_diff_km * 0.02)
                location_adj = max(0.85, min(1.15, location_adj))  # cap
                adj_value *= location_adj

            adjusted_values.append(max(0.0, adj_value))

        arr = np.array(adjusted_values)

        if len(arr) <= 2:
            # With very few comparables, widen the range for safety
            low = float(arr.min() * 0.95)
            high = float(arr.max() * 1.05)
            median = float(np.median(arr))
        else:
            low = float(np.percentile(arr, 25))
            median = float(np.median(arr))
            high = float(np.percentile(arr, 75))

        return (round(low, 2), round(median, 2), round(high, 2))

    # ================================================================== #
    # 3. RECONCILE WITH ML
    # ================================================================== #

    def reconcile_with_ml(
        self,
        ml_estimate: float,
        comparable_range: tuple[float, float, float],
        comparable_count: int,
    ) -> tuple[float, float, str]:
        """Blend the ML estimate with the comparable-derived value.

        The blending weight depends on how much the ML estimate and
        comparable median diverge, and how many comparables were found.

        Divergence tiers:
        - < 10%:   80% ML / 20% comps — high confidence
        - 10-20%:  60% ML / 40% comps — moderate confidence
        - >= 20%:  50% ML / 50% comps — flagged for review

        If fewer than 3 comparables were found, the ML estimate is
        given extra weight (90/10) regardless of divergence.

        Args:
            ml_estimate: Point estimate from the ML model.
            comparable_range: (low, median, high) from
                compute_comparable_range().
            comparable_count: Number of comparables used.

        Returns:
            Tuple of (final_estimate, divergence_pct, note).
        """
        comparable_median = comparable_range[1]

        # Guard against zero / missing comparable median
        if comparable_median <= 0:
            logger.warning(
                "Comparable median is zero — returning ML estimate unchanged"
            )
            return (
                round(ml_estimate, 2),
                0.0,
                "No valid comparable median — ML estimate used directly",
            )

        divergence_pct = (
            abs(ml_estimate - comparable_median) / comparable_median * 100
        )

        # Determine blending weights by divergence tier
        if divergence_pct < 10:
            ml_weight = 0.80
            comp_weight = 0.20
            confidence = "high"
            note = (
                f"ML and comparables agree within {divergence_pct:.1f}%"
            )
        elif divergence_pct < 20:
            ml_weight = 0.60
            comp_weight = 0.40
            confidence = "moderate"
            note = (
                f"ML diverges {divergence_pct:.1f}% from comparables "
                f"-- blended 60/40"
            )
        else:
            ml_weight = 0.50
            comp_weight = 0.50
            confidence = "low"
            note = (
                f"ML diverges {divergence_pct:.1f}% from comparables "
                f"-- flagged for review"
            )

        # Override: limited comparables => lean heavily on ML
        if comparable_count < 3:
            ml_weight = 0.90
            comp_weight = 0.10
            note += " (limited comparables, ML-weighted)"

        final = ml_weight * ml_estimate + comp_weight * comparable_median

        logger.info(
            "Reconciled: ML=$%.0f, comp_median=$%.0f, divergence=%.1f%%, "
            "final=$%.0f (%s confidence)",
            ml_estimate,
            comparable_median,
            divergence_pct,
            final,
            confidence,
        )

        return (round(final, 2), round(divergence_pct, 2), note)

    # ================================================================== #
    # 4. SUMMARY FOR API RESPONSE
    # ================================================================== #

    def get_comparable_summary(
        self, comparables: list[ComparableProperty]
    ) -> dict:
        """Build a summary dict suitable for an API response.

        Args:
            comparables: List of ComparableProperty objects.

        Returns:
            Dict with count, median_value, mean_value, value_range,
            avg_distance_m, and avg_similarity.
        """
        if not comparables:
            return {
                "count": 0,
                "median_value": 0.0,
                "mean_value": 0.0,
                "value_range": (0.0, 0.0),
                "avg_distance_m": 0.0,
                "avg_similarity": 0.0,
            }

        values = [c.assessed_value for c in comparables]
        distances = [c.distance_m for c in comparables]
        similarities = [c.similarity_score for c in comparables]

        return {
            "count": len(comparables),
            "median_value": round(float(np.median(values)), 2),
            "mean_value": round(float(np.mean(values)), 2),
            "value_range": (
                round(float(min(values)), 2),
                round(float(max(values)), 2),
            ),
            "avg_distance_m": round(float(np.mean(distances)), 1),
            "avg_similarity": round(float(np.mean(similarities)), 6),
        }

    # ================================================================== #
    # PRIVATE HELPERS
    # ================================================================== #

    def _apply_filters(
        self,
        subject: dict,
        df: pd.DataFrame,
        max_distance_m: float,
        max_age_months: int,
        same_property_type: bool,
    ) -> pd.DataFrame:
        """Apply geographic, temporal, and type filters to candidates.

        Args:
            subject: Subject property dict.
            df: Full candidate DataFrame.
            max_distance_m: Maximum distance in metres.
            max_age_months: Maximum assessment/sale age in months.
            same_property_type: Require matching property type.

        Returns:
            Filtered DataFrame.
        """
        mask = pd.Series(True, index=df.index)

        # --- Exclude the subject itself ---
        subject_pid = subject.get("pid")
        if subject_pid is not None and "pid" in df.columns:
            mask &= df["pid"].astype(str) != str(subject_pid)

        # --- Geographic filter (bounding box first, then haversine) ---
        s_lat = subject.get("latitude", 0.0)
        s_lon = subject.get("longitude", 0.0)

        if "latitude" in df.columns and "longitude" in df.columns:
            # Approximate bounding box for fast pre-filter
            # 1 degree latitude ~ 111 km; longitude varies with cos(lat)
            lat_delta = max_distance_m / 111_000.0
            lon_delta = max_distance_m / (111_000.0 * max(math.cos(math.radians(s_lat)), 0.01))

            mask &= (df["latitude"] >= s_lat - lat_delta) & (df["latitude"] <= s_lat + lat_delta)
            mask &= (df["longitude"] >= s_lon - lon_delta) & (df["longitude"] <= s_lon + lon_delta)

            # Precise haversine on surviving rows
            candidate_subset = df.loc[mask]
            if not candidate_subset.empty:
                distances = SimilarityScorer._haversine_distance_vec(
                    s_lat, s_lon,
                    candidate_subset["latitude"].values,
                    candidate_subset["longitude"].values,
                )
                dist_mask = distances <= max_distance_m
                mask.loc[mask] = dist_mask

        # --- Temporal filter ---
        date_col = None
        for col in ("assessment_date", "sale_date"):
            if col in df.columns:
                date_col = col
                break

        if date_col is not None:
            dates = pd.to_datetime(df[date_col], errors="coerce")
            cutoff = pd.Timestamp.utcnow() - pd.DateOffset(months=max_age_months)
            # Keep rows with valid dates that are recent enough,
            # plus rows with missing dates (don't discard for missing data)
            temporal_mask = dates.isna() | (dates >= cutoff)
            mask &= temporal_mask

        # --- Property type filter ---
        if same_property_type and "property_type" in df.columns:
            s_type = subject.get("property_type")
            if s_type is not None:
                mask &= df["property_type"].astype(str) == str(s_type)

        return df.loc[mask].copy()

    @staticmethod
    def _cbd_distance(lat: float, lon: float) -> float:
        """Distance in metres from a point to downtown Vancouver CBD.

        Uses Waterfront Station (49.2856, -123.1115) as the CBD
        reference point.

        Args:
            lat: Latitude in decimal degrees.
            lon: Longitude in decimal degrees.

        Returns:
            Distance in metres.
        """
        return SimilarityScorer._haversine_distance(
            lat, lon, _CBD_LAT, _CBD_LON
        )
