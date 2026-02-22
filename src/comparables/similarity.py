"""
Multi-dimensional property similarity scoring.

Computes a composite similarity score between a subject property and
candidate comparables using weighted dimensions. Weights shift based
on data availability (assessment-only vs MLS-enhanced mode).

Lower score = more similar (0.0 = identical, 1.0 = maximally dissimilar).
"""
from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================
# DEFAULT WEIGHT PROFILES
# ============================================================

# Assessment-only mode: no MLS fields (sqft, bedrooms) available.
# Geographic proximity and assessed value carry the signal.
ASSESSMENT_ONLY_WEIGHTS: dict[str, float] = {
    "proximity": 0.30,
    "value_similarity": 0.25,
    "year_built": 0.15,
    "zoning_match": 0.15,
    "recency": 0.15,
}

# MLS-enhanced mode: living area is the strongest single predictor
# of value, so it gets the highest weight.
MLS_ENHANCED_WEIGHTS: dict[str, float] = {
    "proximity": 0.20,
    "living_area": 0.25,
    "value_similarity": 0.10,
    "year_built": 0.10,
    "zoning_match": 0.10,
    "bedroom_match": 0.10,
    "recency": 0.15,
}

# Earth radius in metres for haversine calculation
_EARTH_RADIUS_M = 6_371_000.0


class SimilarityScorer:
    """Score the similarity between a subject property and candidates.

    Each dimension maps to a 0.0-1.0 sub-score (0 = identical,
    1 = maximally dissimilar). The final composite score is a
    weighted sum of all active dimensions.

    Args:
        mls_available: When True, use MLS-enhanced weights that
            include living_area and bedroom_match dimensions.
        custom_weights: Optional dict that overrides the default
            weight profile entirely. Keys must match dimension
            names and values should sum to ~1.0.
    """

    def __init__(
        self,
        mls_available: bool = False,
        custom_weights: Optional[dict[str, float]] = None,
    ) -> None:
        self.mls_available = mls_available

        if custom_weights is not None:
            self.weights = dict(custom_weights)
        elif mls_available:
            self.weights = dict(MLS_ENHANCED_WEIGHTS)
        else:
            self.weights = dict(ASSESSMENT_ONLY_WEIGHTS)

        # Normalise weights so they sum to exactly 1.0
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

        logger.debug(
            "SimilarityScorer initialised — mode=%s, weights=%s",
            "MLS" if mls_available else "assessment-only",
            self.weights,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def score(self, subject: dict, candidate: dict) -> float:
        """Compute a composite similarity score between two properties.

        Args:
            subject: Dict of property features for the subject.
            candidate: Dict of property features for a comparable
                candidate.

        Returns:
            Composite similarity score in [0.0, 1.0].
        """
        breakdown = self._compute_dimensions(subject, candidate)
        composite = sum(
            self.weights.get(dim, 0.0) * value
            for dim, value in breakdown.items()
        )
        return round(min(1.0, max(0.0, composite)), 6)

    def score_with_breakdown(
        self, subject: dict, candidate: dict
    ) -> tuple[float, dict[str, float]]:
        """Score with per-dimension breakdown for explainability.

        Returns:
            Tuple of (composite_score, dimension_breakdown).
        """
        breakdown = self._compute_dimensions(subject, candidate)
        composite = sum(
            self.weights.get(dim, 0.0) * value
            for dim, value in breakdown.items()
        )
        return round(min(1.0, max(0.0, composite)), 6), breakdown

    def score_batch(self, subject: dict, candidates_df: pd.DataFrame) -> pd.Series:
        """Vectorised similarity scoring against a DataFrame of candidates.

        Uses numpy operations where possible to avoid Python-level loops
        on the hot-path dimensions (proximity, value, year_built, recency).

        Args:
            subject: Dict of property features for the subject.
            candidates_df: DataFrame where each row is a candidate
                property with the same feature keys as *subject*.

        Returns:
            pd.Series of similarity scores, indexed like *candidates_df*.
        """
        n = len(candidates_df)
        if n == 0:
            return pd.Series(dtype=float)

        scores = np.zeros(n, dtype=np.float64)

        # --- proximity ---
        if "proximity" in self.weights:
            w = self.weights["proximity"]
            dist = self._haversine_distance_vec(
                subject.get("latitude", 0.0),
                subject.get("longitude", 0.0),
                candidates_df.get("latitude", pd.Series(np.zeros(n))).values,
                candidates_df.get("longitude", pd.Series(np.zeros(n))).values,
            )
            # Exponential decay with 1 km half-life
            prox = 1.0 - np.exp(-np.log(2) * dist / 1000.0)
            prox = np.clip(prox, 0.0, 1.0)
            scores += w * prox

        # --- value_similarity ---
        if "value_similarity" in self.weights:
            w = self.weights["value_similarity"]
            sv = subject.get("assessed_value", 0)
            cv = candidates_df.get(
                "assessed_value", pd.Series(np.ones(n))
            ).values.astype(float)
            # Avoid log(0)
            sv_safe = max(sv, 1.0)
            cv_safe = np.maximum(cv, 1.0)
            val_sim = np.abs(np.log(sv_safe / cv_safe)) / np.log(2)
            val_sim = np.clip(val_sim, 0.0, 1.0)
            scores += w * val_sim

        # --- year_built ---
        if "year_built" in self.weights:
            w = self.weights["year_built"]
            sy = subject.get("year_built")
            if sy is not None:
                cy = candidates_df.get(
                    "year_built", pd.Series(np.full(n, np.nan))
                ).values.astype(float)
                year_diff = np.abs(float(sy) - cy) / 50.0
                # Treat missing years as mid-range penalty
                year_diff = np.where(np.isnan(year_diff), 0.5, year_diff)
                year_diff = np.clip(year_diff, 0.0, 1.0)
            else:
                year_diff = np.full(n, 0.5)
            scores += w * year_diff

        # --- zoning_match ---
        if "zoning_match" in self.weights:
            w = self.weights["zoning_match"]
            sz = subject.get("zoning", "")
            cz = candidates_df.get(
                "zoning", pd.Series([""] * n)
            ).values
            zoning_scores = np.array(
                [self._zoning_similarity(sz, str(z)) for z in cz],
                dtype=np.float64,
            )
            scores += w * zoning_scores

        # --- recency ---
        if "recency" in self.weights:
            w = self.weights["recency"]
            recency_vals = self._compute_recency_vec(candidates_df, n)
            scores += w * recency_vals

        # --- living_area (MLS only) ---
        if "living_area" in self.weights:
            w = self.weights["living_area"]
            s_sqft = subject.get("living_area_sqft") or subject.get("estimated_living_area")
            if s_sqft and s_sqft > 0:
                c_sqft = candidates_df.get(
                    "living_area_sqft",
                    candidates_df.get(
                        "estimated_living_area",
                        pd.Series(np.full(n, np.nan)),
                    ),
                ).values.astype(float)
                max_sqft = np.maximum(float(s_sqft), c_sqft)
                max_sqft = np.where(max_sqft == 0, 1.0, max_sqft)
                area_sim = np.abs(float(s_sqft) - c_sqft) / max_sqft
                area_sim = np.where(np.isnan(area_sim), 0.5, area_sim)
                area_sim = np.clip(area_sim, 0.0, 1.0)
            else:
                area_sim = np.full(n, 0.5)
            scores += w * area_sim

        # --- bedroom_match (MLS only) ---
        if "bedroom_match" in self.weights:
            w = self.weights["bedroom_match"]
            s_bed = subject.get("bedrooms")
            if s_bed is not None:
                c_bed = candidates_df.get(
                    "bedrooms", pd.Series(np.full(n, np.nan))
                ).values.astype(float)
                bed_sim = np.abs(float(s_bed) - c_bed) / 3.0
                bed_sim = np.where(np.isnan(bed_sim), 0.5, bed_sim)
                bed_sim = np.clip(bed_sim, 0.0, 1.0)
            else:
                bed_sim = np.full(n, 0.5)
            scores += w * bed_sim

        result = np.clip(scores, 0.0, 1.0)
        return pd.Series(result, index=candidates_df.index, name="similarity_score")

    # ------------------------------------------------------------------ #
    # Dimension helpers (single-pair)
    # ------------------------------------------------------------------ #

    def _compute_dimensions(
        self, subject: dict, candidate: dict
    ) -> dict[str, float]:
        """Compute all active dimension sub-scores for a single pair."""
        dims: dict[str, float] = {}

        if "proximity" in self.weights:
            dist = self._haversine_distance(
                subject.get("latitude", 0.0),
                subject.get("longitude", 0.0),
                candidate.get("latitude", 0.0),
                candidate.get("longitude", 0.0),
            )
            # Exponential decay: 1 km half-life
            dims["proximity"] = min(
                1.0, 1.0 - math.exp(-math.log(2) * dist / 1000.0)
            )

        if "value_similarity" in self.weights:
            v1 = max(subject.get("assessed_value", 0), 1.0)
            v2 = max(candidate.get("assessed_value", 0), 1.0)
            dims["value_similarity"] = min(
                1.0, abs(math.log(v1 / v2)) / math.log(2)
            )

        if "year_built" in self.weights:
            y1 = subject.get("year_built")
            y2 = candidate.get("year_built")
            if y1 is not None and y2 is not None:
                dims["year_built"] = min(1.0, abs(y1 - y2) / 50.0)
            else:
                dims["year_built"] = 0.5  # missing = mid-range penalty

        if "zoning_match" in self.weights:
            dims["zoning_match"] = self._zoning_similarity(
                subject.get("zoning", ""),
                candidate.get("zoning", ""),
            )

        if "recency" in self.weights:
            dims["recency"] = self._recency_score(candidate)

        if "living_area" in self.weights:
            s = subject.get("living_area_sqft") or subject.get("estimated_living_area")
            c = candidate.get("living_area_sqft") or candidate.get("estimated_living_area")
            if s and c and s > 0 and c > 0:
                dims["living_area"] = min(
                    1.0, abs(s - c) / max(s, c)
                )
            else:
                dims["living_area"] = 0.5

        if "bedroom_match" in self.weights:
            b1 = subject.get("bedrooms")
            b2 = candidate.get("bedrooms")
            if b1 is not None and b2 is not None:
                dims["bedroom_match"] = min(1.0, abs(b1 - b2) / 3.0)
            else:
                dims["bedroom_match"] = 0.5

        return dims

    # ------------------------------------------------------------------ #
    # Recency helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _recency_score(candidate: dict) -> float:
        """Compute recency sub-score for a single candidate.

        Uses *assessment_date* or *sale_date* if available, otherwise
        falls back to a mid-range penalty.

        Returns:
            Score in [0.0, 1.0] where 0 = very recent, 1 = 24+ months old.
        """
        date_val = candidate.get("assessment_date") or candidate.get("sale_date")
        if date_val is None:
            return 0.5

        if isinstance(date_val, str):
            try:
                date_val = datetime.fromisoformat(date_val)
            except (ValueError, TypeError):
                return 0.5

        if isinstance(date_val, datetime):
            months = (datetime.utcnow() - date_val).days / 30.44
            return min(1.0, max(0.0, months / 24.0))

        return 0.5

    @staticmethod
    def _compute_recency_vec(df: pd.DataFrame, n: int) -> np.ndarray:
        """Vectorised recency computation across a DataFrame."""
        date_col = None
        for col in ("assessment_date", "sale_date"):
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            return np.full(n, 0.5)

        dates = pd.to_datetime(df[date_col], errors="coerce")
        now = pd.Timestamp.utcnow()
        months = (now - dates).dt.days / 30.44
        result = (months / 24.0).values.astype(float)
        result = np.where(np.isnan(result), 0.5, result)
        return np.clip(result, 0.0, 1.0)

    # ------------------------------------------------------------------ #
    # Geographic helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _haversine_distance(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Haversine distance between two points in metres.

        Args:
            lat1, lon1: Subject coordinates (decimal degrees).
            lat2, lon2: Candidate coordinates (decimal degrees).

        Returns:
            Great-circle distance in metres.
        """
        lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
        lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)

        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))
        return _EARTH_RADIUS_M * c

    @staticmethod
    def _haversine_distance_vec(
        lat1: float,
        lon1: float,
        lat2: np.ndarray,
        lon2: np.ndarray,
    ) -> np.ndarray:
        """Vectorised haversine distance (one-to-many) in metres."""
        lat1_r = np.radians(lat1)
        lon1_r = np.radians(lon1)
        lat2_r = np.radians(lat2)
        lon2_r = np.radians(lon2)

        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        return _EARTH_RADIUS_M * c

    # ------------------------------------------------------------------ #
    # Zoning helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _zoning_similarity(z1: str, z2: str) -> float:
        """Compare two Metro Vancouver zoning codes.

        Zoning codes follow the pattern PREFIX-DIGIT (e.g. RS-1, RM-4,
        CD-1). This method uses a tiered comparison:

        - Exact match:      0.0  (RS-1 vs RS-1)
        - Same prefix:      0.3  (RS-1 vs RS-3)
        - Same category:    0.5  (RS vs RT — both residential)
        - Different:        1.0  (RS vs C — residential vs commercial)

        Returns:
            Similarity score in [0.0, 1.0].
        """
        if not z1 or not z2:
            return 0.5  # missing zoning = mid-range penalty

        z1_upper = z1.strip().upper()
        z2_upper = z2.strip().upper()

        # Exact match
        if z1_upper == z2_upper:
            return 0.0

        # Extract prefix (letters before the dash or digit)
        prefix1 = z1_upper.split("-")[0].rstrip("0123456789")
        prefix2 = z2_upper.split("-")[0].rstrip("0123456789")

        # Same prefix (e.g. RS-1 vs RS-3)
        if prefix1 == prefix2 and prefix1:
            return 0.3

        # Define broad zoning categories
        residential = {"RS", "RT", "RM", "RR", "RA", "RF"}
        commercial = {"C", "CD", "CR", "CS", "FC"}
        industrial = {"I", "IC", "IM", "IH", "IL"}
        comprehensive = {"CD", "DD", "HA"}

        def _category(prefix: str) -> str:
            if prefix in residential:
                return "residential"
            if prefix in commercial:
                return "commercial"
            if prefix in industrial:
                return "industrial"
            if prefix in comprehensive:
                return "comprehensive"
            return "other"

        # Same category
        if _category(prefix1) == _category(prefix2):
            return 0.5

        # Different categories
        return 1.0
