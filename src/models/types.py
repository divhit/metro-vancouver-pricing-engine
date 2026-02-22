"""
Shared data types for the pricing engine.

Canonical dataclasses used across ingestion, training, prediction,
backtesting, and adjustment modules. All modules should import types
from here rather than defining ad-hoc dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

# Import from feature registry to avoid duplication
from src.features.feature_registry import FeatureTier, PropertyType


# ============================================================
# COMPARABLE PROPERTY
# ============================================================

@dataclass
class ComparableProperty:
    """A comparable property returned alongside a prediction.

    Each comparable includes distance and similarity metrics so the
    end-user can evaluate how relevant the comp is to the subject.
    """

    pid: str
    address: str
    assessed_value: float
    year_built: int | None
    zoning: str | None
    neighbourhood_code: str
    latitude: float
    longitude: float
    distance_m: float
    similarity_score: float
    similarity_breakdown: dict[str, float] = field(default_factory=dict)


# ============================================================
# PREDICTION RESULT
# ============================================================

@dataclass
class PredictionResult:
    """Complete prediction output from the pricing engine.

    Contains the point estimate, confidence interval, comparable
    properties, SHAP explanations, post-model adjustments, market
    context, and risk flags.
    """

    pid: str
    point_estimate: float
    confidence_interval: tuple[float, float]
    confidence_grade: str  # A / B / C
    comparables: list[ComparableProperty]
    shap_values: dict[str, float]
    adjustments: list[tuple[str, float, str]]  # (name, pct, explanation)
    market_context: dict
    risk_flags: list[dict]
    model_segment: str
    model_version: str
    prediction_timestamp: datetime = field(default_factory=datetime.utcnow)


# ============================================================
# TRAINING RESULT
# ============================================================

@dataclass
class TrainingResult:
    """Output from training a single segment model.

    Captures the fitted model object, evaluation metrics,
    feature importances, and hyperparameters for reproducibility.
    """

    segment_key: str
    model: Any  # lgb.Booster or similar
    metrics: dict
    feature_importances: dict[str, float]
    shap_values: np.ndarray | None
    training_samples: int
    validation_mape: float
    hyperparameters: dict
    training_timestamp: datetime = field(default_factory=datetime.utcnow)


# ============================================================
# BACKTEST RESULT
# ============================================================

@dataclass
class BacktestResult:
    """Output from backtesting the model on a holdout year.

    Breaks down accuracy by property type, local area, and model
    segment, and compares against benchmark (e.g. BC Assessment).
    """

    holdout_year: int
    overall_metrics: dict
    by_property_type: dict[str, dict]
    by_local_area: dict[str, dict]
    by_segment: dict[str, dict]
    benchmark_comparison: dict
    predictions_df: pd.DataFrame = field(default_factory=pd.DataFrame)


# ============================================================
# INGESTION RESULT
# ============================================================

@dataclass
class IngestionResult:
    """Result from a data source ingestion run.

    Used by all ingestion modules to report what was fetched and
    whether it succeeded, partially succeeded, or failed.
    """

    source: str
    row_count: int
    timestamp: datetime
    status: str  # 'success', 'partial', 'failed'
    error: str | None = None


# ============================================================
# ADJUSTMENT RESULT
# ============================================================

@dataclass
class AdjustmentResult:
    """Output from Tier 2 (post-model) adjustments.

    Collects all applied adjustments (leasehold, view premium,
    heritage, etc.) and the resulting adjusted value.
    """

    adjusted_value: float
    adjustments: list[tuple[str, float, str]]  # (name, pct, explanation)
    total_adjustment_pct: float


# ============================================================
# FRESHNESS REPORT
# ============================================================

@dataclass
class FreshnessReport:
    """Data freshness report for a single data source.

    Used by the monitoring layer to flag stale data sources
    that may degrade prediction accuracy.
    """

    source: str
    last_updated: datetime | None
    expected_frequency_days: int
    is_stale: bool
    staleness_days: int | None
