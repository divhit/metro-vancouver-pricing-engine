"""
Quantile regression models for prediction intervals.

Property values are right-skewed, so Gaussian confidence intervals
are inappropriate. Quantile regression produces asymmetric intervals
that reflect actual uncertainty.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard quantiles for prediction intervals
DEFAULT_QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

# Mapping from confidence level to (lower_quantile, upper_quantile)
_CONFIDENCE_QUANTILE_MAP = {
    0.50: (0.25, 0.75),
    0.80: (0.10, 0.90),
    0.90: (0.05, 0.95),
}


class QuantileModelTrainer:
    """Trains and manages LightGBM quantile regression models.

    For each segment, trains a suite of quantile models that produce
    prediction intervals at various confidence levels. These intervals
    are asymmetric -- property values are right-skewed, so the upper
    bound extends further than the lower bound, reflecting reality.

    Usage::

        qt = QuantileModelTrainer()
        models = qt.train_quantile_models("Kitsilano__condo", X, y)
        lower, upper = qt.predict_intervals(models, X_new, confidence_level=0.80)

    The quantile models share the same feature space as the main
    regression model but use LightGBM's ``quantile`` objective with
    different ``alpha`` values.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    # ================================================================
    # TRAINING
    # ================================================================

    def train_quantile_models(
        self,
        segment_key: str,
        X: pd.DataFrame,
        y: pd.Series,
        quantiles: Optional[list[float]] = None,
        base_params: Optional[dict] = None,
    ) -> dict[float, Any]:
        """Train LightGBM quantile regression models.

        Trains one model per requested quantile, all sharing the same
        feature matrix. Each model uses LightGBM's native ``quantile``
        objective function which directly optimizes the pinball loss.

        Args:
            segment_key: Canonical segment key for logging context.
            X: Feature matrix (same features as the main model).
            y: Log-transformed target values.
            quantiles: List of quantile values to train models for.
                Defaults to [0.10, 0.25, 0.50, 0.75, 0.90].
            base_params: LightGBM parameters to use as a starting point.
                The ``objective`` and ``alpha`` params are overridden.

        Returns:
            Dict mapping quantile value (float) to trained LightGBM model.
        """
        if quantiles is None:
            quantiles = DEFAULT_QUANTILES.copy()

        if base_params is None:
            base_params = self._get_quantile_params()

        models: dict[float, Any] = {}

        logger.info(
            "Training %d quantile models for segment '%s' (%d samples)",
            len(quantiles),
            segment_key,
            len(X),
        )

        dtrain = lgb.Dataset(X, label=y)

        for q in sorted(quantiles):
            params = base_params.copy()
            params["objective"] = "quantile"
            params["alpha"] = q
            # Remove metric override -- quantile loss is used automatically
            params.pop("metric", None)

            n_estimators = params.pop("n_estimators", 1500)

            model = lgb.train(
                params,
                dtrain,
                num_boost_round=n_estimators,
            )

            models[q] = model

            logger.debug(
                "Trained quantile model q=%.2f for '%s'", q, segment_key
            )

        logger.info(
            "Quantile training complete for '%s': %d models trained",
            segment_key,
            len(models),
        )

        return models

    # ================================================================
    # PREDICTION INTERVALS
    # ================================================================

    def predict_intervals(
        self,
        models: dict[float, Any],
        X: pd.DataFrame,
        confidence_level: float = 0.80,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Produce prediction intervals from quantile models.

        Selects the appropriate lower and upper quantile models for the
        requested confidence level. For example, an 80% interval uses
        the 0.10 and 0.90 quantile models, meaning 80% of actual values
        should fall within the interval.

        The predictions are in log-space and are exponentiated back to
        dollar values.

        Args:
            models: Dict of quantile -> trained LightGBM model.
            X: Feature matrix to predict on.
            confidence_level: Desired confidence level (e.g. 0.80 for
                80% prediction interval).

        Returns:
            Tuple of (lower_bounds, upper_bounds) as numpy arrays in
            original dollar values (not log-transformed).

        Raises:
            ValueError: If the required quantile models are not available.
        """
        # Determine which quantiles bracket the confidence level
        lower_q, upper_q = self._get_quantile_bounds(confidence_level, models)

        if lower_q not in models:
            raise ValueError(
                f"Lower quantile model q={lower_q} not available. "
                f"Available quantiles: {sorted(models.keys())}"
            )
        if upper_q not in models:
            raise ValueError(
                f"Upper quantile model q={upper_q} not available. "
                f"Available quantiles: {sorted(models.keys())}"
            )

        lower_log = models[lower_q].predict(X)
        upper_log = models[upper_q].predict(X)

        # Convert from log-space to dollar values
        lower_bounds = np.expm1(lower_log)
        upper_bounds = np.expm1(upper_log)

        # Ensure lower <= upper (can be violated in edge cases)
        swap_mask = lower_bounds > upper_bounds
        if swap_mask.any():
            logger.warning(
                "Swapping %d inverted intervals (lower > upper)",
                swap_mask.sum(),
            )
            lower_bounds[swap_mask], upper_bounds[swap_mask] = (
                upper_bounds[swap_mask],
                lower_bounds[swap_mask],
            )

        # Ensure non-negative
        lower_bounds = np.maximum(lower_bounds, 0)
        upper_bounds = np.maximum(upper_bounds, 0)

        logger.info(
            "Prediction intervals (%.0f%% CI): median width $%,.0f, "
            "median lower $%,.0f, median upper $%,.0f",
            confidence_level * 100,
            float(np.median(upper_bounds - lower_bounds)),
            float(np.median(lower_bounds)),
            float(np.median(upper_bounds)),
        )

        return lower_bounds, upper_bounds

    @staticmethod
    def _get_quantile_bounds(
        confidence_level: float,
        available_models: dict[float, Any],
    ) -> tuple[float, float]:
        """Determine the lower and upper quantile for a confidence level.

        If the exact quantile pair for the requested confidence level
        is not available, selects the closest available pair.

        Args:
            confidence_level: Desired confidence level (e.g. 0.80).
            available_models: Dict of available quantile models.

        Returns:
            Tuple of (lower_quantile, upper_quantile).
        """
        # Check exact matches first
        if confidence_level in _CONFIDENCE_QUANTILE_MAP:
            lower_q, upper_q = _CONFIDENCE_QUANTILE_MAP[confidence_level]
            if lower_q in available_models and upper_q in available_models:
                return lower_q, upper_q

        # Compute needed quantiles from confidence level
        tail = (1.0 - confidence_level) / 2.0
        needed_lower = tail
        needed_upper = 1.0 - tail

        available = sorted(available_models.keys())

        # Find closest available quantiles
        lower_q = min(available, key=lambda q: abs(q - needed_lower))
        upper_q = min(available, key=lambda q: abs(q - needed_upper))

        # Ensure we actually have a spread
        if lower_q >= upper_q:
            lower_q = available[0]
            upper_q = available[-1]

        return lower_q, upper_q

    # ================================================================
    # PERSISTENCE
    # ================================================================

    def save_quantile_models(
        self,
        segment_key: str,
        models: dict[float, Any],
        model_dir: str | Path,
    ) -> None:
        """Save all quantile models for a segment to disk.

        Each quantile model is saved as a separate pickle file with
        the naming convention: ``{segment_key}_q{quantile}.pkl``.

        Args:
            segment_key: Canonical segment key.
            models: Dict of quantile -> trained model.
            model_dir: Directory to save models into.
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        safe_key = segment_key.replace("__", "-").replace(" ", "_")

        for q, model in models.items():
            q_str = f"{q:.2f}".replace(".", "")
            filename = f"{safe_key}_q{q_str}.pkl"
            filepath = model_dir / filename
            joblib.dump(model, filepath)

        logger.info(
            "Saved %d quantile models for '%s' to %s",
            len(models),
            segment_key,
            model_dir,
        )

    def load_quantile_models(
        self,
        segment_key: str,
        model_dir: str | Path,
    ) -> dict[float, Any]:
        """Load all quantile models for a segment from disk.

        Discovers and loads all files matching the naming convention
        ``{segment_key}_q{quantile}.pkl``.

        Args:
            segment_key: Canonical segment key.
            model_dir: Directory containing saved models.

        Returns:
            Dict of quantile (float) -> trained model.

        Raises:
            FileNotFoundError: If no quantile models are found.
        """
        model_dir = Path(model_dir)
        safe_key = segment_key.replace("__", "-").replace(" ", "_")

        pattern = f"{safe_key}_q*.pkl"
        model_files = sorted(model_dir.glob(pattern))

        if not model_files:
            raise FileNotFoundError(
                f"No quantile models found for segment '{segment_key}' "
                f"in {model_dir} (pattern: {pattern})"
            )

        models: dict[float, Any] = {}

        for filepath in model_files:
            # Extract quantile from filename
            # Pattern: safe_key_q0010.pkl -> q = 0.10
            stem = filepath.stem
            q_part = stem.split("_q")[-1]
            try:
                q_value = int(q_part) / 100.0
            except ValueError:
                logger.warning("Cannot parse quantile from filename: %s", filepath)
                continue

            models[q_value] = joblib.load(filepath)

        logger.info(
            "Loaded %d quantile models for '%s': quantiles=%s",
            len(models),
            segment_key,
            sorted(models.keys()),
        )

        return models

    # ================================================================
    # DEFAULT PARAMS
    # ================================================================

    @staticmethod
    def _get_quantile_params() -> dict:
        """Return default LightGBM parameters for quantile regression.

        Similar to the main model defaults but with slightly more
        regularization to prevent quantile crossing.

        Returns:
            Dict of LightGBM parameters (``objective`` and ``alpha``
            are set per-quantile during training).
        """
        return {
            "boosting_type": "gbdt",
            "num_leaves": 50,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 30,
            "lambda_l1": 1.0,
            "lambda_l2": 1.0,
            "verbose": -1,
            "n_estimators": 1500,
        }
