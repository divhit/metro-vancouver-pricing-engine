"""
LightGBM model training pipeline.

Trains separate models per (sub_region, property_type) segment with:
- Spatial k-fold cross-validation (prevents geographic data leakage)
- Optuna hyperparameter tuning (50 trials per segment)
- SHAP value computation for explainability
- Model persistence with metadata

Target: log(total_assessed_value)
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.model_selection import KFold

from src.features.feature_builder import FeatureBuilder
from src.features.feature_registry import PropertyType
from src.models.subregions import SubRegionEngine
from src.models.types import TrainingResult

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose trial logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelTrainer:
    """LightGBM training pipeline with spatial cross-validation and Optuna.

    Trains one LightGBM model per segment (local area x property type).
    Each segment model is independently tuned via Optuna, validated with
    spatial cross-validation, and explained with SHAP values.

    Segments with fewer properties than ``SubRegionEngine.min_segment_size``
    are skipped; those properties fall back to a broader segment model
    at prediction time.

    Usage::

        trainer = ModelTrainer(model_dir="models")
        results = trainer.train_all_segments(
            properties_df, feature_builder, subregion_engine, n_optuna_trials=50
        )
        trainer.save_model("Kitsilano__condo", model, results["Kitsilano__condo"])

    Args:
        model_dir: Directory for saving trained models and metadata.
        random_state: Random seed for reproducibility across K-Means,
            KFold, LightGBM, and Optuna.
    """

    def __init__(
        self,
        model_dir: str = "models",
        random_state: int = 42,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

        logger.info(
            "ModelTrainer initialized: model_dir=%s, random_state=%d",
            self.model_dir,
            random_state,
        )

    # ================================================================
    # TRAIN ALL SEGMENTS
    # ================================================================

    def train_all_segments(
        self,
        properties_df: pd.DataFrame,
        feature_builder: FeatureBuilder,
        subregion_engine: SubRegionEngine,
        n_optuna_trials: int = 50,
    ) -> dict[str, TrainingResult]:
        """Train models for all qualifying segments.

        Iterates over every (neighbourhood_code, property_type) segment,
        checks whether it has enough data, and trains an optimized model
        for each qualifying segment.

        Args:
            properties_df: Enriched property universe DataFrame with
                ``neighbourhood_code``, ``property_type``, and
                ``total_assessed_value`` columns.
            feature_builder: Initialized FeatureBuilder for computing
                (X, y) feature matrices.
            subregion_engine: Initialized SubRegionEngine for segment
                statistics and fallback decisions.
            n_optuna_trials: Number of Optuna hyperparameter trials per
                segment.

        Returns:
            Dict mapping segment_key to TrainingResult for each trained
            segment. Segments with too few samples are excluded.
        """
        t0 = time.perf_counter()
        results: dict[str, TrainingResult] = {}

        segment_stats = subregion_engine.get_segment_stats(properties_df)
        if segment_stats.empty:
            logger.warning("No segment stats available; cannot train models")
            return results

        n_qualifying = 0
        n_skipped = 0

        for _, row in segment_stats.iterrows():
            segment_key = row["segment_key"]

            if not subregion_engine.should_use_segment_model(
                segment_key, segment_stats
            ):
                logger.info(
                    "Skipping segment '%s': %d samples < min %d",
                    segment_key,
                    int(row["count"]),
                    subregion_engine.min_segment_size,
                )
                n_skipped += 1
                continue

            # Parse segment key to get area and property type
            parts = segment_key.split("__")
            if len(parts) != 2:
                logger.warning("Malformed segment key: %s", segment_key)
                continue

            area_code, ptype_str = parts

            # Filter properties to this segment
            if area_code == "citywide":
                segment_mask = pd.Series(True, index=properties_df.index)
            else:
                segment_mask = properties_df["neighbourhood_code"] == area_code

            if ptype_str != "all":
                segment_mask &= properties_df["property_type"] == ptype_str

            segment_df = properties_df[segment_mask].copy()

            if len(segment_df) < subregion_engine.min_segment_size:
                logger.info(
                    "Segment '%s' filtered to %d rows; skipping",
                    segment_key,
                    len(segment_df),
                )
                n_skipped += 1
                continue

            # Build features for this segment
            try:
                ptype_enum = None
                for pt in PropertyType:
                    if pt.value == ptype_str:
                        ptype_enum = pt
                        break

                X, y = feature_builder.build_features_batch(
                    segment_df, property_type=ptype_enum
                )
            except (ValueError, KeyError) as exc:
                logger.error(
                    "Feature building failed for segment '%s': %s",
                    segment_key,
                    exc,
                )
                n_skipped += 1
                continue

            if len(X) < subregion_engine.min_segment_size:
                logger.info(
                    "Segment '%s' has %d valid samples after feature building; skipping",
                    segment_key,
                    len(X),
                )
                n_skipped += 1
                continue

            # Train the segment model
            logger.info(
                "Training segment '%s': %d samples, %d features",
                segment_key,
                len(X),
                len(X.columns),
            )

            try:
                result = self.train_segment(segment_key, X, y, n_trials=n_optuna_trials)
                results[segment_key] = result

                # Persist model and metadata
                self.save_model(segment_key, result.model, result)
                n_qualifying += 1

                logger.info(
                    "Segment '%s' trained: MAPE=%.2f%%, %d features, best of %d trials",
                    segment_key,
                    result.validation_mape,
                    len(result.feature_importances),
                    n_optuna_trials,
                )
            except Exception as exc:
                logger.error(
                    "Training failed for segment '%s': %s",
                    segment_key,
                    exc,
                    exc_info=True,
                )
                n_skipped += 1

        elapsed = time.perf_counter() - t0
        logger.info(
            "Training complete: %d segments trained, %d skipped in %.1fs",
            n_qualifying,
            n_skipped,
            elapsed,
        )

        return results

    # ================================================================
    # TRAIN SINGLE SEGMENT
    # ================================================================

    def train_segment(
        self,
        segment_key: str,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
    ) -> TrainingResult:
        """Train an optimized LightGBM model for a single segment.

        Pipeline:
          1. Create spatial cross-validation folds
          2. Run Optuna hyperparameter search
          3. Train final model with best params on full data
          4. Compute SHAP values for explainability

        Args:
            segment_key: Canonical segment key (e.g. 'Kitsilano__condo').
            X: Feature matrix (output of FeatureBuilder).
            y: Log-transformed target (log of total_assessed_value).
            n_trials: Number of Optuna hyperparameter trials.

        Returns:
            TrainingResult with model, metrics, SHAP values, and params.
        """
        t0 = time.perf_counter()

        # 1. Create spatial folds
        folds = self._create_spatial_folds(X)

        # 2. Run Optuna hyperparameter search
        best_params = self._get_default_params()
        best_mape = float("inf")

        if n_trials > 0 and len(folds) > 1:
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
            )

            # Use the first fold for Optuna tuning for efficiency
            train_idx, val_idx = folds[0]
            X_train_opt = X.iloc[train_idx]
            y_train_opt = y.iloc[train_idx]
            X_val_opt = X.iloc[val_idx]
            y_val_opt = y.iloc[val_idx]

            study.optimize(
                lambda trial: self._optuna_objective(
                    trial, X_train_opt, y_train_opt, X_val_opt, y_val_opt
                ),
                n_trials=n_trials,
                show_progress_bar=False,
            )

            best_params.update(study.best_params)
            best_mape = study.best_value

            logger.info(
                "Optuna search complete for '%s': best MAPE=%.3f%% after %d trials",
                segment_key,
                best_mape,
                n_trials,
            )

        # Cross-validate with best params to get reliable MAPE estimate
        cv_mapes = []
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

            n_estimators = best_params.pop("n_estimators", 2000)

            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

            model_cv = lgb.train(
                best_params,
                dtrain,
                num_boost_round=n_estimators,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )

            best_params["n_estimators"] = n_estimators

            y_pred_val = model_cv.predict(X_va)
            # MAPE on log scale -> convert to percentage
            mape = float(np.median(np.abs(np.expm1(y_va) - np.expm1(y_pred_val)) / np.expm1(y_va)) * 100)
            cv_mapes.append(mape)

        avg_cv_mape = float(np.mean(cv_mapes)) if cv_mapes else best_mape

        # 3. Train final model on all data with best params
        n_estimators_final = best_params.pop("n_estimators", 2000)
        dtrain_full = lgb.Dataset(X, label=y)

        final_model = lgb.train(
            best_params,
            dtrain_full,
            num_boost_round=n_estimators_final,
        )

        best_params["n_estimators"] = n_estimators_final

        # 4. Compute SHAP values
        shap_importances = self._compute_shap_values(final_model, X)

        # Also get LightGBM native feature importance
        feature_importances = dict(
            zip(X.columns, final_model.feature_importance(importance_type="gain"))
        )

        # Compile metrics
        y_pred_full = final_model.predict(X)
        train_mape = float(
            np.median(np.abs(np.expm1(y) - np.expm1(y_pred_full)) / np.expm1(y)) * 100
        )

        metrics = {
            "train_mape": train_mape,
            "cv_mape": avg_cv_mape,
            "cv_mape_std": float(np.std(cv_mapes)) if cv_mapes else 0.0,
            "n_folds": len(folds),
            "n_features": len(X.columns),
            "n_samples": len(X),
        }

        elapsed = time.perf_counter() - t0
        logger.info(
            "Segment '%s' training complete: CV MAPE=%.2f%% (std=%.2f%%) in %.1fs",
            segment_key,
            avg_cv_mape,
            metrics["cv_mape_std"],
            elapsed,
        )

        return TrainingResult(
            segment_key=segment_key,
            model=final_model,
            metrics=metrics,
            feature_importances=feature_importances,
            shap_values=shap_importances,
            training_samples=len(X),
            validation_mape=avg_cv_mape,
            hyperparameters=best_params,
        )

    # ================================================================
    # SPATIAL CROSS-VALIDATION
    # ================================================================

    def _create_spatial_folds(
        self,
        X: pd.DataFrame,
        n_folds: int = 5,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Create geographically-separated cross-validation folds.

        If the feature matrix includes ``neighbourhood_code``, the 22
        local areas are grouped into ``n_folds`` balanced groups ensuring
        a mix of East Side and West Side areas in each fold. Each fold
        holds out one group, preventing geographic data leakage.

        If ``neighbourhood_code`` is absent, falls back to random KFold.

        Args:
            X: Feature DataFrame.
            n_folds: Number of cross-validation folds.

        Returns:
            List of (train_indices, val_indices) tuples.
        """
        if "neighbourhood_code" in X.columns:
            unique_areas = X["neighbourhood_code"].dropna().unique()

            if len(unique_areas) >= n_folds:
                # Group areas into folds, attempting geographic balance
                # West Side areas (higher-value, western geography)
                west_side = {
                    "ARBUTUS-RIDGE", "DUNBAR-SOUTHLANDS", "KERRISDALE",
                    "KITSILANO", "SHAUGHNESSY", "SOUTH CAMBIE",
                    "WEST END", "WEST POINT GREY", "FAIRVIEW",
                    "MARPOLE", "OAKRIDGE",
                }
                # East Side areas
                east_side = {
                    "DOWNTOWN", "GRANDVIEW-WOODLAND", "HASTINGS-SUNRISE",
                    "KENSINGTON-CEDAR COTTAGE", "KILLARNEY",
                    "MOUNT PLEASANT", "RENFREW-COLLINGWOOD", "RILEY PARK",
                    "STRATHCONA", "SUNSET", "VICTORIA-FRASERVIEW",
                }

                # Split areas into west and east, then interleave into folds
                west_areas = [a for a in unique_areas if a in west_side]
                east_areas = [a for a in unique_areas if a in east_side]
                other_areas = [
                    a for a in unique_areas
                    if a not in west_side and a not in east_side
                ]

                # Distribute evenly across folds
                all_ordered = []
                for i in range(max(len(west_areas), len(east_areas), len(other_areas))):
                    if i < len(west_areas):
                        all_ordered.append(west_areas[i])
                    if i < len(east_areas):
                        all_ordered.append(east_areas[i])
                    if i < len(other_areas):
                        all_ordered.append(other_areas[i])

                fold_assignments = {}
                for idx, area in enumerate(all_ordered):
                    fold_assignments[area] = idx % n_folds

                folds = []
                for fold_id in range(n_folds):
                    holdout_areas = {
                        area for area, fid in fold_assignments.items()
                        if fid == fold_id
                    }
                    val_mask = X["neighbourhood_code"].isin(holdout_areas)
                    train_idx = np.where(~val_mask)[0]
                    val_idx = np.where(val_mask)[0]

                    if len(val_idx) > 0 and len(train_idx) > 0:
                        folds.append((train_idx, val_idx))

                if len(folds) >= 2:
                    logger.info(
                        "Created %d spatial folds from %d areas "
                        "(geographic separation)",
                        len(folds),
                        len(unique_areas),
                    )
                    return folds

        # Fallback: random KFold
        logger.info(
            "Using random KFold (%d folds) -- no neighbourhood_code available",
            n_folds,
        )
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        return list(kf.split(X))

    # ================================================================
    # OPTUNA OBJECTIVE
    # ================================================================

    def _optuna_objective(
        self,
        trial: optuna.Trial,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> float:
        """Optuna objective function for LightGBM hyperparameter tuning.

        Samples hyperparameters from Optuna, trains a LightGBM model
        with early stopping on the validation set, and returns the
        validation MAPE.

        Args:
            trial: Optuna trial object for parameter sampling.
            X_train: Training feature matrix.
            y_train: Training target (log-transformed values).
            X_val: Validation feature matrix.
            y_val: Validation target (log-transformed values).

        Returns:
            Validation MAPE (percentage) -- lower is better.
        """
        params = {
            "objective": "regression",
            "metric": "mape",
            "boosting_type": "gbdt",
            "verbose": -1,
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 5,
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
        }
        n_estimators = trial.suggest_int("n_estimators", 500, 3000)

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        y_pred = model.predict(X_val)

        # Compute MAPE in original value space
        y_true_orig = np.expm1(y_val)
        y_pred_orig = np.expm1(y_pred)

        mape = float(
            np.median(np.abs(y_true_orig - y_pred_orig) / y_true_orig) * 100
        )

        return mape

    # ================================================================
    # DEFAULT PARAMS
    # ================================================================

    @staticmethod
    def _get_default_params() -> dict:
        """Return default LightGBM hyperparameters.

        These serve as a reasonable baseline when Optuna tuning is
        skipped or as the starting point for the search space.

        Returns:
            Dict of LightGBM parameters.
        """
        return {
            "objective": "regression",
            "metric": "mape",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "verbose": -1,
            "n_estimators": 2000,
        }

    # ================================================================
    # SHAP VALUES
    # ================================================================

    @staticmethod
    def _compute_shap_values(model: Any, X: pd.DataFrame) -> dict[str, float]:
        """Compute mean absolute SHAP values for feature importance.

        Uses SHAP's TreeExplainer, which is exact and efficient for
        tree-based models like LightGBM.

        Args:
            model: Trained LightGBM Booster.
            X: Feature matrix used for SHAP computation. A sample is
                taken if the dataset is very large.

        Returns:
            Dict mapping feature_name to mean(|SHAP value|), sorted
            by importance descending.
        """
        # Sample for performance on large datasets
        if len(X) > 5000:
            X_sample = X.sample(5000, random_state=42)
        else:
            X_sample = X

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            importance_dict = dict(zip(X_sample.columns, mean_abs_shap))

            # Sort by importance descending
            importance_dict = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )

            return importance_dict
        except Exception as exc:
            logger.warning("SHAP computation failed: %s", exc)
            return {}

    # ================================================================
    # MODEL PERSISTENCE
    # ================================================================

    def save_model(
        self,
        segment_key: str,
        model: Any,
        training_result: TrainingResult,
    ) -> None:
        """Save a trained model and its metadata to disk.

        Creates two files per segment:
          - ``{segment_key}.pkl``: serialized model (joblib)
          - ``{segment_key}_metadata.json``: training metadata

        Args:
            segment_key: Canonical segment key.
            model: Trained LightGBM model.
            training_result: TrainingResult with metrics and params.
        """
        # Sanitize segment key for filesystem (replace __ with -)
        safe_key = segment_key.replace("__", "-").replace(" ", "_")

        model_path = self.model_dir / f"{safe_key}.pkl"
        meta_path = self.model_dir / f"{safe_key}_metadata.json"

        # Save model
        joblib.dump(model, model_path)

        # Save metadata
        metadata = {
            "segment_key": segment_key,
            "training_samples": training_result.training_samples,
            "validation_mape": training_result.validation_mape,
            "metrics": training_result.metrics,
            "hyperparameters": training_result.hyperparameters,
            "feature_importances": {
                k: float(v) for k, v in training_result.feature_importances.items()
            },
            "training_timestamp": training_result.training_timestamp.isoformat(),
        }

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            "Saved model for '%s' to %s (%.1f KB)",
            segment_key,
            model_path,
            model_path.stat().st_size / 1024,
        )

    def load_model(self, segment_key: str) -> tuple[Any, dict]:
        """Load a trained model and its metadata from disk.

        Args:
            segment_key: Canonical segment key.

        Returns:
            Tuple of (model, metadata_dict).

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        safe_key = segment_key.replace("__", "-").replace(" ", "_")

        model_path = self.model_dir / f"{safe_key}.pkl"
        meta_path = self.model_dir / f"{safe_key}_metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"No model found for segment '{segment_key}' at {model_path}"
            )

        model = joblib.load(model_path)

        metadata = {}
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)

        logger.info("Loaded model for segment '%s' from %s", segment_key, model_path)

        return model, metadata
