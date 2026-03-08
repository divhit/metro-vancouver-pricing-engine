#!/usr/bin/env python3
"""
Train market price models using actual MLS sold prices.

Instead of predicting assessed values (217K records), these models predict
actual sale prices using matched MLS data (~800 records, growing daily).

Strategy:
- Train one model per property type (condo, townhome, detached) city-wide
- Use all BC Assessment features + actual MLS features (bedrooms, floor_area)
- Models are saved separately in models/market/ directory
- As daily data accumulates, retrain to improve accuracy

Usage:
    python -m src.pipeline.train_market_models
    python -m src.pipeline.train_market_models --n-trials 20  # fewer Optuna trials
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline.sold_price_enrichment import build_market_training_data
from src.features.building_footprint import BuildingFootprintEstimator
from src.features.feature_builder import FeatureBuilder
from src.features.feature_registry import PropertyType
from src.features.spatial_features import SpatialFeatureComputer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("market_trainer")

# Suppress Optuna verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_DIR = Path("models/market")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Minimum samples to train a model
MIN_SAMPLES = 50


def train_market_model(
    X: pd.DataFrame,
    y: pd.Series,
    segment_key: str,
    n_trials: int = 30,
) -> dict:
    """Train a LightGBM model for market price prediction.

    Returns dict with model, metrics, and metadata.
    """
    logger.info(f"Training '{segment_key}': {len(X)} samples, {len(X.columns)} features")

    # K-Fold CV (no spatial folds needed — data is too small)
    n_folds = min(5, max(2, len(X) // 30))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    folds = list(kf.split(X))

    # Optuna hyperparameter search
    best_params = {
        "objective": "regression",
        "metric": "mape",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 10,
        "verbose": -1,
        "n_estimators": 1000,
    }

    if n_trials > 0 and len(folds) >= 2:
        train_idx, val_idx = folds[0]
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

        def objective(trial):
            params = {
                "objective": "regression",
                "metric": "mape",
                "boosting_type": "gbdt",
                "verbose": -1,
                "num_leaves": trial.suggest_int("num_leaves", 15, 80),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "bagging_freq": 5,
                "min_child_samples": trial.suggest_int("min_child_samples", 3, 50),
                "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 3.0),
                "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 3.0),
            }
            n_est = trial.suggest_int("n_estimators", 300, 2000)

            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

            model = lgb.train(
                params, dtrain,
                num_boost_round=n_est,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            y_pred = model.predict(X_va)
            y_true_orig = np.expm1(y_va)
            y_pred_orig = np.expm1(y_pred)
            mape = float(np.median(np.abs(y_true_orig - y_pred_orig) / y_true_orig) * 100)
            return mape

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params.update(study.best_params)
        logger.info(f"  Optuna best MAPE: {study.best_value:.2f}% after {n_trials} trials")

    # Cross-validate with best params
    cv_mapes = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

        n_est = best_params.pop("n_estimators", 1000)
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

        model_cv = lgb.train(
            best_params, dtrain,
            num_boost_round=n_est,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        best_params["n_estimators"] = n_est

        y_pred = model_cv.predict(X_va)
        y_true_orig = np.expm1(y_va)
        y_pred_orig = np.expm1(y_pred)
        mape = float(np.median(np.abs(y_true_orig - y_pred_orig) / y_true_orig) * 100)
        cv_mapes.append(mape)

    avg_mape = float(np.mean(cv_mapes))
    std_mape = float(np.std(cv_mapes))

    # Train final model on all data
    n_est_final = best_params.pop("n_estimators", 1000)
    dtrain_full = lgb.Dataset(X, label=y)
    final_model = lgb.train(
        best_params, dtrain_full,
        num_boost_round=n_est_final,
    )
    best_params["n_estimators"] = n_est_final

    # Feature importance
    feature_importances = dict(
        zip(X.columns, final_model.feature_importance(importance_type="gain"))
    )
    top_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]

    logger.info(f"  CV MAPE: {avg_mape:.2f}% (±{std_mape:.2f}%)")
    logger.info(f"  Top features: {', '.join(f'{k}' for k, v in top_features[:5])}")

    return {
        "model": final_model,
        "segment_key": segment_key,
        "cv_mape": avg_mape,
        "cv_mape_std": std_mape,
        "n_samples": len(X),
        "n_features": len(X.columns),
        "n_folds": n_folds,
        "hyperparameters": best_params,
        "feature_importances": feature_importances,
        "feature_names": list(X.columns),
    }


def save_market_model(result: dict) -> Path:
    """Save model and metadata to models/market/."""
    key = result["segment_key"]
    model_path = MODEL_DIR / f"{key}.pkl"
    meta_path = MODEL_DIR / f"{key}_metadata.json"

    joblib.dump(result["model"], model_path)

    metadata = {
        "segment_key": key,
        "target": "sold_price",
        "cv_mape": result["cv_mape"],
        "cv_mape_std": result["cv_mape_std"],
        "n_samples": result["n_samples"],
        "n_features": result["n_features"],
        "n_folds": result["n_folds"],
        "hyperparameters": {k: v for k, v in result["hyperparameters"].items()
                           if not isinstance(v, (np.integer, np.floating))
                           or not np.isnan(v)},
        "feature_importances": {k: float(v) for k, v in result["feature_importances"].items()},
        "feature_names": result["feature_names"],
        "trained_at": datetime.now().isoformat(),
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"  Saved: {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")
    return model_path


def run_market_training(n_trials: int = 30):
    """Main entry point for market model training."""
    t0 = time.perf_counter()

    # Build training data: MLS sold + BC Assessment features
    logger.info("=" * 60)
    logger.info("Building market training dataset...")
    logger.info("=" * 60)
    market_df = build_market_training_data()

    if market_df.empty:
        logger.error("No market training data available. Ingest MLS CSVs first.")
        return

    logger.info(f"Total matched sales: {len(market_df)}")

    # Initialize feature builder with sold_price as target
    spatial = SpatialFeatureComputer()
    footprint = BuildingFootprintEstimator()
    feature_builder = FeatureBuilder(
        spatial_computer=spatial,
        footprint_estimator=footprint,
        phase=1,
        mls_available=True,  # We have MLS data for these records
        target_column="sold_price",
    )

    # Train one model per property type
    results = {}
    type_map = {
        "condo": PropertyType.CONDO,
        "townhome": PropertyType.TOWNHOME,
        "detached": PropertyType.DETACHED,
    }

    for ptype_str, ptype_enum in type_map.items():
        segment_df = market_df[market_df["property_type"] == ptype_str].copy()

        if len(segment_df) < MIN_SAMPLES:
            logger.warning(
                f"Skipping {ptype_str}: only {len(segment_df)} samples (need {MIN_SAMPLES})"
            )
            continue

        logger.info("=" * 60)
        logger.info(f"Training market model: {ptype_str} ({len(segment_df)} sales)")
        logger.info("=" * 60)

        try:
            X, y = feature_builder.build_features_batch(segment_df, property_type=ptype_enum)
        except Exception as e:
            logger.error(f"Feature building failed for {ptype_str}: {e}")
            continue

        if len(X) < MIN_SAMPLES:
            logger.warning(f"Only {len(X)} valid samples for {ptype_str} after feature building")
            continue

        segment_key = f"market_{ptype_str}"
        result = train_market_model(X, y, segment_key, n_trials=n_trials)
        save_market_model(result)
        results[segment_key] = result

    # Summary
    elapsed = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info("MARKET MODEL TRAINING SUMMARY")
    logger.info("=" * 60)
    for key, r in results.items():
        logger.info(
            f"  {key}: MAPE={r['cv_mape']:.2f}% (±{r['cv_mape_std']:.2f}%), "
            f"{r['n_samples']} samples, {r['n_features']} features"
        )
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info(f"Models saved to: {MODEL_DIR}")

    # Compare with assessed value models
    logger.info("")
    logger.info("COMPARISON: Market vs Assessment models")
    assessed_model_dir = Path("models")
    for key in results:
        ptype = key.replace("market_", "")
        # Find any assessed value model for this property type
        assessed_metas = list(assessed_model_dir.glob(f"*-{ptype}_metadata.json"))
        if assessed_metas:
            with open(assessed_metas[0]) as f:
                assessed_meta = json.load(f)
            assessed_mape = assessed_meta.get("validation_mape", assessed_meta.get("metrics", {}).get("cv_mape", "?"))
            logger.info(
                f"  {ptype}: Market MAPE={results[key]['cv_mape']:.2f}% vs "
                f"Assessment MAPE={assessed_mape}%"
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="Train market price models from MLS sold data")
    parser.add_argument("--n-trials", type=int, default=30, help="Optuna trials per model")
    args = parser.parse_args()

    run_market_training(n_trials=args.n_trials)


if __name__ == "__main__":
    main()
