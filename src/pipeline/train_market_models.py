#!/usr/bin/env python3
"""
Train SAR (Sale-to-Assessment Ratio) models using actual MLS sold prices.

Instead of predicting absolute dollar values from scratch, these models predict
the ratio of sold_price / assessed_value. This is fundamentally better because:
  - Assessed values already capture 90%+ of property-specific info
  - The model only learns the market premium/discount (a much simpler task)
  - SAR varies in a tight range (0.7–1.3x) vs absolute prices ($400K–$5M)
  - Neighbourhood trends and property type drive SAR — exactly our signal

Prediction at inference:
  market_estimate = assessed_value × predicted_SAR

Usage:
    python -m src.pipeline.train_market_models
    python -m src.pipeline.train_market_models --n-trials 20
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline.sold_price_enrichment import build_market_training_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("market_trainer")

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_DIR = Path("models/market")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MIN_SAMPLES = 30  # Lower threshold since SAR is easier to learn


# Features that are useful for predicting SAR (market premium/discount).
# We deliberately exclude assessed values and derived value features
# since those are what we're adjusting, not predicting from.
SAR_FEATURES = [
    # Location
    "neighbourhood_code",
    "latitude",
    "longitude",
    "dist_downtown_m",
    "dist_waterfront_m",
    # Transit
    "dist_nearest_transit_m",
    "dist_nearest_skytrain_m",
    "transit_stops_400m",
    "has_skytrain_800m",
    # Amenities
    "dist_nearest_park_m",
    "parks_within_500m",
    # Demographics
    "census_median_income",
    "census_pop_density",
    "census_pct_owner_occupied",
    "census_pct_immigrants",
    "census_pct_university",
    # Property characteristics
    "year_built",
    "effective_age",
    "living_area_sqft",
    "bedrooms",
    "bathrooms",
    "parking_spaces",
    # Market signals
    "days_on_market",
    "list_to_sold_ratio",
    # Environmental
    "in_floodplain",
    "contaminated_sites_500m",
    # STR (short-term rental) density
    "str_count_500m",
    "str_density_per_km2",
    # Value structure (ratios are OK — they indicate property character)
    "land_to_total_ratio",
    "improvement_to_land_ratio",
]


def prepare_sar_features(market_df: pd.DataFrame, property_type: str) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix X and target y (SAR) for a property type.

    Returns (X, y) where y = sold_price / total_assessed_value.
    """
    segment = market_df[market_df["property_type"] == property_type].copy()

    if segment.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    # Compute SAR target
    valid = (
        segment["sold_price"].notna()
        & (segment["sold_price"] > 0)
        & segment["total_assessed_value"].notna()
        & (segment["total_assessed_value"] > 0)
    )
    segment = segment[valid].copy()
    y = segment["sold_price"] / segment["total_assessed_value"]

    # Filter extreme outliers (SAR outside 0.5–2.0 likely data errors)
    reasonable = (y >= 0.5) & (y <= 2.0)
    segment = segment[reasonable]
    y = y[reasonable]

    # Remove IQR outliers within each neighbourhood
    if "neighbourhood_code" in segment.columns:
        keep = pd.Series(True, index=segment.index)
        for hood, grp_idx in segment.groupby("neighbourhood_code").groups.items():
            if len(grp_idx) >= 5:
                grp_sar = y.loc[grp_idx]
                q1 = grp_sar.quantile(0.25)
                q3 = grp_sar.quantile(0.75)
                iqr = q3 - q1
                lo = q1 - 1.5 * iqr
                hi = q3 + 1.5 * iqr
                outliers = grp_sar[(grp_sar < lo) | (grp_sar > hi)].index
                keep.loc[outliers] = False
        n_iqr = (~keep).sum()
        segment = segment[keep]
        y = y[keep]
        if n_iqr > 0:
            logger.info(f"  Removed {n_iqr} IQR outliers within neighbourhoods")

    # Build feature matrix from available columns
    available = [f for f in SAR_FEATURES if f in segment.columns]
    X = segment[available].copy()

    # Convert non-numeric columns
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # Drop features with >60% missing
    null_pcts = X.isnull().mean()
    keep = null_pcts[null_pcts < 0.6].index.tolist()
    X = X[keep]

    logger.info(
        f"  {property_type}: {len(X)} samples, {len(X.columns)} features, "
        f"SAR range: {y.min():.2f}–{y.max():.2f}, median: {y.median():.3f}"
    )

    return X, y


def train_sar_model(
    X: pd.DataFrame,
    y: pd.Series,
    segment_key: str,
    n_trials: int = 30,
) -> dict:
    """Train a LightGBM model to predict SAR.

    Target is raw SAR (not log-transformed) since it's already in a tight range.
    """
    logger.info(f"Training '{segment_key}': {len(X)} samples, {len(X.columns)} features")

    n_folds = min(5, max(2, len(X) // 20))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    folds = list(kf.split(X))

    # Default hyperparameters — conservative for small datasets
    best_params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 20,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 5,
        "verbose": -1,
        "n_estimators": 500,
    }

    # Optuna search
    if n_trials > 0 and len(folds) >= 2:
        train_idx, val_idx = folds[0]
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

        def objective(trial):
            params = {
                "objective": "regression",
                "metric": "mae",
                "boosting_type": "gbdt",
                "verbose": -1,
                "num_leaves": trial.suggest_int("num_leaves", 8, 50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "bagging_freq": 5,
                "min_child_samples": trial.suggest_int("min_child_samples", 3, 30),
                "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 2.0),
                "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 2.0),
            }
            n_est = trial.suggest_int("n_estimators", 100, 1000)

            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

            model = lgb.train(
                params, dtrain,
                num_boost_round=n_est,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            y_pred = model.predict(X_va)
            # MAE on SAR (e.g. 0.05 means off by 5% of assessed value)
            mae = float(np.mean(np.abs(y_pred - y_va)))
            return mae

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params.update(study.best_params)
        logger.info(f"  Optuna best MAE: {study.best_value:.4f} after {n_trials} trials")

    # Cross-validate
    cv_maes = []
    cv_mapes = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

        n_est = best_params.pop("n_estimators", 500)
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
        mae = float(np.mean(np.abs(y_pred - y_va)))
        # MAPE on the final dollar estimate: |predicted_SAR - actual_SAR| / actual_SAR
        mape = float(np.median(np.abs(y_pred - y_va) / y_va) * 100)
        cv_maes.append(mae)
        cv_mapes.append(mape)

    avg_mae = float(np.mean(cv_maes))
    avg_mape = float(np.mean(cv_mapes))

    # Train final model on all data
    n_est_final = best_params.pop("n_estimators", 500)
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

    # Compute neighbourhood-level SAR stats for fallback
    logger.info(f"  CV MAE on SAR: {avg_mae:.4f} (±{np.std(cv_maes):.4f})")
    logger.info(f"  CV MAPE on dollar estimate: {avg_mape:.2f}%")
    logger.info(f"  Median SAR in training data: {y.median():.3f}")
    logger.info(f"  Top features: {', '.join(f'{k}' for k, v in top_features[:5])}")

    return {
        "model": final_model,
        "segment_key": segment_key,
        "target": "sar",
        "cv_mae": avg_mae,
        "cv_mae_std": float(np.std(cv_maes)),
        "cv_mape": avg_mape,
        "median_sar": float(y.median()),
        "mean_sar": float(y.mean()),
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
        "target": "sar",
        "cv_mae": result["cv_mae"],
        "cv_mae_std": result["cv_mae_std"],
        "cv_mape": result["cv_mape"],
        "median_sar": result["median_sar"],
        "mean_sar": result["mean_sar"],
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
    """Main entry point for SAR model training."""
    t0 = time.perf_counter()

    logger.info("=" * 60)
    logger.info("Building market training dataset...")
    logger.info("=" * 60)
    market_df = build_market_training_data()

    if market_df.empty:
        logger.error("No market training data available. Ingest MLS CSVs first.")
        return

    logger.info(f"Total matched sales: {len(market_df)}")

    # Log SAR distribution
    valid_sar = market_df["sale_to_assessment_ratio"].dropna()
    logger.info(
        f"Overall SAR: median={valid_sar.median():.3f}, "
        f"mean={valid_sar.mean():.3f}, "
        f"std={valid_sar.std():.3f}"
    )

    # Train one SAR model per property type
    results = {}
    for ptype in ["condo", "detached", "townhome"]:
        segment = market_df[market_df["property_type"] == ptype]
        if len(segment) < MIN_SAMPLES:
            logger.warning(f"Skipping {ptype}: only {len(segment)} samples (need {MIN_SAMPLES})")
            # Save the median SAR as a simple fallback
            if not segment.empty:
                median_sar = float(
                    (segment["sold_price"] / segment["total_assessed_value"]).median()
                )
                fallback_meta = {
                    "segment_key": f"market_{ptype}",
                    "target": "sar",
                    "fallback": True,
                    "median_sar": median_sar,
                    "n_samples": len(segment),
                    "trained_at": datetime.now().isoformat(),
                }
                meta_path = MODEL_DIR / f"market_{ptype}_metadata.json"
                with open(meta_path, "w") as f:
                    json.dump(fallback_meta, f, indent=2)
                logger.info(f"  Saved fallback SAR for {ptype}: {median_sar:.3f} ({len(segment)} samples)")
            continue

        logger.info("=" * 60)
        logger.info(f"Training SAR model: {ptype} ({len(segment)} sales)")
        logger.info("=" * 60)

        X, y = prepare_sar_features(market_df, ptype)

        if len(X) < MIN_SAMPLES:
            logger.warning(f"Only {len(X)} valid samples for {ptype} after prep")
            continue

        segment_key = f"market_{ptype}"
        result = train_sar_model(X, y, segment_key, n_trials=n_trials)
        save_market_model(result)
        results[segment_key] = result

    # Summary
    elapsed = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info("SAR MODEL TRAINING SUMMARY")
    logger.info("=" * 60)
    for key, r in results.items():
        logger.info(
            f"  {key}: SAR MAE={r['cv_mae']:.4f}, MAPE={r['cv_mape']:.2f}%, "
            f"median SAR={r['median_sar']:.3f}, {r['n_samples']} samples"
        )
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info(f"Models saved to: {MODEL_DIR}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train SAR models from MLS sold data")
    parser.add_argument("--n-trials", type=int, default=30, help="Optuna trials per model")
    args = parser.parse_args()

    run_market_training(n_trials=args.n_trials)


if __name__ == "__main__":
    main()
