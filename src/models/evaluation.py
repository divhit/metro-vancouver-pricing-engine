"""
Model evaluation and backtesting framework.

Comprehensive evaluation with spatial awareness, property-type breakdowns,
and comparison against naive benchmarks.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features.feature_builder import FeatureBuilder
from src.features.feature_registry import PropertyType
from src.models.subregions import SubRegionEngine
from src.models.trainer import ModelTrainer
from src.models.types import BacktestResult, TrainingResult

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Backtesting and evaluation framework for the pricing engine.

    Evaluates model accuracy using temporal holdout (train on all years
    except the holdout year, predict the holdout year). Computes a full
    suite of error metrics overall, by property type, and by local area.

    Results are compared against a naive benchmark (median assessed value
    for the same neighbourhood + property type) to quantify the value
    added by the ML model.

    Usage::

        evaluator = ModelEvaluator()
        result = evaluator.backtest(
            properties_df, feature_builder, subregion_engine,
            holdout_year=2024,
        )
        report = evaluator.generate_evaluation_report(result)
        print(report)

    """

    def __init__(self, model_dir: str = "models", random_state: int = 42) -> None:
        self.model_dir = model_dir
        self.random_state = random_state

    # ================================================================
    # BACKTESTING
    # ================================================================

    def backtest(
        self,
        properties_df: pd.DataFrame,
        feature_builder: FeatureBuilder,
        subregion_engine: SubRegionEngine,
        holdout_year: int,
    ) -> BacktestResult:
        """Run a temporal backtest on the property universe.

        Splits the data by assessment year: all years before the holdout
        year are used for training, and the holdout year is used for
        evaluation. This simulates the real-world scenario of predicting
        current-year values using historical data.

        Args:
            properties_df: Multi-year property universe DataFrame with
                ``tax_assessment_year`` column.
            feature_builder: Initialized FeatureBuilder.
            subregion_engine: Initialized SubRegionEngine.
            holdout_year: Year to hold out for testing.

        Returns:
            BacktestResult with overall, per-type, per-area metrics
            and benchmark comparison.
        """
        t0 = time.perf_counter()

        if "tax_assessment_year" not in properties_df.columns:
            raise ValueError(
                "Cannot backtest: 'tax_assessment_year' column not found. "
                "Use PropertyUniverseBuilder.build_multi_year_panel() to "
                "include multiple years."
            )

        # Split train / test
        train_df = properties_df[
            properties_df["tax_assessment_year"] < holdout_year
        ].copy()
        test_df = properties_df[
            properties_df["tax_assessment_year"] == holdout_year
        ].copy()

        logger.info(
            "Backtest: holdout_year=%d, train=%d properties (%d years), "
            "test=%d properties",
            holdout_year,
            len(train_df),
            train_df["tax_assessment_year"].nunique(),
            len(test_df),
        )

        if len(train_df) == 0:
            raise ValueError(
                f"No training data: no properties with year < {holdout_year}"
            )
        if len(test_df) == 0:
            raise ValueError(
                f"No test data: no properties with year == {holdout_year}"
            )

        # Train models on training set
        trainer = ModelTrainer(
            model_dir=self.model_dir, random_state=self.random_state
        )
        training_results = trainer.train_all_segments(
            train_df,
            feature_builder,
            subregion_engine,
            n_optuna_trials=20,  # Fewer trials for backtesting speed
        )

        if not training_results:
            logger.warning(
                "No segment models were trained; backtest will have no predictions"
            )

        # Predict on test set
        predictions = self._predict_test_set(
            test_df, feature_builder, subregion_engine, trainer, training_results
        )

        if predictions.empty:
            logger.warning("No predictions generated for holdout year")
            return BacktestResult(
                holdout_year=holdout_year,
                overall_metrics={},
                by_property_type={},
                by_local_area={},
                by_segment={},
                benchmark_comparison={},
                predictions_df=predictions,
            )

        # Compute metrics
        y_true = predictions["actual_value"].values
        y_pred = predictions["predicted_value"].values

        overall_metrics = self.compute_metrics(y_true, y_pred)

        # Per property type
        by_property_type = {}
        if "property_type" in predictions.columns:
            for ptype, group in predictions.groupby("property_type"):
                by_property_type[ptype] = self.compute_metrics(
                    group["actual_value"].values,
                    group["predicted_value"].values,
                )

        # Per local area
        by_local_area = {}
        if "neighbourhood_code" in predictions.columns:
            for area, group in predictions.groupby("neighbourhood_code"):
                by_local_area[area] = self.compute_metrics(
                    group["actual_value"].values,
                    group["predicted_value"].values,
                )

        # Per segment
        by_segment = {}
        if "segment_key" in predictions.columns:
            for seg, group in predictions.groupby("segment_key"):
                by_segment[seg] = self.compute_metrics(
                    group["actual_value"].values,
                    group["predicted_value"].values,
                )

        # Benchmark comparison
        benchmark_comparison = self.compare_to_benchmark(
            y_true, y_pred, predictions
        )

        elapsed = time.perf_counter() - t0
        logger.info(
            "Backtest complete: holdout_year=%d, MAPE=%.2f%%, "
            "benchmark improvement=%.1f%%, elapsed=%.1fs",
            holdout_year,
            overall_metrics.get("mape", float("nan")),
            benchmark_comparison.get("improvement_pct", 0.0),
            elapsed,
        )

        return BacktestResult(
            holdout_year=holdout_year,
            overall_metrics=overall_metrics,
            by_property_type=by_property_type,
            by_local_area=by_local_area,
            by_segment=by_segment,
            benchmark_comparison=benchmark_comparison,
            predictions_df=predictions,
        )

    def _predict_test_set(
        self,
        test_df: pd.DataFrame,
        feature_builder: FeatureBuilder,
        subregion_engine: SubRegionEngine,
        trainer: ModelTrainer,
        training_results: dict[str, TrainingResult],
    ) -> pd.DataFrame:
        """Generate predictions for the test set using trained models.

        For each property in the test set, determines the appropriate
        segment model and generates a prediction. If the property's
        specific segment model was not trained, walks up the fallback
        hierarchy until a model is found.

        Args:
            test_df: Test set DataFrame.
            feature_builder: FeatureBuilder for computing features.
            subregion_engine: SubRegionEngine for segment assignment.
            trainer: ModelTrainer with trained models.
            training_results: Dict of segment_key -> TrainingResult.

        Returns:
            DataFrame with actual_value, predicted_value, and metadata.
        """
        records = []

        # Group test properties by segment for batch prediction
        segment_groups: dict[str, list[int]] = {}

        for idx, row in test_df.iterrows():
            segment_key = subregion_engine.assign_segment(row.to_dict())

            # Walk fallback hierarchy to find a trained segment
            effective_segment = segment_key
            while effective_segment not in training_results:
                fallback = subregion_engine.get_fallback_segment(effective_segment)
                if fallback == effective_segment:
                    break  # At the top of the hierarchy
                effective_segment = fallback

            if effective_segment not in segment_groups:
                segment_groups[effective_segment] = []
            segment_groups[effective_segment].append(idx)

        for segment_key, indices in segment_groups.items():
            if segment_key not in training_results:
                logger.warning(
                    "No model available for segment '%s' (%d properties); skipping",
                    segment_key,
                    len(indices),
                )
                continue

            segment_test = test_df.loc[indices]
            result = training_results[segment_key]

            # Build features for this segment's test data
            try:
                ptype_str = segment_key.split("__")[-1] if "__" in segment_key else "all"
                ptype_enum = None
                for pt in PropertyType:
                    if pt.value == ptype_str:
                        ptype_enum = pt
                        break

                X_test, y_test = feature_builder.build_features_batch(
                    segment_test, property_type=ptype_enum
                )
            except (ValueError, KeyError) as exc:
                logger.warning(
                    "Feature building failed for test segment '%s': %s",
                    segment_key,
                    exc,
                )
                continue

            # Align features with the trained model
            model = result.model
            model_features = model.feature_name()
            missing_features = set(model_features) - set(X_test.columns)
            extra_features = set(X_test.columns) - set(model_features)

            # Add missing features as NaN
            for feat in missing_features:
                X_test[feat] = np.nan

            # Select only features the model expects, in the right order
            X_test = X_test[model_features]

            # Predict
            y_pred_log = model.predict(X_test)
            y_pred = np.expm1(y_pred_log)
            y_actual = np.expm1(y_test)

            for i, (test_idx, pred_val, actual_val) in enumerate(
                zip(X_test.index, y_pred, y_actual)
            ):
                row = test_df.loc[test_idx] if test_idx in test_df.index else segment_test.iloc[i]
                records.append({
                    "pid": row.get("pid", ""),
                    "actual_value": float(actual_val),
                    "predicted_value": float(pred_val),
                    "property_type": row.get("property_type", ""),
                    "neighbourhood_code": row.get("neighbourhood_code", ""),
                    "segment_key": segment_key,
                })

        predictions_df = pd.DataFrame(records)

        logger.info(
            "Test set predictions: %d out of %d properties predicted (%.1f%%)",
            len(predictions_df),
            len(test_df),
            len(predictions_df) / max(len(test_df), 1) * 100,
        )

        return predictions_df

    # ================================================================
    # METRICS COMPUTATION
    # ================================================================

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        segment_labels: Optional[np.ndarray] = None,
    ) -> dict:
        """Compute a comprehensive suite of evaluation metrics.

        All dollar-denominated metrics are computed in original value
        space (not log-transformed). Percentage metrics use absolute
        percentage error relative to the true value.

        Args:
            y_true: Actual property values (dollar amounts).
            y_pred: Predicted property values (dollar amounts).
            segment_labels: Optional array of segment keys for per-segment
                metric breakdown.

        Returns:
            Dict with the following keys:
              - mape: Median absolute percentage error
              - mean_ape: Mean absolute percentage error
              - mae: Mean absolute error (dollars)
              - median_ae: Median absolute error (dollars)
              - r2: R-squared score
              - rmse: Root mean squared error (dollars)
              - hit_rate_5pct: % within 5% of actual
              - hit_rate_10pct: % within 10%
              - hit_rate_15pct: % within 15%
              - bias: Systematic over/under-estimation (%)
              - n_samples: Sample count
              - by_segment: per-segment metrics (if segment_labels given)
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        # Guard against zero/negative true values
        valid = (y_true > 0) & np.isfinite(y_true) & np.isfinite(y_pred)
        if not valid.any():
            logger.warning("No valid predictions to evaluate")
            return {"n_samples": 0}

        y_t = y_true[valid]
        y_p = y_pred[valid]

        # Absolute percentage errors
        ape = np.abs(y_t - y_p) / y_t * 100

        metrics = {
            "mape": float(np.median(ape)),
            "mean_ape": float(np.mean(ape)),
            "mae": float(mean_absolute_error(y_t, y_p)),
            "median_ae": float(np.median(np.abs(y_t - y_p))),
            "r2": float(r2_score(y_t, y_p)),
            "rmse": float(np.sqrt(mean_squared_error(y_t, y_p))),
            "hit_rate_5pct": float((ape <= 5).mean() * 100),
            "hit_rate_10pct": float((ape <= 10).mean() * 100),
            "hit_rate_15pct": float((ape <= 15).mean() * 100),
            "bias": float(np.mean(y_p - y_t) / np.mean(y_t) * 100),
            "n_samples": int(valid.sum()),
        }

        # Per-segment breakdown
        if segment_labels is not None:
            seg_labels = np.asarray(segment_labels)[valid]
            by_segment = {}
            for seg in np.unique(seg_labels):
                seg_mask = seg_labels == seg
                if seg_mask.sum() >= 5:
                    seg_ape = ape[seg_mask]
                    by_segment[str(seg)] = {
                        "mape": float(np.median(seg_ape)),
                        "mean_ape": float(np.mean(seg_ape)),
                        "hit_rate_10pct": float((seg_ape <= 10).mean() * 100),
                        "bias": float(
                            np.mean(y_p[seg_mask] - y_t[seg_mask])
                            / np.mean(y_t[seg_mask]) * 100
                        ),
                        "n_samples": int(seg_mask.sum()),
                    }
            metrics["by_segment"] = by_segment

        return metrics

    # ================================================================
    # BENCHMARK COMPARISON
    # ================================================================

    def compare_to_benchmark(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        properties_df: pd.DataFrame,
    ) -> dict:
        """Compare model predictions against a naive benchmark.

        The naive benchmark assigns each property the median assessed
        value for its neighbourhood + property type combination. This
        represents what you would get without any ML model -- just
        looking up the local median.

        Args:
            y_true: Actual property values.
            y_pred: Model-predicted values.
            properties_df: DataFrame with ``neighbourhood_code``,
                ``property_type``, and ``actual_value`` columns.

        Returns:
            Dict with benchmark_mape, model_mape, improvement_pct,
            and model_wins_pct.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        # Compute naive benchmark predictions
        benchmark_pred = np.full_like(y_true, np.nan)

        if (
            "neighbourhood_code" in properties_df.columns
            and "property_type" in properties_df.columns
            and "actual_value" in properties_df.columns
        ):
            # Compute median per (area, property_type)
            medians = properties_df.groupby(
                ["neighbourhood_code", "property_type"], observed=True
            )["actual_value"].median()

            for i, (_, row) in enumerate(properties_df.iterrows()):
                area = row.get("neighbourhood_code", "")
                ptype = row.get("property_type", "")
                key = (area, ptype)
                if key in medians.index:
                    benchmark_pred[i] = medians[key]
                else:
                    # Fall back to overall median for the property type
                    type_median = properties_df.loc[
                        properties_df["property_type"] == ptype, "actual_value"
                    ].median()
                    benchmark_pred[i] = type_median if not np.isnan(type_median) else np.median(y_true)
        else:
            # No grouping columns -- benchmark is just the global median
            benchmark_pred[:] = np.median(y_true)

        # Compute MAPEs
        valid = (y_true > 0) & np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(benchmark_pred)
        if not valid.any():
            return {
                "benchmark_mape": float("nan"),
                "model_mape": float("nan"),
                "improvement_pct": 0.0,
                "model_wins_pct": 0.0,
            }

        y_t = y_true[valid]
        y_p = y_pred[valid]
        b_p = benchmark_pred[valid]

        model_ape = np.abs(y_t - y_p) / y_t * 100
        benchmark_ape = np.abs(y_t - b_p) / y_t * 100

        model_mape = float(np.median(model_ape))
        benchmark_mape = float(np.median(benchmark_ape))

        improvement_pct = (
            (benchmark_mape - model_mape) / benchmark_mape * 100
            if benchmark_mape > 0
            else 0.0
        )

        model_wins = (model_ape < benchmark_ape).sum()
        model_wins_pct = float(model_wins / len(y_t) * 100)

        result = {
            "benchmark_mape": benchmark_mape,
            "model_mape": model_mape,
            "improvement_pct": float(improvement_pct),
            "model_wins_pct": model_wins_pct,
            "n_evaluated": int(valid.sum()),
        }

        logger.info(
            "Benchmark comparison: model MAPE=%.2f%% vs benchmark MAPE=%.2f%% "
            "(%.1f%% improvement, model wins %.1f%% of properties)",
            model_mape,
            benchmark_mape,
            improvement_pct,
            model_wins_pct,
        )

        return result

    # ================================================================
    # EVALUATION REPORT
    # ================================================================

    def generate_evaluation_report(self, backtest_result: BacktestResult) -> str:
        """Generate a formatted markdown evaluation report.

        Produces a human-readable report with tables for overall metrics,
        per property type, per local area, and worst-performing segments.

        Args:
            backtest_result: Output from ``backtest()``.

        Returns:
            Formatted markdown string.
        """
        br = backtest_result
        lines = []

        lines.append(f"# Model Evaluation Report -- Holdout Year {br.holdout_year}")
        lines.append("")

        # --- Overall Metrics ---
        lines.append("## Overall Metrics")
        lines.append("")
        if br.overall_metrics:
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            metric_labels = {
                "mape": "Median APE (%)",
                "mean_ape": "Mean APE (%)",
                "mae": "Mean Absolute Error ($)",
                "median_ae": "Median Absolute Error ($)",
                "r2": "R-squared",
                "rmse": "RMSE ($)",
                "hit_rate_5pct": "Hit Rate <= 5% (%)",
                "hit_rate_10pct": "Hit Rate <= 10% (%)",
                "hit_rate_15pct": "Hit Rate <= 15% (%)",
                "bias": "Systematic Bias (%)",
                "n_samples": "Sample Count",
            }
            for key, label in metric_labels.items():
                val = br.overall_metrics.get(key)
                if val is not None:
                    if key in ("mae", "median_ae", "rmse"):
                        lines.append(f"| {label} | ${val:,.0f} |")
                    elif key == "r2":
                        lines.append(f"| {label} | {val:.4f} |")
                    elif key == "n_samples":
                        lines.append(f"| {label} | {val:,} |")
                    else:
                        lines.append(f"| {label} | {val:.2f} |")
            lines.append("")
        else:
            lines.append("No overall metrics available.")
            lines.append("")

        # --- By Property Type ---
        lines.append("## By Property Type")
        lines.append("")
        if br.by_property_type:
            lines.append(
                "| Property Type | MAPE (%) | Hit Rate 10% (%) | "
                "Bias (%) | MAE ($) | N |"
            )
            lines.append("|---------------|----------|-------------------|----------|---------|---|")
            for ptype in sorted(br.by_property_type.keys()):
                m = br.by_property_type[ptype]
                lines.append(
                    f"| {ptype} | {m.get('mape', 0):.2f} | "
                    f"{m.get('hit_rate_10pct', 0):.1f} | "
                    f"{m.get('bias', 0):.2f} | "
                    f"${m.get('mae', 0):,.0f} | "
                    f"{m.get('n_samples', 0):,} |"
                )
            lines.append("")
        else:
            lines.append("No property type breakdown available.")
            lines.append("")

        # --- By Local Area ---
        lines.append("## By Local Area")
        lines.append("")
        if br.by_local_area:
            lines.append(
                "| Local Area | MAPE (%) | Hit Rate 10% (%) | "
                "Bias (%) | MAE ($) | N |"
            )
            lines.append("|------------|----------|-------------------|----------|---------|---|")
            # Sort by MAPE ascending (best first)
            sorted_areas = sorted(
                br.by_local_area.items(),
                key=lambda x: x[1].get("mape", 999),
            )
            for area, m in sorted_areas:
                lines.append(
                    f"| {area} | {m.get('mape', 0):.2f} | "
                    f"{m.get('hit_rate_10pct', 0):.1f} | "
                    f"{m.get('bias', 0):.2f} | "
                    f"${m.get('mae', 0):,.0f} | "
                    f"{m.get('n_samples', 0):,} |"
                )
            lines.append("")
        else:
            lines.append("No local area breakdown available.")
            lines.append("")

        # --- Top 5 Worst Segments ---
        lines.append("## Top 5 Worst-Performing Segments")
        lines.append("")
        if br.by_segment:
            sorted_segments = sorted(
                br.by_segment.items(),
                key=lambda x: x[1].get("mape", 0),
                reverse=True,
            )
            lines.append("| Segment | MAPE (%) | Bias (%) | N |")
            lines.append("|---------|----------|----------|---|")
            for seg, m in sorted_segments[:5]:
                lines.append(
                    f"| {seg} | {m.get('mape', 0):.2f} | "
                    f"{m.get('bias', 0):.2f} | "
                    f"{m.get('n_samples', 0):,} |"
                )
            lines.append("")
        else:
            lines.append("No segment breakdown available.")
            lines.append("")

        # --- Hit Rate Distribution ---
        lines.append("## Hit Rate Distribution")
        lines.append("")
        if br.overall_metrics:
            lines.append("| Threshold | Cumulative Hit Rate (%) |")
            lines.append("|-----------|------------------------|")
            for threshold_key, threshold_label in [
                ("hit_rate_5pct", "<= 5%"),
                ("hit_rate_10pct", "<= 10%"),
                ("hit_rate_15pct", "<= 15%"),
            ]:
                val = br.overall_metrics.get(threshold_key, 0)
                bar = "#" * int(val / 2)  # Simple text bar chart
                lines.append(f"| {threshold_label} | {val:.1f}% {bar} |")
            lines.append("")

        # --- Benchmark Comparison ---
        lines.append("## Benchmark Comparison")
        lines.append("")
        if br.benchmark_comparison:
            bc = br.benchmark_comparison
            lines.append(f"- **Model MAPE**: {bc.get('model_mape', 0):.2f}%")
            lines.append(f"- **Benchmark MAPE** (neighbourhood median): {bc.get('benchmark_mape', 0):.2f}%")
            lines.append(f"- **Improvement**: {bc.get('improvement_pct', 0):.1f}%")
            lines.append(f"- **Model wins**: {bc.get('model_wins_pct', 0):.1f}% of properties")
            lines.append("")
        else:
            lines.append("No benchmark comparison available.")
            lines.append("")

        report = "\n".join(lines)

        logger.info(
            "Evaluation report generated: %d lines, holdout_year=%d",
            len(lines),
            br.holdout_year,
        )

        return report
