"""
Master data pipeline orchestrator.

Coordinates all data ingestion, feature enrichment, model training,
and provides a single entry point for rebuilding the entire system.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.features.building_footprint import BuildingFootprintEstimator
from src.features.feature_builder import FeatureBuilder
from src.features.feature_registry import PropertyType
from src.features.spatial_features import SpatialFeatureComputer
from src.models.evaluation import ModelEvaluator
from src.models.quantile_models import QuantileModelTrainer
from src.models.subregions import SubRegionEngine
from src.models.trainer import ModelTrainer
from src.models.types import TrainingResult
from src.pipeline.feature_enrichment import FeatureEnrichmentPipeline
from src.pipeline.property_universe import PropertyUniverseBuilder

logger = logging.getLogger(__name__)


class DataOrchestrator:
    """Master orchestrator for the full data and training pipeline.

    Coordinates all stages of the property pricing engine:
      1. Property universe construction (BC Assessment data)
      2. Feature enrichment (spatial, census, building footprint, etc.)
      3. Sub-region segmentation (22 local areas x property type)
      4. Model training (LightGBM with Optuna tuning)
      5. Quantile model training (for prediction intervals)
      6. Backtesting and evaluation
      7. Model persistence

    Provides both a full end-to-end pipeline and selective runs
    (data-only, training-only) for iterative development.

    Usage::

        orchestrator = DataOrchestrator()
        summary = orchestrator.run_full_pipeline(assessment_year=2024)
        print(summary)

    Args:
        data_dir: Root directory for raw and processed data files.
        model_dir: Directory for trained model files.
        cache_dir: Directory for intermediate cached data.
    """

    def __init__(
        self,
        data_dir: str = "data",
        model_dir: str = "models",
        cache_dir: str = "data/processed",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.cache_dir = Path(cache_dir)

        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "DataOrchestrator initialized: data_dir=%s, model_dir=%s, cache_dir=%s",
            self.data_dir,
            self.model_dir,
            self.cache_dir,
        )

    # ================================================================
    # FULL PIPELINE
    # ================================================================

    def run_full_pipeline(
        self,
        assessment_year: Optional[int] = None,
        n_optuna_trials: int = 50,
    ) -> dict:
        """Run the complete end-to-end pipeline.

        Steps:
          1. Build property universe from BC Assessment data
          2. Enrich features (spatial, census, building footprint, etc.)
          3. Define sub-regions and micro-neighbourhoods
          4. Train LightGBM models per segment
          5. Train quantile models for prediction intervals
          6. Evaluate via backtest
          7. Save everything to disk

        Args:
            assessment_year: Tax assessment year to build for. If None,
                fetches the latest available year.
            n_optuna_trials: Number of Optuna hyperparameter tuning trials
                per segment model. More trials = better models but slower.

        Returns:
            Summary dict with timings, counts, and metrics for each stage.
        """
        t0 = time.perf_counter()
        summary: dict = {
            "start_time": datetime.utcnow().isoformat(),
            "assessment_year": assessment_year,
            "n_optuna_trials": n_optuna_trials,
            "stages": {},
        }

        logger.info(
            "Starting full pipeline: year=%s, n_optuna_trials=%d",
            assessment_year,
            n_optuna_trials,
        )

        # ------------------------------------------------------------------
        # 1. Build property universe
        # ------------------------------------------------------------------
        stage_t0 = time.perf_counter()
        logger.info("Stage 1/6: Building property universe...")

        try:
            universe_builder = PropertyUniverseBuilder(
                cache_dir=str(self.data_dir / "raw"),
            )
            properties_df = universe_builder.build_universe(year=assessment_year)

            summary["stages"]["1_property_universe"] = {
                "status": "success",
                "row_count": len(properties_df),
                "elapsed_s": round(time.perf_counter() - stage_t0, 1),
            }
            logger.info(
                "Stage 1 complete: %d properties in %.1fs",
                len(properties_df),
                time.perf_counter() - stage_t0,
            )
        except Exception as exc:
            logger.error("Stage 1 failed: %s", exc, exc_info=True)
            summary["stages"]["1_property_universe"] = {
                "status": "failed",
                "error": str(exc),
                "elapsed_s": round(time.perf_counter() - stage_t0, 1),
            }
            summary["total_elapsed_s"] = round(time.perf_counter() - t0, 1)
            return summary

        # ------------------------------------------------------------------
        # 2. Enrich features
        # ------------------------------------------------------------------
        stage_t0 = time.perf_counter()
        logger.info("Stage 2/6: Enriching features...")

        try:
            enrichment = FeatureEnrichmentPipeline(
                data_dir=str(self.data_dir),
                cache_dir=str(self.cache_dir),
            )
            enriched_df = enrichment.enrich_all(properties_df, phase=1)

            # Save enriched data for later use
            enriched_path = self.cache_dir / "enriched_properties.parquet"
            enriched_df.to_parquet(enriched_path, index=False)

            summary["stages"]["2_feature_enrichment"] = {
                "status": "success",
                "row_count": len(enriched_df),
                "feature_count": len(enriched_df.columns),
                "elapsed_s": round(time.perf_counter() - stage_t0, 1),
            }
            logger.info(
                "Stage 2 complete: %d properties, %d features in %.1fs",
                len(enriched_df),
                len(enriched_df.columns),
                time.perf_counter() - stage_t0,
            )
        except Exception as exc:
            logger.error("Stage 2 failed: %s", exc, exc_info=True)
            summary["stages"]["2_feature_enrichment"] = {
                "status": "failed",
                "error": str(exc),
                "elapsed_s": round(time.perf_counter() - stage_t0, 1),
            }
            # Continue with un-enriched data
            enriched_df = properties_df

        # ------------------------------------------------------------------
        # 3. Define sub-regions
        # ------------------------------------------------------------------
        stage_t0 = time.perf_counter()
        logger.info("Stage 3/6: Defining sub-regions...")

        try:
            subregion_engine = SubRegionEngine(min_segment_size=200)
            enriched_df = subregion_engine.define_micro_neighborhoods(enriched_df)
            segment_stats = subregion_engine.get_segment_stats(enriched_df)

            summary["stages"]["3_subregions"] = {
                "status": "success",
                "total_segments": len(segment_stats),
                "qualifying_segments": int(
                    (segment_stats["count"] >= 200).sum()
                ) if not segment_stats.empty else 0,
                "elapsed_s": round(time.perf_counter() - stage_t0, 1),
            }
            logger.info(
                "Stage 3 complete: %d segments in %.1fs",
                len(segment_stats),
                time.perf_counter() - stage_t0,
            )
        except Exception as exc:
            logger.error("Stage 3 failed: %s", exc, exc_info=True)
            subregion_engine = SubRegionEngine(min_segment_size=200)
            summary["stages"]["3_subregions"] = {
                "status": "failed",
                "error": str(exc),
                "elapsed_s": round(time.perf_counter() - stage_t0, 1),
            }

        # ------------------------------------------------------------------
        # 4. Train models
        # ------------------------------------------------------------------
        stage_t0 = time.perf_counter()
        logger.info("Stage 4/6: Training segment models...")

        training_results: dict[str, TrainingResult] = {}
        try:
            spatial = SpatialFeatureComputer()
            footprint = BuildingFootprintEstimator()
            feature_builder = FeatureBuilder(
                spatial_computer=spatial,
                footprint_estimator=footprint,
                phase=1,
                mls_available=False,
            )

            trainer = ModelTrainer(
                model_dir=str(self.model_dir),
            )
            training_results = trainer.train_all_segments(
                enriched_df,
                feature_builder,
                subregion_engine,
                n_optuna_trials=n_optuna_trials,
            )

            # Compute aggregate metrics
            if training_results:
                mapes = [r.validation_mape for r in training_results.values()]
                avg_mape = sum(mapes) / len(mapes)
                best_mape = min(mapes)
                worst_mape = max(mapes)
            else:
                avg_mape = best_mape = worst_mape = 0.0

            summary["stages"]["4_model_training"] = {
                "status": "success",
                "models_trained": len(training_results),
                "avg_cv_mape": round(avg_mape, 2),
                "best_cv_mape": round(best_mape, 2),
                "worst_cv_mape": round(worst_mape, 2),
                "elapsed_s": round(time.perf_counter() - stage_t0, 1),
            }
            logger.info(
                "Stage 4 complete: %d models trained, avg MAPE=%.2f%% in %.1fs",
                len(training_results),
                avg_mape,
                time.perf_counter() - stage_t0,
            )
        except Exception as exc:
            logger.error("Stage 4 failed: %s", exc, exc_info=True)
            summary["stages"]["4_model_training"] = {
                "status": "failed",
                "error": str(exc),
                "elapsed_s": round(time.perf_counter() - stage_t0, 1),
            }

        # ------------------------------------------------------------------
        # 5. Train quantile models
        # ------------------------------------------------------------------
        stage_t0 = time.perf_counter()
        logger.info("Stage 5/6: Training quantile models...")

        try:
            quantile_trainer = QuantileModelTrainer()
            n_quantile_models = 0

            for segment_key, result in training_results.items():
                # Re-build features for this segment to get X, y
                parts = segment_key.split("__")
                if len(parts) != 2:
                    continue

                area_code, ptype_str = parts

                # Filter properties
                if area_code == "citywide":
                    segment_mask = pd.Series(True, index=enriched_df.index)
                else:
                    segment_mask = enriched_df["neighbourhood_code"] == area_code

                if ptype_str != "all":
                    segment_mask &= enriched_df["property_type"] == ptype_str

                segment_df = enriched_df[segment_mask]

                if len(segment_df) < 50:
                    continue

                try:
                    ptype_enum = None
                    for pt in PropertyType:
                        if pt.value == ptype_str:
                            ptype_enum = pt
                            break

                    X, y = feature_builder.build_features_batch(
                        segment_df, property_type=ptype_enum,
                    )

                    q_models = quantile_trainer.train_quantile_models(
                        segment_key, X, y,
                        base_params=result.hyperparameters.copy(),
                    )

                    quantile_trainer.save_quantile_models(
                        segment_key, q_models, str(self.model_dir),
                    )
                    n_quantile_models += 1
                except Exception as exc:
                    logger.warning(
                        "Quantile training failed for '%s': %s",
                        segment_key,
                        exc,
                    )

            summary["stages"]["5_quantile_models"] = {
                "status": "success",
                "segments_with_quantiles": n_quantile_models,
                "elapsed_s": round(time.perf_counter() - stage_t0, 1),
            }
            logger.info(
                "Stage 5 complete: %d quantile model sets trained in %.1fs",
                n_quantile_models,
                time.perf_counter() - stage_t0,
            )
        except Exception as exc:
            logger.error("Stage 5 failed: %s", exc, exc_info=True)
            summary["stages"]["5_quantile_models"] = {
                "status": "failed",
                "error": str(exc),
                "elapsed_s": round(time.perf_counter() - stage_t0, 1),
            }

        # ------------------------------------------------------------------
        # 6. Evaluate via backtest
        # ------------------------------------------------------------------
        stage_t0 = time.perf_counter()
        logger.info("Stage 6/6: Running backtest evaluation...")

        try:
            evaluator = ModelEvaluator(
                model_dir=str(self.model_dir),
            )

            # Determine holdout year
            if (
                "tax_assessment_year" in enriched_df.columns
                and enriched_df["tax_assessment_year"].nunique() > 1
            ):
                holdout_year = int(enriched_df["tax_assessment_year"].max())
                backtest_result = evaluator.backtest(
                    enriched_df,
                    feature_builder,
                    subregion_engine,
                    holdout_year=holdout_year,
                )

                report = evaluator.generate_evaluation_report(backtest_result)

                # Save evaluation report
                report_path = self.model_dir / "evaluation_report.md"
                report_path.write_text(report)

                summary["stages"]["6_evaluation"] = {
                    "status": "success",
                    "holdout_year": holdout_year,
                    "overall_mape": backtest_result.overall_metrics.get("mape"),
                    "hit_rate_10pct": backtest_result.overall_metrics.get(
                        "hit_rate_10pct",
                    ),
                    "benchmark_improvement": backtest_result.benchmark_comparison.get(
                        "improvement_pct",
                    ),
                    "elapsed_s": round(time.perf_counter() - stage_t0, 1),
                }
                logger.info(
                    "Stage 6 complete: MAPE=%.2f%%, report saved in %.1fs",
                    backtest_result.overall_metrics.get("mape", 0),
                    time.perf_counter() - stage_t0,
                )
            else:
                logger.info(
                    "Skipping backtest: only one assessment year available",
                )
                summary["stages"]["6_evaluation"] = {
                    "status": "skipped",
                    "reason": "Only one assessment year available",
                    "elapsed_s": round(time.perf_counter() - stage_t0, 1),
                }
        except Exception as exc:
            logger.error("Stage 6 failed: %s", exc, exc_info=True)
            summary["stages"]["6_evaluation"] = {
                "status": "failed",
                "error": str(exc),
                "elapsed_s": round(time.perf_counter() - stage_t0, 1),
            }

        # ------------------------------------------------------------------
        # Finalize
        # ------------------------------------------------------------------
        total_elapsed = time.perf_counter() - t0
        summary["total_elapsed_s"] = round(total_elapsed, 1)
        summary["end_time"] = datetime.utcnow().isoformat()

        # Count successes
        n_success = sum(
            1
            for stage in summary["stages"].values()
            if stage.get("status") == "success"
        )
        n_failed = sum(
            1
            for stage in summary["stages"].values()
            if stage.get("status") == "failed"
        )
        summary["overall_status"] = (
            "success" if n_failed == 0 else "partial" if n_success > 0 else "failed"
        )

        logger.info(
            "Full pipeline complete: %d stages succeeded, %d failed in %.1fs",
            n_success,
            n_failed,
            total_elapsed,
        )

        return summary

    # ================================================================
    # DATA-ONLY PIPELINE
    # ================================================================

    def run_data_only(
        self,
        assessment_year: Optional[int] = None,
    ) -> pd.DataFrame:
        """Run only data ingestion and feature enrichment (no training).

        Useful for data exploration, feature analysis, and QA before
        committing to a full training run.

        Args:
            assessment_year: Tax assessment year. If None, fetches the
                latest available.

        Returns:
            Enriched property DataFrame.
        """
        t0 = time.perf_counter()
        logger.info("Starting data-only pipeline (year=%s)", assessment_year)

        # 1. Build property universe
        universe_builder = PropertyUniverseBuilder(
            cache_dir=str(self.data_dir / "raw"),
        )
        properties_df = universe_builder.build_universe(year=assessment_year)
        logger.info(
            "Property universe built: %d properties", len(properties_df),
        )

        # 2. Enrich features
        enrichment = FeatureEnrichmentPipeline(
            data_dir=str(self.data_dir),
            cache_dir=str(self.cache_dir),
        )
        enriched_df = enrichment.enrich_all(properties_df, phase=1)

        # Save enriched data
        enriched_path = self.cache_dir / "enriched_properties.parquet"
        enriched_df.to_parquet(enriched_path, index=False)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Data-only pipeline complete: %d properties, %d features in %.1fs",
            len(enriched_df),
            len(enriched_df.columns),
            elapsed,
        )

        return enriched_df

    # ================================================================
    # TRAINING-ONLY PIPELINE
    # ================================================================

    def run_training_only(
        self,
        properties_df: pd.DataFrame,
        n_trials: int = 50,
    ) -> dict:
        """Run only model training on pre-prepared data.

        Assumes the input DataFrame is already enriched with all features.
        Runs sub-region definition, model training, quantile training,
        and evaluation.

        Args:
            properties_df: Enriched property DataFrame (output of
                run_data_only or FeatureEnrichmentPipeline.enrich_all).
            n_trials: Number of Optuna trials per segment.

        Returns:
            Summary dict with training results and metrics.
        """
        t0 = time.perf_counter()
        logger.info(
            "Starting training-only pipeline: %d properties, %d trials",
            len(properties_df),
            n_trials,
        )

        summary: dict = {
            "start_time": datetime.utcnow().isoformat(),
            "input_rows": len(properties_df),
            "n_trials": n_trials,
        }

        # 3. Define sub-regions
        subregion_engine = SubRegionEngine(min_segment_size=200)
        properties_df = subregion_engine.define_micro_neighborhoods(properties_df)

        # 4. Train models
        spatial = SpatialFeatureComputer()
        footprint = BuildingFootprintEstimator()
        feature_builder = FeatureBuilder(
            spatial_computer=spatial,
            footprint_estimator=footprint,
            phase=1,
            mls_available=False,
        )

        trainer = ModelTrainer(model_dir=str(self.model_dir))
        training_results = trainer.train_all_segments(
            properties_df,
            feature_builder,
            subregion_engine,
            n_optuna_trials=n_trials,
        )

        summary["models_trained"] = len(training_results)
        if training_results:
            mapes = [r.validation_mape for r in training_results.values()]
            summary["avg_cv_mape"] = round(sum(mapes) / len(mapes), 2)
            summary["best_cv_mape"] = round(min(mapes), 2)
            summary["worst_cv_mape"] = round(max(mapes), 2)

        # 5. Train quantile models
        quantile_trainer = QuantileModelTrainer()
        n_quantile = 0

        for segment_key, result in training_results.items():
            parts = segment_key.split("__")
            if len(parts) != 2:
                continue

            area_code, ptype_str = parts
            if area_code == "citywide":
                seg_mask = pd.Series(True, index=properties_df.index)
            else:
                seg_mask = properties_df["neighbourhood_code"] == area_code

            if ptype_str != "all":
                seg_mask &= properties_df["property_type"] == ptype_str

            segment_df = properties_df[seg_mask]
            if len(segment_df) < 50:
                continue

            try:
                ptype_enum = None
                for pt in PropertyType:
                    if pt.value == ptype_str:
                        ptype_enum = pt
                        break

                X, y = feature_builder.build_features_batch(
                    segment_df, property_type=ptype_enum,
                )
                q_models = quantile_trainer.train_quantile_models(
                    segment_key, X, y,
                    base_params=result.hyperparameters.copy(),
                )
                quantile_trainer.save_quantile_models(
                    segment_key, q_models, str(self.model_dir),
                )
                n_quantile += 1
            except Exception as exc:
                logger.warning(
                    "Quantile training failed for '%s': %s", segment_key, exc,
                )

        summary["quantile_model_sets"] = n_quantile

        # 6. Evaluate
        if (
            "tax_assessment_year" in properties_df.columns
            and properties_df["tax_assessment_year"].nunique() > 1
        ):
            evaluator = ModelEvaluator(model_dir=str(self.model_dir))
            holdout_year = int(properties_df["tax_assessment_year"].max())

            try:
                backtest_result = evaluator.backtest(
                    properties_df,
                    feature_builder,
                    subregion_engine,
                    holdout_year=holdout_year,
                )
                summary["backtest"] = {
                    "holdout_year": holdout_year,
                    "overall_mape": backtest_result.overall_metrics.get("mape"),
                    "hit_rate_10pct": backtest_result.overall_metrics.get(
                        "hit_rate_10pct",
                    ),
                }

                report = evaluator.generate_evaluation_report(backtest_result)
                report_path = self.model_dir / "evaluation_report.md"
                report_path.write_text(report)
            except Exception as exc:
                logger.error("Backtest failed: %s", exc, exc_info=True)
                summary["backtest"] = {"status": "failed", "error": str(exc)}
        else:
            summary["backtest"] = {"status": "skipped", "reason": "single year"}

        elapsed = time.perf_counter() - t0
        summary["total_elapsed_s"] = round(elapsed, 1)
        summary["end_time"] = datetime.utcnow().isoformat()

        logger.info(
            "Training-only pipeline complete: %d models, %d quantile sets in %.1fs",
            len(training_results),
            n_quantile,
            elapsed,
        )

        return summary

    # ================================================================
    # STATUS CHECK
    # ================================================================

    def get_status(self) -> dict:
        """Check the status of data files and trained models.

        Inspects the filesystem for enriched data, model files, quantile
        models, and evaluation reports.

        Returns:
            Status dict with existence flags, counts, and timestamps.
        """
        status: dict = {
            "data_dir": str(self.data_dir),
            "model_dir": str(self.model_dir),
            "cache_dir": str(self.cache_dir),
        }

        # Check enriched data
        enriched_path = self.cache_dir / "enriched_properties.parquet"
        if enriched_path.exists():
            status["enriched_data"] = {
                "exists": True,
                "path": str(enriched_path),
                "size_mb": round(enriched_path.stat().st_size / 1024 / 1024, 1),
                "modified": datetime.fromtimestamp(
                    enriched_path.stat().st_mtime
                ).isoformat(),
            }
        else:
            status["enriched_data"] = {"exists": False}

        # Check trained models
        model_files = list(self.model_dir.glob("*.pkl"))
        # Separate main models from quantile models
        main_models = [f for f in model_files if "_q" not in f.stem]
        quantile_models = [f for f in model_files if "_q" in f.stem]

        status["models"] = {
            "main_model_count": len(main_models),
            "quantile_model_count": len(quantile_models),
            "total_size_mb": round(
                sum(f.stat().st_size for f in model_files) / 1024 / 1024, 1
            ) if model_files else 0,
        }

        if main_models:
            latest_model = max(main_models, key=lambda f: f.stat().st_mtime)
            status["models"]["latest_model"] = latest_model.stem
            status["models"]["latest_modified"] = datetime.fromtimestamp(
                latest_model.stat().st_mtime
            ).isoformat()

        # Check metadata files
        metadata_files = list(self.model_dir.glob("*_metadata.json"))
        status["models"]["metadata_count"] = len(metadata_files)

        # Check evaluation report
        report_path = self.model_dir / "evaluation_report.md"
        if report_path.exists():
            status["evaluation"] = {
                "exists": True,
                "path": str(report_path),
                "modified": datetime.fromtimestamp(
                    report_path.stat().st_mtime
                ).isoformat(),
            }
        else:
            status["evaluation"] = {"exists": False}

        return status
