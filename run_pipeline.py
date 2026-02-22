#!/usr/bin/env python3
"""
Run the Metro Vancouver Pricing Engine pipeline.

Usage:
    python run_pipeline.py              # Full pipeline (data + train)
    python run_pipeline.py --data-only  # Just download and enrich data
    python run_pipeline.py --status     # Check current status
"""

import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from src.pipeline.orchestrator import DataOrchestrator


def main():
    orchestrator = DataOrchestrator()

    if "--status" in sys.argv:
        status = orchestrator.get_status()
        print(json.dumps(status, indent=2))
        return

    if "--data-only" in sys.argv:
        print("Running data-only pipeline (download + enrich, no training)...\n")
        df = orchestrator.run_data_only()
        print(f"\nDone! Enriched {len(df)} properties with {len(df.columns)} features.")
        print(f"Saved to: data/processed/enriched_properties.parquet")
        return

    # Full pipeline: use fewer Optuna trials for speed (10 instead of 50)
    trials = 10
    for arg in sys.argv[1:]:
        if arg.startswith("--trials="):
            trials = int(arg.split("=")[1])

    print(f"Running full pipeline (data + train, {trials} Optuna trials)...\n")
    summary = orchestrator.run_full_pipeline(n_optuna_trials=trials)

    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
