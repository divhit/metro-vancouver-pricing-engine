"""
Metro Vancouver Property Pricing Engine API.

Endpoints:
- POST /api/predict         -- Generate property valuation
- GET  /api/property/{pid}  -- Enriched property details
- GET  /api/market/{code}   -- Market summary for a neighbourhood
- GET  /api/market/all      -- Market summaries for all 22 local areas
- GET  /api/neighbourhoods  -- List all 22 neighbourhood codes and names
- GET  /api/health          -- Service health check
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.api.cache import PredictionCache
from src.api.schemas import (
    AdjustmentDTO,
    ComparableDTO,
    ConfidenceInterval,
    HealthResponse,
    MarketContext,
    MarketSummary,
    PredictionMetadata,
    PredictionRequest,
    PredictionResponse,
    PropertyDetail,
    RiskFlag,
    ShapFeature,
)
from src.models.predictor import PropertyPredictor
from src.pipeline.property_universe import VANCOUVER_LOCAL_AREAS

logger = logging.getLogger(__name__)

# ============================================================
# GLOBAL STATE
# ============================================================

# These are populated during startup
_predictor: Optional[PropertyPredictor] = None
_properties_df: Optional[pd.DataFrame] = None
_cache: Optional[PredictionCache] = None

# Configuration from environment
_MODEL_DIR = os.environ.get("MODEL_DIR", "models")
_DATA_DIR = os.environ.get("DATA_DIR", "data")
_REDIS_URL = os.environ.get("REDIS_URL", None)
_MLS_AVAILABLE = os.environ.get("MLS_AVAILABLE", "false").lower() == "true"


# ============================================================
# LIFESPAN
# ============================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load data and models on startup, cleanup on shutdown."""
    global _predictor, _properties_df, _cache

    logger.info("Starting Metro Vancouver Pricing Engine API...")

    # Initialize cache
    _cache = PredictionCache(redis_url=_REDIS_URL)

    # Initialize predictor
    _predictor = PropertyPredictor(
        model_dir=_MODEL_DIR,
        mls_available=_MLS_AVAILABLE,
    )

    # Load enriched property data
    enriched_path = Path(_DATA_DIR) / "processed" / "enriched_properties.parquet"
    if enriched_path.exists():
        try:
            _properties_df = pd.read_parquet(enriched_path)
            logger.info(
                "Loaded enriched properties: %d rows, %d columns from %s",
                len(_properties_df),
                len(_properties_df.columns),
                enriched_path,
            )
        except Exception as exc:
            logger.error("Failed to load enriched properties: %s", exc)
            _properties_df = pd.DataFrame()
    else:
        logger.warning(
            "Enriched properties file not found at %s; "
            "API will operate with limited functionality",
            enriched_path,
        )
        _properties_df = pd.DataFrame()

    logger.info("API startup complete")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down Metro Vancouver Pricing Engine API")
    if _cache:
        _cache.clear()


# ============================================================
# APP CREATION
# ============================================================

app = FastAPI(
    title="Metro Vancouver Property Pricing Engine",
    description=(
        "ML-powered property valuation engine for Metro Vancouver. "
        "Combines LightGBM models with comparable sales analysis and "
        "rules-based adjustments to produce transparent, explainable "
        "property valuations."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware -- allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ENDPOINTS
# ============================================================


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Generate a property valuation.

    Accepts a property identifier (PID, address, or coordinates) and
    returns a full valuation with confidence interval, comparables,
    SHAP explanations, and risk flags.

    Args:
        request: PredictionRequest with at least one identifier.

    Returns:
        PredictionResponse with the complete valuation result.

    Raises:
        HTTPException 422: Invalid request (validation error).
        HTTPException 500: Internal prediction error.
    """
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")

    # Check cache
    cache_key = PredictionCache.make_prediction_key(
        pid=request.pid,
        lat=request.latitude,
        lon=request.longitude,
        property_type=request.property_type,
        overrides=request.overrides,
    )

    if _cache:
        cached = _cache.get(cache_key)
        if cached is not None:
            logger.info("Cache hit for prediction: %s", cache_key)
            return PredictionResponse(**cached)

    try:
        result = _predictor.predict(
            pid=request.pid,
            lat=request.latitude,
            lon=request.longitude,
            address=request.address,
            property_type=request.property_type,
            override_features=request.overrides,
            properties_df=_properties_df,
        )
    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )

    # Convert PredictionResult to PredictionResponse
    response = _convert_prediction_result(result)

    # Cache the result
    if _cache:
        try:
            _cache.set(cache_key, response.model_dump(mode="json"))
        except Exception as exc:
            logger.warning("Failed to cache prediction: %s", exc)

    return response


@app.get("/api/property/{pid}", response_model=PropertyDetail)
async def get_property(pid: str) -> PropertyDetail:
    """Look up enriched property details by PID.

    Args:
        pid: BC Assessment Property Identifier.

    Returns:
        PropertyDetail with full property information.

    Raises:
        HTTPException 404: Property not found.
    """
    if _properties_df is None or _properties_df.empty:
        raise HTTPException(
            status_code=503,
            detail="Property data not loaded",
        )

    match = _properties_df[_properties_df["pid"].astype(str) == str(pid)]
    if match.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Property with PID '{pid}' not found",
        )

    row = match.iloc[0]
    neighbourhood_code = str(row.get("neighbourhood_code", ""))

    return PropertyDetail(
        pid=str(row.get("pid", "")),
        address=str(row.get("full_address", row.get("address", ""))),
        neighbourhood_code=neighbourhood_code,
        neighbourhood_name=VANCOUVER_LOCAL_AREAS.get(
            neighbourhood_code, neighbourhood_code.replace("-", " ").title(),
        ),
        property_type=str(row.get("property_type", "")),
        zoning=str(row.get("zoning_district", "")) or None,
        year_built=(
            int(row["year_built"])
            if pd.notna(row.get("year_built"))
            else None
        ),
        land_value=float(row.get("current_land_value", 0)),
        improvement_value=float(row.get("current_improvement_value", 0)),
        total_assessed_value=float(row.get("total_assessed_value", 0)),
        estimated_living_area_sqft=(
            float(row["estimated_living_area_sqft"])
            if pd.notna(row.get("estimated_living_area_sqft"))
            else None
        ),
        latitude=float(row.get("latitude", 0.0)),
        longitude=float(row.get("longitude", 0.0)),
    )


@app.get("/api/market/all", response_model=list[MarketSummary])
async def get_all_market_summaries() -> list[MarketSummary]:
    """Return market summaries for all 22 Vancouver local areas.

    Returns:
        List of MarketSummary objects, one per neighbourhood.
    """
    if _properties_df is None or _properties_df.empty:
        raise HTTPException(
            status_code=503,
            detail="Property data not loaded",
        )

    summaries = []
    for area_code in sorted(VANCOUVER_LOCAL_AREAS.keys()):
        summary = _compute_market_summary(area_code)
        if summary is not None:
            summaries.append(summary)

    return summaries


@app.get("/api/market/{neighbourhood_code}", response_model=MarketSummary)
async def get_market_summary(neighbourhood_code: str) -> MarketSummary:
    """Return market summary for a specific neighbourhood.

    Args:
        neighbourhood_code: Vancouver local area code (e.g. "KITSILANO").

    Returns:
        MarketSummary with property counts, median values, and YoY changes.

    Raises:
        HTTPException 404: Neighbourhood not found.
    """
    if _properties_df is None or _properties_df.empty:
        raise HTTPException(
            status_code=503,
            detail="Property data not loaded",
        )

    # Normalize the code to uppercase
    code = neighbourhood_code.upper()

    if code not in VANCOUVER_LOCAL_AREAS:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Neighbourhood '{neighbourhood_code}' not found. "
                f"Valid codes: {sorted(VANCOUVER_LOCAL_AREAS.keys())}"
            ),
        )

    summary = _compute_market_summary(code)
    if summary is None:
        raise HTTPException(
            status_code=404,
            detail=f"No data available for neighbourhood '{code}'",
        )

    return summary


@app.get("/api/neighbourhoods")
async def get_neighbourhoods() -> list[dict[str, str]]:
    """Return list of all 22 Vancouver neighbourhood codes and names.

    Returns:
        List of dicts with 'code' and 'name' keys.
    """
    return [
        {"code": code, "name": name}
        for code, name in sorted(VANCOUVER_LOCAL_AREAS.items())
    ]


@app.get("/api/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Service health check.

    Returns model count, data freshness, and version.
    """
    model_count = 0
    if _predictor:
        model_count = _predictor.get_available_model_count()

    data_freshness: dict[str, str] = {}

    # Check enriched data freshness
    enriched_path = Path(_DATA_DIR) / "processed" / "enriched_properties.parquet"
    if enriched_path.exists():
        mtime = datetime.fromtimestamp(enriched_path.stat().st_mtime)
        data_freshness["enriched_properties"] = mtime.isoformat()
    else:
        data_freshness["enriched_properties"] = "not_available"

    # Check properties_df
    if _properties_df is not None and not _properties_df.empty:
        data_freshness["property_count"] = str(len(_properties_df))
        if "tax_assessment_year" in _properties_df.columns:
            latest_year = int(_properties_df["tax_assessment_year"].max())
            data_freshness["latest_assessment_year"] = str(latest_year)
    else:
        data_freshness["property_count"] = "0"

    # Determine status
    if model_count > 0 and _properties_df is not None and not _properties_df.empty:
        status = "ok"
    elif _properties_df is not None and not _properties_df.empty:
        status = "degraded"
    else:
        status = "error"

    return HealthResponse(
        status=status,
        model_count=model_count,
        data_freshness=data_freshness,
        version="0.1.0",
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def _compute_market_summary(area_code: str) -> Optional[MarketSummary]:
    """Compute market summary statistics for a neighbourhood.

    Args:
        area_code: Vancouver local area code.

    Returns:
        MarketSummary or None if no data.
    """
    if _properties_df is None or _properties_df.empty:
        return None

    if "neighbourhood_code" not in _properties_df.columns:
        return None

    hood_df = _properties_df[_properties_df["neighbourhood_code"] == area_code]
    if hood_df.empty:
        return None

    # Property counts by type
    property_counts: dict[str, int] = {}
    median_values: dict[str, float] = {}
    yoy_changes: dict[str, Optional[float]] = {}

    if "property_type" in hood_df.columns:
        for ptype, group in hood_df.groupby("property_type", observed=True):
            ptype_str = str(ptype)
            property_counts[ptype_str] = len(group)

            if "total_assessed_value" in group.columns:
                median_values[ptype_str] = round(
                    float(group["total_assessed_value"].median()), 2,
                )

            # YoY change
            if (
                "total_assessed_value" in group.columns
                and "previous_land_value" in group.columns
                and "previous_improvement_value" in group.columns
            ):
                current = group["total_assessed_value"].median()
                previous = (
                    group["previous_land_value"].fillna(0)
                    + group["previous_improvement_value"].fillna(0)
                ).median()
                if previous > 0:
                    yoy = round(float((current - previous) / previous * 100), 2)
                    yoy_changes[ptype_str] = yoy
                else:
                    yoy_changes[ptype_str] = None
            else:
                yoy_changes[ptype_str] = None
    else:
        property_counts["all"] = len(hood_df)
        if "total_assessed_value" in hood_df.columns:
            median_values["all"] = round(
                float(hood_df["total_assessed_value"].median()), 2,
            )

    return MarketSummary(
        neighbourhood_code=area_code,
        neighbourhood_name=VANCOUVER_LOCAL_AREAS.get(
            area_code, area_code.replace("-", " ").title(),
        ),
        property_counts=property_counts,
        median_values=median_values,
        yoy_changes=yoy_changes,
        interest_rate=None,  # Populated when BoC data is available
    )


def _convert_prediction_result(result) -> PredictionResponse:
    """Convert an internal PredictionResult to the API PredictionResponse.

    Args:
        result: PredictionResult dataclass instance.

    Returns:
        PredictionResponse Pydantic model.
    """
    # Convert comparables
    comparables = [
        ComparableDTO(
            pid=comp.pid,
            address=comp.address,
            assessed_value=comp.assessed_value,
            distance_m=comp.distance_m,
            similarity_score=comp.similarity_score,
            year_built=comp.year_built,
            zoning=comp.zoning,
            neighbourhood=comp.neighbourhood_code,
        )
        for comp in result.comparables
    ]

    # Convert SHAP values to ShapFeature objects
    shap_features = []
    for feature_name, shap_value in result.shap_values.items():
        direction = "up" if shap_value > 0 else "down"
        abs_val = abs(shap_value)

        # Generate a human-readable description
        if direction == "up":
            desc = f"{feature_name} pushes the estimate up by {abs_val:.4f} log-units"
        else:
            desc = f"{feature_name} pushes the estimate down by {abs_val:.4f} log-units"

        shap_features.append(
            ShapFeature(
                feature_name=feature_name,
                feature_value=None,
                shap_value=shap_value,
                direction=direction,
                description=desc,
            )
        )

    # Convert adjustments: (name, pct, explanation) tuples
    adjustments = []
    for adj_tuple in result.adjustments:
        name, pct, explanation = adj_tuple
        dollar_amount = result.point_estimate * (pct / 100.0) if result.point_estimate > 0 else 0.0
        adjustments.append(
            AdjustmentDTO(
                name=name,
                percentage=pct,
                dollar_amount=round(dollar_amount, 2),
                explanation=explanation,
            )
        )

    # Convert market context
    mc = result.market_context
    market_context = MarketContext(
        neighbourhood_code=mc.get("neighbourhood_code", ""),
        neighbourhood_name=mc.get("neighbourhood_name", ""),
        median_assessed_value=mc.get("median_assessed_value", 0.0),
        yoy_change_pct=mc.get("yoy_change_pct"),
        interest_rate_5yr=mc.get("interest_rate_5yr"),
        property_count=mc.get("property_count", 0),
        assessment_year=mc.get("assessment_year", datetime.utcnow().year),
    )

    # Convert risk flags
    risk_flags = [
        RiskFlag(
            category=flag.get("category", "unknown"),
            severity=flag.get("severity", "medium"),
            description=flag.get("description", ""),
        )
        for flag in result.risk_flags
    ]

    # Build confidence interval
    ci_lower, ci_upper = result.confidence_interval
    confidence_interval = ConfidenceInterval(
        lower=ci_lower,
        upper=ci_upper,
        level=0.80,
    )

    # Build metadata
    metadata = PredictionMetadata(
        model_segment=result.model_segment,
        model_version=result.model_version,
        prediction_timestamp=result.prediction_timestamp.isoformat(),
        data_freshness=None,
        mls_available=False,
    )

    return PredictionResponse(
        point_estimate=result.point_estimate,
        confidence_interval=confidence_interval,
        confidence_grade=result.confidence_grade,
        comparables=comparables,
        shap_features=shap_features,
        adjustments=adjustments,
        market_context=market_context,
        risk_flags=risk_flags,
        metadata=metadata,
    )
