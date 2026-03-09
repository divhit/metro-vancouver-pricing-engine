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

logger = logging.getLogger(__name__)

# Mapping of Vancouver neighbourhood numeric codes to human-readable names.
# 22 official City of Vancouver local areas, assigned by NeighbourhoodAssigner
# via spatial join against the City's local-area-boundary GeoJSON polygons.
NEIGHBOURHOOD_CODE_NAMES: dict[str, str] = {
    "1": "West Point Grey",
    "2": "Kitsilano",
    "3": "Dunbar-Southlands",
    "4": "Arbutus Ridge",
    "5": "Kerrisdale",
    "6": "Shaughnessy",
    "7": "Fairview",
    "8": "South Cambie",
    "9": "Oakridge",
    "10": "Marpole",
    "11": "Riley Park",
    "12": "Sunset",
    "13": "Mount Pleasant",
    "14": "Grandview-Woodland",
    "15": "Hastings-Sunrise",
    "16": "Kensington-Cedar Cottage",
    "17": "Killarney",
    "18": "Victoria-Fraserview",
    "19": "Strathcona",
    "20": "Renfrew-Collingwood",
    "21": "Downtown",
    "22": "West End",
}

# ============================================================
# GLOBAL STATE
# ============================================================

# These are populated during startup
_predictor: Optional[PropertyPredictor] = None
_properties_df: Optional[pd.DataFrame] = None
_cache: Optional[PredictionCache] = None
_boundary_gdf = None  # City of Vancouver local area boundary GeoDataFrame
_trends_summary: Optional[pd.DataFrame] = None  # Pre-computed trends from raw multi-year CSV

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
            # Compute unified civic_number if not already present.
            # BC Assessment stores strata unit numbers in from_civic_number
            # but the actual street address in to_civic_number. We need both
            # for accurate address matching.
            if "civic_number" not in _properties_df.columns:
                from_c = _properties_df.get(
                    "from_civic_number", pd.Series(dtype=float)
                ).fillna(0).astype(int)
                to_c = _properties_df.get(
                    "to_civic_number", pd.Series(dtype=float)
                ).fillna(0).astype(int)
                _properties_df["civic_number"] = from_c.where(from_c > 0, to_c)
                logger.info(
                    "Computed civic_number: %d properties with civic > 0 "
                    "(was %d from from_civic_number alone)",
                    (_properties_df["civic_number"] > 0).sum(),
                    (from_c > 0).sum(),
                )
            # Fix R1-1 (Residential Inclusive) misclassification.
            # R1-1 is Vancouver's 2024 city-wide rezone — still single-family
            # lots. STRATA on R1-1 are bare-land strata, NOT condos.
            if "zoning_district" in _properties_df.columns:
                r1_strata_mask = (
                    (_properties_df["zoning_district"] == "R1-1")
                    & (_properties_df["property_type"] == "condo")
                )
                n_fixed = r1_strata_mask.sum()
                if n_fixed > 0:
                    _properties_df.loc[r1_strata_mask, "property_type"] = "detached"
                    logger.info(
                        "Reclassified %d R1-1 STRATA properties from "
                        "'condo' to 'detached'",
                        n_fixed,
                    )
            # ----------------------------------------------------------
            # Year_built propagation for strata subdivisions.
            # Strata PIDs often have NaN year_built because BC Assessment
            # creates new PIDs without carrying over the parent lot's data.
            # Fix by looking up same (civic_number, street_name) properties.
            # ----------------------------------------------------------
            if "year_built" in _properties_df.columns and "street_name" in _properties_df.columns:
                missing_yb = _properties_df["year_built"].isna()
                n_missing_before = missing_yb.sum()
                if n_missing_before > 0:
                    # Build lookup: (civic_number, street_name) -> max year_built
                    has_yb = _properties_df[~missing_yb].copy()
                    if "civic_number" in has_yb.columns:
                        has_yb["_civic"] = has_yb["civic_number"].fillna(0).astype(int)
                    else:
                        has_yb["_civic"] = has_yb.get(
                            "to_civic_number", pd.Series(0, index=has_yb.index)
                        ).fillna(0).astype(int)
                    has_yb = has_yb[has_yb["_civic"] > 0]
                    yb_lookup = (
                        has_yb.groupby(["_civic", "street_name"])["year_built"]
                        .max()
                        .to_dict()
                    )

                    # Apply lookup to missing rows
                    if "civic_number" in _properties_df.columns:
                        missing_civic = (
                            _properties_df.loc[missing_yb, "civic_number"]
                            .fillna(0).astype(int)
                        )
                    else:
                        missing_civic = (
                            _properties_df.loc[missing_yb, "to_civic_number"]
                            .fillna(0).astype(int)
                        )
                    missing_street = _properties_df.loc[missing_yb, "street_name"]

                    filled = 0
                    for idx in _properties_df.index[missing_yb]:
                        cv = int(missing_civic.get(idx, 0))
                        st = missing_street.get(idx, "")
                        if cv <= 0 or not st:
                            continue
                        # Exact address match first
                        if (cv, st) in yb_lookup:
                            _properties_df.at[idx, "year_built"] = yb_lookup[(cv, st)]
                            filled += 1
                            continue
                        # Nearby civic numbers (±4) — strata twins on same lot
                        for offset in range(1, 5):
                            for neighbor in (cv - offset, cv + offset):
                                if (neighbor, st) in yb_lookup:
                                    _properties_df.at[idx, "year_built"] = yb_lookup[(neighbor, st)]
                                    filled += 1
                                    break
                            else:
                                continue
                            break

                    if filled > 0:
                        logger.info(
                            "Filled year_built for %d/%d strata properties "
                            "from parent lots",
                            filled, n_missing_before,
                        )

            # ----------------------------------------------------------
            # Year_built from building permits (Vancouver Open Data).
            # For remaining NaN year_built, fetch "New Building" permits
            # and use issue year as construction year proxy.
            # ----------------------------------------------------------
            still_missing = _properties_df["year_built"].isna()
            n_still_missing = still_missing.sum()
            if n_still_missing > 0:
                try:
                    import requests as _requests

                    logger.info(
                        "Fetching building permits to fill %d remaining "
                        "missing year_built values...",
                        n_still_missing,
                    )
                    permits_url = (
                        "https://opendata.vancouver.ca/api/records/1.0/search/"
                    )
                    all_permits: list[dict] = []
                    offset = 0
                    page_size = 100
                    # Vancouver Open Data API v1 has a hard limit of 10,000
                    # records per dataset query. Cap pagination accordingly.
                    max_offset = 9900
                    while offset <= max_offset:
                        resp = _requests.get(
                            permits_url,
                            params={
                                "dataset": "issued-building-permits",
                                "rows": page_size,
                                "start": offset,
                                "refine.typeofwork": "New Building",
                            },
                            timeout=30,
                        )
                        resp.raise_for_status()
                        records = resp.json().get("records", [])
                        if not records:
                            break
                        all_permits.extend(r["fields"] for r in records)
                        offset += len(records)
                        if len(records) < page_size:
                            break

                    if all_permits:
                        permits_df = pd.DataFrame(all_permits)
                        logger.info(
                            "Fetched %d 'New Building' permits", len(permits_df)
                        )

                        # Build lookup: parse civic + street from address,
                        # map to max issueyear
                        import re

                        permit_lookup: dict[tuple[int, str], int] = {}
                        for _, pr in permits_df.iterrows():
                            addr = str(pr.get("address", ""))
                            issue_year = pr.get("issueyear")
                            if not addr or issue_year is None:
                                continue
                            m = re.match(r"^(\d+)\s+(.+?)(?:,|$)", addr.upper())
                            if m:
                                pcivic = int(m.group(1))
                                pstreet = m.group(2).strip()
                                # Normalize: "FREMLIN STREET" -> "FREMLIN ST"
                                pstreet = re.sub(
                                    r"\b(STREET|AVENUE|DRIVE|ROAD|PLACE|"
                                    r"CRESCENT|BOULEVARD|COURT|WAY|LANE|"
                                    r"TERRACE|CIRCLE)\b",
                                    lambda x: {
                                        "STREET": "ST", "AVENUE": "AVE",
                                        "DRIVE": "DR", "ROAD": "RD",
                                        "PLACE": "PL", "CRESCENT": "CRES",
                                        "BOULEVARD": "BLVD", "COURT": "CT",
                                        "WAY": "WAY", "LANE": "LANE",
                                        "TERRACE": "TERR", "CIRCLE": "CIR",
                                    }.get(x.group(0), x.group(0)),
                                    pstreet,
                                )
                                key = (pcivic, pstreet)
                                existing = permit_lookup.get(key, 0)
                                yr = int(issue_year)
                                if yr > existing:
                                    permit_lookup[key] = yr

                        # Apply permit lookup
                        if "civic_number" in _properties_df.columns:
                            mc = (
                                _properties_df.loc[still_missing, "civic_number"]
                                .fillna(0).astype(int)
                            )
                        else:
                            mc = (
                                _properties_df.loc[still_missing, "to_civic_number"]
                                .fillna(0).astype(int)
                            )
                        ms = _properties_df.loc[still_missing, "street_name"]

                        permit_filled = 0
                        for idx in _properties_df.index[still_missing]:
                            cv = int(mc.get(idx, 0))
                            st = ms.get(idx, "")
                            if cv > 0 and st:
                                key = (cv, st)
                                if key in permit_lookup:
                                    _properties_df.at[idx, "year_built"] = float(
                                        permit_lookup[key]
                                    )
                                    permit_filled += 1

                        if permit_filled > 0:
                            logger.info(
                                "Filled year_built for %d properties from "
                                "building permits",
                                permit_filled,
                            )
                except Exception as permit_exc:
                    logger.warning(
                        "Failed to fetch building permits for year_built "
                        "enrichment: %s",
                        permit_exc,
                    )

            # Recompute effective_age after year_built propagation
            if (
                "year_built" in _properties_df.columns
                and "tax_assessment_year" in _properties_df.columns
            ):
                yb = _properties_df["year_built"]
                tay = _properties_df["tax_assessment_year"]
                _properties_df["effective_age"] = np.where(
                    yb.notna(), tay - yb, np.nan
                )

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

    # ----------------------------------------------------------
    # Load City of Vancouver local area boundary GeoJSON.
    # Used for spatial neighbourhood assignment of synthetic
    # properties (new builds not in BC Assessment) via
    # point-in-polygon test against the 22 official boundaries.
    # BC Assessment-matched properties already have correct
    # neighbourhood codes from the training data pipeline.
    # ----------------------------------------------------------
    global _boundary_gdf
    try:
        import geopandas as _gpd

        boundary_url = (
            "https://opendata.vancouver.ca/api/explore/v2.1/"
            "catalog/datasets/local-area-boundary/exports/geojson"
        )
        _boundary_gdf = _gpd.read_file(boundary_url)
        # Ensure CRS is WGS84 (EPSG:4326) for lat/lon queries
        if _boundary_gdf.crs is None or _boundary_gdf.crs.to_epsg() != 4326:
            _boundary_gdf = _boundary_gdf.to_crs(epsg=4326)

        logger.info(
            "Loaded local area boundaries: %d polygons (columns: %s)",
            len(_boundary_gdf),
            list(_boundary_gdf.columns),
        )
    except Exception as boundary_exc:
        logger.warning(
            "Failed to load local area boundaries: %s — "
            "will fall back to BC Assessment neighbourhood codes",
            boundary_exc,
        )
        _boundary_gdf = None

    # ----------------------------------------------------------
    # Pre-compute multi-year trends from raw property tax CSV.
    # The enriched parquet is deduplicated (one row per PID for
    # the latest year), so it lacks historical data. The raw CSV
    # has ~220K records per year for 7 years (2020-2026).
    # ----------------------------------------------------------
    global _trends_summary
    raw_csv_path = Path(_DATA_DIR) / "raw" / "property_tax_all.csv"
    if raw_csv_path.exists() and _properties_df is not None and not _properties_df.empty:
        try:
            logger.info("Loading raw property tax CSV for multi-year trends...")
            raw = pd.read_csv(
                raw_csv_path,
                usecols=[
                    "pid", "tax_assessment_year", "current_land_value",
                    "current_improvement_value", "legal_type",
                ],
                dtype={"pid": str},
            )
            # Normalize PIDs (raw has dashes: 011-307-561, enriched has digits: 011307561)
            raw["pid"] = raw["pid"].str.replace("-", "", regex=False)

            # Map PIDs to City of Vancouver neighbourhood codes via enriched parquet
            pid_to_code = dict(zip(
                _properties_df["pid"].astype(str),
                _properties_df["neighbourhood_code"].astype(str),
            ))
            raw["neighbourhood_code"] = raw["pid"].map(pid_to_code)
            matched = raw["neighbourhood_code"].notna().sum()
            logger.info(
                "Trends PID match rate: %d/%d (%.1f%%)",
                matched, len(raw), matched / len(raw) * 100,
            )
            raw = raw[raw["neighbourhood_code"].notna()].copy()

            # Compute total assessed value
            raw["total_assessed_value"] = (
                raw["current_land_value"].fillna(0)
                + raw["current_improvement_value"].fillna(0)
            )
            raw = raw[raw["total_assessed_value"] > 0]

            # Derive property_type from legal_type (STRATA→condo, LAND→detached)
            raw["property_type"] = raw["legal_type"].map({
                "STRATA": "condo",
                "LAND": "detached",
                "OTHER": "other",
            }).fillna("other")

            _trends_summary = raw[[
                "neighbourhood_code", "tax_assessment_year",
                "total_assessed_value", "property_type",
            ]].copy()
            logger.info(
                "Trends data ready: %d records across years %s",
                len(_trends_summary),
                sorted(_trends_summary["tax_assessment_year"].dropna().unique().astype(int)),
            )
        except Exception as trends_exc:
            logger.warning("Failed to load raw CSV for trends: %s", trends_exc)
            _trends_summary = None
    else:
        if not raw_csv_path.exists():
            logger.info("Raw property tax CSV not found at %s; trends will use enriched data", raw_csv_path)

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
        address=request.address,
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
            boundary_gdf=_boundary_gdf,
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
        neighbourhood_name=NEIGHBOURHOOD_CODE_NAMES.get(
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
    for area_code in sorted(NEIGHBOURHOOD_CODE_NAMES.keys()):
        summary = _compute_market_summary(area_code)
        if summary is not None:
            summaries.append(summary)

    return summaries


@app.get("/api/market/trends")
async def get_market_trends(
    property_type: Optional[str] = Query(None, description="Filter by property type"),
) -> list[dict]:
    """Return year-over-year median value trends for all neighbourhoods.

    Uses the raw multi-year property tax CSV (7 years, ~220K records/year)
    to compute median assessed values per neighbourhood per year.
    Falls back to enriched parquet if raw CSV was not loaded.
    """
    # Use pre-computed trends from raw CSV if available
    if _trends_summary is not None and not _trends_summary.empty:
        df = _trends_summary.copy()

        if property_type and property_type != "all":
            df = df[df["property_type"] == property_type]

        if df.empty:
            return []

        results = []
        for code in sorted(NEIGHBOURHOOD_CODE_NAMES.keys()):
            hood = df[df["neighbourhood_code"] == code]
            if hood.empty:
                continue

            agg = hood.groupby("tax_assessment_year")["total_assessed_value"].agg(
                ["median", "count"]
            )
            trends = [
                {
                    "year": int(year),
                    "median_value": round(float(row["median"]), 0),
                    "count": int(row["count"]),
                }
                for year, row in agg.iterrows()
                if row["count"] >= 20
            ]
            trends.sort(key=lambda t: t["year"])

            if len(trends) >= 2:
                results.append({
                    "neighbourhood_code": code,
                    "neighbourhood_name": NEIGHBOURHOOD_CODE_NAMES.get(code, code),
                    "trends": trends,
                })

        return results

    # Fallback: use enriched parquet (mostly 2026 data)
    if _properties_df is None or _properties_df.empty:
        return []

    df = _properties_df.copy()
    if property_type and property_type != "all" and "property_type" in df.columns:
        df = df[df["property_type"] == property_type]

    if df.empty:
        return []

    results = []
    for code in sorted(NEIGHBOURHOOD_CODE_NAMES.keys()):
        hood = df[df["neighbourhood_code"] == code]
        if hood.empty:
            continue

        trends = []
        if "tax_assessment_year" in hood.columns:
            for year, group in hood.groupby("tax_assessment_year"):
                yr = int(year)
                median_val = float(group["total_assessed_value"].median())
                count = len(group)
                if count >= 20:
                    trends.append({
                        "year": yr,
                        "median_value": round(median_val, 0),
                        "count": count,
                    })

        trends.sort(key=lambda t: t["year"])
        if len(trends) >= 2:
            results.append({
                "neighbourhood_code": code,
                "neighbourhood_name": NEIGHBOURHOOD_CODE_NAMES.get(code, code),
                "trends": trends,
            })

    return results


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

    if code not in NEIGHBOURHOOD_CODE_NAMES:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Neighbourhood '{neighbourhood_code}' not found. "
                f"Valid codes: {sorted(NEIGHBOURHOOD_CODE_NAMES.keys())}"
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
        for code, name in sorted(NEIGHBOURHOOD_CODE_NAMES.items())
    ]


@app.get("/api/search")
async def search_properties(
    q: str = Query(..., min_length=2, description="Search query (address or PID)"),
    limit: int = Query(10, ge=1, le=50, description="Max results to return"),
) -> list[dict]:
    """Search properties by address or PID for autocomplete.

    Smart search: splits query into numeric (civic number) and text (street name)
    parts. Matches street name even if civic number doesn't exist — shows closest
    civic numbers on that street. Also searches PIDs.
    """
    if _properties_df is None or _properties_df.empty:
        return []

    query = q.strip().upper()
    df = _properties_df

    # Build unified civic number: prefer civic_number (from property universe),
    # else from_civic_number, else to_civic_number.  Many non-strata parcels
    # store the real street address only in to_civic_number.
    if "civic_number" in df.columns:
        civic_col = df["civic_number"].fillna(0).astype(int)
    else:
        from_c = df["from_civic_number"].fillna(0).astype(int)
        to_c = df.get("to_civic_number", pd.Series(0, index=df.index)).fillna(0).astype(int)
        civic_col = from_c.where(from_c > 0, to_c)
    street_col = df["street_name"].fillna("")

    # Separate numeric and text parts of query
    parts = query.split()
    numeric_parts = [p for p in parts if p.replace("-", "").isdigit()]
    text_parts = [p for p in parts if not p.replace("-", "").isdigit()]

    # Pure PID search (all digits)
    is_pid_only = len(text_parts) == 0 and len(numeric_parts) > 0
    if is_pid_only:
        pid_clean = query.replace("-", "").replace(" ", "")
        mask = df["pid"].astype(str).str.contains(pid_clean, na=False)
        results = df[mask].head(limit)
    else:
        # Street name matching: match text parts against start of street name words
        # Use word-boundary anchoring so "FRE" matches "FREMLIN" but not "RENFREW"
        street_mask = pd.Series(True, index=df.index)
        for part in text_parts:
            # Match at start of any word in the street name (\\b = word boundary)
            pattern = r"(?:^|[\s\-])" + part
            street_mask = street_mask & street_col.str.contains(
                pattern, na=False, regex=True,
            )

        if numeric_parts:
            target_civic = int(numeric_parts[0].replace("-", ""))
            # Try exact civic + street match first
            exact_mask = street_mask & (civic_col == target_civic)
            if exact_mask.sum() > 0:
                # Sort by most recent assessment year first
                exact_df = df[exact_mask]
                if "tax_assessment_year" in exact_df.columns:
                    exact_df = exact_df.sort_values("tax_assessment_year", ascending=False, na_position="last")
                results = exact_df.head(limit)
            else:
                # No exact match — show closest civic numbers on that street
                street_matches = df[street_mask & (civic_col > 0)].copy()
                if len(street_matches) > 0:
                    street_matches["_civic_dist"] = (
                        civic_col[street_matches.index] - target_civic
                    ).abs()
                    results = street_matches.nsmallest(limit, "_civic_dist")
                else:
                    # Fall back to street name only (including those without civic)
                    results = df[street_mask].head(limit)
        else:
            # Text-only search (just street name)
            # Prefer results that have a civic number
            has_civic = street_mask & (civic_col > 0)
            if has_civic.sum() > 0:
                results = df[has_civic].head(limit)
            else:
                results = df[street_mask].head(limit)

    out = []
    for _, row in results.iterrows():
        # Use unified civic_number if available, else from_civic → to_civic
        _cn = row.get("civic_number")
        if pd.notna(_cn) and int(_cn) > 0:
            civic_num = int(_cn)
        elif pd.notna(row.get("from_civic_number")) and int(row["from_civic_number"]) > 0:
            civic_num = int(row["from_civic_number"])
        elif pd.notna(row.get("to_civic_number")) and int(row["to_civic_number"]) > 0:
            civic_num = int(row["to_civic_number"])
        else:
            civic_num = None
        street_nm = str(row.get("street_name", ""))
        addr = f"{civic_num} {street_nm}" if civic_num else street_nm
        neighbourhood_code = str(row.get("neighbourhood_code", ""))
        out.append({
            "pid": str(row.get("pid", "")),
            "address": addr,
            "property_type": str(row.get("property_type", "")),
            "neighbourhood": NEIGHBOURHOOD_CODE_NAMES.get(
                neighbourhood_code, neighbourhood_code,
            ),
            "assessed_value": float(row.get("total_assessed_value", 0)),
        })

    return out


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
        neighbourhood_name=NEIGHBOURHOOD_CODE_NAMES.get(
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

    # Build metadata
    metadata = PredictionMetadata(
        model_segment=result.model_segment,
        model_version=result.model_version,
        prediction_timestamp=result.prediction_timestamp.isoformat(),
        data_freshness=None,
        mls_available=False,
    )

    # Use market estimate as primary if available (trained on actual sold prices)
    primary_estimate = result.point_estimate
    if result.market_estimate is not None:
        primary_estimate = result.market_estimate

    # Build confidence interval — scale to match primary estimate
    ci_lower, ci_upper = result.confidence_interval
    if result.point_estimate > 0 and primary_estimate != result.point_estimate:
        # CI was computed around the assessment model estimate; rescale
        # to center around the market model estimate
        scale = primary_estimate / result.point_estimate
        ci_lower *= scale
        ci_upper *= scale

    # Ensure CI is sensible: brackets the estimate with reasonable width
    ci_width_pct = (
        (ci_upper - ci_lower) / primary_estimate * 100
        if primary_estimate > 0 else 0
    )
    if (
        ci_lower <= 0
        or ci_upper <= 0
        or ci_lower > primary_estimate
        or ci_upper < primary_estimate
        or ci_width_pct < 5  # absurdly narrow (< 5% spread)
    ):
        # Fallback: use MAPE-based interval from market model
        mape_pct = 0.15  # default 15%
        if result.market_model_info:
            # Extract MAPE from info string like "market_detached (MAPE=9.22%, n=236)"
            import re
            m = re.search(r"MAPE=([\d.]+)%", result.market_model_info)
            if m:
                mape_pct = float(m.group(1)) / 100.0
        ci_lower = primary_estimate * (1 - mape_pct)
        ci_upper = primary_estimate * (1 + mape_pct)

    confidence_interval = ConfidenceInterval(
        lower=round(ci_lower, 2),
        upper=round(ci_upper, 2),
        level=0.80,
    )

    return PredictionResponse(
        point_estimate=round(primary_estimate, 2),
        confidence_interval=confidence_interval,
        confidence_grade=result.confidence_grade,
        comparables=comparables,
        shap_features=shap_features,
        adjustments=adjustments,
        market_context=market_context,
        risk_flags=risk_flags,
        metadata=metadata,
        market_estimate=round(result.market_estimate, 2) if result.market_estimate else None,
        market_model_info=result.market_model_info,
    )
