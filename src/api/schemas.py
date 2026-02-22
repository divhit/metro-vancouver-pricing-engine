"""Pydantic request/response models for the prediction API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ============================================================
# REQUEST
# ============================================================


class PredictionRequest(BaseModel):
    """Request body for the /api/predict endpoint.

    At least one of pid, address, or (latitude + longitude) must be
    provided so the engine can identify or locate the property.
    """

    pid: Optional[str] = Field(
        None,
        description="BC Assessment Property Identifier (PID)",
        examples=["012-345-678"],
    )
    address: Optional[str] = Field(
        None,
        description="Street address for display and fallback lookup",
        examples=["1234 Main St, Vancouver, BC"],
    )
    latitude: Optional[float] = Field(
        None,
        description="Latitude in decimal degrees",
        ge=48.5,
        le=49.6,
        examples=[49.2827],
    )
    longitude: Optional[float] = Field(
        None,
        description="Longitude in decimal degrees",
        ge=-123.5,
        le=-122.5,
        examples=[-123.1207],
    )
    property_type: Optional[str] = Field(
        None,
        description="Property type override (condo, townhome, detached, development_land)",
        examples=["condo"],
    )
    overrides: Optional[dict[str, float | int | str]] = Field(
        None,
        description=(
            "Feature overrides provided by the user. Keys are feature names "
            "(e.g. living_area_sqft, bedrooms, year_built) and values replace "
            "computed features."
        ),
        examples=[{"living_area_sqft": 1200, "bedrooms": 2}],
    )

    @model_validator(mode="after")
    def validate_at_least_one_identifier(self) -> PredictionRequest:
        """Ensure at least one of pid, address, or (lat+lon) is provided."""
        has_pid = self.pid is not None
        has_address = self.address is not None
        has_coords = self.latitude is not None and self.longitude is not None

        if not (has_pid or has_address or has_coords):
            raise ValueError(
                "At least one of 'pid', 'address', or both "
                "'latitude' and 'longitude' must be provided."
            )

        return self


# ============================================================
# RESPONSE SUB-MODELS
# ============================================================


class ConfidenceInterval(BaseModel):
    """Prediction interval bounds."""

    lower: float = Field(description="Lower bound of the confidence interval ($)")
    upper: float = Field(description="Upper bound of the confidence interval ($)")
    level: float = Field(
        default=0.80,
        description="Confidence level (e.g. 0.80 for 80% interval)",
    )


class ShapFeature(BaseModel):
    """A single SHAP feature contribution to the prediction."""

    feature_name: str = Field(description="Name of the feature")
    feature_value: Optional[float | int | str] = Field(
        None,
        description="The property's value for this feature",
    )
    shap_value: float = Field(
        description="SHAP contribution (positive = pushes value up)",
    )
    direction: str = Field(
        description="Direction of contribution: 'up' or 'down'",
    )
    description: str = Field(
        description="Human-readable explanation of this feature's impact",
    )


class ComparableDTO(BaseModel):
    """A comparable property used in the valuation."""

    pid: str = Field(description="PID of the comparable property")
    address: str = Field(description="Street address")
    assessed_value: float = Field(description="Total assessed value ($)")
    distance_m: float = Field(description="Distance from subject property (metres)")
    similarity_score: float = Field(
        description="Similarity score (lower = more similar)",
    )
    year_built: Optional[int] = Field(None, description="Year of construction")
    zoning: Optional[str] = Field(None, description="Zoning code")
    neighbourhood: Optional[str] = Field(None, description="Local area name")


class AdjustmentDTO(BaseModel):
    """A Tier 2 post-model adjustment applied to the estimate."""

    name: str = Field(description="Name of the adjustment")
    percentage: float = Field(description="Adjustment percentage")
    dollar_amount: float = Field(description="Dollar impact of the adjustment")
    explanation: str = Field(description="Human-readable explanation")


class RiskFlag(BaseModel):
    """A risk condition identified for the property."""

    category: str = Field(
        description="Risk category (leasehold, environmental, building_envelope, land_use, data_quality)",
    )
    severity: str = Field(
        description="Risk severity: low, medium, or high",
    )
    description: str = Field(
        description="Detailed description of the risk condition",
    )


class MarketContext(BaseModel):
    """Neighbourhood-level market context for the prediction."""

    neighbourhood_code: str = Field(description="Vancouver local area code")
    neighbourhood_name: str = Field(description="Human-readable local area name")
    median_assessed_value: float = Field(
        description="Median assessed value in this neighbourhood ($)",
    )
    yoy_change_pct: Optional[float] = Field(
        None,
        description="Year-over-year change in median value (%)",
    )
    interest_rate_5yr: Optional[float] = Field(
        None,
        description="Current 5-year fixed mortgage rate (%)",
    )
    property_count: int = Field(
        description="Number of properties in this neighbourhood",
    )
    assessment_year: int = Field(description="Assessment year of the data")


class PredictionMetadata(BaseModel):
    """Metadata about the prediction process."""

    model_segment: str = Field(
        description="Segment model used for this prediction",
    )
    model_version: str = Field(
        default="0.1.0",
        description="Version of the pricing engine",
    )
    prediction_timestamp: str = Field(
        description="ISO 8601 timestamp of the prediction",
    )
    data_freshness: Optional[str] = Field(
        None,
        description="Date of the most recent data used",
    )
    mls_available: bool = Field(
        default=False,
        description="Whether MLS listing data was available",
    )


# ============================================================
# MAIN RESPONSE
# ============================================================


class PredictionResponse(BaseModel):
    """Complete prediction response from the pricing engine."""

    point_estimate: float = Field(
        description="Final blended property value estimate ($)",
    )
    confidence_interval: ConfidenceInterval = Field(
        description="Prediction interval",
    )
    confidence_grade: str = Field(
        description="Confidence grade: A (high), B (moderate), C (low)",
    )
    comparables: list[ComparableDTO] = Field(
        description="Top comparable properties used in valuation",
    )
    shap_features: list[ShapFeature] = Field(
        description="Top SHAP feature contributions",
    )
    adjustments: list[AdjustmentDTO] = Field(
        description="Tier 2 adjustments applied to the ML estimate",
    )
    market_context: MarketContext = Field(
        description="Neighbourhood-level market context",
    )
    risk_flags: list[RiskFlag] = Field(
        description="Risk conditions identified for the property",
    )
    metadata: PredictionMetadata = Field(
        description="Prediction metadata",
    )


# ============================================================
# PROPERTY DETAIL
# ============================================================


class PropertyDetail(BaseModel):
    """Enriched property detail response for the /api/property/{pid} endpoint."""

    pid: str = Field(description="BC Assessment PID")
    address: str = Field(description="Full street address")
    neighbourhood_code: str = Field(description="Vancouver local area code")
    neighbourhood_name: str = Field(description="Human-readable local area name")
    property_type: str = Field(description="Classified property type")
    zoning: Optional[str] = Field(None, description="Zoning district code")
    year_built: Optional[int] = Field(None, description="Year of construction")
    land_value: float = Field(description="Current land value ($)")
    improvement_value: float = Field(description="Current improvement value ($)")
    total_assessed_value: float = Field(description="Total assessed value ($)")
    estimated_living_area_sqft: Optional[float] = Field(
        None,
        description="Estimated living area in square feet",
    )
    latitude: float = Field(description="Latitude in decimal degrees")
    longitude: float = Field(description="Longitude in decimal degrees")


# ============================================================
# MARKET SUMMARY
# ============================================================


class MarketSummary(BaseModel):
    """Market summary for a neighbourhood."""

    neighbourhood_code: str = Field(description="Vancouver local area code")
    neighbourhood_name: str = Field(description="Human-readable local area name")
    property_counts: dict[str, int] = Field(
        description="Property counts by property type",
    )
    median_values: dict[str, float] = Field(
        description="Median assessed values by property type ($)",
    )
    yoy_changes: dict[str, Optional[float]] = Field(
        description="Year-over-year changes by property type (%)",
    )
    interest_rate: Optional[float] = Field(
        None,
        description="Current 5-year fixed mortgage rate (%)",
    )


# ============================================================
# HEALTH CHECK
# ============================================================


class HealthResponse(BaseModel):
    """Service health check response."""

    status: str = Field(description="Service status (ok, degraded, error)")
    model_count: int = Field(description="Number of trained models available")
    data_freshness: dict[str, str] = Field(
        description="Data freshness per source",
    )
    version: str = Field(description="API version string")
