"""
Tier 2 adjustment engine.

Orchestrates all rules-based adjustments that are applied after the
ML model (Tier 1) prediction. Adjustments are applied sequentially
and multiplicatively.

Order:
1. Market time adjustment (assessment -> current)
2. Leasehold discount (if applicable)
3. Assembly premium (if in development corridor)
4. Supply pipeline (nearby construction effect)
5. Strata health (if strata property with Form B data)
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from src.adjustments.assembly_premium import AssemblyPremiumCalculator
from src.adjustments.leasehold import LeaseholdType, apply_leasehold_adjustment
from src.adjustments.market_adjustments import MarketAdjustmentEngine
from src.adjustments.strata_health import StrataHealthAdjuster
from src.adjustments.supply_pipeline import SupplyPipelineAdjuster
from src.features.feature_registry import PropertyType
from src.models.types import AdjustmentResult

logger = logging.getLogger(__name__)


class AdjustmentEngine:
    """Master orchestrator for all Tier 2 (post-model) adjustments.

    Takes an ML model's raw prediction and applies a sequence of
    rules-based corrections to account for factors that are difficult
    to learn purely from data:

    1. **Market time adjustment**: Bridges the gap between BC Assessment's
       July 1 valuation date and the current market using interest rate
       movements and seasonal patterns.

    2. **Leasehold discount**: Applies the non-linear leasehold discount
       curve for prepaid, non-prepaid, and First Nations leaseholds.

    3. **Assembly premium**: Adds a premium for properties in active
       development corridors (Broadway Plan, Cambie Corridor, etc.).

    4. **Supply pipeline**: Adjusts for nearby construction activity —
       supply pressure for condos, demand signal for development land.

    5. **Strata health**: Penalizes buildings with financial, governance,
       or construction-era risk factors.

    Adjustments are applied multiplicatively in sequence. Each adjustment
    modifies the running estimate, and all adjustments are tracked for
    transparency and auditability.
    """

    def __init__(self) -> None:
        self.market_engine = MarketAdjustmentEngine()
        self.assembly_calculator = AssemblyPremiumCalculator()
        self.supply_adjuster = SupplyPipelineAdjuster()
        self.strata_adjuster = StrataHealthAdjuster()

    def apply_all_adjustments(
        self,
        ml_estimate: float,
        property_data: dict,
        valuation_date: Optional[date] = None,
    ) -> AdjustmentResult:
        """Apply all applicable Tier 2 adjustments to the ML estimate.

        Args:
            ml_estimate: The Tier 1 ML model's point estimate ($)
            property_data: Dictionary of property features. Expected keys:
                - property_type (str): "condo", "townhome", "detached", "development_land"
                - latitude (float): Property latitude
                - longitude (float): Property longitude
                - assessment_date (str/date): Assessment valuation date
                - assessment_rate (float, optional): Mortgage rate at assessment date
                - current_rate (float, optional): Current mortgage rate
                - lease_remaining_years (float, optional): Years remaining on lease
                - lease_type (str, optional): "freehold", "prepaid", "non_prepaid", "first_nations"
                - zoning_code (str, optional): Current zoning
                - neighbourhood_code (str, optional): GVR sub-area
                - lot_frontage_ft (float, optional): Lot frontage in feet
                - is_corner_lot (bool, optional): Corner lot flag
                - year_built (int, optional): Year of construction
                - depreciation_report_age_months (int, optional)
                - contingency_reserve_pct (float, optional)
                - special_assessment_count_5yr (int, optional)
                - special_assessment_total_5yr (float, optional)
                - insurance_deductible (float, optional)
                - rainscreen_status (str, optional)
                - crt_dispute_count_3yr (int, optional)
            valuation_date: Target date for the valuation (defaults to today)

        Returns:
            AdjustmentResult with adjusted_value, adjustments list, and total_adjustment_pct
        """
        if valuation_date is None:
            valuation_date = date.today()

        property_type = self._resolve_property_type(
            property_data.get("property_type", "detached")
        )

        running_estimate = ml_estimate
        adjustments: list[tuple[str, float, str]] = []

        # ------------------------------------------------------------------
        # 1. Market time adjustment (always apply — assessment date is known)
        # ------------------------------------------------------------------
        assessment_date = property_data.get("assessment_date")
        if assessment_date is not None:
            if isinstance(assessment_date, str):
                assessment_date = date.fromisoformat(assessment_date)

            adjusted, pct, explanation = self.market_engine.adjust_assessment_to_current(
                assessed_value=running_estimate,
                assessment_date=assessment_date,
                valuation_date=valuation_date,
                property_type=property_type,
                assessment_rate=property_data.get("assessment_rate"),
                current_rate=property_data.get("current_rate"),
            )
            running_estimate = adjusted
            adjustments.append(("Market Time Adjustment", pct, explanation))

            logger.info(
                "Market adjustment applied: %+.2f%% -> $%,.0f",
                pct, running_estimate,
            )

        # ------------------------------------------------------------------
        # 2. Leasehold discount (if applicable)
        # ------------------------------------------------------------------
        if self._should_apply_leasehold(property_data):
            lease_type_str = property_data.get("lease_type", "prepaid")
            try:
                lease_type = LeaseholdType(lease_type_str)
            except ValueError:
                lease_type = LeaseholdType.PREPAID

            adjusted, discount_pct, explanation = apply_leasehold_adjustment(
                freehold_estimate=running_estimate,
                remaining_years=property_data.get("lease_remaining_years"),
                lease_type=lease_type,
            )
            if discount_pct != 0.0:
                running_estimate = adjusted
                adjustments.append(("Leasehold Discount", -discount_pct, explanation))

                logger.info(
                    "Leasehold adjustment applied: -%.2f%% -> $%,.0f",
                    discount_pct, running_estimate,
                )

        # ------------------------------------------------------------------
        # 3. Assembly premium (if in development corridor)
        # ------------------------------------------------------------------
        lat = property_data.get("latitude")
        lon = property_data.get("longitude")

        if lat is not None and lon is not None:
            premium_pct, explanation = self.assembly_calculator.compute_assembly_premium(
                lat=lat,
                lon=lon,
                zoning_code=property_data.get("zoning_code"),
                neighbourhood_code=property_data.get("neighbourhood_code"),
                lot_frontage_ft=property_data.get("lot_frontage_ft"),
                is_corner=property_data.get("is_corner_lot", False),
            )
            if premium_pct > 0.0:
                running_estimate *= (1.0 + premium_pct)
                adjustments.append(
                    ("Assembly Premium", round(premium_pct * 100, 2), explanation)
                )

                logger.info(
                    "Assembly premium applied: +%.2f%% -> $%,.0f",
                    premium_pct * 100, running_estimate,
                )

        # ------------------------------------------------------------------
        # 4. Supply pipeline (always apply if lat/lon available)
        # ------------------------------------------------------------------
        if lat is not None and lon is not None:
            supply_pct, explanation = self.supply_adjuster.compute_supply_adjustment(
                lat=lat,
                lon=lon,
                property_type=property_type,
            )
            if supply_pct != 0.0:
                running_estimate *= (1.0 + supply_pct)
                adjustments.append(
                    ("Supply Pipeline", round(supply_pct * 100, 2), explanation)
                )

                logger.info(
                    "Supply pipeline adjustment applied: %+.2f%% -> $%,.0f",
                    supply_pct * 100, running_estimate,
                )

        # ------------------------------------------------------------------
        # 5. Strata health (condos and townhomes with strata data)
        # ------------------------------------------------------------------
        if self._should_apply_strata(property_data, property_type):
            strata_pct, explanation = self.strata_adjuster.compute_strata_adjustment(
                year_built=property_data.get("year_built"),
                depreciation_report_age_months=property_data.get("depreciation_report_age_months"),
                contingency_reserve_pct=property_data.get("contingency_reserve_pct"),
                special_assessment_count_5yr=property_data.get("special_assessment_count_5yr"),
                special_assessment_total_5yr=property_data.get("special_assessment_total_5yr"),
                insurance_deductible=property_data.get("insurance_deductible"),
                rainscreen_status=property_data.get("rainscreen_status"),
                crt_dispute_count_3yr=property_data.get("crt_dispute_count_3yr"),
            )
            if strata_pct != 0.0:
                running_estimate *= (1.0 + strata_pct)
                adjustments.append(
                    ("Strata Health", round(strata_pct * 100, 2), explanation)
                )

                logger.info(
                    "Strata health adjustment applied: %.2f%% -> $%,.0f",
                    strata_pct * 100, running_estimate,
                )

        # ------------------------------------------------------------------
        # Compute total adjustment percentage
        # ------------------------------------------------------------------
        if ml_estimate > 0:
            total_adjustment_pct = round(
                ((running_estimate - ml_estimate) / ml_estimate) * 100, 2
            )
        else:
            total_adjustment_pct = 0.0

        running_estimate = round(running_estimate, 2)

        logger.info(
            "All adjustments complete: $%,.0f -> $%,.0f (net %+.2f%%, %d adjustments)",
            ml_estimate, running_estimate, total_adjustment_pct, len(adjustments),
        )

        return AdjustmentResult(
            adjusted_value=running_estimate,
            adjustments=adjustments,
            total_adjustment_pct=total_adjustment_pct,
        )

    @staticmethod
    def _should_apply_leasehold(property_data: dict) -> bool:
        """Determine whether leasehold adjustment should be applied.

        Args:
            property_data: Property feature dictionary

        Returns:
            True if the property is leasehold and has remaining years data
        """
        lease_remaining = property_data.get("lease_remaining_years")
        lease_type = property_data.get("lease_type", "freehold")

        if lease_remaining is None:
            return False
        if lease_type == "freehold" or lease_type == LeaseholdType.FREEHOLD:
            return False

        return True

    @staticmethod
    def _should_apply_strata(
        property_data: dict,
        property_type: PropertyType,
    ) -> bool:
        """Determine whether strata health adjustment should be applied.

        Only applies to condos and townhomes, and only when at least
        some strata data fields are present.

        Args:
            property_data: Property feature dictionary
            property_type: Resolved PropertyType enum

        Returns:
            True if strata adjustment should be applied
        """
        if property_type not in (PropertyType.CONDO, PropertyType.TOWNHOME):
            return False

        # Check if any strata-specific data fields are present
        strata_fields = [
            "depreciation_report_age_months",
            "contingency_reserve_pct",
            "special_assessment_count_5yr",
            "special_assessment_total_5yr",
            "insurance_deductible",
            "rainscreen_status",
            "crt_dispute_count_3yr",
        ]

        return any(
            property_data.get(field) is not None
            for field in strata_fields
        )

    @staticmethod
    def _resolve_property_type(raw: str) -> PropertyType:
        """Resolve a raw property type string to a PropertyType enum.

        Args:
            raw: Property type as a string (e.g., "condo", "DETACHED")

        Returns:
            PropertyType enum value
        """
        normalized = raw.strip().lower()
        try:
            return PropertyType(normalized)
        except ValueError:
            logger.warning(
                "Unknown property type '%s' — defaulting to DETACHED", raw
            )
            return PropertyType.DETACHED
