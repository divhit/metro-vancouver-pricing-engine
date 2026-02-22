"""
Market time adjustment for BC Assessment values.

BC Assessment values have a July 1 valuation date and reflect the market
at that point. This module adjusts from assessment date to current market
using interest rate changes and market indices.

Key insight: Every 100bp rate change moves Vancouver property prices ~3-5%,
with condos being more rate-sensitive than detached homes.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from src.features.feature_registry import PropertyType

logger = logging.getLogger(__name__)

# Rate sensitivity by property type (% change per +100bp rate increase)
# Condos are most rate-sensitive because buyers are most mortgage-dependent.
# Land value is relatively rate-insensitive (supply-constrained, optionality value).
RATE_SENSITIVITY = {
    PropertyType.CONDO: -0.05,             # -5% per +100bp
    PropertyType.TOWNHOME: -0.04,          # -4% per +100bp
    PropertyType.DETACHED: -0.03,          # -3% per +100bp
    PropertyType.DEVELOPMENT_LAND: -0.02,  # -2% per +100bp
}


class MarketAdjustmentEngine:
    """Adjusts assessed values from assessment date to current market conditions.

    BC Assessment values are set as of July 1 each year and mailed the
    following January. By the time a buyer or seller is using the assessment
    as a reference, 6-18 months may have passed. This engine bridges that
    gap using interest rate movements and seasonal patterns.
    """

    def adjust_assessment_to_current(
        self,
        assessed_value: float,
        assessment_date: date,
        valuation_date: date,
        property_type: PropertyType,
        assessment_rate: Optional[float] = None,
        current_rate: Optional[float] = None,
    ) -> tuple[float, float, str]:
        """Time-adjust an assessed value to the valuation date.

        Args:
            assessed_value: BC Assessment value (as of assessment_date)
            assessment_date: The date the assessment reflects (typically July 1)
            valuation_date: The target date for the adjusted value
            property_type: Property type for rate sensitivity lookup
            assessment_rate: 5-yr fixed mortgage rate at assessment_date (%)
            current_rate: 5-yr fixed mortgage rate at valuation_date (%)

        Returns:
            Tuple of (adjusted_value, adjustment_pct, explanation)
        """
        months_delta = (
            (valuation_date.year - assessment_date.year) * 12
            + (valuation_date.month - assessment_date.month)
        )

        if months_delta == 0:
            return assessed_value, 0.0, "Assessment date matches valuation date — no time adjustment"

        # Interest rate adjustment
        rate_factor = 1.0
        rate_explanation = ""

        if assessment_rate is not None and current_rate is not None:
            rate_change_bps = (current_rate - assessment_rate) * 100
            rate_factor = self.compute_interest_rate_sensitivity(
                rate_change_bps, property_type
            )
            direction = "increase" if rate_change_bps > 0 else "decrease"
            rate_explanation = (
                f"Rate {direction} of {abs(rate_change_bps):.0f}bp "
                f"({assessment_rate:.2f}% -> {current_rate:.2f}%) "
                f"implies {(rate_factor - 1.0) * 100:+.1f}% price effect"
            )
        else:
            rate_explanation = "No rate data available — rate adjustment skipped"

        # Seasonal adjustment
        seasonal_factor = self.compute_seasonal_factor(valuation_date.month)
        seasonal_pct = (seasonal_factor - 1.0) * 100

        # Combined adjustment
        combined_factor = rate_factor * seasonal_factor
        adjusted_value = assessed_value * combined_factor
        adjustment_pct = (combined_factor - 1.0) * 100

        # Build explanation
        parts = [
            f"Time adjustment over {months_delta} months "
            f"({assessment_date.isoformat()} -> {valuation_date.isoformat()})",
        ]
        if rate_explanation:
            parts.append(rate_explanation)
        if seasonal_pct != 0.0:
            season = self._month_to_season(valuation_date.month)
            parts.append(
                f"Seasonal factor ({season}): {seasonal_pct:+.1f}%"
            )
        parts.append(f"Net adjustment: {adjustment_pct:+.1f}%")
        explanation = "; ".join(parts)

        logger.debug(
            "Market adjustment: %s -> %s (%.1f%%)",
            f"${assessed_value:,.0f}",
            f"${adjusted_value:,.0f}",
            adjustment_pct,
        )

        return round(adjusted_value, 2), round(adjustment_pct, 2), explanation

    def compute_interest_rate_sensitivity(
        self,
        rate_change_bps: float,
        property_type: PropertyType,
    ) -> float:
        """Compute the multiplicative price factor from an interest rate change.

        Args:
            rate_change_bps: Change in 5-yr fixed rate in basis points
                (positive = rates went up)
            property_type: Property type for sensitivity lookup

        Returns:
            Multiplicative factor (e.g., 0.95 means 5% price decrease)
        """
        sensitivity = RATE_SENSITIVITY.get(property_type, -0.03)

        # sensitivity is per 100bp, so scale linearly
        factor = 1.0 + sensitivity * (rate_change_bps / 100.0)

        # Clamp to reasonable bounds — rates alone shouldn't move prices >25%
        factor = max(0.75, min(1.25, factor))

        return round(factor, 4)

    def compute_seasonal_factor(self, month: int) -> float:
        """Compute seasonal adjustment factor for a given month.

        Vancouver market seasonality:
        - Spring (March-May): peak listing/buying season, strongest prices
        - Fall (Sept-Nov): secondary season, moderate strength
        - Winter (Dec-Feb): weakest period, fewer buyers
        - Summer (June-Aug): neutral, families settled

        Args:
            month: Calendar month (1-12)

        Returns:
            Multiplicative seasonal factor
        """
        if month in (3, 4, 5):
            return 1.02  # Spring: +2%
        elif month in (9, 10, 11):
            return 1.01  # Fall: +1%
        elif month in (12, 1, 2):
            return 0.98  # Winter: -2%
        else:
            return 1.00  # Summer: neutral

    @staticmethod
    def _month_to_season(month: int) -> str:
        """Convert month number to season name."""
        if month in (3, 4, 5):
            return "spring"
        elif month in (6, 7, 8):
            return "summer"
        elif month in (9, 10, 11):
            return "fall"
        else:
            return "winter"
