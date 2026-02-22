"""
Strata health adjustment.

Strata-specific risk factors that affect property values.
Most of this data requires Form B certificates (available via eStrataHub)
which are not yet in the pipeline. This module provides the adjustment
calculations for when the data becomes available.

Key risk factors:
- Depreciation report currency (mandatory every 5 years as of July 2024)
- Contingency reserve adequacy (25% of annual operating minimum)
- Special assessment history
- Insurance deductible levels (post-2019 crisis)
- Construction era (1981-1999 leaky condo risk)
- Rainscreen remediation status
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Leaky condo era boundaries
LEAKY_CONDO_ERA_START = 1981
LEAKY_CONDO_ERA_END = 1999

# Maximum total penalty cap
MAX_PENALTY = -0.25  # -25%


class StrataHealthAdjuster:
    """Adjusts property values based on strata corporation health indicators.

    Strata buildings carry risk factors that are not captured by structural
    features alone. This adjuster evaluates governance, financial health,
    maintenance status, and construction-era risk to produce a penalty
    (or zero if the building is healthy).

    All adjustments are penalties (negative or zero). A well-maintained
    building with current reports and adequate reserves receives no penalty.
    """

    def compute_strata_adjustment(
        self,
        year_built: Optional[int] = None,
        depreciation_report_age_months: Optional[int] = None,
        contingency_reserve_pct: Optional[float] = None,
        special_assessment_count_5yr: Optional[int] = None,
        special_assessment_total_5yr: Optional[float] = None,
        insurance_deductible: Optional[float] = None,
        rainscreen_status: Optional[str] = None,
        crt_dispute_count_3yr: Optional[int] = None,
    ) -> tuple[float, str]:
        """Compute the total strata health penalty.

        Each factor is assessed independently. Penalties are additive
        (not multiplicative) and the total is capped at -25%.

        Args:
            year_built: Year the building was constructed
            depreciation_report_age_months: Months since last depreciation report
            contingency_reserve_pct: Contingency reserve as % of annual operating
            special_assessment_count_5yr: Number of special assessments in past 5 years
            special_assessment_total_5yr: Total special assessment dollars in past 5 years
            insurance_deductible: Building insurance deductible amount ($)
            rainscreen_status: "Yes" / "No" / "Partial" / None
            crt_dispute_count_3yr: CRT strata disputes in past 3 years

        Returns:
            Tuple of (adjustment_pct, explanation)
            adjustment_pct is negative (penalty) or 0.0
        """
        penalties = []

        # 1. Depreciation report currency
        dep_penalty, dep_note = self._assess_depreciation_report(
            depreciation_report_age_months
        )
        if dep_penalty != 0.0:
            penalties.append((dep_penalty, dep_note))

        # 2. Contingency reserve adequacy
        res_penalty, res_note = self._assess_contingency_reserve(
            contingency_reserve_pct
        )
        if res_penalty != 0.0:
            penalties.append((res_penalty, res_note))

        # 3. Special assessment history
        sa_penalty, sa_note = self._assess_special_assessments(
            special_assessment_count_5yr, special_assessment_total_5yr
        )
        if sa_penalty != 0.0:
            penalties.append((sa_penalty, sa_note))

        # 4. Insurance deductible
        ins_penalty, ins_note = self._assess_insurance_deductible(
            insurance_deductible
        )
        if ins_penalty != 0.0:
            penalties.append((ins_penalty, ins_note))

        # 5. Leaky condo era risk
        lc_penalty, lc_note = self._assess_leaky_condo_risk(
            year_built, rainscreen_status
        )
        if lc_penalty != 0.0:
            penalties.append((lc_penalty, lc_note))

        # 6. CRT dispute frequency (governance red flag)
        crt_penalty, crt_note = self._assess_crt_disputes(
            crt_dispute_count_3yr
        )
        if crt_penalty != 0.0:
            penalties.append((crt_penalty, crt_note))

        # Sum all penalties
        if not penalties:
            return 0.0, "No strata health data available or no penalties apply"

        total_penalty = sum(p for p, _ in penalties)

        # Cap total penalty
        if total_penalty < MAX_PENALTY:
            total_penalty = MAX_PENALTY

        # Build explanation
        penalty_lines = [f"{note}: {pct*100:.1f}%" for pct, note in penalties]
        explanation = (
            "Strata health penalties: "
            + "; ".join(penalty_lines)
            + f" | Total: {total_penalty*100:.1f}%"
        )

        if total_penalty == MAX_PENALTY:
            explanation += " (capped at -25%)"

        logger.debug(
            "Strata health adjustment: %.1f%% from %d factors",
            total_penalty * 100, len(penalties),
        )

        return round(total_penalty, 4), explanation

    @staticmethod
    def _assess_depreciation_report(
        age_months: Optional[int],
    ) -> tuple[float, str]:
        """Assess penalty for depreciation report currency.

        As of July 2024, depreciation reports are mandatory every 5 years.
        Stale or missing reports indicate potential hidden maintenance costs.

        Args:
            age_months: Months since last depreciation report

        Returns:
            Tuple of (penalty, note)
        """
        if age_months is None:
            # No data — assume missing report
            return -0.03, "Depreciation report missing or not provided"
        elif age_months > 60:
            return -0.03, f"Depreciation report overdue ({age_months} months old, >60 month limit)"
        elif age_months > 36:
            return -0.01, f"Depreciation report aging ({age_months} months old)"
        else:
            return 0.0, ""

    @staticmethod
    def _assess_contingency_reserve(
        reserve_pct: Optional[float],
    ) -> tuple[float, str]:
        """Assess penalty for inadequate contingency reserve.

        BC Strata Property Act requires a minimum 25% contingency reserve
        (as a percentage of annual operating budget). Low reserves
        indicate future special assessment risk.

        Args:
            reserve_pct: Contingency reserve as % of annual operating budget

        Returns:
            Tuple of (penalty, note)
        """
        if reserve_pct is None:
            return 0.0, ""
        elif reserve_pct < 25:
            return -0.05, f"Contingency reserve critically low ({reserve_pct:.0f}% of operating, <25% minimum)"
        elif reserve_pct < 50:
            return -0.02, f"Contingency reserve below ideal ({reserve_pct:.0f}% of operating)"
        else:
            return 0.0, ""

    @staticmethod
    def _assess_special_assessments(
        count_5yr: Optional[int],
        total_5yr: Optional[float],
    ) -> tuple[float, str]:
        """Assess penalty for special assessment history.

        Frequent or large special assessments indicate deferred maintenance
        or inadequate budgeting by the strata council.

        Args:
            count_5yr: Number of special assessments in past 5 years
            total_5yr: Total dollars of special assessments in past 5 years

        Returns:
            Tuple of (penalty, note)
        """
        if total_5yr is None or total_5yr <= 0:
            return 0.0, ""

        # -2% per $10,000 in special assessments over 5 years
        penalty = -0.02 * (total_5yr / 10_000)

        note = (
            f"{count_5yr or '?'} special assessment(s) totalling "
            f"${total_5yr:,.0f} in past 5 years"
        )

        return round(penalty, 4), note

    @staticmethod
    def _assess_insurance_deductible(
        deductible: Optional[float],
    ) -> tuple[float, str]:
        """Assess penalty for high insurance deductibles.

        Post-2019, BC strata insurance costs and deductibles have
        skyrocketed. High deductibles expose individual owners to
        significant out-of-pocket risk from claims.

        Args:
            deductible: Building insurance deductible amount in dollars

        Returns:
            Tuple of (penalty, note)
        """
        if deductible is None:
            return 0.0, ""
        elif deductible > 500_000:
            return -0.05, f"Very high insurance deductible (${deductible:,.0f} — >$500K)"
        elif deductible > 250_000:
            return -0.03, f"High insurance deductible (${deductible:,.0f} — >$250K)"
        elif deductible > 100_000:
            return -0.01, f"Elevated insurance deductible (${deductible:,.0f} — >$100K)"
        else:
            return 0.0, ""

    @staticmethod
    def _assess_leaky_condo_risk(
        year_built: Optional[int],
        rainscreen_status: Optional[str],
    ) -> tuple[float, str]:
        """Assess penalty for leaky condo era construction.

        Buildings constructed 1981-1999 in Metro Vancouver are at high risk
        for moisture infiltration due to construction practices of the era
        (face-sealed building envelopes, insufficient flashing, etc.).

        Rainscreen remediation significantly reduces but does not eliminate
        the risk — some stigma persists even after full remediation.

        Args:
            year_built: Year the building was constructed
            rainscreen_status: "Yes" (fully rainscreened), "Partial",
                "No" (not remediated), or None

        Returns:
            Tuple of (penalty, note)
        """
        if year_built is None:
            return 0.0, ""

        if year_built < LEAKY_CONDO_ERA_START or year_built > LEAKY_CONDO_ERA_END:
            return 0.0, ""

        # Property is in the leaky condo era (1981-1999)
        status = (rainscreen_status or "").strip().lower()

        if status == "yes":
            return -0.02, (
                f"Leaky condo era construction ({year_built}) — "
                f"fully rainscreened, residual stigma only"
            )
        elif status == "partial":
            return -0.05, (
                f"Leaky condo era construction ({year_built}) — "
                f"partially remediated, moderate ongoing risk"
            )
        else:
            # Not remediated or unknown
            return -0.10, (
                f"Leaky condo era construction ({year_built}) — "
                f"not rainscreened, high moisture infiltration risk"
            )

    @staticmethod
    def _assess_crt_disputes(
        count_3yr: Optional[int],
    ) -> tuple[float, str]:
        """Assess penalty for CRT (Civil Resolution Tribunal) disputes.

        A high frequency of strata disputes at the CRT is a strong signal
        of governance dysfunction, which affects property values through
        buyer wariness and potential for costly litigation.

        Args:
            count_3yr: Number of CRT strata disputes in past 3 years

        Returns:
            Tuple of (penalty, note)
        """
        if count_3yr is None or count_3yr <= 0:
            return 0.0, ""
        elif count_3yr > 5:
            return -0.05, (
                f"{count_3yr} CRT disputes in 3 years — "
                f"severe governance problems indicated"
            )
        elif count_3yr > 3:
            return -0.02, (
                f"{count_3yr} CRT disputes in 3 years — "
                f"above-average governance friction"
            )
        else:
            return 0.0, ""
