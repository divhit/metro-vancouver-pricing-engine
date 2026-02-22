"""
Leasehold discount adjustment curve.

Key insight: The discount is non-linear:
- 0% discount above 70 years remaining
- Modest 5-10% at 40-70 years
- Accelerating sharply below 30 years (mortgage amortization threshold)
- Near-total value destruction below 25 years (financing cliff)
- Most banks refuse financing below 30-40 years remaining

Special cases:
- UBC 99-year prepaid leasehold (~2102 expiry): near freehold parity
- False Creek South (2036-2046 expiry): massive discounts
- Musqueam non-prepaid: 50%+ down payment required, 30-60% discount
"""

import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class LeaseholdType(Enum):
    FREEHOLD = "freehold"
    PREPAID = "prepaid"           # e.g., UBC leasehold
    NON_PREPAID = "non_prepaid"   # e.g., Musqueam leasehold
    FIRST_NATIONS = "first_nations"


def compute_leasehold_discount(
    remaining_years: float,
    lease_type: LeaseholdType = LeaseholdType.PREPAID,
) -> float:
    """Compute the estimated discount vs. freehold equivalent.

    Returns a value between 0.0 (no discount) and 1.0 (total loss).

    The curve is calibrated from observed Metro Vancouver transactions:
    - UBC prepaid leasehold (70+ years): ~0-3% discount
    - False Creek South (10-20 years): ~70-85% discount
    - Musqueam non-prepaid: additional 15-25% penalty

    Args:
        remaining_years: Years remaining on the lease
        lease_type: Type of leasehold arrangement

    Returns:
        Discount factor (0.0 = no discount, 1.0 = worthless)
    """
    if lease_type == LeaseholdType.FREEHOLD:
        return 0.0

    # Base discount curve (prepaid leasehold)
    if remaining_years >= 80:
        base_discount = 0.0
    elif remaining_years >= 70:
        # 0% to 3% — minimal discount for very long leases
        base_discount = 0.03 * (80 - remaining_years) / 10
    elif remaining_years >= 50:
        # 3% to 10% — moderate discount
        base_discount = 0.03 + 0.07 * (70 - remaining_years) / 20
    elif remaining_years >= 40:
        # 10% to 20% — significant discount, some lender hesitancy
        base_discount = 0.10 + 0.10 * (50 - remaining_years) / 10
    elif remaining_years >= 30:
        # 20% to 35% — approaching financing cliff
        base_discount = 0.20 + 0.15 * (40 - remaining_years) / 10
    elif remaining_years >= 25:
        # 35% to 50% — FINANCING CLIFF: most banks refuse new mortgages
        base_discount = 0.35 + 0.15 * (30 - remaining_years) / 5
    elif remaining_years >= 15:
        # 50% to 75% — cash-only buyers, extreme discount
        base_discount = 0.50 + 0.25 * (25 - remaining_years) / 10
    elif remaining_years >= 5:
        # 75% to 90% — near-total loss
        base_discount = 0.75 + 0.15 * (15 - remaining_years) / 10
    else:
        # < 5 years — essentially worthless as real estate
        base_discount = 0.90 + 0.10 * max(0, (5 - remaining_years)) / 5

    # Non-prepaid leasehold penalty (e.g., Musqueam)
    # These require ground rent payments and have renewal risk
    if lease_type == LeaseholdType.NON_PREPAID:
        base_discount = min(1.0, base_discount + 0.15)

    # First Nations leasehold (additional uncertainty premium)
    if lease_type == LeaseholdType.FIRST_NATIONS:
        base_discount = min(1.0, base_discount + 0.20)

    return round(min(1.0, max(0.0, base_discount)), 4)


def apply_leasehold_adjustment(
    freehold_estimate: float,
    remaining_years: Optional[float],
    lease_type: LeaseholdType = LeaseholdType.FREEHOLD,
) -> tuple[float, float, str]:
    """Apply leasehold discount to a freehold-equivalent estimate.

    Args:
        freehold_estimate: The model's estimate assuming freehold
        remaining_years: Years remaining on lease (None if freehold)
        lease_type: Type of leasehold

    Returns:
        Tuple of (adjusted_value, discount_pct, explanation)
    """
    if lease_type == LeaseholdType.FREEHOLD or remaining_years is None:
        return freehold_estimate, 0.0, "Freehold property — no adjustment"

    discount = compute_leasehold_discount(remaining_years, lease_type)
    adjusted = freehold_estimate * (1.0 - discount)

    # Generate explanation
    if discount < 0.05:
        explanation = (
            f"Long-term leasehold ({remaining_years:.0f} years remaining) — "
            f"minimal {discount*100:.1f}% discount"
        )
    elif discount < 0.20:
        explanation = (
            f"Leasehold with {remaining_years:.0f} years remaining — "
            f"{discount*100:.1f}% discount reflects moderate lender caution"
        )
    elif discount < 0.40:
        explanation = (
            f"Leasehold with {remaining_years:.0f} years remaining — "
            f"{discount*100:.1f}% discount near financing cliff "
            f"(most lenders require 30-40 year minimum)"
        )
    elif discount < 0.60:
        explanation = (
            f"Short leasehold ({remaining_years:.0f} years remaining) — "
            f"{discount*100:.1f}% discount; limited to cash/high-down-payment buyers"
        )
    else:
        explanation = (
            f"Very short leasehold ({remaining_years:.0f} years remaining) — "
            f"{discount*100:.1f}% discount; severely limited buyer pool"
        )

    if lease_type == LeaseholdType.NON_PREPAID:
        explanation += " (non-prepaid lease adds renewal risk premium)"
    elif lease_type == LeaseholdType.FIRST_NATIONS:
        explanation += " (First Nations leasehold adds additional uncertainty premium)"

    return round(adjusted, 2), round(discount * 100, 2), explanation
