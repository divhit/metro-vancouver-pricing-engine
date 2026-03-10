"""
Comparative Market Analysis (CMA) engine.

Finds recent comparable SOLD properties near a subject property and produces
a CMA report with adjusted sale prices, giving realtors a defensible price
range backed by actual transactions.

Key differences from the assessment-based comparable engine:
- Uses actual MLS sold prices (not assessed values)
- Filters to recent sales only (default 60 days)
- Adjusts comparable sold prices toward the subject (age, size, location)
- Produces a CMA price range alongside the SAR-based market estimate
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.daily_intel.storage.database import get_connection

logger = logging.getLogger(__name__)

# MLS property_type → canonical type
_MLS_TYPE_MAP = {
    "Apartment/Condo": "condo",
    "Townhouse": "townhome",
    "1/2 Duplex": "detached",
    "House/Single Family": "detached",
    "HOUSE": "detached",
    "House with Acreage": "detached",
}

# Reverse map: canonical → MLS types
_CANONICAL_TO_MLS = {
    "condo": ["Apartment/Condo"],
    "townhome": ["Townhouse"],
    "detached": ["1/2 Duplex", "House/Single Family", "HOUSE", "House with Acreage"],
}


def _parse_sold_date(date_str: str) -> Optional[datetime]:
    """Parse sold_date from various MLS formats."""
    if not date_str:
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y", "%B %d, %Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in metres between two points."""
    R = 6_371_000.0
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


class CMAEngine:
    """Finds comparable sold properties and produces CMA reports."""

    def __init__(self, properties_df: pd.DataFrame):
        """
        Args:
            properties_df: Enriched properties DataFrame (217K rows with lat/lon).
                Used to geocode sold listings via address matching.
        """
        self.properties_df = properties_df
        self._sold_with_coords: Optional[pd.DataFrame] = None

    def _load_sold_listings(self) -> pd.DataFrame:
        """Load sold listings from SQLite and enrich with lat/lon from assessments."""
        if self._sold_with_coords is not None:
            return self._sold_with_coords

        conn = get_connection()
        rows = conn.execute(
            "SELECT * FROM sold_listings WHERE sold_price IS NOT NULL AND sold_price > 0"
        ).fetchall()
        if not rows:
            self._sold_with_coords = pd.DataFrame()
            conn.close()
            return self._sold_with_coords

        cols = [d[0] for d in conn.execute("SELECT * FROM sold_listings LIMIT 1").description]
        conn.close()

        sold = pd.DataFrame([dict(zip(cols, r)) for r in rows])

        # Parse sold_date to datetime
        sold["sold_dt"] = sold["sold_date"].apply(
            lambda x: _parse_sold_date(str(x)) if pd.notna(x) else None
        )

        # Map property_type to canonical
        sold["canonical_type"] = sold["property_type"].map(_MLS_TYPE_MAP).fillna("other")

        # Match to enriched properties for lat/lon via address
        if self.properties_df is not None and not self.properties_df.empty:
            sold = self._geocode_sold(sold)

        self._sold_with_coords = sold
        logger.info("CMA engine loaded %d sold listings (%d with coordinates)",
                     len(sold), sold["latitude"].notna().sum())
        return self._sold_with_coords

    def _geocode_sold(self, sold: pd.DataFrame) -> pd.DataFrame:
        """Match sold listings to enriched properties for lat/lon."""
        import re

        # Build lookup from enriched properties: (civic_number, street_name_upper) -> (lat, lon, assessed_value)
        props = self.properties_df
        if "civic_number" not in props.columns:
            return sold

        lookup = {}
        has_coords = props["latitude"].notna() & props["longitude"].notna()
        subset = props[has_coords].copy()
        civic = subset["civic_number"].fillna(0).astype(int)
        street = subset["street_name"].fillna("").str.upper().str.strip()
        lats = subset["latitude"].values
        lons = subset["longitude"].values
        assessed = subset["total_assessed_value"].values

        for i in range(len(subset)):
            c = int(civic.iloc[i])
            s = street.iloc[i]
            if c > 0 and s:
                key = (c, s)
                if key not in lookup:
                    lookup[key] = (float(lats[i]), float(lons[i]), float(assessed[i]))

        # Parse MLS addresses and match
        latitudes = []
        longitudes = []
        assessed_values = []

        for _, row in sold.iterrows():
            addr = str(row.get("address", ""))
            lat, lon, av = None, None, None

            if addr:
                # Parse "1234 MAIN ST" or "302 5TH AVE W" etc
                addr_upper = addr.upper().strip()
                m = re.match(r"^(\d+)\s+(.+)", addr_upper)
                if m:
                    civic_num = int(m.group(1))
                    street_part = m.group(2).strip()
                    # Remove unit/suite prefixes
                    street_part = re.sub(r"^(UNIT|SUITE|APT|#)\s*\S+\s*", "", street_part)

                    # Try exact match
                    if (civic_num, street_part) in lookup:
                        lat, lon, av = lookup[(civic_num, street_part)]
                    else:
                        # Try common abbreviation variants
                        for suffix_map in [
                            ("STREET", "ST"), ("ST", "STREET"),
                            ("AVENUE", "AVE"), ("AVE", "AVENUE"),
                            ("DRIVE", "DR"), ("DR", "DRIVE"),
                            ("ROAD", "RD"), ("RD", "ROAD"),
                            ("BOULEVARD", "BLVD"), ("BLVD", "BOULEVARD"),
                            ("CRESCENT", "CRES"), ("CRES", "CRESCENT"),
                            ("PLACE", "PL"), ("PL", "PLACE"),
                            ("COURT", "CT"), ("CT", "COURT"),
                        ]:
                            variant = street_part.replace(suffix_map[0], suffix_map[1])
                            if variant != street_part and (civic_num, variant) in lookup:
                                lat, lon, av = lookup[(civic_num, variant)]
                                break

            latitudes.append(lat)
            longitudes.append(lon)
            assessed_values.append(av)

        sold["latitude"] = latitudes
        sold["longitude"] = longitudes
        sold["assessed_value"] = assessed_values

        return sold

    def find_comparables(
        self,
        subject: dict,
        max_comps: int = 10,
        max_radius_m: float = 2000,
        max_age_days: int = 60,
        same_type: bool = True,
    ) -> list[dict]:
        """Find comparable sold properties near the subject.

        Args:
            subject: Dict with keys: latitude, longitude, property_type,
                     year_built, floor_area/living_area_sqft, bedrooms, bathrooms,
                     total_assessed_value
            max_comps: Maximum number of comparables to return (up to 10).
            max_radius_m: Maximum search radius in metres.
            max_age_days: Maximum age of sale in days from today.
            same_type: Require same canonical property type.

        Returns:
            List of comparable dicts sorted by similarity score (best first).
        """
        sold = self._load_sold_listings()
        if sold.empty:
            return []

        s_lat = subject.get("latitude", 0.0)
        s_lon = subject.get("longitude", 0.0)
        s_type = subject.get("property_type", "")
        s_year = subject.get("year_built")
        # For floor area: prefer explicit floor_area override, then living_area_sqft.
        # estimated_living_area_sqft from BC Assessment is often lot size for detached homes,
        # so only use it if it's reasonable (< 5000 sqft for detached, < 3000 for others).
        s_sqft = subject.get("floor_area") or subject.get("living_area_sqft")
        if not s_sqft:
            est = subject.get("estimated_living_area_sqft")
            if est:
                max_reasonable = 5000 if s_type == "detached" else 3000
                if est <= max_reasonable:
                    s_sqft = est
                else:
                    logger.info("Ignoring estimated_living_area_sqft=%s (likely lot size)", est)
        s_beds = subject.get("bedrooms")
        s_baths = subject.get("bathrooms")
        s_assessed = subject.get("total_assessed_value", 0)

        # Filter by recency
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        mask = sold["sold_dt"].notna() & (sold["sold_dt"] >= cutoff_date)

        # Filter by type
        if same_type and s_type:
            mask &= sold["canonical_type"] == s_type

        # Filter by geography (need coordinates)
        has_coords = sold["latitude"].notna() & sold["longitude"].notna()
        mask &= has_coords

        candidates = sold[mask].copy()

        # If too few candidates, widen search
        if len(candidates) < 3:
            logger.info("Only %d candidates in %d days — widening to 120 days, 5km",
                        len(candidates), max_age_days)
            cutoff_date = datetime.now() - timedelta(days=120)
            mask = sold["sold_dt"].notna() & (sold["sold_dt"] >= cutoff_date)
            if same_type and s_type:
                mask &= sold["canonical_type"] == s_type
            mask &= has_coords
            candidates = sold[mask].copy()
            max_radius_m = 5000

        if candidates.empty:
            return []

        # Compute distances
        if s_lat and s_lon:
            candidates["distance_m"] = candidates.apply(
                lambda r: _haversine(s_lat, s_lon, r["latitude"], r["longitude"]),
                axis=1,
            )
            candidates = candidates[candidates["distance_m"] <= max_radius_m]
        else:
            # No coordinates — filter by sub_area match
            s_sub = subject.get("sub_area", subject.get("neighbourhood_code", ""))
            if s_sub:
                candidates = candidates[candidates["sub_area"] == s_sub]
            candidates["distance_m"] = 0.0

        if candidates.empty:
            return []

        # Score each candidate for similarity
        scores = []
        for _, row in candidates.iterrows():
            score = self._score_comparable(
                subject, row, s_lat, s_lon, s_year, s_sqft, s_beds, s_baths, s_assessed,
            )
            scores.append(score)

        candidates["similarity_score"] = scores
        candidates = candidates.sort_values("similarity_score").head(max_comps)

        # Build result dicts
        results = []
        for _, row in candidates.iterrows():
            comp = {
                "mls_number": row.get("mls_number", ""),
                "address": row.get("address", ""),
                "sold_price": int(row.get("sold_price", 0)),
                "list_price": int(row["list_price"]) if pd.notna(row.get("list_price")) else None,
                "sold_date": row.get("sold_date", ""),
                "dom": int(row["dom"]) if pd.notna(row.get("dom")) else None,
                "bedrooms": int(row["bedrooms"]) if pd.notna(row.get("bedrooms")) else None,
                "bathrooms": float(row["bathrooms"]) if pd.notna(row.get("bathrooms")) else None,
                "floor_area": int(row["floor_area"]) if pd.notna(row.get("floor_area")) else None,
                "year_built": int(row["year_built"]) if pd.notna(row.get("year_built")) else None,
                "property_type": row.get("property_type", ""),
                "distance_m": round(row["distance_m"], 0),
                "similarity_score": round(row["similarity_score"], 4),
                "latitude": float(row["latitude"]) if pd.notna(row.get("latitude")) else None,
                "longitude": float(row["longitude"]) if pd.notna(row.get("longitude")) else None,
                "assessed_value": int(row["assessed_value"]) if pd.notna(row.get("assessed_value")) else None,
            }

            # Compute price adjustments toward subject
            adj_price, adjustments = self._adjust_price(subject, row, comp["sold_price"])
            comp["adjusted_price"] = round(adj_price)
            comp["adjustments"] = adjustments

            # Sale-to-assessment ratio
            if comp["assessed_value"] and comp["assessed_value"] > 0:
                comp["sar"] = round(comp["sold_price"] / comp["assessed_value"], 3)
            else:
                comp["sar"] = None

            # List-to-sold ratio
            if comp["list_price"] and comp["list_price"] > 0:
                comp["list_to_sold"] = round(comp["sold_price"] / comp["list_price"], 3)
            else:
                comp["list_to_sold"] = None

            results.append(comp)

        return results

    def generate_cma_report(
        self,
        subject: dict,
        comparables: list[dict],
        market_estimate: Optional[float] = None,
        assessed_value: Optional[float] = None,
    ) -> dict:
        """Generate a full CMA report combining comparables with other estimates.

        Args:
            subject: Subject property dict.
            comparables: List of comparable dicts from find_comparables().
            market_estimate: SAR-based market estimate (if available).
            assessed_value: BC Assessment total assessed value.

        Returns:
            CMA report dict with price ranges and recommendation.
        """
        # Use same floor_area logic — avoid inflated lot sizes
        s_type = subject.get("property_type", "")
        report_sqft = subject.get("floor_area") or subject.get("living_area_sqft")
        if not report_sqft:
            est = subject.get("estimated_living_area_sqft")
            if est:
                max_reasonable = 5000 if s_type == "detached" else 3000
                report_sqft = est if est <= max_reasonable else None

        report = {
            "subject": {
                "address": subject.get("address", subject.get("full_address", "")),
                "property_type": subject.get("property_type", ""),
                "bedrooms": subject.get("bedrooms"),
                "bathrooms": subject.get("bathrooms"),
                "floor_area": report_sqft,
                "year_built": subject.get("year_built"),
                "assessed_value": assessed_value,
            },
            "comparables": comparables,
            "comparable_count": len(comparables),
        }

        if not comparables:
            report["cma_estimate"] = None
            report["cma_range"] = None
            report["recommendation"] = {
                "estimated_range": None,
                "confidence": "low",
                "note": "Insufficient comparable sales data for CMA analysis",
            }
            return report

        # CMA price range from adjusted comparable prices
        adj_prices = [c["adjusted_price"] for c in comparables if c.get("adjusted_price")]
        raw_prices = [c["sold_price"] for c in comparables if c.get("sold_price")]

        if adj_prices:
            cma_low = int(np.percentile(adj_prices, 25))
            cma_median = int(np.median(adj_prices))
            cma_high = int(np.percentile(adj_prices, 75))
            cma_mean = int(np.mean(adj_prices))
        else:
            cma_low = cma_median = cma_high = cma_mean = 0

        report["cma_estimate"] = cma_median
        report["cma_range"] = {
            "low": cma_low,
            "median": cma_median,
            "high": cma_high,
            "mean": cma_mean,
        }

        # Market stats from comparables
        if raw_prices:
            sars = [c["sar"] for c in comparables if c.get("sar")]
            l2s = [c["list_to_sold"] for c in comparables if c.get("list_to_sold")]
            doms = [c["dom"] for c in comparables if c.get("dom") is not None]

            report["market_stats"] = {
                "median_sold_price": int(np.median(raw_prices)),
                "avg_sar": round(np.mean(sars), 3) if sars else None,
                "avg_list_to_sold": round(np.mean(l2s), 3) if l2s else None,
                "avg_dom": round(np.mean(doms), 1) if doms else None,
                "avg_distance_m": round(np.mean([c["distance_m"] for c in comparables]), 0),
            }

        # Build recommendation combining all signals
        estimates = []
        labels = []

        if cma_median > 0:
            estimates.append(cma_median)
            labels.append("CMA")

        if market_estimate and market_estimate > 0:
            estimates.append(market_estimate)
            labels.append("SAR Model")
            report["sar_estimate"] = round(market_estimate)

        if assessed_value and assessed_value > 0:
            report["assessed_value"] = round(assessed_value)

        if estimates:
            # Weighted blend: CMA gets more weight when we have good comps
            if len(estimates) == 2 and len(comparables) >= 5:
                # Good comps: 60% CMA, 40% SAR
                blended = 0.60 * estimates[0] + 0.40 * estimates[1]
                method = "60% CMA / 40% SAR Model"
            elif len(estimates) == 2:
                # Few comps: 40% CMA, 60% SAR
                blended = 0.40 * estimates[0] + 0.60 * estimates[1]
                method = "40% CMA / 60% SAR Model (limited comparables)"
            else:
                blended = estimates[0]
                method = labels[0]

            # Confidence based on number of comps and agreement
            if len(comparables) >= 7:
                confidence = "high"
            elif len(comparables) >= 4:
                confidence = "moderate"
            else:
                confidence = "low"

            # Check agreement between methods
            if len(estimates) >= 2:
                spread = abs(estimates[0] - estimates[1]) / max(estimates) * 100
                if spread > 15:
                    confidence = "low"
                    method += f" (methods diverge by {spread:.0f}%)"
                elif spread > 8:
                    if confidence == "high":
                        confidence = "moderate"

            # Recommended range: ±5% of blended estimate
            range_pct = 0.05 if confidence == "high" else 0.08 if confidence == "moderate" else 0.10
            rec_low = int(blended * (1 - range_pct))
            rec_high = int(blended * (1 + range_pct))

            report["recommendation"] = {
                "estimated_value": int(blended),
                "estimated_range": {"low": rec_low, "high": rec_high},
                "confidence": confidence,
                "method": method,
            }
        else:
            report["recommendation"] = {
                "estimated_range": None,
                "confidence": "low",
                "note": "No pricing signals available",
            }

        return report

    def _score_comparable(
        self, subject: dict, row: pd.Series,
        s_lat: float, s_lon: float,
        s_year, s_sqft, s_beds, s_baths, s_assessed,
    ) -> float:
        """Score a comparable for similarity to the subject (lower = better)."""
        score = 0.0

        # Distance (30% weight) — closer is better
        dist = row.get("distance_m", 0)
        # Exponential decay: 0 at 0m, ~0.5 at 1km, ~1.0 at 5km
        score += 0.30 * min(1.0, 1.0 - math.exp(-math.log(2) * dist / 1000.0))

        # Floor area similarity (25% weight)
        c_sqft = row.get("floor_area")
        if s_sqft and c_sqft and s_sqft > 0 and c_sqft > 0:
            area_diff = abs(s_sqft - c_sqft) / max(s_sqft, c_sqft)
            score += 0.25 * min(1.0, area_diff)
        else:
            score += 0.25 * 0.5  # Unknown = mid penalty

        # Age similarity (15% weight)
        c_year = row.get("year_built")
        if s_year and c_year:
            try:
                year_diff = abs(int(s_year) - int(c_year)) / 30.0
                score += 0.15 * min(1.0, year_diff)
            except (ValueError, TypeError):
                score += 0.15 * 0.5
        else:
            score += 0.15 * 0.5

        # Bedroom match (10% weight)
        c_beds = row.get("bedrooms")
        if s_beds is not None and c_beds is not None:
            try:
                bed_diff = abs(int(s_beds) - int(c_beds)) / 3.0
                score += 0.10 * min(1.0, bed_diff)
            except (ValueError, TypeError):
                score += 0.10 * 0.5
        else:
            score += 0.10 * 0.5

        # Bathroom match (5% weight)
        c_baths = row.get("bathrooms")
        if s_baths is not None and c_baths is not None:
            try:
                bath_diff = abs(float(s_baths) - float(c_baths)) / 2.0
                score += 0.05 * min(1.0, bath_diff)
            except (ValueError, TypeError):
                score += 0.05 * 0.5
        else:
            score += 0.05 * 0.5

        # Assessed value similarity (10% weight) — sanity check
        c_assessed = row.get("assessed_value")
        if s_assessed and c_assessed and s_assessed > 0 and c_assessed > 0:
            val_ratio = abs(math.log(s_assessed / c_assessed)) / math.log(2)
            score += 0.10 * min(1.0, val_ratio)
        else:
            score += 0.10 * 0.5

        # Recency (5% weight) — more recent sales are better
        sold_dt = row.get("sold_dt")
        if sold_dt:
            days_ago = (datetime.now() - sold_dt).days
            score += 0.05 * min(1.0, days_ago / 60.0)
        else:
            score += 0.05 * 0.5

        return score

    def _adjust_price(self, subject: dict, comp_row: pd.Series, sold_price: float) -> tuple[float, list[dict]]:
        """Adjust comparable sold price toward the subject property.

        Returns (adjusted_price, list_of_adjustments).
        """
        adjustments = []
        adj_price = float(sold_price)

        s_year = subject.get("year_built")
        c_year = comp_row.get("year_built")
        # Use same floor_area logic as find_comparables — avoid inflated lot sizes
        s_type = subject.get("property_type", "")
        s_sqft = subject.get("floor_area") or subject.get("living_area_sqft")
        if not s_sqft:
            est = subject.get("estimated_living_area_sqft")
            if est:
                max_reasonable = 5000 if s_type == "detached" else 3000
                s_sqft = est if est <= max_reasonable else None
        c_sqft = comp_row.get("floor_area")
        s_beds = subject.get("bedrooms")
        c_beds = comp_row.get("bedrooms")

        # Age adjustment: ±0.5% per year difference, capped at ±15%
        if s_year and c_year:
            try:
                year_diff = int(s_year) - int(c_year)
                if year_diff != 0:
                    pct = year_diff * 0.005
                    pct = max(-0.15, min(0.15, pct))
                    dollar = adj_price * pct
                    adj_price += dollar
                    adjustments.append({
                        "name": "Age",
                        "detail": f"Subject {abs(year_diff)} years {'newer' if year_diff > 0 else 'older'}",
                        "percentage": round(pct * 100, 1),
                        "dollar": round(dollar),
                    })
            except (ValueError, TypeError):
                pass

        # Size adjustment: $/sqft pro-rata
        if s_sqft and c_sqft:
            try:
                s_sqft = float(s_sqft)
                c_sqft = float(c_sqft)
                if c_sqft > 0 and s_sqft > 0:
                    sqft_diff = s_sqft - c_sqft
                    if abs(sqft_diff) > 10:  # Ignore trivial differences
                        ppsf = sold_price / c_sqft
                        dollar = ppsf * sqft_diff
                        # Cap at ±30%
                        dollar = max(-sold_price * 0.30, min(sold_price * 0.30, dollar))
                        adj_price += dollar
                        adjustments.append({
                            "name": "Size",
                            "detail": f"Subject {abs(sqft_diff):.0f} sqft {'larger' if sqft_diff > 0 else 'smaller'}",
                            "percentage": round(dollar / sold_price * 100, 1),
                            "dollar": round(dollar),
                        })
            except (ValueError, TypeError):
                pass

        # Bedroom adjustment: ±$25K per bedroom difference for condos, ±$50K for detached
        if s_beds is not None and c_beds is not None:
            try:
                bed_diff = int(s_beds) - int(c_beds)
                if bed_diff != 0:
                    s_type = subject.get("property_type", "")
                    per_bed = 25000 if s_type == "condo" else 50000
                    dollar = bed_diff * per_bed
                    adj_price += dollar
                    adjustments.append({
                        "name": "Bedrooms",
                        "detail": f"Subject has {abs(bed_diff)} {'more' if bed_diff > 0 else 'fewer'} bedroom(s)",
                        "percentage": round(dollar / sold_price * 100, 1),
                        "dollar": round(dollar),
                    })
            except (ValueError, TypeError):
                pass

        return adj_price, adjustments
