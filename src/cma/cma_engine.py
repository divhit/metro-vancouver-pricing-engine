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
    "1/2 Duplex": "duplex",
    "House/Single Family": "detached",
    "HOUSE": "detached",
    "House with Acreage": "detached",
}

# Reverse map: canonical → MLS types
_CANONICAL_TO_MLS = {
    "condo": ["Apartment/Condo"],
    "townhome": ["Townhouse"],
    "duplex": ["1/2 Duplex"],
    "detached": ["House/Single Family", "HOUSE", "House with Acreage"],
}

# Types that can fall back to each other if too few comps
_TYPE_FALLBACKS = {
    "duplex": ["detached"],  # duplex can widen to detached if needed
    "detached": [],
    "condo": [],
    "townhome": [],
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

    @staticmethod
    def _normalize_mls_address(addr: str) -> tuple[int, str]:
        """Normalize an MLS address to (civic_number, street_name) matching BC Assessment format.

        MLS examples → BC Assessment format:
          "1 3090 VANNESS AVENUE"  → (3090, "VANNESS AVE")       # unit prefix stripped
          "1577 E 58TH AVENUE"    → (1577, "58TH AVE E")         # direction moved to end
          "323 N KAMLOOPS STREET" → (323, "KAMLOOPS ST")          # N stripped (BC has no N/S for streets)
          "2103 E 33RD AVENUE"    → (2103, "33RD AVE E")
          "8271 LAUREL STREET"    → (8271, "LAUREL ST")
        """
        import re

        addr_upper = addr.upper().strip()

        # Step 1: Extract unit prefix if present
        # Condo: "601 1850 COMOX STREET" → unit=601, civic=1850, street="COMOX STREET"
        # Duplex: "1 3090 VANNESS AVENUE" → unit=1, civic=3090, street="VANNESS AVENUE"
        # Pattern: first_num second_num street_name — first is unit, second is civic
        m = re.match(r"^(\d+)\s+(\d+)\s+(.+)", addr_upper)
        if m:
            civic_num = int(m.group(2))
            street_part = m.group(3).strip()
        else:
            m = re.match(r"^(\d+)\s+(.+)", addr_upper)
            if not m:
                return 0, ""
            civic_num = int(m.group(1))
            street_part = m.group(2).strip()

        # Step 2: Extract direction (E/W/N/S) from multiple positions
        # "E 58TH AVENUE" → dir="E", street="58TH AVENUE"
        # "13TH E AVENUE" → dir="E", street="13TH AVENUE"
        # "58TH AVENUE E" → dir="E", street="58TH AVENUE"
        direction = ""

        # Direction at start: "E 58TH AVENUE"
        dir_match = re.match(r"^(E|W|N|S|EAST|WEST|NORTH|SOUTH)\s+", street_part)
        if dir_match:
            direction = dir_match.group(1)[0]
            street_part = street_part[dir_match.end():].strip()

        # Direction at end: "58TH AVENUE E"
        end_dir = re.search(r"\s+(E|W|N|S|EAST|WEST|NORTH|SOUTH)$", street_part)
        if end_dir:
            direction = end_dir.group(1)[0]
            street_part = street_part[:end_dir.start()].strip()

        # Direction embedded between ordinal and suffix: "13TH E AVENUE"
        mid_dir = re.match(r"^(\d+\w*)\s+(E|W|N|S)\s+(.+)$", street_part)
        if mid_dir:
            direction = mid_dir.group(2)[0]
            street_part = f"{mid_dir.group(1)} {mid_dir.group(3)}"

        # Step 3: Abbreviate suffixes to match BC Assessment format
        suffix_map = {
            "AVENUE": "AVE", "STREET": "ST", "DRIVE": "DR", "ROAD": "RD",
            "BOULEVARD": "BLVD", "CRESCENT": "CRES", "PLACE": "PL",
            "COURT": "CT", "LANE": "LANE", "WAY": "WAY",
        }
        for long_form, short_form in suffix_map.items():
            street_part = re.sub(rf"\b{long_form}\b", short_form, street_part)

        # Step 4: Add ordinal suffix to bare numbers — "44 AVE" → "44TH AVE"
        # BC Assessment always uses ordinals: 1ST, 2ND, 3RD, 4TH, etc.
        bare_num = re.match(r"^(\d+)\s+(AVE|ST|DR|RD|BLVD|CRES|PL|CT)", street_part)
        if bare_num:
            num = int(bare_num.group(1))
            if num % 100 in (11, 12, 13):
                suffix = "TH"
            elif num % 10 == 1:
                suffix = "ST"
            elif num % 10 == 2:
                suffix = "ND"
            elif num % 10 == 3:
                suffix = "RD"
            else:
                suffix = "TH"
            street_part = f"{num}{suffix}{street_part[bare_num.end(1):]}"

        # Step 5: Append direction at end (BC Assessment format: "58TH AVE E")
        # Skip N/S for non-numbered streets (BC Assessment typically doesn't use N/S)
        if direction in ("E", "W"):
            street_part = f"{street_part} {direction}"
        elif direction in ("N", "S"):
            # Only append for numbered streets (e.g. "1ST AVE N"), skip for named streets
            if re.match(r"^\d", street_part):
                street_part = f"{street_part} {direction}"

        return civic_num, street_part

    def _geocode_sold(self, sold: pd.DataFrame) -> pd.DataFrame:
        """Match sold listings to enriched properties for lat/lon and neighbourhood."""
        import re

        # Build lookup from enriched properties:
        # (civic_number, street_name_upper) -> (lat, lon, assessed_value, neighbourhood_code)
        # Also index by to_civic_number for duplex units
        props = self.properties_df
        if "civic_number" not in props.columns:
            return sold

        lookup = {}
        has_coords = props["latitude"].notna() & props["longitude"].notna()
        subset = props[has_coords].copy()
        civic = subset["civic_number"].fillna(0).astype(int)
        to_civic = subset["to_civic_number"].fillna(0).astype(int) if "to_civic_number" in subset.columns else civic
        street = subset["street_name"].fillna("").str.upper().str.strip()
        lats = subset["latitude"].values
        lons = subset["longitude"].values
        assessed = subset["total_assessed_value"].values
        hoods = subset["neighbourhood_code"].fillna("").values if "neighbourhood_code" in subset.columns else [""] * len(subset)

        for i in range(len(subset)):
            s = street.iloc[i]
            if not s:
                continue
            val = (float(lats[i]), float(lons[i]), float(assessed[i]), str(hoods[i]))
            c = int(civic.iloc[i])
            tc = int(to_civic.iloc[i])
            if c > 0:
                key = (c, s)
                if key not in lookup:
                    lookup[key] = val
            # Also index by to_civic for duplex street addresses
            if tc > 0 and tc != c:
                key = (tc, s)
                if key not in lookup:
                    lookup[key] = val

        # Parse and normalize MLS addresses, then match
        latitudes = []
        longitudes = []
        assessed_values = []
        hood_codes = []

        for _, row in sold.iterrows():
            addr = str(row.get("address", ""))
            lat, lon, av, hood = None, None, None, ""

            if addr:
                civic_num, street_norm = self._normalize_mls_address(addr)
                if civic_num > 0 and street_norm:
                    # Try normalized match
                    if (civic_num, street_norm) in lookup:
                        lat, lon, av, hood = lookup[(civic_num, street_norm)]
                    else:
                        # Try without direction suffix
                        base = re.sub(r"\s+[EWNS]$", "", street_norm)
                        if base != street_norm and (civic_num, base) in lookup:
                            lat, lon, av, hood = lookup[(civic_num, base)]
                        else:
                            # Try with opposite abbreviation (AVE↔AVENUE shouldn't
                            # happen after normalization, but handle ST.↔ST etc.)
                            for alt_suffix in [
                                ("ST", "STREET"), ("STREET", "ST"),
                                ("AVE", "AVENUE"), ("AVENUE", "AVE"),
                            ]:
                                variant = street_norm.replace(alt_suffix[0], alt_suffix[1])
                                if variant != street_norm and (civic_num, variant) in lookup:
                                    lat, lon, av, hood = lookup[(civic_num, variant)]
                                    break

            latitudes.append(lat)
            longitudes.append(lon)
            assessed_values.append(av)
            hood_codes.append(hood)

        sold["latitude"] = latitudes
        sold["longitude"] = longitudes
        sold["assessed_value"] = assessed_values
        sold["neighbourhood_code"] = hood_codes

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

        s_hood = subject.get("neighbourhood_code", "")
        has_coords = sold["latitude"].notna() & sold["longitude"].notna()

        # ── Comparable selection criteria ──
        # A valid comp must meet ALL of:
        #   1. Same sub-region (MLS sub_area)
        #   2. Within 2km of subject
        #   3. Assessed value within ±10% of subject's assessed value
        #   4. Same property type
        #   5. Recent sale
        #
        # Progressive widening only relaxes time window and value tolerance,
        # NEVER the sub-region or distance requirements.

        # Map subject neighbourhood to MLS sub_areas
        matched_sold = sold[(sold["neighbourhood_code"] != "") & sold["sub_area"].notna()]
        hood_to_subs = matched_sold.groupby("neighbourhood_code")["sub_area"].apply(set).to_dict()
        subject_subs = hood_to_subs.get(s_hood, set())
        adjacent_subs = set()
        if s_hood:
            try:
                hood_int = int(s_hood)
                for h in range(max(1, hood_int - 2), hood_int + 3):
                    adjacent_subs.update(hood_to_subs.get(str(h), set()))
            except ValueError:
                adjacent_subs = subject_subs

        candidates = pd.DataFrame()
        types_to_search = [s_type] if same_type and s_type else []

        def _search(
            sub_areas: set[str], types: list[str], days: int,
            max_dist: float, value_pct: float,
        ) -> pd.DataFrame:
            """Find comps matching sub_area, type, recency, distance, and assessed value."""
            cutoff = datetime.now() - timedelta(days=days)
            m = sold["sold_dt"].notna() & (sold["sold_dt"] >= cutoff)
            m &= has_coords
            if types:
                m &= sold["canonical_type"].isin(types)
            if sub_areas:
                m &= sold["sub_area"].isin(sub_areas)
            # Assessed value within ±X% of subject
            if s_assessed and s_assessed > 0 and value_pct > 0:
                av = sold["assessed_value"]
                m &= av.notna() & (av >= s_assessed * (1 - value_pct)) & (av <= s_assessed * (1 + value_pct))
            result = sold[m].copy()
            if s_lat and s_lon and not result.empty:
                result["distance_m"] = result.apply(
                    lambda r: _haversine(s_lat, s_lon, r["latitude"], r["longitude"]),
                    axis=1,
                )
                result = result[result["distance_m"] <= max_dist]
            else:
                result["distance_m"] = 0.0
            return result

        # Step 1: Same sub_area, ±10% value, 2km, 60 days
        if subject_subs:
            candidates = _search(subject_subs, types_to_search, max_age_days, 2000, 0.10)
            if len(candidates) >= 3:
                logger.info("CMA step 1: %d comps (same sub_area, ±10%%, %dd)",
                            len(candidates), max_age_days)

        # Step 2: Same sub_area, ±10% value, 2km, 120 days
        if len(candidates) < 3 and subject_subs:
            candidates = _search(subject_subs, types_to_search, 120, 2000, 0.10)
            if len(candidates) >= 3:
                logger.info("CMA step 2: %d comps (same sub_area, ±10%%, 120d)",
                            len(candidates))

        # Step 3: Adjacent sub_areas, ±10% value, 2km, 120 days
        if len(candidates) < 3 and adjacent_subs:
            candidates = _search(adjacent_subs, types_to_search, 120, 2000, 0.10)
            if len(candidates) >= 3:
                logger.info("CMA step 3: %d comps (adjacent subs, ±10%%, 120d)",
                            len(candidates))

        # Step 4: Adjacent sub_areas, ±20% value, 2km, 120 days
        if len(candidates) < 3 and adjacent_subs:
            candidates = _search(adjacent_subs, types_to_search, 120, 2000, 0.20)
            if len(candidates) >= 3:
                logger.info("CMA step 4: %d comps (adjacent subs, ±20%%, 120d)",
                            len(candidates))

        # Step 5: Fallback types, adjacent sub_areas, ±20% value, 2km
        if len(candidates) < 3 and same_type and s_type:
            fallbacks = _TYPE_FALLBACKS.get(s_type, [])
            if fallbacks and adjacent_subs:
                all_types = [s_type] + fallbacks
                candidates = _search(adjacent_subs, all_types, 120, 2000, 0.20)
                logger.info("CMA step 5: %d comps (fallback types, ±20%%)",
                            len(candidates))

        # Step 6: Last resort — relax value to ±30%, extend to 180 days
        if len(candidates) < 3:
            logger.info("CMA step 6: last resort (±30%%, 180d, 2km)")
            candidates = _search(
                adjacent_subs or subject_subs, types_to_search, 180, 2000, 0.30,
            )

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

        # Size adjustment: use BUILDING value per sqft, not total price per sqft.
        # In Vancouver, most detached/duplex value is in the LAND, not the building.
        # The building depreciates over ~20 years and then it's basically land value.
        # We estimate the improvement (building) portion and adjust based on that.
        if s_sqft and c_sqft:
            try:
                s_sqft = float(s_sqft)
                c_sqft = float(c_sqft)
                if c_sqft > 0 and s_sqft > 0:
                    sqft_diff = s_sqft - c_sqft
                    if abs(sqft_diff) > 10:  # Ignore trivial differences
                        # For condos/townhomes, more of the value is in the unit itself
                        if s_type in ("condo", "townhome"):
                            building_pct = 0.70
                        else:
                            # For detached/duplex: building portion depends on age
                            # New builds: ~40-50% building. Old homes (20+ yr): ~15-25% building.
                            c_age = None
                            if c_year:
                                try:
                                    c_age = datetime.now().year - int(c_year)
                                except (ValueError, TypeError):
                                    pass
                            if c_age is not None and c_age <= 5:
                                building_pct = 0.45  # New construction
                            elif c_age is not None and c_age <= 15:
                                building_pct = 0.35  # Relatively new
                            elif c_age is not None and c_age <= 25:
                                building_pct = 0.25  # Depreciating
                            else:
                                building_pct = 0.15  # Old home — mostly land value

                        # Only the building portion drives size-based value differences
                        building_value = sold_price * building_pct
                        ppsf = building_value / c_sqft
                        dollar = ppsf * sqft_diff
                        # Cap at ±15% for detached/duplex, ±25% for condos
                        max_cap = 0.25 if s_type in ("condo", "townhome") else 0.15
                        dollar = max(-sold_price * max_cap, min(sold_price * max_cap, dollar))
                        adj_price += dollar
                        adjustments.append({
                            "name": "Size",
                            "detail": f"Subject {abs(sqft_diff):.0f} sqft {'larger' if sqft_diff > 0 else 'smaller'} (bldg value ~{building_pct:.0%} of total)",
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
