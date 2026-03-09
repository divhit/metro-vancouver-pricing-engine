"""Match MLS sold listings to BC Assessment records and compute market ratios.

Every property in Vancouver has a PID and assessed value. This module matches
MLS sold listings to BC Assessment records.

Strategy:
1. Build a comprehensive lookup from BC Assessment keyed on (civic_number, building_number, street)
2. For each MLS address, try multiple normalized forms of the street name
3. Handle edge cases: PH units, non-numeric units, compound street names, UBC lands
"""
import logging
import re
from typing import Optional

import pandas as pd

from src.daily_intel.storage.database import get_connection

logger = logging.getLogger(__name__)

_ASSESSMENT_DF: Optional[pd.DataFrame] = None
_STREET_LOOKUP: Optional[dict] = None


def _load_assessments() -> pd.DataFrame:
    global _ASSESSMENT_DF
    if _ASSESSMENT_DF is not None:
        return _ASSESSMENT_DF
    try:
        df = pd.read_parquet("data/processed/enriched_properties.parquet")
        _ASSESSMENT_DF = df
        logger.info(f"Loaded {len(df)} BC Assessment records")
        return df
    except Exception as e:
        logger.error(f"Failed to load assessment data: {e}")
        return pd.DataFrame()


def _build_street_alias_map(assessments: pd.DataFrame) -> dict:
    """Build a map from various MLS street forms → canonical assessment street name.

    E.g., "COMOX STREET" → "COMOX ST", "KINGSWAY WAY" → "KINGSWAY"
    """
    global _STREET_LOOKUP
    if _STREET_LOOKUP is not None:
        return _STREET_LOOKUP

    canonical_streets = set(assessments["street_name"].dropna().str.upper().str.strip().unique())

    alias_map = {}
    # Each canonical street maps to itself
    for s in canonical_streets:
        alias_map[s] = s

    # Generate aliases: AVENUE↔AVE, STREET↔ST, etc.
    expansions = {
        "AVE": "AVENUE", "ST": "STREET", "DR": "DRIVE", "RD": "ROAD",
        "BLVD": "BOULEVARD", "CRES": "CRESCENT", "PL": "PLACE",
        "CT": "COURT", "CRT": "COURT", "TERR": "TERRACE", "HWY": "HIGHWAY",
        "SQ": "SQUARE",
    }

    for s in canonical_streets:
        # Try expanding abbreviations
        expanded = s
        for abbr, full in expansions.items():
            expanded = re.sub(r'\b' + abbr + r'\b', full, expanded)
        if expanded != s:
            alias_map[expanded] = s

        # Try abbreviating full words
        contracted = s
        for abbr, full in expansions.items():
            contracted = re.sub(r'\b' + full + r'\b', abbr, contracted)
        if contracted != s:
            alias_map[contracted] = s

        # LANE — assessment keeps "LANE", MLS might use "LN"
        if " LANE" in s:
            alias_map[s.replace(" LANE", " LN")] = s

        # Bare street names: if assessment has "KINGSWAY", map "KINGSWAY ST/WAY/etc"
        if not re.search(r'\b(AVE|ST|DR|RD|BLVD|CRES|PL|CT|CRT|TERR|HWY|SQ|LANE|LN|WALK|WAY|MALL)\b', s):
            for suffix in ["ST", "WAY", "STREET", "RD"]:
                alias_map[f"{s} {suffix}"] = s

        # "RIVER DISTRICT CROSS" → "RIVER DISTRICT CROSSING"
        if s.endswith(" CROSS"):
            alias_map[s + "ING"] = s

    # Direction-first variants: "BROADWAY W" → also match "W BROADWAY", "W BROADWAY ST"
    for s in canonical_streets:
        dm = re.match(r'^(.+)\s+(E|W|N|S|NE|NW|SE|SW)$', s)
        if dm:
            base = dm.group(1)
            dir_ = dm.group(2)
            # "W BROADWAY", "W BROADWAY ST", "W BROADWAY STREET"
            alias_map[f"{dir_} {base}"] = s
            for abbr, full in expansions.items():
                if base.endswith(f" {abbr}"):
                    expanded_base = base[:-len(abbr)] + full
                    alias_map[f"{dir_} {expanded_base}"] = s

    # Direction between ordinal and type: "13TH E AVE" → "13TH AVE E"
    for s in canonical_streets:
        dm = re.match(r'^(\d+\w+)\s+(AVE|ST|DR|BLVD|CRES|RD)\s+(E|W|N|S)$', s)
        if dm:
            alt = f"{dm.group(1)} {dm.group(3)} {dm.group(2)}"
            alias_map[alt] = s
            # Also with full word
            for abbr, full in expansions.items():
                if dm.group(2) == abbr:
                    alias_map[f"{dm.group(1)} {dm.group(3)} {full}"] = s

    _STREET_LOOKUP = alias_map
    return alias_map


def _normalize_mls_street(street: str, alias_map: dict) -> list[str]:
    """Convert an MLS street name to assessment canonical form(s).

    Tries multiple normalization strategies and returns matching assessment streets.
    """
    s = street.upper().strip()
    candidates = set()

    # Strategy 1: Direct lookup
    if s in alias_map:
        candidates.add(alias_map[s])

    # Strategy 2: Move direction from start to end
    # "E 8TH AVE" → "8TH AVE E"
    dm = re.match(r'^(E|W|N|S|NE|NW|SE|SW)\s+(.+)$', s)
    if dm:
        moved = f"{dm.group(2)} {dm.group(1)}"
        if moved in alias_map:
            candidates.add(alias_map[moved])
        # Also try with abbreviation
        for trial in _expand_contract(moved):
            if trial in alias_map:
                candidates.add(alias_map[trial])

    # Strategy 3: Try all abbreviation/expansion variants
    for trial in _expand_contract(s):
        if trial in alias_map:
            candidates.add(alias_map[trial])
        # Also try with direction moved
        dm2 = re.match(r'^(E|W|N|S|NE|NW|SE|SW)\s+(.+)$', trial)
        if dm2:
            moved2 = f"{dm2.group(2)} {dm2.group(1)}"
            if moved2 in alias_map:
                candidates.add(alias_map[moved2])

    # Strategy 4: Fix ordinal — "38 AVE" → "38TH AVE"
    fixed = _fix_ordinal(s)
    if fixed != s:
        for trial in [fixed] + list(_expand_contract(fixed)):
            dm3 = re.match(r'^(E|W|N|S)\s+(.+)$', trial)
            if dm3:
                trial = f"{dm3.group(2)} {dm3.group(1)}"
            if trial in alias_map:
                candidates.add(alias_map[trial])

    return list(candidates) if candidates else [s]


def _expand_contract(s: str) -> list[str]:
    """Generate abbreviation variants of a street name."""
    abbrevs = {
        "AVENUE": "AVE", "STREET": "ST", "DRIVE": "DR", "ROAD": "RD",
        "BOULEVARD": "BLVD", "CRESCENT": "CRES", "PLACE": "PL",
        "COURT": "CT", "TERRACE": "TERR", "HIGHWAY": "HWY", "SQUARE": "SQ",
        "LANE": "LN",
    }
    results = []
    # Try abbreviating
    contracted = s
    for full, abbr in abbrevs.items():
        contracted = re.sub(r'\b' + full + r'\b', abbr, contracted)
    if contracted != s:
        results.append(contracted)
    # Try expanding
    expanded = s
    for full, abbr in abbrevs.items():
        expanded = re.sub(r'\b' + abbr + r'\b', full, expanded)
    if expanded != s:
        results.append(expanded)
    return results


def _fix_ordinal(s: str) -> str:
    """Fix missing ordinal suffixes: '38 AVE' → '38TH AVE', '1 AVE' → '1ST AVE'."""
    def _add_suffix(m):
        n = int(m.group(1))
        rest = m.group(2)
        if n % 100 in (11, 12, 13):
            suf = "TH"
        elif n % 10 == 1:
            suf = "ST"
        elif n % 10 == 2:
            suf = "ND"
        elif n % 10 == 3:
            suf = "RD"
        else:
            suf = "TH"
        return f"{n}{suf} {rest}"
    return re.sub(r'\b(\d+)\s+(AVE|AVENUE|ST|STREET)\b', _add_suffix, s)


def _parse_mls_address(addr: str) -> dict:
    """Parse MLS address into components."""
    raw = addr.upper().strip()

    # Merge PH/TH + space + number: "PH 401" → "PH401"
    norm = re.sub(r'\b(PH|TH)\s+(\d)', r'\1\2', raw)

    # Pattern 1: unit building street — strata
    # "601 1850 COMOX STREET", "PH1 688 E 17TH AVENUE", "8A 199 DRAKE STREET"
    m = re.match(r'^([A-Z]*\d+[A-Z]*)\s+(\d+)\s+(.+)$', norm)
    if m:
        unit_str = m.group(1)
        bldg = int(m.group(2))
        street = m.group(3).strip()
        unit_num = re.sub(r'[^0-9]', '', unit_str)
        return {
            "unit_str": unit_str,
            "unit_int": int(unit_num) if unit_num else None,
            "building_num": bldg,
            "street_raw": street,
            "is_strata": True,
        }

    # Pattern 2: direction number street — detached
    # "2432 W 49TH AVENUE" or "2168 E 8TH AVENUE"
    dm = re.match(r'^(\d+)\s+(E|W|N|S|NE|NW|SE|SW)\s+(.+)$', norm)
    if dm:
        return {
            "unit_str": None, "unit_int": None,
            "building_num": int(dm.group(1)),
            "street_raw": f"{dm.group(2)} {dm.group(3).strip()}",
            "is_strata": False,
        }

    # Pattern 3: number street — simple
    pm = re.match(r'^(\d+)\s+(.+)$', norm)
    if pm:
        return {
            "unit_str": None, "unit_int": None,
            "building_num": int(pm.group(1)),
            "street_raw": pm.group(2).strip(),
            "is_strata": False,
        }

    return {
        "unit_str": None, "unit_int": None, "building_num": None,
        "street_raw": norm, "is_strata": False,
    }


def _build_lookups(assessments: pd.DataFrame) -> dict:
    """Build lookup indices from BC Assessment data."""
    assessments = assessments.copy()
    assessments["sn"] = assessments["street_name"].str.upper().str.strip()

    strata = assessments[assessments["legal_type"] == "STRATA"].copy()

    # Lookup 1: (unit_civic, building_to_civic, street) → assessment row
    strata_exact = {}
    for _, r in strata.iterrows():
        tcn = r["to_civic_number"]
        if pd.notna(tcn):
            key = (int(r["civic_number"]), int(tcn), r["sn"])
            if key not in strata_exact:
                strata_exact[key] = r

    # Lookup 2: (unit_civic, street) → list of assessment rows
    strata_by_unit_street = {}
    for _, r in strata.iterrows():
        key = (int(r["civic_number"]), r["sn"])
        if key not in strata_by_unit_street:
            strata_by_unit_street[key] = []
        strata_by_unit_street[key].append(r)

    # Lookup 3: full_address → assessment row (for non-strata and fallback)
    # Prefer STRATA over LAND when both exist at the same address (e.g. duplexes
    # where the strata civic_number equals the building address — 2835 OLIVER CRES
    # has both a LAND record for the full lot and a STRATA record for each half).
    addr_lookup = {}
    for _, r in assessments.iterrows():
        fa = str(r["full_address"]).upper().strip()
        if fa not in addr_lookup:
            addr_lookup[fa] = r
        elif r.get("legal_type") == "STRATA" and addr_lookup[fa].get("legal_type") == "LAND":
            addr_lookup[fa] = r

    # Street alias map
    alias_map = _build_street_alias_map(assessments)

    return {
        "strata_exact": strata_exact,
        "strata_by_unit_street": strata_by_unit_street,
        "addr_lookup": addr_lookup,
        "alias_map": alias_map,
    }


def _find_match(parsed: dict, lookups: dict) -> Optional[pd.Series]:
    """Find the best BC Assessment match for a parsed MLS address."""
    alias_map = lookups["alias_map"]
    street_raw = parsed.get("street_raw", "")
    canon_streets = _normalize_mls_street(street_raw, alias_map)

    if parsed["is_strata"]:
        unit = parsed["unit_int"]
        bldg = parsed["building_num"]

        for street in canon_streets:
            # Try exact: (unit, building, street)
            if unit is not None:
                key = (unit, bldg, street)
                if key in lookups["strata_exact"]:
                    return lookups["strata_exact"][key]

            # Try unit+street (building-agnostic)
            if unit is not None:
                key = (unit, street)
                if key in lookups["strata_by_unit_street"]:
                    matches = lookups["strata_by_unit_street"][key]
                    if len(matches) == 1:
                        return matches[0]
                    # Multiple buildings — match by building number
                    for m in matches:
                        if pd.notna(m["to_civic_number"]) and int(m["to_civic_number"]) == bldg:
                            return m
                    return matches[0]  # best effort

        # Fallback: building address — but only if it's a STRATA record
        # (not a LAND parcel for the whole building, which would have a
        # wildly different value than an individual unit)
        for street in canon_streets:
            fa = f"{bldg} {street}"
            if fa in lookups["addr_lookup"]:
                candidate = lookups["addr_lookup"][fa]
                if candidate.get("legal_type") != "LAND":
                    return candidate

    else:
        # Non-strata
        bldg = parsed.get("building_num")
        if bldg:
            for street in canon_streets:
                fa = f"{bldg} {street}"
                if fa in lookups["addr_lookup"]:
                    return lookups["addr_lookup"][fa]

    return None


def match_sales_to_assessments() -> pd.DataFrame:
    """Match all sold listings to BC Assessment records."""
    assessments = _load_assessments()
    if assessments.empty:
        return pd.DataFrame()

    conn = get_connection()
    sold = pd.DataFrame([dict(r) for r in conn.execute(
        "SELECT * FROM sold_listings WHERE sold_price IS NOT NULL"
    ).fetchall()])
    conn.close()

    if sold.empty:
        return pd.DataFrame()

    lookups = _build_lookups(assessments)

    assessed_values = []
    land_values = []
    improvement_values = []

    for _, row in sold.iterrows():
        parsed = _parse_mls_address(row["address"])
        match = _find_match(parsed, lookups)

        if match is not None:
            assessed_values.append(match["total_assessed_value"])
            land_values.append(match["current_land_value"])
            improvement_values.append(match["current_improvement_value"])
        else:
            assessed_values.append(None)
            land_values.append(None)
            improvement_values.append(None)

    sold["assessed_value"] = assessed_values
    sold["land_value"] = land_values
    sold["improvement_value"] = improvement_values

    sold["sale_to_assessment_ratio"] = None
    mask = sold["assessed_value"].notna() & (sold["assessed_value"] > 0)
    sold.loc[mask, "sale_to_assessment_ratio"] = (
        sold.loc[mask, "sold_price"] / sold.loc[mask, "assessed_value"]
    ).round(3)

    matched = mask.sum()
    total = len(sold)
    logger.info(f"Matched {matched}/{total} sold listings to BC Assessment ({matched/total*100:.1f}%)")

    unmatched = total - matched
    if unmatched > 0:
        logger.info(f"  {unmatched} unmatched (likely UBC/UEL lands or new construction not yet assessed)")

    return sold


def get_market_summary() -> dict:
    """Compute market-wide and per-area sale-to-assessment ratios."""
    df = match_sales_to_assessments()
    if df.empty:
        return {"match_rate": 0, "total_sales": 0}

    matched = df[df["sale_to_assessment_ratio"].notna()].copy()
    total = len(df)
    n_matched = len(matched)

    summary = {
        "total_sales": total,
        "matched": n_matched,
        "match_rate": round(n_matched / total * 100, 1) if total > 0 else 0,
        "overall_sar": round(matched["sale_to_assessment_ratio"].median(), 3) if n_matched > 0 else None,
        "overall_sar_mean": round(matched["sale_to_assessment_ratio"].mean(), 3) if n_matched > 0 else None,
        "avg_sold_price": int(matched["sold_price"].mean()) if n_matched > 0 else None,
        "avg_assessed": int(matched["assessed_value"].mean()) if n_matched > 0 else None,
        "pct_above_assessment": round(
            (matched["sale_to_assessment_ratio"] > 1.0).mean() * 100, 1
        ) if n_matched > 0 else None,
        "pct_below_assessment": round(
            (matched["sale_to_assessment_ratio"] < 1.0).mean() * 100, 1
        ) if n_matched > 0 else None,
    }

    if n_matched > 0:
        by_area = matched.groupby("sub_area_name").agg(
            count=("sale_to_assessment_ratio", "size"),
            median_sar=("sale_to_assessment_ratio", "median"),
            avg_sold=("sold_price", "mean"),
            avg_assessed=("assessed_value", "mean"),
        ).sort_values("count", ascending=False)

        summary["by_area"] = [
            {
                "area": area,
                "count": int(row["count"]),
                "median_sar": round(row["median_sar"], 3),
                "avg_sold": int(row["avg_sold"]),
                "avg_assessed": int(row["avg_assessed"]),
            }
            for area, row in by_area.iterrows()
        ]

        matched["sold_month"] = pd.to_datetime(
            matched["sold_date"], format="mixed", dayfirst=False
        ).dt.to_period("M").astype(str)

        by_month = matched.groupby("sold_month").agg(
            count=("sale_to_assessment_ratio", "size"),
            median_sar=("sale_to_assessment_ratio", "median"),
            avg_sold=("sold_price", "mean"),
        ).sort_index()

        summary["by_month"] = [
            {
                "month": month,
                "count": int(row["count"]),
                "median_sar": round(row["median_sar"], 3),
                "avg_sold": int(row["avg_sold"]),
            }
            for month, row in by_month.iterrows()
        ]

    return summary


def get_matched_listings() -> list[dict]:
    """Get all matched listings with assessed values for the report."""
    df = match_sales_to_assessments()
    if df.empty:
        return []

    cols = [
        "mls_number", "address", "sub_area_name", "city",
        "sold_price", "list_price", "assessed_value", "land_value",
        "sale_to_assessment_ratio", "sold_date", "dom",
        "property_type", "floor_area", "bedrooms", "bathrooms",
        "year_built", "price_diff_pct",
    ]
    available = [c for c in cols if c in df.columns]
    result = df[available].copy()
    result = result[result["sale_to_assessment_ratio"].notna()]
    result = result.sort_values("sold_date", ascending=False)

    return result.to_dict("records")
