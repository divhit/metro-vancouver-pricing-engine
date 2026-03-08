"""Match MLS sold listings to BC Assessment records and compute market ratios.

Computes sale-to-assessment ratio (SAR) to show where the market actually trades
relative to assessed values. SAR > 1.0 = selling above assessment, < 1.0 = below.

Address matching handles two cases:
1. Strata (condos): MLS "601 1850 COMOX STREET" → BC Assessment unit 601 at building 1850
   Match on civic_number=601, to_civic_number=1850, street_name=COMOX ST
2. Detached: MLS "2168 E 8TH AVENUE" → BC Assessment "2168 8TH AVE E"
   Match on normalized full_address
"""
import logging
import re
from typing import Optional

import pandas as pd

from src.daily_intel.storage.database import get_connection

logger = logging.getLogger(__name__)

_ASSESSMENT_DF: Optional[pd.DataFrame] = None

_STREET_ABBREVS = {
    "AVENUE": "AVE", "STREET": "ST", "DRIVE": "DR", "ROAD": "RD",
    "BOULEVARD": "BLVD", "CRESCENT": "CRES", "PLACE": "PL",
    "COURT": "CT", "LANE": "LN", "TERRACE": "TERR", "HIGHWAY": "HWY",
}


def _load_assessments() -> pd.DataFrame:
    global _ASSESSMENT_DF
    if _ASSESSMENT_DF is not None:
        return _ASSESSMENT_DF
    try:
        df = pd.read_parquet("data/processed/enriched_properties.parquet")
        df["street_name_norm"] = df["street_name"].str.upper().str.strip()
        _ASSESSMENT_DF = df
        logger.info(f"Loaded {len(df)} BC Assessment records")
        return df
    except Exception as e:
        logger.error(f"Failed to load assessment data: {e}")
        return pd.DataFrame()


def _normalize_street(street: str) -> str:
    """Normalize street name: abbreviate types, move direction to end."""
    s = street.upper().strip()
    for full, abbr in _STREET_ABBREVS.items():
        s = s.replace(full, abbr)
    return s


def _parse_mls_address(addr: str) -> dict:
    """Parse MLS address into components.

    Returns dict with keys: unit, building_num, street, is_strata, full_normalized
    """
    addr = addr.upper().strip()
    for full, abbr in _STREET_ABBREVS.items():
        addr = re.sub(r'\b' + full + r'\b', abbr, addr)

    # Move direction from middle to end of street name for any address component
    def _fix_direction(street: str) -> str:
        m = re.match(r"^(E|W|N|S)\s+(.+)$", street)
        if m:
            return f"{m.group(2)} {m.group(1)}"
        return street

    # Strata pattern: "601 1850 COMOX ST" -> unit=601, building=1850, street=COMOX ST
    strata_match = re.match(r"^(\d+)\s+(\d+)\s+(.+)$", addr)
    if strata_match:
        street = _fix_direction(strata_match.group(3).strip())
        return {
            "unit": int(strata_match.group(1)),
            "building_num": int(strata_match.group(2)),
            "street": street,
            "is_strata": True,
            "full_normalized": addr,
        }

    # Detached with direction: "2168 E 8TH AVE" -> "2168 8TH AVE E"
    dir_match = re.match(r"^(\d+)\s+(E|W|N|S)\s+(.+)$", addr)
    if dir_match:
        normalized = f"{dir_match.group(1)} {dir_match.group(3)} {dir_match.group(2)}"
        return {
            "unit": None,
            "building_num": int(dir_match.group(1)),
            "street": f"{dir_match.group(3)} {dir_match.group(2)}",
            "is_strata": False,
            "full_normalized": normalized,
        }

    # Plain address: "868 KINGSWAY"
    plain_match = re.match(r"^(\d+)\s+(.+)$", addr)
    if plain_match:
        return {
            "unit": None,
            "building_num": int(plain_match.group(1)),
            "street": plain_match.group(2).strip(),
            "is_strata": False,
            "full_normalized": addr,
        }

    return {
        "unit": None, "building_num": None, "street": addr,
        "is_strata": False, "full_normalized": addr,
    }


def match_sales_to_assessments() -> pd.DataFrame:
    """Match sold listings to BC Assessment records."""
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

    # Parse all MLS addresses
    parsed = sold["address"].apply(_parse_mls_address).apply(pd.Series)
    sold = pd.concat([sold, parsed], axis=1)

    # Build assessment lookup indices
    # For strata: index by (civic_number, to_civic_number, street_name_norm)
    strata_mask = assessments["legal_type"] == "STRATA"
    strata_df = assessments[strata_mask].copy()
    strata_df["lookup_key"] = (
        strata_df["civic_number"].astype(str) + "|" +
        strata_df["to_civic_number"].fillna(0).astype(int).astype(str) + "|" +
        strata_df["street_name_norm"]
    )
    strata_lookup = strata_df.drop_duplicates("lookup_key").set_index("lookup_key")

    # For non-strata: index by full_address
    non_strata_df = assessments[~strata_mask].copy()
    non_strata_df["match_addr"] = non_strata_df["full_address"].str.upper().str.strip()
    non_strata_lookup = non_strata_df.drop_duplicates("match_addr").set_index("match_addr")

    # Match each listing
    assessed_values = []
    land_values = []
    improvement_values = []
    bc_year_built = []

    for _, row in sold.iterrows():
        match = None

        if row["is_strata"] and pd.notna(row["unit"]):
            # Convert to int (pandas may store as float due to NaN in other rows)
            unit = int(row["unit"])
            bldg = int(row["building_num"])
            key = f"{unit}|{bldg}|{row['street']}"
            if key in strata_lookup.index:
                match = strata_lookup.loc[key]
            else:
                # Try with reversed street direction
                street = row["street"]
                dir_match = re.match(r"^(E|W|N|S)\s+(.+)$", street)
                if dir_match:
                    alt_street = f"{dir_match.group(2)} {dir_match.group(1)}"
                    alt_key = f"{unit}|{bldg}|{alt_street}"
                    if alt_key in strata_lookup.index:
                        match = strata_lookup.loc[alt_key]
                if match is None:
                    dir_end = re.match(r"^(.+)\s+(E|W|N|S)$", street)
                    if dir_end:
                        alt_street = f"{dir_end.group(2)} {dir_end.group(1)}"
                        alt_key = f"{unit}|{bldg}|{alt_street}"
                        if alt_key in strata_lookup.index:
                            match = strata_lookup.loc[alt_key]

        if match is None:
            # Try non-strata match on full normalized address
            norm = row["full_normalized"]
            if norm in non_strata_lookup.index:
                match = non_strata_lookup.loc[norm]

        if match is not None:
            assessed_values.append(match["total_assessed_value"])
            land_values.append(match["current_land_value"])
            improvement_values.append(match["current_improvement_value"])
            bc_year_built.append(match.get("year_built"))
        else:
            assessed_values.append(None)
            land_values.append(None)
            improvement_values.append(None)
            bc_year_built.append(None)

    sold["assessed_value"] = assessed_values
    sold["land_value"] = land_values
    sold["improvement_value"] = improvement_values
    sold["bc_year_built"] = bc_year_built

    # Compute sale-to-assessment ratio
    sold["sale_to_assessment_ratio"] = None
    mask = sold["assessed_value"].notna() & (sold["assessed_value"] > 0)
    sold.loc[mask, "sale_to_assessment_ratio"] = (
        sold.loc[mask, "sold_price"] / sold.loc[mask, "assessed_value"]
    ).round(3)

    matched = mask.sum()
    total = len(sold)
    logger.info(f"Matched {matched}/{total} sold listings to BC Assessment ({matched/total*100:.1f}%)")

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
        # Per sub-area breakdown
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

        # By month
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
