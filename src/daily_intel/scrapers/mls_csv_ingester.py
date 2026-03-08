"""Ingest MLS Paragon CSV exports (sold listings for Van East/West)."""
import csv
import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import Optional

from ..storage.database import get_connection

logger = logging.getLogger(__name__)

# Sub-area code to human-readable mapping (Vancouver East & West only)
SUB_AREA_MAP = {
    # Vancouver West
    "VVWDT": "Downtown", "VVWWE": "West End", "VVWCB": "Coal Harbour",
    "VVWYA": "Yaletown", "VVWKT": "Kitsilano", "VVWFA": "Fairview",
    "VVWPG": "Point Grey", "VVWKE": "Kerrisdale", "VVWSH": "Shaughnessy",
    "VVWOA": "Oakridge", "VVWMR": "Marpole", "VVWCA": "Cambie",
    "VVWSA": "South Cambie", "VVWUL": "University/UBC", "VVWDN": "Dunbar",
    "VVWSO": "South Granville", "VVWQU": "Quilchena", "VVWAR": "Arbutus",
    "VVWMK": "MacKenzie Heights",
    # Vancouver East
    "VVEST": "Strathcona", "VVEDT": "Downtown East", "VVEGW": "Grandview-Woodland",
    "VVERE": "Renfrew", "VVESU": "Sunrise", "VVEKO": "Kensington-Cedar Cottage",
    "VVEKN": "Knight", "VVEVI": "Victoria", "VVEFR": "Fraser",
    "VVECO": "Collingwood", "VVESV": "South Vancouver", "VVEKL": "Killarney",
    "VVEMP": "Mount Pleasant", "VVEHA": "Hastings",
}


def _clean_price(price_str: str) -> Optional[int]:
    """Parse price strings like '$1,350,000' to integer."""
    if not price_str or not price_str.strip():
        return None
    cleaned = re.sub(r"[^\d.]", "", price_str)
    try:
        return int(float(cleaned)) if cleaned else None
    except (ValueError, TypeError):
        return None


def _clean_int(val: str) -> Optional[int]:
    if not val or not val.strip():
        return None
    cleaned = re.sub(r"[^\d.]", "", val)
    try:
        return int(float(cleaned)) if cleaned else None
    except (ValueError, TypeError):
        return None


def _clean_float(val: str) -> Optional[float]:
    if not val or not val.strip():
        return None
    cleaned = re.sub(r"[^\d.]", "", val)
    try:
        return float(cleaned) if cleaned else None
    except (ValueError, TypeError):
        return None


def _detect_format(headers: list[str]) -> str:
    """Detect if CSV is attached or detached format."""
    header_set = set(h.strip().strip('"') for h in headers)
    if "MaintFee" in header_set or "Locker" in header_set:
        return "attached"
    if "Frontage - Feet" in header_set or "#Kitchens" in header_set:
        return "detached"
    return "unknown"


def _resolve_area(sub_area_code: str) -> str:
    return SUB_AREA_MAP.get(sub_area_code, sub_area_code)


def _side(sub_area_code: str) -> str:
    if sub_area_code.startswith("VVW"):
        return "Vancouver West"
    elif sub_area_code.startswith("VVE"):
        return "Vancouver East"
    return "Other"


def ingest_csv(csv_path: str, source_label: Optional[str] = None) -> dict:
    """Ingest a Paragon MLS CSV export into the database.

    Handles both attached (condo/townhome) and detached (house) formats.
    Stores ALL fields from the CSV.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    today = date.today().isoformat()
    source = source_label or path.name

    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        fmt = _detect_format(headers)
        logger.info(f"Detected format: {fmt} for {path.name} ({len(headers)} columns: {headers})")

        conn = get_connection()
        inserted = 0
        updated = 0
        skipped = 0

        for row in reader:
            mls_number = row.get("ML #", "").strip()
            if not mls_number:
                skipped += 1
                continue

            sub_area = row.get("S/A", "").strip()
            address = row.get("Address", "").strip()
            status = row.get("Status", "").strip()

            # Price field = sold price in these exports (Status=F means firm/sold)
            sold_price = _clean_price(row.get("Sold Price") or row.get("Price", ""))
            list_price = _clean_price(row.get("List Price", ""))

            # Calculate price difference
            price_diff = None
            price_diff_pct = None
            if sold_price and list_price and list_price > 0:
                price_diff = sold_price - list_price
                price_diff_pct = round(price_diff / list_price * 100, 1)

            dom = _clean_int(row.get("DOM", ""))
            days_on_mls = _clean_int(row.get("Days On MLS", ""))
            bedrooms = _clean_int(row.get("Tot BR", ""))
            bathrooms = _clean_int(row.get("Tot Baths", ""))
            floor_area = _clean_int(row.get("TotFlArea", ""))
            year_built = _clean_int(row.get("Yr Blt", ""))
            age = _clean_int(row.get("Age", ""))
            type_dwel = row.get("TypeDwel", "").strip()
            list_date = row.get("List Date", "").strip() or None
            sold_date = row.get("Sold Date", "").strip() or None
            postal_code = row.get("Postal Code", "").strip() or None
            zoning = row.get("Zoning", "").strip() or None
            view = row.get("View - Specify", "").strip() or None
            pic_count = _clean_int(row.get("PicCount", ""))
            pic_url = row.get("Pics", "").strip() or None

            # Attached-specific
            locker = row.get("Locker", "").strip() or None
            parking = _clean_int(row.get("TotalPrkng", ""))
            maint_fee = _clean_float(row.get("MaintFee", ""))
            bylaw = row.get("Bylaw Restrictions", "").strip() or None

            # Detached-specific
            frontage = _clean_float(row.get("Frontage - Feet", ""))
            depth = _clean_float(row.get("Depth", ""))
            kitchens = _clean_int(row.get("#Kitchens", ""))
            style = row.get("Style of Home", "").strip() or None

            area_name = _resolve_area(sub_area)
            city = _side(sub_area)

            try:
                # Use INSERT OR REPLACE to update if MLS number already exists
                conn.execute("""
                    INSERT OR REPLACE INTO sold_listings (
                        mls_number, status, address, sub_area, sub_area_name, city,
                        postal_code, sold_price, list_price, price_diff, price_diff_pct,
                        list_date, sold_date, dom, days_on_mls,
                        bedrooms, bathrooms, floor_area, year_built, age,
                        property_type, style, zoning, view,
                        locker, parking, maint_fee, bylaw_restrictions,
                        frontage, depth, kitchens,
                        pic_count, pic_url, source_file, source_format, first_seen_date
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?, ?, ?, ?
                    )
                """, (
                    mls_number, status, address, sub_area, area_name, city,
                    postal_code, sold_price, list_price, price_diff, price_diff_pct,
                    list_date, sold_date, dom, days_on_mls,
                    bedrooms, bathrooms, floor_area, year_built, age,
                    type_dwel, style, zoning, view,
                    locker, parking, maint_fee, bylaw,
                    frontage, depth, kitchens,
                    pic_count, pic_url, source, fmt, today,
                ))
                inserted += 1
            except Exception as e:
                logger.warning(f"Failed to insert {mls_number}: {e}")
                skipped += 1

        conn.commit()
        conn.close()

    stats = {
        "file": path.name,
        "format": fmt,
        "total_rows": inserted + skipped,
        "inserted": inserted,
        "skipped": skipped,
        "source": source,
        "date": today,
    }
    logger.info(f"Ingested {inserted} listings from {path.name} ({skipped} skipped)")
    return stats


def ingest_directory(directory: str, pattern: str = "ML_*.csv") -> list[dict]:
    """Ingest all matching CSV files from a directory."""
    dir_path = Path(directory)
    results = []
    for csv_file in sorted(dir_path.glob(pattern)):
        logger.info(f"Processing: {csv_file.name}")
        stats = ingest_csv(str(csv_file))
        results.append(stats)
    return results


def watch_directory(directory: str, pattern: str = "ML_*.csv") -> list[dict]:
    """Check for new CSV files and ingest them. Tracks processed files."""
    dir_path = Path(directory)
    conn = get_connection()

    new_files = []
    for csv_file in sorted(dir_path.glob(pattern)):
        row = conn.execute(
            "SELECT 1 FROM ingested_files WHERE file_path = ?",
            (str(csv_file),)
        ).fetchone()
        if not row:
            new_files.append(csv_file)

    results = []
    for csv_file in new_files:
        logger.info(f"New file detected: {csv_file.name}")
        stats = ingest_csv(str(csv_file))
        conn.execute(
            "INSERT OR REPLACE INTO ingested_files (file_path, ingested_at, row_count, stats_json) VALUES (?, ?, ?, ?)",
            (str(csv_file), date.today().isoformat(), stats["inserted"], json.dumps(stats))
        )
        conn.commit()
        results.append(stats)

    conn.close()
    return results
