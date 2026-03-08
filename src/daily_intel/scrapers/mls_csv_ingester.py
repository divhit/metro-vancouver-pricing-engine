"""Ingest MLS Paragon CSV exports (sold listings for Van East/West)."""
import csv
import logging
import re
from datetime import date
from pathlib import Path
from typing import Optional

from ..storage.database import get_connection

logger = logging.getLogger(__name__)

# Sub-area code to human-readable mapping
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

# Attached property columns
ATTACHED_COLUMNS = [
    "PicCount", "Pics", "ML #", "Status", "Address", "S/A", "Price",
    "List Date", "DOM", "Tot BR", "Tot Baths", "TotFlArea", "Yr Blt",
    "Age", "Locker", "TotalPrkng", "MaintFee", "TypeDwel", "Bylaw Restrictions",
]

# Detached property columns
DETACHED_COLUMNS = [
    "PicCount", "Pics", "ML #", "Status", "Address", "S/A", "Price",
    "List Date", "DOM", "Tot BR", "Tot Baths", "TotFlArea", "Yr Blt",
    "Age", "Frontage - Feet", "Depth", "#Kitchens", "TypeDwel", "Style of Home",
]

# Extended columns (when Aparna adds Sold Price + Sold Date)
EXTENDED_COLUMNS_EXTRA = ["Sold Price", "Sold Date", "Area"]


def _clean_price(price_str: str) -> Optional[int]:
    """Parse price strings like '$1,350,000' to integer."""
    if not price_str:
        return None
    cleaned = re.sub(r"[^\d.]", "", price_str)
    try:
        return int(float(cleaned))
    except (ValueError, TypeError):
        return None


def _clean_number(val: str) -> Optional[int]:
    """Parse numeric strings, handling commas."""
    if not val or val.strip() == "":
        return None
    cleaned = re.sub(r"[^\d.]", "", val)
    try:
        return int(float(cleaned))
    except (ValueError, TypeError):
        return None


def _clean_float(val: str) -> Optional[float]:
    """Parse float strings."""
    if not val or val.strip() == "":
        return None
    cleaned = re.sub(r"[^\d.]", "", val)
    try:
        return float(cleaned)
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
    """Convert S/A code to readable area name."""
    return SUB_AREA_MAP.get(sub_area_code, sub_area_code)


def _is_west_side(sub_area_code: str) -> bool:
    """Check if sub-area is Vancouver West."""
    return sub_area_code.startswith("VVW")


def ingest_csv(csv_path: str, source_label: Optional[str] = None) -> dict:
    """Ingest a Paragon MLS CSV export into the database.

    Returns summary stats about what was ingested.
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
        logger.info(f"Detected format: {fmt} for {path.name} ({len(headers)} columns)")

        conn = get_connection()
        inserted = 0
        skipped = 0
        rows = []

        for row in reader:
            mls_number = row.get("ML #", "").strip()
            if not mls_number:
                skipped += 1
                continue

            status = row.get("Status", "").strip()
            sub_area = row.get("S/A", "").strip()
            address = row.get("Address", "").strip()
            price = _clean_price(row.get("Price", ""))
            list_date = row.get("List Date", "").strip()
            dom = _clean_number(row.get("DOM", ""))
            bedrooms = _clean_number(row.get("Tot BR", ""))
            bathrooms = _clean_number(row.get("Tot Baths", ""))
            floor_area = _clean_number(row.get("TotFlArea", ""))
            year_built = _clean_number(row.get("Yr Blt", ""))
            type_dwel = row.get("TypeDwel", "").strip()

            # Extended fields (if Aparna adds them)
            sold_price = _clean_price(row.get("Sold Price", ""))
            sold_date = row.get("Sold Date", "").strip() or None

            # Format-specific fields
            if fmt == "attached":
                maint_fee = _clean_float(row.get("MaintFee", ""))
                parking = _clean_number(row.get("TotalPrkng", ""))
                locker = row.get("Locker", "").strip()
                extra_json = f'{{"maint_fee": {maint_fee or "null"}, "parking": {parking or "null"}, "locker": "{locker}"}}'
            elif fmt == "detached":
                frontage = _clean_float(row.get("Frontage - Feet", ""))
                depth = _clean_float(row.get("Depth", ""))
                kitchens = _clean_number(row.get("#Kitchens", ""))
                style = row.get("Style of Home", "").strip()
                extra_json = f'{{"frontage": {frontage or "null"}, "depth": {depth or "null"}, "kitchens": {kitchens or "null"}, "style": "{style}"}}'
            else:
                extra_json = "{}"

            area_name = _resolve_area(sub_area)
            side = "Vancouver West" if _is_west_side(sub_area) else "Vancouver East"

            try:
                conn.execute("""
                    INSERT OR IGNORE INTO sold_listings (
                        mls_number, sold_date, entry_date, sold_price, listing_price,
                        listing_price_orig, street_address, city, area_name, province,
                        postal_code, bedroom_count, bathroom_count, house_size, lot_size,
                        property_type, listing_agent, listing_brokerage, latitude, longitude,
                        images_json, first_seen_date, raw_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    mls_number,
                    sold_date,
                    list_date if list_date else None,
                    sold_price or price,  # Use sold_price if available, else list price
                    price,
                    price,
                    address,
                    side,
                    area_name,
                    "BC",
                    None,  # postal code not in export
                    bedrooms,
                    bathrooms,
                    floor_area,
                    None,  # lot size not directly available
                    type_dwel,
                    None,  # agent not in export
                    None,  # brokerage not in export
                    None, None,  # lat/lon not in export
                    row.get("Pics", ""),
                    today,
                    extra_json,
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


def watch_directory(directory: str, pattern: str = "ML_*.csv"):
    """Check for new CSV files and ingest them.

    Tracks which files have been processed to avoid re-ingestion.
    """
    dir_path = Path(directory)
    conn = get_connection()

    # Create tracking table if needed
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ingested_files (
            file_path TEXT PRIMARY KEY,
            ingested_at TEXT NOT NULL,
            stats_json TEXT
        )
    """)
    conn.commit()

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
        import json
        stats = ingest_csv(str(csv_file))
        conn.execute(
            "INSERT INTO ingested_files (file_path, ingested_at, stats_json) VALUES (?, ?, ?)",
            (str(csv_file), date.today().isoformat(), json.dumps(stats))
        )
        conn.commit()
        results.append(stats)

    conn.close()
    return results
