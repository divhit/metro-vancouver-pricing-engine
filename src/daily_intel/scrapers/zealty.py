"""Zealty tRPC API client for sold listings and active listing snapshots."""
import json
import logging
import time
from datetime import date
from typing import Optional
from urllib.parse import urlencode

import httpx

from ..config import ZEALTY_TRPC_URL, ZEALTY_COOKIE_FILE, TRACKED_REGIONS, PROPERTY_TYPES

logger = logging.getLogger(__name__)

# Session cookies loaded from exported browser session
_cookies: Optional[dict] = None


def _load_cookies() -> dict:
    """Load Zealty session cookies from file."""
    global _cookies
    if _cookies is not None:
        return _cookies
    if ZEALTY_COOKIE_FILE.exists():
        with open(ZEALTY_COOKIE_FILE) as f:
            cookie_data = json.load(f)
        # Support both formats: [{name, value}] or {name: value}
        if isinstance(cookie_data, list):
            _cookies = {c["name"]: c["value"] for c in cookie_data}
        else:
            _cookies = cookie_data
        logger.info(f"Loaded {len(_cookies)} Zealty cookies")
    else:
        _cookies = {}
        logger.warning(f"No cookie file at {ZEALTY_COOKIE_FILE} — requests may fail for auth-required data")
    return _cookies


def _build_search_input(
    status: str = "sold",
    region: str = "metro",
    month: str = "7",
    page: int = 1,
    limit: int = 100,
    property_types: Optional[list[str]] = None,
    sort_by: str = "daysAtCurrentPrice",
    sort_direction: str = "asc",
) -> dict:
    """Build tRPC input for properties.search."""
    return {
        "0": {
            "json": {
                "propertyTypes": property_types or PROPERTY_TYPES,
                "status": status,
                "province": "BC",
                "cityOrRegion": region,
                "sortBy": sort_by,
                "sortDirection": sort_direction,
                "month": month,
                "features": [],
                "showOnly": [],
                "minBeds": "any",
                "minBedsExact": "false",
                "minBaths": "any",
                "minBathsExact": "false",
                "orPriceChanged": "false",
                "landSizeMetric": "false",
                "page": page,
                "limit": limit,
                "areaNames": None,
                "postalCode": None,
            },
            "meta": {
                "values": {
                    "areaNames": ["undefined"],
                    "postalCode": ["undefined"],
                },
                "v": 1,
            },
        }
    }


def _fetch_trpc(input_data: dict, procedure: str = "properties.search") -> Optional[dict]:
    """Make a tRPC batch request to Zealty."""
    cookies = _load_cookies()
    params = {
        "batch": "1",
        "input": json.dumps(input_data),
    }
    url = f"{ZEALTY_TRPC_URL}/{procedure}?{urlencode(params)}"

    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(url, cookies=cookies, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Accept": "application/json",
                "Referer": "https://www.zealty.ca/region/bc/metro",
            })
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0].get("result", {}).get("data", {}).get("json")
            return data
    except Exception as e:
        logger.error(f"Zealty tRPC request failed: {e}")
        return None


def fetch_sold_listings(
    region: str = "metro",
    days_back: str = "7",
    max_pages: int = 10,
) -> list[dict]:
    """Fetch all sold listings for a region within the last N days."""
    all_listings = []
    page = 1

    while page <= max_pages:
        logger.info(f"Fetching sold listings for {region}, page {page}...")
        input_data = _build_search_input(
            status="sold",
            region=region,
            month=days_back,
            page=page,
            limit=100,
        )
        result = _fetch_trpc(input_data)
        if not result or not result.get("properties"):
            break

        listings = result["properties"]
        all_listings.extend(listings)
        logger.info(f"  Got {len(listings)} listings (total: {len(all_listings)})")

        # If we got fewer than the limit, we've reached the end
        if len(listings) < 100:
            break
        page += 1
        time.sleep(0.5)  # Be respectful

    return all_listings


def fetch_active_listings(
    region: str = "metro",
    max_pages: int = 50,
) -> list[dict]:
    """Fetch all active listings for a region (for daily snapshot)."""
    all_listings = []
    page = 1

    while page <= max_pages:
        logger.info(f"Fetching active listings for {region}, page {page}...")
        input_data = _build_search_input(
            status="active",
            region=region,
            month="0",  # All active
            page=page,
            limit=100,
            sort_by="listedDate",
            sort_direction="desc",
        )
        result = _fetch_trpc(input_data)
        if not result or not result.get("properties"):
            break

        listings = result["properties"]
        all_listings.extend(listings)
        logger.info(f"  Got {len(listings)} active listings (total: {len(all_listings)})")

        if len(listings) < 100:
            break
        page += 1
        time.sleep(0.5)

    return all_listings


def fetch_all_regions_sold(days_back: str = "7") -> dict[str, list[dict]]:
    """Fetch sold listings across all tracked regions."""
    results = {}
    for region_info in TRACKED_REGIONS:
        region_id = region_info["cityOrRegion"]
        label = region_info["label"]
        logger.info(f"Fetching sold for {label}...")
        listings = fetch_sold_listings(region=region_id, days_back=days_back)
        results[label] = listings
        time.sleep(1)  # Rate limit between regions
    return results


def fetch_metro_sold(days_back: str = "7") -> list[dict]:
    """Fetch all Metro Vancouver sold listings (single query, deduped)."""
    return fetch_sold_listings(region="metro", days_back=days_back)


def compute_market_stats(sold_listings: list[dict], active_listings: list[dict], region: str) -> dict:
    """Compute summary market statistics from listings data."""
    today = date.today().isoformat()

    sold_prices = [l["soldPrice"] for l in sold_listings if l.get("soldPrice")]
    list_prices = [l["listingPrice"] for l in active_listings if l.get("listingPrice")]

    # Price change percentages (sold vs original list)
    price_changes = []
    for l in sold_listings:
        if l.get("soldPrice") and l.get("listingPriceOrig") and l["listingPriceOrig"] > 0:
            pct = (l["soldPrice"] - l["listingPriceOrig"]) / l["listingPriceOrig"] * 100
            price_changes.append(pct)

    stats = {
        "stat_date": today,
        "region": region,
        "property_type": "ALL",
        "total_active": len(active_listings),
        "total_sold_7d": len(sold_listings),
        "total_sold_30d": None,
        "avg_sold_price": round(sum(sold_prices) / len(sold_prices)) if sold_prices else None,
        "median_sold_price": round(sorted(sold_prices)[len(sold_prices) // 2]) if sold_prices else None,
        "avg_list_price": round(sum(list_prices) / len(list_prices)) if list_prices else None,
        "avg_price_change_pct": round(sum(price_changes) / len(price_changes), 1) if price_changes else None,
        "sales_to_active_ratio": round(len(sold_listings) / len(active_listings) * 100, 1) if active_listings else None,
        "avg_days_on_market": None,
    }
    return stats
