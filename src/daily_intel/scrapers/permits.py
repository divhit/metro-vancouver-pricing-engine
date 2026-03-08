"""Fetch building permits from City of Vancouver and Surrey Open Data APIs."""
import logging
from datetime import date, timedelta
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# City of Vancouver Open Data API (ODS v2.1)
VANCOUVER_API = "https://opendata.vancouver.ca/api/explore/v2.1/catalog/datasets/issued-building-permits/records"

# City of Surrey Open Data API (CKAN)
SURREY_PERMITS_API = "https://data.surrey.ca/api/3/action/datastore_search"
SURREY_PERMITS_RESOURCE = "building-permits"  # Will need actual resource ID


def fetch_vancouver_permits(
    days_back: int = 7,
    type_of_work: Optional[str] = None,
    min_project_value: Optional[int] = None,
) -> list[dict]:
    """Fetch recently issued building permits from City of Vancouver.

    Args:
        days_back: How many days back to look
        type_of_work: Filter by type - "New Building", "Addition / Alteration", "Demolition"
        min_project_value: Minimum project value to include
    """
    since_date = (date.today() - timedelta(days=days_back)).isoformat()

    # Build ODS SQL-like where clause
    where_parts = [f"issuedate >= '{since_date}'"]
    if type_of_work:
        where_parts.append(f"typeofwork = '{type_of_work}'")
    if min_project_value:
        where_parts.append(f"projectvalue >= {min_project_value}")

    params = {
        "where": " AND ".join(where_parts),
        "order_by": "issuedate DESC",
        "limit": 100,
    }

    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(VANCOUVER_API, params=params)
            resp.raise_for_status()
            data = resp.json()

        permits = []
        for record in data.get("results", []):
            permits.append({
                "permit_number": record.get("permitnumber"),
                "permit_number_created_date": record.get("permitnumbercreateddate"),
                "issue_date": record.get("issuedate"),
                "type_of_work": record.get("typeofwork"),
                "address": record.get("address"),
                "project_description": record.get("projectdescription"),
                "project_value": record.get("projectvalue"),
                "property_use": record.get("propertyuse"),
                "specific_use": record.get("specificusecategory"),
                "building_contractor": record.get("buildingcontractor"),
                "applicant": record.get("applicant"),
                "geo_point": record.get("geo_point_2d"),
                "source": "City of Vancouver",
            })

        logger.info(f"Fetched {len(permits)} Vancouver permits (last {days_back} days)")
        return permits

    except Exception as e:
        logger.error(f"Vancouver permits API failed: {e}")
        return []


def fetch_vancouver_new_buildings(days_back: int = 30, min_value: int = 500000) -> list[dict]:
    """Fetch new building permits — the ones that matter for supply pipeline."""
    return fetch_vancouver_permits(
        days_back=days_back,
        type_of_work="New Building",
        min_project_value=min_value,
    )


def fetch_vancouver_demolitions(days_back: int = 30) -> list[dict]:
    """Fetch demolition permits — signals redevelopment."""
    return fetch_vancouver_permits(
        days_back=days_back,
        type_of_work="Demolition / Deconstruction",
    )


def fetch_surrey_permits(days_back: int = 7) -> list[dict]:
    """Fetch building permits from City of Surrey Open Data.

    Surrey uses CKAN API — we need the resource ID for the building permits dataset.
    """
    since_date = (date.today() - timedelta(days=days_back)).isoformat()

    # Surrey's CKAN datastore search
    params = {
        "resource_id": SURREY_PERMITS_RESOURCE,
        "limit": 100,
        "sort": "ISSUED_DATE desc",
        "filters": f'{{"PERMIT_TYPE":"Building"}}',
    }

    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(SURREY_PERMITS_API, params=params)
            resp.raise_for_status()
            data = resp.json()

        if not data.get("success"):
            logger.warning("Surrey API returned success=false")
            return []

        permits = []
        for record in data.get("result", {}).get("records", []):
            permits.append({
                "permit_number": record.get("PERMIT_NUMBER"),
                "issue_date": record.get("ISSUED_DATE"),
                "type_of_work": record.get("WORK_TYPE"),
                "address": record.get("ADDRESS"),
                "project_description": record.get("DESCRIPTION"),
                "project_value": record.get("CONSTRUCTION_VALUE"),
                "source": "City of Surrey",
            })

        logger.info(f"Fetched {len(permits)} Surrey permits")
        return permits

    except Exception as e:
        logger.error(f"Surrey permits API failed: {e}")
        return []


def get_permit_summary(permits: list[dict]) -> dict:
    """Summarize permits into report-friendly stats."""
    if not permits:
        return {"total": 0, "new_buildings": 0, "demolitions": 0, "total_value": 0}

    new_buildings = [p for p in permits if "New" in (p.get("type_of_work") or "")]
    demolitions = [p for p in permits if "Demol" in (p.get("type_of_work") or "")]
    total_value = sum(p.get("project_value") or 0 for p in permits)

    return {
        "total": len(permits),
        "new_buildings": len(new_buildings),
        "demolitions": len(demolitions),
        "total_value": total_value,
        "top_projects": sorted(
            [p for p in permits if p.get("project_value")],
            key=lambda p: p["project_value"],
            reverse=True,
        )[:5],
    }
