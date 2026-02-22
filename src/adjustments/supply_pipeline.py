"""
Supply pipeline adjustment.

Nearby construction projects affect property values:
- Short-term: construction disruption, temporary supply pressure (-2% to -5%)
- Long-term: amenity creation, neighbourhood improvement (+1% to +3%)
- Rental supply primarily impacts condos, not detached homes

Key Metro Vancouver projects (2024-2030):
- Oakridge Park: 4,300 homes, Phase 1 opening spring 2026
- Senakw: 6,000 rental units, First Nations leasehold
- Broadway Plan: 30,000+ units over 30 years
- Jericho Lands: 13,000 units over 30+ years
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from src.features.feature_registry import PropertyType

logger = logging.getLogger(__name__)

# Major development projects in Metro Vancouver
# Each project includes location, unit count, tenure type, and status.
# Status values: "completed", "under_construction", "approved", "planned"
MAJOR_PROJECTS = [
    {
        "name": "Oakridge Park",
        "lat": 49.2275,
        "lon": -123.1170,
        "units": 4300,
        "type": "strata",
        "status": "under_construction",
        "description": "Mixed-use redevelopment of Oakridge Centre",
    },
    {
        "name": "Senakw",
        "lat": 49.2726,
        "lon": -123.1380,
        "units": 6000,
        "type": "rental",
        "status": "under_construction",
        "description": "Squamish Nation rental towers near Burrard Bridge",
    },
    {
        "name": "Broadway Plan SkyTrain Corridor",
        "lat": 49.2630,
        "lon": -123.1150,
        "units": 30000,
        "type": "mixed",
        "status": "planned",
        "description": "30,000+ units along Broadway SkyTrain extension over 30 years",
    },
    {
        "name": "Jericho Lands",
        "lat": 49.2680,
        "lon": -123.1900,
        "units": 13000,
        "type": "mixed",
        "status": "planned",
        "description": "MST Nations and Canada Lands Company partnership",
    },
    {
        "name": "River District",
        "lat": 49.2070,
        "lon": -123.0360,
        "units": 7000,
        "type": "strata",
        "status": "under_construction",
        "description": "East Fraser Lands master-planned community",
    },
    {
        "name": "Lougheed Town Centre",
        "lat": 49.2490,
        "lon": -122.8960,
        "units": 5500,
        "type": "mixed",
        "status": "approved",
        "description": "SkyTrain-adjacent high-density redevelopment in Burnaby",
    },
    {
        "name": "Brentwood Town Centre",
        "lat": 49.2662,
        "lon": -122.9920,
        "units": 8000,
        "type": "strata",
        "status": "under_construction",
        "description": "Multi-tower redevelopment around Brentwood SkyTrain",
    },
    {
        "name": "Metrotown South",
        "lat": 49.2240,
        "lon": -123.0000,
        "units": 4000,
        "type": "mixed",
        "status": "approved",
        "description": "Continued densification south of Metrotown station",
    },
    {
        "name": "Heather Lands",
        "lat": 49.2450,
        "lon": -123.1140,
        "units": 1700,
        "type": "mixed",
        "status": "approved",
        "description": "MST Nations development near Queen Elizabeth Park",
    },
    {
        "name": "Fraser Mills",
        "lat": 49.2190,
        "lon": -122.8530,
        "units": 6200,
        "type": "mixed",
        "status": "under_construction",
        "description": "Former mill site waterfront community in Coquitlam",
    },
    {
        "name": "Capstan Village",
        "lat": 49.1850,
        "lon": -123.1360,
        "units": 5000,
        "type": "strata",
        "status": "under_construction",
        "description": "High-density node around future Capstan Canada Line station",
    },
]

# Supply sensitivity by property type
# (adjustment_pct per 500 units of each tenure type within radius)
SUPPLY_SENSITIVITY = {
    PropertyType.CONDO: {"strata": -0.005, "rental": -0.003, "mixed": -0.004},
    PropertyType.TOWNHOME: {"strata": -0.003, "rental": -0.001, "mixed": -0.002},
    PropertyType.DETACHED: {"strata": -0.001, "rental": -0.001, "mixed": -0.001},
    PropertyType.DEVELOPMENT_LAND: {"strata": 0.005, "rental": 0.005, "mixed": 0.005},
}

# Completed project amenity premium (per 500 units, positive effect)
COMPLETED_AMENITY_PREMIUM = 0.002  # +0.2% per 500 units


class SupplyPipelineAdjuster:
    """Adjusts property values based on nearby construction pipeline.

    New supply affects property values in two ways:
    1. Under construction / approved: supply pressure and disruption (negative)
    2. Completed: amenity creation and neighbourhood maturation (positive)

    The magnitude depends on property type — condos are most affected by
    new strata supply, while development land actually benefits from
    nearby density signals.
    """

    def compute_supply_adjustment(
        self,
        lat: float,
        lon: float,
        property_type: PropertyType,
        radius_m: float = 1000,
    ) -> tuple[float, str]:
        """Compute the supply pipeline adjustment for a property.

        Args:
            lat: Property latitude
            lon: Property longitude
            property_type: Property type for sensitivity lookup
            radius_m: Search radius in meters (default 1km)

        Returns:
            Tuple of (adjustment_pct, explanation)
            adjustment_pct is negative for supply pressure, positive for amenity premium
        """
        nearby_projects = self._find_nearby_projects(lat, lon, radius_m)

        if not nearby_projects:
            return 0.0, "No major development projects within search radius — no supply adjustment"

        sensitivity = SUPPLY_SENSITIVITY.get(
            property_type,
            {"strata": -0.001, "rental": -0.001, "mixed": -0.001},
        )

        total_adjustment = 0.0
        project_details = []

        for project, distance in nearby_projects:
            units = project["units"]
            tenure = project["type"]
            status = project["status"]

            # Distance weighting: closer projects have more impact
            # Linear decay from 1.0 at 0m to 0.5 at radius_m
            distance_weight = max(0.5, 1.0 - 0.5 * (distance / radius_m))

            if status == "completed":
                # Completed projects provide amenity premium
                unit_blocks = units / 500.0
                project_adj = COMPLETED_AMENITY_PREMIUM * unit_blocks * distance_weight
                direction = "amenity premium"
            else:
                # Active/planned projects create supply pressure
                sens = sensitivity.get(tenure, sensitivity.get("mixed", -0.001))
                unit_blocks = units / 500.0
                project_adj = sens * unit_blocks * distance_weight
                direction = "supply pressure"

            total_adjustment += project_adj
            project_details.append(
                f"{project['name']} ({project['units']:,} {tenure} units, "
                f"{status}, {distance:.0f}m away): "
                f"{project_adj * 100:+.2f}% {direction}"
            )

        # Clamp adjustment to reasonable bounds
        total_adjustment = max(-0.05, min(0.03, total_adjustment))

        # Build explanation
        explanation_parts = [
            f"{len(nearby_projects)} project(s) within {radius_m:.0f}m",
        ]
        explanation_parts.extend(project_details)
        explanation_parts.append(f"Net supply adjustment: {total_adjustment * 100:+.2f}%")
        explanation = "; ".join(explanation_parts)

        logger.debug(
            "Supply adjustment for (%.4f, %.4f) [%s]: %.2f%% from %d projects",
            lat, lon, property_type.value, total_adjustment * 100,
            len(nearby_projects),
        )

        return round(total_adjustment, 4), explanation

    def _find_nearby_projects(
        self,
        lat: float,
        lon: float,
        radius_m: float,
    ) -> list[tuple[dict, float]]:
        """Find all major projects within the given radius.

        Args:
            lat: Property latitude
            lon: Property longitude
            radius_m: Search radius in meters

        Returns:
            List of (project_dict, distance_m) tuples sorted by distance
        """
        nearby = []
        for project in MAJOR_PROJECTS:
            dist = self._haversine_m(lat, lon, project["lat"], project["lon"])
            if dist <= radius_m:
                nearby.append((project, dist))

        nearby.sort(key=lambda x: x[1])
        return nearby

    @staticmethod
    def _haversine_m(
        lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Haversine distance between two points in meters."""
        R = 6_371_000  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)

        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        )
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
