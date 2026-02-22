"""
Land assembly premium/discount calculator.

Properties in active development corridors can command 20-40% premiums
over comparable residential sales. Key corridors:
- Broadway Plan (Broadway to UBC SkyTrain extension)
- Cambie Corridor (Cambie Street stations)
- Oakridge Municipal Town Centre
- Metrotown (Burnaby)
"""

from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

# Assembly corridors defined by bounding boxes (lat_min, lat_max, lon_min, lon_max)
# and metadata about each corridor's development characteristics.
ASSEMBLY_CORRIDORS = [
    {
        "name": "Broadway Plan",
        "lat_min": 49.26,
        "lat_max": 49.27,
        "lon_min": -123.18,
        "lon_max": -123.07,
        "base_premium": 0.15,
        "description": "Broadway to UBC SkyTrain extension — high-density rezoning",
    },
    {
        "name": "Cambie Corridor",
        "lat_min": 49.21,
        "lat_max": 49.26,
        "lon_min": -123.12,
        "lon_max": -123.10,
        "base_premium": 0.12,
        "description": "Canada Line stations — established transit corridor",
    },
    {
        "name": "Oakridge Municipal Town Centre",
        "lat_min": 49.225,
        "lat_max": 49.235,
        "lon_min": -123.12,
        "lon_max": -123.11,
        "base_premium": 0.13,
        "description": "Oakridge redevelopment area — major mixed-use node",
    },
    {
        "name": "Joyce-Collingwood",
        "lat_min": 49.237,
        "lat_max": 49.243,
        "lon_min": -123.03,
        "lon_max": -123.02,
        "base_premium": 0.10,
        "description": "Joyce-Collingwood station area plan",
    },
    {
        "name": "Metrotown",
        "lat_min": 49.224,
        "lat_max": 49.232,
        "lon_min": -123.005,
        "lon_max": -122.995,
        "base_premium": 0.14,
        "description": "Burnaby Metrotown high-density core",
    },
    {
        "name": "Brentwood-Holdom",
        "lat_min": 49.264,
        "lat_max": 49.270,
        "lon_min": -123.00,
        "lon_max": -122.98,
        "base_premium": 0.11,
        "description": "Brentwood Town Centre redevelopment",
    },
    {
        "name": "Lougheed Town Centre",
        "lat_min": 49.248,
        "lat_max": 49.254,
        "lon_min": -122.90,
        "lon_max": -122.88,
        "base_premium": 0.10,
        "description": "Lougheed and Burquitlam transit node",
    },
]

# Key SkyTrain stations for transit-oriented development proximity
# (lat, lon) — used for TOD premium calculation within 400m
SKYTRAIN_STATIONS = [
    (49.2633, -123.0694),  # Commercial-Broadway
    (49.2627, -123.1146),  # Cambie-Broadway (Millennium Line)
    (49.2625, -123.0686),  # Renfrew
    (49.2483, -123.1160),  # King Edward (Canada Line)
    (49.2294, -123.1166),  # Oakridge (Canada Line)
    (49.2125, -123.1168),  # Langara-49th (Canada Line)
    (49.2384, -123.0318),  # Joyce-Collingwood
    (49.2267, -122.9997),  # Metrotown
    (49.2660, -122.9920),  # Brentwood
    (49.2486, -122.8967),  # Lougheed
    (49.2530, -122.9180),  # Burquitlam
    (49.2050, -123.1174),  # Marine Drive (Canada Line)
    (49.2855, -123.1117),  # VCC-Clark
    (49.2839, -123.0536),  # Rupert
    (49.2627, -123.0687),  # Commercial-Broadway (Expo)
]

# Multi-family zoning codes that reduce rezoning risk
MULTI_FAMILY_ZONES = {
    "RM", "FM", "CD", "C-2", "C-3", "MC", "RM-3A", "RM-4", "RM-5",
    "RM-6", "RM-7", "RM-8", "RM-9", "RM-10", "RM-11", "RM-12",
    "FC-1", "FC-2", "FCCDD",
}


class AssemblyPremiumCalculator:
    """Calculates land assembly premiums for properties in development corridors.

    When multiple adjacent lots are assembled for higher-density
    redevelopment, individual properties can command significant premiums
    over their standalone residential value. This calculator estimates
    the assembly premium based on corridor location, lot characteristics,
    and transit proximity.
    """

    def compute_assembly_premium(
        self,
        lat: float,
        lon: float,
        zoning_code: Optional[str] = None,
        neighbourhood_code: Optional[str] = None,
        lot_frontage_ft: Optional[float] = None,
        is_corner: bool = False,
    ) -> tuple[float, str]:
        """Compute the assembly premium for a property.

        Args:
            lat: Property latitude
            lon: Property longitude
            zoning_code: Current zoning designation
            neighbourhood_code: GVR sub-area code
            lot_frontage_ft: Lot frontage in feet (None if unknown)
            is_corner: Whether the property is a corner lot

        Returns:
            Tuple of (premium_pct, explanation)
            premium_pct is 0.0 to 0.40 (i.e., 0% to 40%)
        """
        # Check if property is in any assembly corridor
        corridor = self._find_corridor(lat, lon)

        if corridor is None:
            return 0.0, "Property is not in a known assembly corridor — no premium"

        premium = corridor["base_premium"]
        factors = [f"In {corridor['name']} corridor: base premium {corridor['base_premium']*100:.0f}%"]

        # Standard lot frontage bonus (33ft is the classic Vancouver lot)
        if lot_frontage_ft is not None and 30 <= lot_frontage_ft <= 40:
            premium += 0.05
            factors.append(
                f"Standard lot frontage ({lot_frontage_ft:.0f}ft) adds 5% "
                f"(ideal assemblable width)"
            )

        # Corner lot bonus — extra access for development site
        if is_corner:
            premium += 0.05
            factors.append("Corner lot adds 5% (dual street access)")

        # Transit-oriented development proximity bonus
        tod_dist = self._distance_to_nearest_station(lat, lon)
        if tod_dist is not None and tod_dist <= 400:
            tod_premium = 0.10 if tod_dist <= 200 else 0.05
            premium += tod_premium
            factors.append(
                f"Within {tod_dist:.0f}m of SkyTrain station: "
                f"+{tod_premium*100:.0f}% TOD premium"
            )

        # Multi-family zoning bonus (less rezoning risk)
        if zoning_code is not None:
            # Check prefix match against multi-family zones
            zone_prefix = zoning_code.split("-")[0] if "-" in zoning_code else zoning_code
            if zoning_code in MULTI_FAMILY_ZONES or zone_prefix in MULTI_FAMILY_ZONES:
                premium += 0.05
                factors.append(
                    f"Current zoning ({zoning_code}) allows multi-family: "
                    f"+5% (reduced rezoning risk)"
                )

        # Cap total premium at 40%
        if premium > 0.40:
            premium = 0.40
            factors.append("Total premium capped at 40%")

        explanation = "; ".join(factors)

        logger.debug(
            "Assembly premium for (%.4f, %.4f): %.1f%% — %s",
            lat, lon, premium * 100, corridor["name"],
        )

        return round(premium, 4), explanation

    def _find_corridor(
        self, lat: float, lon: float
    ) -> Optional[dict]:
        """Find the assembly corridor containing the given coordinates.

        Args:
            lat: Property latitude
            lon: Property longitude

        Returns:
            Corridor dict if found, None otherwise
        """
        for corridor in ASSEMBLY_CORRIDORS:
            if (
                corridor["lat_min"] <= lat <= corridor["lat_max"]
                and corridor["lon_min"] <= lon <= corridor["lon_max"]
            ):
                return corridor
        return None

    def _distance_to_nearest_station(
        self, lat: float, lon: float
    ) -> Optional[float]:
        """Compute distance to the nearest SkyTrain station in meters.

        Uses the Haversine formula for accurate distance at Vancouver's latitude.

        Args:
            lat: Property latitude
            lon: Property longitude

        Returns:
            Distance in meters to the nearest station, or None if no stations
        """
        if not SKYTRAIN_STATIONS:
            return None

        min_dist = float("inf")
        for station_lat, station_lon in SKYTRAIN_STATIONS:
            dist = self._haversine_m(lat, lon, station_lat, station_lon)
            if dist < min_dist:
                min_dist = dist

        return round(min_dist, 1)

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
