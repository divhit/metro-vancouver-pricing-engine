"""
Walk Score API ingestion.

Endpoints:
- Walk Score (0-100)
- Transit Score (0-100)
- Bike Score (0-100)

Free tier: 5,000 calls/day.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class WalkScoreResult:
    """Walk Score API response for a single location."""

    latitude: float
    longitude: float
    walk_score: Optional[int] = None
    walk_description: Optional[str] = None
    transit_score: Optional[int] = None
    transit_description: Optional[str] = None
    bike_score: Optional[int] = None
    bike_description: Optional[str] = None


class WalkScoreClient:
    """Client for the Walk Score API."""

    BASE_URL = "https://api.walkscore.com/score"
    TRANSIT_URL = "https://transit.walkscore.com/transit/score/"
    RATE_LIMIT_PER_DAY = 5_000

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._call_count = 0

    def get_scores(
        self, latitude: float, longitude: float, address: str = ""
    ) -> WalkScoreResult:
        """Get Walk Score, Transit Score, and Bike Score for a location.

        Args:
            latitude: Property latitude
            longitude: Property longitude
            address: Optional address string for better accuracy

        Returns:
            WalkScoreResult with all available scores
        """
        if self._call_count >= self.RATE_LIMIT_PER_DAY:
            logger.warning("Daily rate limit reached (5,000 calls)")
            return WalkScoreResult(latitude=latitude, longitude=longitude)

        params = {
            "format": "json",
            "lat": latitude,
            "lon": longitude,
            "transit": 1,
            "bike": 1,
            "wsapikey": self.api_key,
        }
        if address:
            params["address"] = address

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            self._call_count += 1

            result = WalkScoreResult(
                latitude=latitude,
                longitude=longitude,
                walk_score=data.get("walkscore"),
                walk_description=data.get("description"),
            )

            # Transit score is in a nested object
            transit = data.get("transit", {})
            if transit:
                result.transit_score = transit.get("score")
                result.transit_description = transit.get("description")

            # Bike score
            bike = data.get("bike", {})
            if bike:
                result.bike_score = bike.get("score")
                result.bike_description = bike.get("description")

            return result

        except requests.RequestException as e:
            logger.error(f"Walk Score API error for ({latitude}, {longitude}): {e}")
            return WalkScoreResult(latitude=latitude, longitude=longitude)

    def batch_score(
        self,
        locations: list[tuple[float, float, str]],
        delay: float = 0.2,
    ) -> list[WalkScoreResult]:
        """Score multiple locations with rate limiting.

        Args:
            locations: List of (latitude, longitude, address) tuples
            delay: Seconds between requests to respect rate limits

        Returns:
            List of WalkScoreResult objects
        """
        results = []
        for lat, lon, addr in locations:
            result = self.get_scores(lat, lon, addr)
            results.append(result)
            time.sleep(delay)
        return results
