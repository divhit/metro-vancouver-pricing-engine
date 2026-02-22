"""
TransLink GTFS static data ingestion.

Source: TransLink Open API / GTFS static feed
URL: https://www.translink.ca/about-us/doing-business-with-translink/app-developer-resources

TransLink publishes a GTFS (General Transit Feed Specification) static
data package as a ZIP of CSV files. This includes:
- stops.txt: 8,000+ bus stops + SkyTrain stations with lat/lon
- routes.txt: All bus, SkyTrain, SeaBus routes
- trips.txt: Individual trip schedules
- stop_times.txt: Arrival/departure times at each stop
- calendar.txt / calendar_dates.txt: Service schedules

This module parses the GTFS data and computes transit accessibility
metrics for any location, including:
- Number of transit stops within a given radius
- Route diversity (unique routes serving nearby stops)
- Frequency scores based on stop_times
- SkyTrain station proximity
"""

import logging
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

logger = logging.getLogger(__name__)

# TransLink GTFS static feed URL
GTFS_URL = "https://gtfs.translink.ca/static/latest"

# GTFS file names within the ZIP
GTFS_FILES = {
    "stops": "stops.txt",
    "routes": "routes.txt",
    "trips": "trips.txt",
    "stop_times": "stop_times.txt",
    "calendar": "calendar.txt",
    "calendar_dates": "calendar_dates.txt",
    "shapes": "shapes.txt",
}

# GTFS route types
ROUTE_TYPES = {
    0: "streetcar",
    1: "subway",  # SkyTrain
    2: "rail",    # West Coast Express
    3: "bus",
    4: "ferry",   # SeaBus
}

# SkyTrain line identifiers (TransLink route_short_name patterns)
SKYTRAIN_LINES = {"expo", "millennium", "canada", "evergreen"}

# Default radius for transit accessibility (meters)
DEFAULT_RADIUS_M = 400  # ~5 minute walk
SKYTRAIN_RADIUS_M = 800  # ~10 minute walk for rapid transit


class TransLinkGTFSClient:
    """Parse TransLink GTFS static data and compute transit accessibility.

    Downloads the GTFS ZIP, parses the CSV files, and provides methods
    to compute transit accessibility metrics for any lat/lon point.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the GTFS client.

        Args:
            cache_dir: Directory to cache the GTFS download.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "gtfs_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._stops: Optional[gpd.GeoDataFrame] = None
        self._routes: Optional[pd.DataFrame] = None
        self._trips: Optional[pd.DataFrame] = None
        self._stop_times: Optional[pd.DataFrame] = None
        self._calendar: Optional[pd.DataFrame] = None
        self._stop_route_map: Optional[pd.DataFrame] = None

    def download_gtfs(self, url: Optional[str] = None) -> Path:
        """Download the GTFS ZIP file.

        Args:
            url: Custom URL. Uses TransLink default if None.

        Returns:
            Path to the downloaded ZIP file.
        """
        import requests

        url = url or GTFS_URL
        zip_path = self.cache_dir / "translink_gtfs.zip"

        logger.info(f"Downloading GTFS data from {url}")
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            zip_path.write_bytes(response.content)
            logger.info(f"Downloaded GTFS ZIP: {len(response.content) / 1e6:.1f} MB")
            return zip_path
        except requests.RequestException as e:
            logger.error(f"Failed to download GTFS data: {e}")
            raise

    def load_gtfs(self, zip_path: Optional[str] = None) -> None:
        """Load GTFS data from a ZIP file into memory.

        Parses stops, routes, trips, stop_times, and calendar files.

        Args:
            zip_path: Path to GTFS ZIP. Downloads if None.
        """
        if zip_path is None:
            zip_path = str(self.download_gtfs())

        logger.info(f"Loading GTFS data from {zip_path}")

        with zipfile.ZipFile(zip_path, "r") as zf:
            available = zf.namelist()

            # Load stops
            if GTFS_FILES["stops"] in available:
                with zf.open(GTFS_FILES["stops"]) as f:
                    stops_df = pd.read_csv(f)
                self._stops = gpd.GeoDataFrame(
                    stops_df,
                    geometry=gpd.points_from_xy(stops_df["stop_lon"], stops_df["stop_lat"]),
                    crs="EPSG:4326",
                )
                logger.info(f"Loaded {len(self._stops):,} stops")

            # Load routes
            if GTFS_FILES["routes"] in available:
                with zf.open(GTFS_FILES["routes"]) as f:
                    self._routes = pd.read_csv(f)
                logger.info(f"Loaded {len(self._routes):,} routes")

            # Load trips
            if GTFS_FILES["trips"] in available:
                with zf.open(GTFS_FILES["trips"]) as f:
                    self._trips = pd.read_csv(f)
                logger.info(f"Loaded {len(self._trips):,} trips")

            # Load stop_times (large file -- read only needed columns)
            if GTFS_FILES["stop_times"] in available:
                with zf.open(GTFS_FILES["stop_times"]) as f:
                    self._stop_times = pd.read_csv(
                        f,
                        usecols=["trip_id", "arrival_time", "departure_time", "stop_id"],
                    )
                logger.info(f"Loaded {len(self._stop_times):,} stop_times")

            # Load calendar
            if GTFS_FILES["calendar"] in available:
                with zf.open(GTFS_FILES["calendar"]) as f:
                    self._calendar = pd.read_csv(f)

        # Build stop-to-route mapping
        self._build_stop_route_map()

    def _build_stop_route_map(self) -> None:
        """Build a mapping from stop_id to set of route_ids serving it."""
        if self._trips is None or self._stop_times is None or self._routes is None:
            return

        # Join stop_times -> trips -> routes
        st_trips = self._stop_times[["stop_id", "trip_id"]].drop_duplicates()
        trip_routes = self._trips[["trip_id", "route_id"]].drop_duplicates()
        merged = st_trips.merge(trip_routes, on="trip_id")

        # Add route metadata
        route_info = self._routes[["route_id", "route_short_name", "route_type"]].drop_duplicates()
        merged = merged.merge(route_info, on="route_id")

        self._stop_route_map = merged[
            ["stop_id", "route_id", "route_short_name", "route_type"]
        ].drop_duplicates()

        logger.info(
            f"Built stop-route map: {len(self._stop_route_map):,} stop-route pairs"
        )

    def get_stops(self) -> gpd.GeoDataFrame:
        """Return all transit stops as a GeoDataFrame.

        Returns:
            GeoDataFrame with stop_id, stop_name, geometry, and route info.
        """
        if self._stops is None:
            raise RuntimeError("GTFS data not loaded. Call load_gtfs() first.")
        return self._stops.copy()

    def get_skytrain_stations(self) -> gpd.GeoDataFrame:
        """Return only SkyTrain stations.

        Returns:
            GeoDataFrame of SkyTrain station stops.
        """
        if self._stops is None or self._stop_route_map is None:
            raise RuntimeError("GTFS data not loaded. Call load_gtfs() first.")

        # Filter to route_type 1 (subway/metro = SkyTrain)
        skytrain_stops = self._stop_route_map[
            self._stop_route_map["route_type"] == 1
        ]["stop_id"].unique()

        stations = self._stops[self._stops["stop_id"].isin(skytrain_stops)].copy()
        logger.info(f"Found {len(stations)} SkyTrain stations")
        return stations

    def stops_within_radius(
        self,
        lat: float,
        lon: float,
        radius_m: float = DEFAULT_RADIUS_M,
    ) -> gpd.GeoDataFrame:
        """Find transit stops within a radius of a point.

        Uses a projected CRS (UTM 10N for Vancouver) for accurate distance.

        Args:
            lat: Latitude of the point.
            lon: Longitude of the point.
            radius_m: Search radius in meters.

        Returns:
            GeoDataFrame of stops within the radius, with distance_m column.
        """
        if self._stops is None:
            raise RuntimeError("GTFS data not loaded. Call load_gtfs() first.")

        point = gpd.GeoDataFrame(
            geometry=[Point(lon, lat)], crs="EPSG:4326"
        )

        # Project to UTM 10N for Vancouver area
        stops_proj = self._stops.to_crs(epsg=32610)
        point_proj = point.to_crs(epsg=32610)

        # Compute distances
        stops_proj["distance_m"] = stops_proj.geometry.distance(
            point_proj.geometry.iloc[0]
        )

        nearby = stops_proj[stops_proj["distance_m"] <= radius_m].copy()
        nearby = nearby.sort_values("distance_m")

        # Convert back to WGS84
        nearby = nearby.to_crs(epsg=4326)
        return nearby

    def compute_transit_score(
        self,
        lat: float,
        lon: float,
        bus_radius_m: float = DEFAULT_RADIUS_M,
        skytrain_radius_m: float = SKYTRAIN_RADIUS_M,
    ) -> dict:
        """Compute transit accessibility metrics for a location.

        Returns a dict with:
        - stops_400m: Number of transit stops within 400m
        - stops_800m: Number of transit stops within 800m
        - unique_routes_400m: Number of unique routes within 400m
        - has_skytrain_800m: Whether a SkyTrain station is within 800m
        - nearest_skytrain_m: Distance to nearest SkyTrain station
        - route_types: Set of route types serving nearby stops
        - frequency_score: Estimated daily departures from nearby stops

        Args:
            lat: Latitude.
            lon: Longitude.
            bus_radius_m: Radius for bus stop search.
            skytrain_radius_m: Radius for SkyTrain station search.

        Returns:
            Dict of transit accessibility metrics.
        """
        if self._stops is None:
            raise RuntimeError("GTFS data not loaded. Call load_gtfs() first.")

        # Get nearby stops at both radii
        nearby_400 = self.stops_within_radius(lat, lon, bus_radius_m)
        nearby_800 = self.stops_within_radius(lat, lon, skytrain_radius_m)

        # Get SkyTrain stations
        skytrain = self.get_skytrain_stations()
        skytrain_proj = skytrain.to_crs(epsg=32610)
        point_proj = gpd.GeoDataFrame(
            geometry=[Point(lon, lat)], crs="EPSG:4326"
        ).to_crs(epsg=32610)

        if not skytrain_proj.empty:
            skytrain_proj["distance_m"] = skytrain_proj.geometry.distance(
                point_proj.geometry.iloc[0]
            )
            nearest_skytrain = skytrain_proj["distance_m"].min()
        else:
            nearest_skytrain = None

        # Count unique routes within 400m
        unique_routes_400 = set()
        route_types = set()
        if self._stop_route_map is not None and not nearby_400.empty:
            stop_ids = nearby_400["stop_id"].tolist()
            serving = self._stop_route_map[
                self._stop_route_map["stop_id"].isin(stop_ids)
            ]
            unique_routes_400 = set(serving["route_id"].unique())
            route_types = {
                ROUTE_TYPES.get(rt, "unknown")
                for rt in serving["route_type"].unique()
            }

        # Estimate frequency score (daily departures from nearby stops)
        frequency_score = 0
        if self._stop_times is not None and not nearby_400.empty:
            stop_ids = nearby_400["stop_id"].tolist()
            departures = self._stop_times[
                self._stop_times["stop_id"].isin(stop_ids)
            ]
            # Rough estimate: unique departure times
            frequency_score = len(departures)

        return {
            "stops_400m": len(nearby_400),
            "stops_800m": len(nearby_800),
            "unique_routes_400m": len(unique_routes_400),
            "has_skytrain_800m": (
                nearest_skytrain is not None and nearest_skytrain <= skytrain_radius_m
            ),
            "nearest_skytrain_m": round(nearest_skytrain, 1) if nearest_skytrain else None,
            "route_types": list(route_types),
            "frequency_score": frequency_score,
        }

    def batch_transit_scores(
        self,
        locations: list[tuple[float, float]],
    ) -> pd.DataFrame:
        """Compute transit scores for multiple locations.

        Args:
            locations: List of (latitude, longitude) tuples.

        Returns:
            DataFrame with one row per location and transit metric columns.
        """
        results = []
        for i, (lat, lon) in enumerate(locations):
            if i > 0 and i % 100 == 0:
                logger.info(f"Processing transit score {i}/{len(locations)}")
            score = self.compute_transit_score(lat, lon)
            score["latitude"] = lat
            score["longitude"] = lon
            results.append(score)

        df = pd.DataFrame(results)
        return df
