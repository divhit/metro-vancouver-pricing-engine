"""
Environmental data ingestion for Metro Vancouver.

Unified client for free environmental datasets relevant to property
valuation and risk assessment:

1. Agricultural Land Reserve (ALR) boundaries — DataBC WFS
   Properties within or adjacent to ALR face development restrictions.

2. Floodplain maps — DataBC WFS
   Properties in designated floodplains face insurance and mortgage
   implications. Includes coastal and river floodplain designations.

3. Contaminated sites — DataBC / Site Registry
   Properties near contaminated sites may face value impacts and
   remediation requirements.

4. Seismic microzonation — NRCan / Metro Vancouver
   Soil amplification factors affecting earthquake risk by area.

All data is returned as GeoDataFrames for spatial joins with property
locations.

DataBC WFS endpoint:
  https://openmaps.gov.bc.ca/geo/pub/wfs
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import pandas as pd
import geopandas as gpd
import requests

logger = logging.getLogger(__name__)

# DataBC WFS base endpoint
DATABC_WFS = "https://openmaps.gov.bc.ca/geo/pub/wfs"

# Layer names in the DataBC WFS
WFS_LAYERS = {
    "alr_polygons": {
        "layer": "WHSE_LEGAL_ADMIN_BOUNDARIES.OATS_ALR_POLYS",
        "description": "Agricultural Land Reserve boundaries",
    },
    "alr_lines": {
        "layer": "WHSE_LEGAL_ADMIN_BOUNDARIES.OATS_ALR_BOUNDARY_LINES_SVW",
        "description": "ALR boundary lines",
    },
    "floodplain": {
        "layer": "WHSE_WATER_MANAGEMENT.WLS_BC_FLOODPLAIN_AREA_SP",
        "description": "BC designated floodplain areas",
    },
    "contaminated_sites": {
        "layer": "WHSE_WASTE.SITE_ENV_RMDTN_SITES_SVW",
        "description": "Contaminated sites from BC Site Registry",
    },
    "drinking_water_protection": {
        "layer": "WHSE_WATER_MANAGEMENT.WLS_COMMUNITY_WS_PB_SVW",
        "description": "Community watershed protection boundaries",
    },
    "old_growth": {
        "layer": "WHSE_FOREST_VEGETATION.OGSR_TAP_PRIORITY_DEF_AREA_SP",
        "description": "Old growth management areas",
    },
    "parks_protected": {
        "layer": "WHSE_TANTALIS.TA_PARK_ECORES_PA_SVW",
        "description": "BC Parks and protected areas",
    },
    "wildfire_risk": {
        "layer": "WHSE_LAND_AND_NATURAL_RESOURCE.PROT_DANGER_RATING_SP",
        "description": "Fire danger rating polygons",
    },
}

# Metro Vancouver bounding box (WGS84)
METRO_VANCOUVER_BBOX = {
    "west": -123.35,
    "south": 49.00,
    "east": -122.40,
    "north": 49.45,
}


class EnvironmentalDataClient:
    """Unified client for free environmental spatial data.

    Downloads environmental layers from DataBC WFS and other open sources.
    Returns GeoDataFrames suitable for spatial joins with property locations.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the environmental data client.

        Args:
            cache_dir: Directory to cache downloaded data.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "env_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "MetroVancouverPricingEngine/1.0"

    # ----------------------------------------------------------------
    # DataBC WFS client
    # ----------------------------------------------------------------

    def _fetch_wfs_layer(
        self,
        layer_name: str,
        bbox: Optional[dict] = None,
        max_features: int = 50_000,
        cql_filter: Optional[str] = None,
    ) -> gpd.GeoDataFrame:
        """Fetch a layer from the DataBC WFS service.

        Args:
            layer_name: Full WFS layer name.
            bbox: Bounding box dict with west, south, east, north keys.
                  Uses Metro Vancouver bbox if None.
            max_features: Maximum number of features to return.
            cql_filter: Optional CQL filter expression.

        Returns:
            GeoDataFrame of the requested layer.
        """
        if bbox is None:
            bbox = METRO_VANCOUVER_BBOX

        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeName": layer_name,
            "outputFormat": "json",
            "count": max_features,
            "srsName": "EPSG:4326",
            "BBOX": f"{bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']},EPSG:4326",
        }

        if cql_filter:
            params["CQL_FILTER"] = cql_filter

        url = f"{DATABC_WFS}?{urlencode(params)}"
        logger.info(f"Fetching WFS layer: {layer_name}")

        try:
            response = self.session.get(url, timeout=120)
            response.raise_for_status()

            gdf = gpd.read_file(response.text, driver="GeoJSON")
            logger.info(f"Fetched {len(gdf):,} features from {layer_name}")

            # Ensure CRS is WGS84
            if gdf.crs is None:
                gdf = gdf.set_crs(epsg=4326)
            elif gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)

            return gdf

        except requests.RequestException as e:
            logger.error(f"WFS request failed for {layer_name}: {e}")
            return gpd.GeoDataFrame()
        except Exception as e:
            logger.error(f"Failed to parse WFS response for {layer_name}: {e}")
            return gpd.GeoDataFrame()

    # ----------------------------------------------------------------
    # Individual layer fetchers
    # ----------------------------------------------------------------

    def fetch_alr_boundaries(self) -> gpd.GeoDataFrame:
        """Fetch Agricultural Land Reserve polygon boundaries.

        The ALR is a provincial land use zone that restricts subdivision
        and non-farm development. Properties within or adjacent to ALR
        face significant development constraints.

        Returns:
            GeoDataFrame of ALR polygons in Metro Vancouver.
        """
        return self._fetch_wfs_layer(WFS_LAYERS["alr_polygons"]["layer"])

    def fetch_floodplain(self) -> gpd.GeoDataFrame:
        """Fetch designated floodplain area polygons.

        Floodplain designation affects:
        - Mortgage availability and insurance costs
        - Building elevation requirements
        - Property value (negative premium)

        Returns:
            GeoDataFrame of floodplain polygons.
        """
        return self._fetch_wfs_layer(WFS_LAYERS["floodplain"]["layer"])

    def fetch_contaminated_sites(self) -> gpd.GeoDataFrame:
        """Fetch contaminated sites from BC Site Registry.

        Contaminated sites affect nearby property values and may require
        remediation before development.

        Returns:
            GeoDataFrame of contaminated site locations.
        """
        return self._fetch_wfs_layer(WFS_LAYERS["contaminated_sites"]["layer"])

    def fetch_parks_protected_areas(self) -> gpd.GeoDataFrame:
        """Fetch BC Parks and ecological reserves.

        Proximity to parks and protected areas is generally a positive
        amenity factor for property values.

        Returns:
            GeoDataFrame of park and protected area polygons.
        """
        return self._fetch_wfs_layer(WFS_LAYERS["parks_protected"]["layer"])

    def fetch_wildfire_risk(self) -> gpd.GeoDataFrame:
        """Fetch wildfire danger rating polygons.

        Relevant for properties in the WUI (wildland-urban interface),
        particularly in North Shore and eastern Metro Vancouver.

        Returns:
            GeoDataFrame of wildfire risk polygons.
        """
        return self._fetch_wfs_layer(WFS_LAYERS["wildfire_risk"]["layer"])

    def fetch_drinking_water_protection(self) -> gpd.GeoDataFrame:
        """Fetch community watershed protection boundaries.

        Development restrictions apply within watershed protection areas.

        Returns:
            GeoDataFrame of watershed protection polygons.
        """
        return self._fetch_wfs_layer(WFS_LAYERS["drinking_water_protection"]["layer"])

    # ----------------------------------------------------------------
    # Spatial analysis utilities
    # ----------------------------------------------------------------

    def check_alr_status(
        self,
        lat: float,
        lon: float,
        alr_gdf: Optional[gpd.GeoDataFrame] = None,
    ) -> dict:
        """Check if a property is within or near the ALR.

        Args:
            lat: Property latitude.
            lon: Property longitude.
            alr_gdf: Pre-fetched ALR GeoDataFrame. Fetches if None.

        Returns:
            Dict with in_alr (bool) and distance_to_alr_m (float).
        """
        if alr_gdf is None:
            alr_gdf = self.fetch_alr_boundaries()

        if alr_gdf.empty:
            return {"in_alr": None, "distance_to_alr_m": None}

        from shapely.geometry import Point

        point = gpd.GeoDataFrame(
            geometry=[Point(lon, lat)], crs="EPSG:4326"
        )

        # Check containment
        in_alr = alr_gdf.contains(point.geometry.iloc[0]).any()

        # Compute distance if not in ALR
        distance = None
        if not in_alr:
            alr_proj = alr_gdf.to_crs(epsg=32610)
            point_proj = point.to_crs(epsg=32610)
            distances = alr_proj.geometry.distance(point_proj.geometry.iloc[0])
            distance = round(distances.min(), 1)

        return {
            "in_alr": bool(in_alr),
            "distance_to_alr_m": distance if not in_alr else 0.0,
        }

    def check_floodplain_status(
        self,
        lat: float,
        lon: float,
        floodplain_gdf: Optional[gpd.GeoDataFrame] = None,
    ) -> dict:
        """Check if a property is within a designated floodplain.

        Args:
            lat: Property latitude.
            lon: Property longitude.
            floodplain_gdf: Pre-fetched floodplain GeoDataFrame.

        Returns:
            Dict with in_floodplain (bool) and distance_to_floodplain_m.
        """
        if floodplain_gdf is None:
            floodplain_gdf = self.fetch_floodplain()

        if floodplain_gdf.empty:
            return {"in_floodplain": None, "distance_to_floodplain_m": None}

        from shapely.geometry import Point

        point = gpd.GeoDataFrame(
            geometry=[Point(lon, lat)], crs="EPSG:4326"
        )

        in_flood = floodplain_gdf.contains(point.geometry.iloc[0]).any()

        distance = None
        if not in_flood:
            flood_proj = floodplain_gdf.to_crs(epsg=32610)
            point_proj = point.to_crs(epsg=32610)
            distances = flood_proj.geometry.distance(point_proj.geometry.iloc[0])
            distance = round(distances.min(), 1)

        return {
            "in_floodplain": bool(in_flood),
            "distance_to_floodplain_m": distance if not in_flood else 0.0,
        }

    def count_contaminated_sites_nearby(
        self,
        lat: float,
        lon: float,
        radius_m: float = 500,
        contam_gdf: Optional[gpd.GeoDataFrame] = None,
    ) -> dict:
        """Count contaminated sites within a radius.

        Args:
            lat: Property latitude.
            lon: Property longitude.
            radius_m: Search radius in meters.
            contam_gdf: Pre-fetched contaminated sites GeoDataFrame.

        Returns:
            Dict with count, nearest_m, and site details.
        """
        if contam_gdf is None:
            contam_gdf = self.fetch_contaminated_sites()

        if contam_gdf.empty:
            return {"contaminated_sites_nearby": 0, "nearest_contaminated_m": None}

        from shapely.geometry import Point

        point = gpd.GeoDataFrame(
            geometry=[Point(lon, lat)], crs="EPSG:4326"
        )

        contam_proj = contam_gdf.to_crs(epsg=32610)
        point_proj = point.to_crs(epsg=32610)

        contam_proj["distance_m"] = contam_proj.geometry.distance(
            point_proj.geometry.iloc[0]
        )

        nearby = contam_proj[contam_proj["distance_m"] <= radius_m]

        return {
            "contaminated_sites_nearby": len(nearby),
            "nearest_contaminated_m": (
                round(contam_proj["distance_m"].min(), 1)
                if not contam_proj.empty else None
            ),
        }

    def get_environmental_risk_profile(
        self,
        lat: float,
        lon: float,
        alr_gdf: Optional[gpd.GeoDataFrame] = None,
        floodplain_gdf: Optional[gpd.GeoDataFrame] = None,
        contam_gdf: Optional[gpd.GeoDataFrame] = None,
    ) -> dict:
        """Get a complete environmental risk profile for a property.

        Combines ALR, floodplain, and contamination checks into a
        single assessment.

        Args:
            lat: Property latitude.
            lon: Property longitude.
            alr_gdf: Pre-fetched ALR boundaries.
            floodplain_gdf: Pre-fetched floodplain boundaries.
            contam_gdf: Pre-fetched contaminated sites.

        Returns:
            Dict with all environmental risk indicators.
        """
        profile = {}
        profile.update(self.check_alr_status(lat, lon, alr_gdf))
        profile.update(self.check_floodplain_status(lat, lon, floodplain_gdf))
        profile.update(self.count_contaminated_sites_nearby(lat, lon, contam_gdf=contam_gdf))

        # Composite risk flag
        risks = []
        if profile.get("in_alr"):
            risks.append("alr")
        if profile.get("in_floodplain"):
            risks.append("floodplain")
        if profile.get("contaminated_sites_nearby", 0) > 0:
            risks.append("contamination")

        profile["environmental_risks"] = risks
        profile["risk_count"] = len(risks)

        return profile

    # ----------------------------------------------------------------
    # Batch operations
    # ----------------------------------------------------------------

    def fetch_all_layers(self) -> dict[str, gpd.GeoDataFrame]:
        """Fetch all environmental layers for Metro Vancouver.

        Returns:
            Dict mapping layer name to GeoDataFrame.
        """
        results = {}
        fetchers = {
            "alr": self.fetch_alr_boundaries,
            "floodplain": self.fetch_floodplain,
            "contaminated_sites": self.fetch_contaminated_sites,
            "parks": self.fetch_parks_protected_areas,
            "wildfire_risk": self.fetch_wildfire_risk,
        }

        for name, fetcher in fetchers.items():
            try:
                results[name] = fetcher()
            except Exception as e:
                logger.error(f"Failed to fetch {name}: {e}")
                results[name] = gpd.GeoDataFrame()

        return results
