"""
Inside Airbnb Vancouver data ingestion.

Source: Inside Airbnb (http://insideairbnb.com/get-the-data/)
License: Creative Commons CC0 1.0

Inside Airbnb scrapes public Airbnb listings and publishes periodic
snapshots. The Vancouver dataset contains ~6,000+ active listings with
106 attributes per listing, including:

- Location (latitude, longitude, neighbourhood)
- Pricing (price, cleaning_fee, weekly/monthly pricing)
- Availability (calendar, minimum/maximum nights)
- Reviews (number, scores, dates)
- Host info (superhost, response time, listings count)
- Property characteristics (room_type, property_type, bedrooms, etc.)

This data enables computation of short-term rental (STR) density metrics,
which correlate with property values in high-tourism neighbourhoods like
downtown, Kitsilano, and Mount Pleasant.
"""

import gzip
import logging
import tempfile
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import Point

logger = logging.getLogger(__name__)

# Inside Airbnb data URLs for Vancouver
# These follow a consistent pattern: /data/{city}/{date}/{filename}
INSIDE_AIRBNB_BASE = "http://insideairbnb.com/get-the-data/"
VANCOUVER_DATA_BASE = "http://data.insideairbnb.com/canada/bc/vancouver"

# File types available for each snapshot
DATA_FILES = {
    "listings": "listings.csv.gz",           # Full listing data (106 columns)
    "listings_summary": "listings.csv",      # Summary listing data
    "calendar": "calendar.csv.gz",           # 365-day availability calendar
    "reviews": "reviews.csv.gz",             # All reviews
    "reviews_summary": "reviews.csv",        # Review summary
    "neighbourhoods": "neighbourhoods.csv",  # Neighbourhood list
    "neighbourhoods_geo": "neighbourhoods.geojson",  # Neighbourhood boundaries
}

# Key columns for real estate analysis
LISTING_COLUMNS = [
    "id",
    "name",
    "host_id",
    "host_is_superhost",
    "neighbourhood_cleansed",
    "neighbourhood_group_cleansed",
    "latitude",
    "longitude",
    "property_type",
    "room_type",
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms_text",
    "price",
    "minimum_nights",
    "maximum_nights",
    "availability_30",
    "availability_60",
    "availability_90",
    "availability_365",
    "number_of_reviews",
    "number_of_reviews_ltm",  # Last twelve months
    "review_scores_rating",
    "review_scores_cleanliness",
    "review_scores_location",
    "review_scores_value",
    "instant_bookable",
    "calculated_host_listings_count",
    "reviews_per_month",
    "license",
    "last_scraped",
]


class InsideAirbnbClient:
    """Client for Inside Airbnb Vancouver data.

    Downloads and parses listing snapshots to compute short-term rental
    density metrics for use in property valuation models.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the Inside Airbnb client.

        Args:
            cache_dir: Directory to cache downloaded files.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "airbnb_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "MetroVancouverPricingEngine/1.0"

    def _build_data_url(self, snapshot_date: str, filename: str) -> str:
        """Build the URL for a specific data file.

        Args:
            snapshot_date: Date string in YYYY-MM-DD format.
            filename: One of the DATA_FILES values.

        Returns:
            Full URL to the data file.
        """
        return f"{VANCOUVER_DATA_BASE}/{snapshot_date}/data/{filename}"

    def download_listings(
        self, snapshot_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Download the full listings dataset (106 columns).

        Args:
            snapshot_date: Date string (YYYY-MM-DD). Uses a recent known
                          date if None. Check insideairbnb.com for latest.

        Returns:
            DataFrame with all listing attributes.
        """
        if snapshot_date is None:
            # Try recent known snapshot dates (updated ~quarterly)
            # These need to be updated periodically
            candidates = self._get_recent_dates()
        else:
            candidates = [snapshot_date]

        for date_str in candidates:
            url = self._build_data_url(date_str, DATA_FILES["listings"])
            logger.info(f"Trying listings download for {date_str}: {url}")

            try:
                response = self.session.get(url, timeout=120)
                if response.status_code == 200:
                    # Decompress gzip and parse CSV
                    content = gzip.decompress(response.content)
                    df = pd.read_csv(BytesIO(content), low_memory=False)
                    logger.info(
                        f"Downloaded {len(df):,} listings from {date_str} "
                        f"({len(df.columns)} columns)"
                    )
                    return df
                else:
                    logger.debug(f"No data for {date_str} (HTTP {response.status_code})")
            except Exception as e:
                logger.debug(f"Failed for {date_str}: {e}")
                continue

        logger.error("Could not download listings for any known snapshot date")
        return pd.DataFrame()

    def _get_recent_dates(self) -> list[str]:
        """Generate candidate snapshot dates to try.

        Inside Airbnb updates roughly quarterly. We generate dates
        for the last few quarters.

        Returns:
            List of date strings to try.
        """
        now = datetime.now()
        dates = []
        # Try recent months (Inside Airbnb publishes mid-month)
        for months_back in range(0, 18):
            year = now.year
            month = now.month - months_back
            while month <= 0:
                month += 12
                year -= 1
            # Try common days
            for day in [15, 1, 10, 20]:
                dates.append(f"{year}-{month:02d}-{day:02d}")
        return dates

    def download_neighbourhoods_geo(
        self, snapshot_date: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """Download neighbourhood boundary GeoJSON.

        Args:
            snapshot_date: Snapshot date. Tries recent dates if None.

        Returns:
            GeoDataFrame of neighbourhood polygons.
        """
        candidates = [snapshot_date] if snapshot_date else self._get_recent_dates()

        for date_str in candidates:
            url = self._build_data_url(date_str, DATA_FILES["neighbourhoods_geo"])
            try:
                response = self.session.get(url, timeout=60)
                if response.status_code == 200:
                    gdf = gpd.read_file(StringIO(response.text))
                    logger.info(f"Downloaded {len(gdf)} neighbourhood boundaries")
                    return gdf
            except Exception:
                continue

        logger.error("Could not download neighbourhood boundaries")
        return gpd.GeoDataFrame()

    def load_from_file(self, file_path: str) -> pd.DataFrame:
        """Load listings from a local file (CSV or CSV.gz).

        Args:
            file_path: Path to listings file.

        Returns:
            DataFrame of listings.
        """
        path = Path(file_path)
        if path.suffix == ".gz":
            df = pd.read_csv(file_path, compression="gzip", low_memory=False)
        else:
            df = pd.read_csv(file_path, low_memory=False)
        logger.info(f"Loaded {len(df):,} listings from {file_path}")
        return df

    # ----------------------------------------------------------------
    # Data cleaning
    # ----------------------------------------------------------------

    def clean_listings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize listing data.

        - Parses price strings to floats
        - Filters to relevant columns
        - Drops listings without coordinates

        Args:
            df: Raw listings DataFrame.

        Returns:
            Cleaned DataFrame.
        """
        if df.empty:
            return df

        # Filter to columns that exist
        available_cols = [c for c in LISTING_COLUMNS if c in df.columns]
        df = df[available_cols].copy()

        # Parse price: "$1,234.00" -> 1234.0
        if "price" in df.columns:
            df["price"] = (
                df["price"]
                .astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
            )
            df["price"] = pd.to_numeric(df["price"], errors="coerce")

        # Drop rows without coordinates
        if "latitude" in df.columns and "longitude" in df.columns:
            df = df.dropna(subset=["latitude", "longitude"])

        # Parse boolean columns
        for col in ["host_is_superhost", "instant_bookable"]:
            if col in df.columns:
                df[col] = df[col].map({"t": True, "f": False})

        return df

    def to_geodataframe(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """Convert listings DataFrame to GeoDataFrame.

        Args:
            df: Listings DataFrame with latitude/longitude columns.

        Returns:
            GeoDataFrame with point geometries.
        """
        if "latitude" not in df.columns or "longitude" not in df.columns:
            logger.error("DataFrame missing latitude/longitude columns")
            return gpd.GeoDataFrame()

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs="EPSG:4326",
        )
        return gdf

    # ----------------------------------------------------------------
    # STR density metrics
    # ----------------------------------------------------------------

    def compute_str_density(
        self,
        lat: float,
        lon: float,
        listings_gdf: gpd.GeoDataFrame,
        radius_m: float = 500,
    ) -> dict:
        """Compute short-term rental density metrics around a point.

        Args:
            lat: Latitude.
            lon: Longitude.
            listings_gdf: GeoDataFrame of Airbnb listings.
            radius_m: Search radius in meters.

        Returns:
            Dict with density metrics.
        """
        point = gpd.GeoDataFrame(
            geometry=[Point(lon, lat)], crs="EPSG:4326"
        )

        # Project for accurate distance
        listings_proj = listings_gdf.to_crs(epsg=32610)
        point_proj = point.to_crs(epsg=32610)

        listings_proj["distance_m"] = listings_proj.geometry.distance(
            point_proj.geometry.iloc[0]
        )

        nearby = listings_proj[listings_proj["distance_m"] <= radius_m].copy()

        metrics = {
            "str_count_500m": len(nearby),
            "str_density_per_km2": round(
                len(nearby) / (3.14159 * (radius_m / 1000) ** 2), 1
            ),
        }

        if "price" in nearby.columns and not nearby["price"].isna().all():
            metrics["str_avg_price"] = round(nearby["price"].mean(), 2)
            metrics["str_median_price"] = round(nearby["price"].median(), 2)

        if "room_type" in nearby.columns:
            room_counts = nearby["room_type"].value_counts().to_dict()
            metrics["str_entire_home_count"] = room_counts.get("Entire home/apt", 0)
            metrics["str_private_room_count"] = room_counts.get("Private room", 0)

        if "number_of_reviews_ltm" in nearby.columns:
            metrics["str_avg_reviews_ltm"] = round(
                nearby["number_of_reviews_ltm"].mean(), 1
            )

        if "availability_365" in nearby.columns:
            metrics["str_avg_availability"] = round(
                nearby["availability_365"].mean(), 0
            )

        return metrics

    def compute_neighbourhood_stats(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute STR statistics by neighbourhood.

        Args:
            df: Cleaned listings DataFrame.

        Returns:
            DataFrame with per-neighbourhood STR metrics.
        """
        if df.empty or "neighbourhood_cleansed" not in df.columns:
            return pd.DataFrame()

        agg_dict = {
            "id": "count",
        }
        if "price" in df.columns:
            agg_dict["price"] = ["mean", "median"]
        if "number_of_reviews" in df.columns:
            agg_dict["number_of_reviews"] = "mean"
        if "availability_365" in df.columns:
            agg_dict["availability_365"] = "mean"
        if "room_type" in df.columns:
            # Count entire homes separately
            df["is_entire_home"] = df["room_type"] == "Entire home/apt"
            agg_dict["is_entire_home"] = "sum"

        stats = df.groupby("neighbourhood_cleansed").agg(agg_dict)
        stats.columns = ["_".join(c).strip("_") for c in stats.columns]
        stats = stats.rename(columns={"id_count": "listing_count"}).reset_index()

        return stats.sort_values("listing_count", ascending=False)
