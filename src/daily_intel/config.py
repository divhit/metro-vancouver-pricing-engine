"""Configuration for daily market intelligence."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INTEL_DIR = DATA_DIR / "daily_intel"
REPORTS_DIR = INTEL_DIR / "reports"
SNAPSHOTS_DIR = INTEL_DIR / "snapshots"
DB_PATH = INTEL_DIR / "market_intel.db"

# Ensure directories exist
INTEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# Zealty tRPC API (authenticated via browser cookies)
ZEALTY_BASE_URL = "https://www.zealty.ca"
ZEALTY_TRPC_URL = f"{ZEALTY_BASE_URL}/api/trpc"

# Zealty cookie file — exported from browser session
ZEALTY_COOKIE_FILE = INTEL_DIR / "zealty_cookies.json"

# Metro Vancouver regions to track
TRACKED_REGIONS = [
    {"cityOrRegion": "vancouver-westside", "label": "Vancouver Westside"},
    {"cityOrRegion": "vancouver-eastside", "label": "Vancouver Eastside"},
    {"cityOrRegion": "vancouver-downtown", "label": "Vancouver Downtown"},
    {"cityOrRegion": "mapby", "label": "Burnaby"},
    {"cityOrRegion": "mapri", "label": "Richmond"},
    {"cityOrRegion": "mapnv", "label": "North Vancouver"},
    {"cityOrRegion": "mapwv", "label": "West Vancouver"},
    {"cityOrRegion": "coquitlam", "label": "Coquitlam"},
    {"cityOrRegion": "north-surrey", "label": "North Surrey"},
    {"cityOrRegion": "south-surrey", "label": "South Surrey"},
    {"cityOrRegion": "langley", "label": "Langley"},
    {"cityOrRegion": "maple-ridge", "label": "Maple Ridge"},
]

# Property types
PROPERTY_TYPES = ["HSE", "APT", "TWN", "PAD", "MUF"]

# RSS feeds for Vancouver real estate news — curated for RE relevance
NEWS_FEEDS = [
    {
        "name": "STOREYS",
        "url": "https://storeys.com/feeds/cities/vancouver.rss",
        "category": "news",
        "min_relevance": 0.0,  # STOREYS is RE-only, always relevant
    },
    {
        "name": "Rennie",
        "url": "https://renfrewrealty.ca/feed/",
        "category": "market_analysis",
        "min_relevance": 0.0,
        "optional": True,  # May not have RSS feed
    },
    {
        "name": "UrbanYVR",
        "url": "https://urbanyvr.com/feed",
        "category": "development",
        "min_relevance": 0.0,  # UrbanYVR is development-focused
    },
    {
        "name": "Vancouver Sun Real Estate",
        "url": "https://vancouversun.com/category/business/real-estate/feed",
        "category": "news",
        "min_relevance": 0.0,
    },
    {
        "name": "Daily Hive Urbanized",
        "url": "https://dailyhive.com/feed/vancouver",
        "category": "news",
        "min_relevance": 0.33,  # General feed — only include RE-relevant articles
    },
    {
        "name": "REBGV",
        "url": "https://www.gvrealtors.ca/news.html",
        "category": "market_data",
        "min_relevance": 0.0,
        "optional": True,  # RSS may not be available
    },
    {
        "name": "BCREA",
        "url": "https://www.bcrea.bc.ca/feed/",
        "category": "market_data",
        "min_relevance": 0.0,
    },
]

# Email configuration
EMAIL_FROM = os.environ.get("INTEL_EMAIL_FROM", "")
EMAIL_TO = os.environ.get("INTEL_EMAIL_TO", "")
SMTP_HOST = os.environ.get("INTEL_SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("INTEL_SMTP_PORT", "587"))
SMTP_USER = os.environ.get("INTEL_SMTP_USER", "")
SMTP_PASS = os.environ.get("INTEL_SMTP_PASS", "")

# Real estate keywords for filtering news (used to score relevance)
RE_KEYWORDS = [
    # Core RE terms
    "real estate", "housing market", "home sales", "home prices",
    "condo", "townhouse", "townhome", "detached", "single-family",
    "presale", "pre-sale", "new construction", "new build",
    # Development & zoning
    "development", "rezoning", "rezone", "density", "tower",
    "housing start", "building permit", "construction",
    # Market & finance
    "mortgage", "interest rate", "bank of canada", "benchmark price",
    "sales-to-active", "price index", "affordability",
    "property tax", "assessment", "property value",
    # Industry
    "MLS", "REBGV", "GVR", "BCREA", "CMHC", "realtor", "listing",
    "rennie", "sold", "asking price", "over asking",
    # Location (boost general articles mentioning RE areas)
    "strata", "rental", "vacancy", "lease",
    "Vancouver housing", "Metro Vancouver",
]
