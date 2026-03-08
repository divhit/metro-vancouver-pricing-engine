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

# RSS feeds for Vancouver real estate news
NEWS_FEEDS = [
    {
        "name": "STOREYS Vancouver",
        "url": "https://storeys.com/feeds/cities/vancouver.rss",
        "category": "news",
    },
    {
        "name": "Daily Hive Urbanized",
        "url": "https://dailyhive.com/feed/vancouver",
        "category": "news",
    },
{
        "name": "Vancouver Sun Real Estate",
        "url": "https://vancouversun.com/category/business/real-estate/feed",
        "category": "news",
    },
]

# Email configuration
EMAIL_FROM = os.environ.get("INTEL_EMAIL_FROM", "")
EMAIL_TO = os.environ.get("INTEL_EMAIL_TO", "")
SMTP_HOST = os.environ.get("INTEL_SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("INTEL_SMTP_PORT", "587"))
SMTP_USER = os.environ.get("INTEL_SMTP_USER", "")
SMTP_PASS = os.environ.get("INTEL_SMTP_PASS", "")

# Real estate keywords for filtering news
RE_KEYWORDS = [
    "real estate", "housing", "condo", "townhouse", "detached",
    "presale", "pre-sale", "development", "rezoning", "rezone",
    "mortgage", "interest rate", "benchmark price", "sales-to-active",
    "MLS", "REBGV", "GVR", "property tax", "assessment",
    "Vancouver", "Burnaby", "Richmond", "North Vancouver",
    "strata", "rental", "vacancy", "housing start",
]
