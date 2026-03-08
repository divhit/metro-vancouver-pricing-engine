"""SQLite storage for daily market intelligence snapshots."""
import sqlite3
import json
from datetime import datetime, date
from pathlib import Path
from typing import Optional

from ..config import DB_PATH


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sold_listings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mls_number TEXT NOT NULL,
            sold_date TEXT,
            entry_date TEXT,
            sold_price INTEGER,
            listing_price INTEGER,
            listing_price_orig INTEGER,
            street_address TEXT,
            city TEXT,
            area_name TEXT,
            province TEXT,
            postal_code TEXT,
            bedroom_count INTEGER,
            bathroom_count INTEGER,
            house_size INTEGER,
            lot_size INTEGER,
            property_type TEXT,
            listing_agent TEXT,
            listing_brokerage TEXT,
            latitude REAL,
            longitude REAL,
            images_json TEXT,
            first_seen_date TEXT NOT NULL,
            raw_json TEXT,
            UNIQUE(mls_number, sold_date)
        );

        CREATE TABLE IF NOT EXISTS active_listings_snapshot (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date TEXT NOT NULL,
            mls_number TEXT NOT NULL,
            listing_price INTEGER,
            listing_price_prev INTEGER,
            street_address TEXT,
            city TEXT,
            area_name TEXT,
            bedroom_count INTEGER,
            bathroom_count INTEGER,
            house_size INTEGER,
            lot_size INTEGER,
            property_type TEXT,
            listing_date TEXT,
            days_on_market INTEGER,
            latitude REAL,
            longitude REAL,
            raw_json TEXT,
            UNIQUE(snapshot_date, mls_number)
        );

        CREATE TABLE IF NOT EXISTS market_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stat_date TEXT NOT NULL,
            region TEXT NOT NULL,
            property_type TEXT,
            total_active INTEGER,
            total_sold_7d INTEGER,
            total_sold_30d INTEGER,
            avg_sold_price REAL,
            median_sold_price REAL,
            avg_list_price REAL,
            avg_price_change_pct REAL,
            sales_to_active_ratio REAL,
            avg_days_on_market REAL,
            raw_json TEXT,
            UNIQUE(stat_date, region, property_type)
        );

        CREATE TABLE IF NOT EXISTS news_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feed_name TEXT NOT NULL,
            title TEXT NOT NULL,
            url TEXT NOT NULL UNIQUE,
            published_date TEXT,
            summary TEXT,
            category TEXT,
            relevance_score REAL,
            first_seen_date TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS report_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_date TEXT NOT NULL UNIQUE,
            generated_at TEXT NOT NULL,
            report_path TEXT,
            email_sent INTEGER DEFAULT 0,
            stats_json TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_sold_date ON sold_listings(sold_date);
        CREATE INDEX IF NOT EXISTS idx_sold_city ON sold_listings(city);
        CREATE INDEX IF NOT EXISTS idx_sold_mls ON sold_listings(mls_number);
        CREATE INDEX IF NOT EXISTS idx_active_date ON active_listings_snapshot(snapshot_date);
        CREATE INDEX IF NOT EXISTS idx_active_mls ON active_listings_snapshot(mls_number);
        CREATE INDEX IF NOT EXISTS idx_news_date ON news_articles(published_date);
        CREATE INDEX IF NOT EXISTS idx_stats_date ON market_stats(stat_date);
    """)
    conn.close()


def store_sold_listings(listings: list[dict], source_date: Optional[str] = None):
    """Store sold listings, skipping duplicates."""
    conn = get_connection()
    today = source_date or date.today().isoformat()
    inserted = 0
    for listing in listings:
        try:
            conn.execute("""
                INSERT OR IGNORE INTO sold_listings (
                    mls_number, sold_date, entry_date, sold_price, listing_price,
                    listing_price_orig, street_address, city, area_name, province,
                    postal_code, bedroom_count, bathroom_count, house_size, lot_size,
                    property_type, listing_agent, listing_brokerage, latitude, longitude,
                    images_json, first_seen_date, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                listing.get("mlsNumber"),
                listing.get("soldDate", "")[:10] if listing.get("soldDate") else None,
                listing.get("entryDate", "")[:10] if listing.get("entryDate") else None,
                listing.get("soldPrice"),
                listing.get("listingPrice"),
                listing.get("listingPriceOrig"),
                listing.get("streetAddress"),
                listing.get("city"),
                listing.get("areaName"),
                listing.get("province"),
                listing.get("postalCode"),
                listing.get("bedroomCount"),
                listing.get("bathroomCount"),
                listing.get("houseSize"),
                listing.get("lotSize"),
                listing.get("type"),
                listing.get("listingAgent"),
                listing.get("listingBrokerage"),
                listing.get("latitude"),
                listing.get("longitude"),
                json.dumps(listing.get("images", [])),
                today,
                json.dumps(listing),
            ))
            inserted += 1
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    conn.close()
    return inserted


def store_active_snapshot(listings: list[dict], snapshot_date: Optional[str] = None):
    """Store today's active listings snapshot for diff tracking."""
    conn = get_connection()
    today = snapshot_date or date.today().isoformat()
    inserted = 0
    for listing in listings:
        try:
            conn.execute("""
                INSERT OR IGNORE INTO active_listings_snapshot (
                    snapshot_date, mls_number, listing_price, listing_price_prev,
                    street_address, city, area_name, bedroom_count, bathroom_count,
                    house_size, lot_size, property_type, listing_date, days_on_market,
                    latitude, longitude, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                today,
                listing.get("mlsNumber"),
                listing.get("listingPrice"),
                listing.get("listingPricePrev"),
                listing.get("streetAddress"),
                listing.get("city"),
                listing.get("areaName"),
                listing.get("bedroomCount"),
                listing.get("bathroomCount"),
                listing.get("houseSize"),
                listing.get("lotSize"),
                listing.get("type"),
                listing.get("listingDate", "")[:10] if listing.get("listingDate") else None,
                listing.get("sortValue"),
                listing.get("latitude"),
                listing.get("longitude"),
                json.dumps(listing),
            ))
            inserted += 1
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    conn.close()
    return inserted


def store_news_articles(articles: list[dict]):
    """Store news articles, skipping duplicates by URL."""
    conn = get_connection()
    today = date.today().isoformat()
    inserted = 0
    for article in articles:
        try:
            conn.execute("""
                INSERT OR IGNORE INTO news_articles (
                    feed_name, title, url, published_date, summary,
                    category, relevance_score, first_seen_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                article.get("feed_name"),
                article.get("title"),
                article.get("url"),
                article.get("published_date"),
                article.get("summary"),
                article.get("category"),
                article.get("relevance_score", 0),
                today,
            ))
            inserted += 1
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    conn.close()
    return inserted


def store_market_stats(stats: list[dict]):
    """Store daily market statistics."""
    conn = get_connection()
    inserted = 0
    for stat in stats:
        try:
            conn.execute("""
                INSERT OR REPLACE INTO market_stats (
                    stat_date, region, property_type, total_active, total_sold_7d,
                    total_sold_30d, avg_sold_price, median_sold_price, avg_list_price,
                    avg_price_change_pct, sales_to_active_ratio, avg_days_on_market, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                stat.get("stat_date"),
                stat.get("region"),
                stat.get("property_type"),
                stat.get("total_active"),
                stat.get("total_sold_7d"),
                stat.get("total_sold_30d"),
                stat.get("avg_sold_price"),
                stat.get("median_sold_price"),
                stat.get("avg_list_price"),
                stat.get("avg_price_change_pct"),
                stat.get("sales_to_active_ratio"),
                stat.get("avg_days_on_market"),
                json.dumps(stat),
            ))
            inserted += 1
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    conn.close()
    return inserted


def get_sold_listings_for_date(target_date: str) -> list[dict]:
    """Get all sold listings first seen on a specific date."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM sold_listings WHERE first_seen_date = ? ORDER BY sold_price DESC",
        (target_date,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_newly_removed_listings(current_date: str, previous_date: str) -> list[dict]:
    """Find listings that were active yesterday but not today (likely sold or expired)."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT prev.* FROM active_listings_snapshot prev
        LEFT JOIN active_listings_snapshot curr
            ON prev.mls_number = curr.mls_number AND curr.snapshot_date = ?
        WHERE prev.snapshot_date = ? AND curr.mls_number IS NULL
        ORDER BY prev.listing_price DESC
    """, (current_date, previous_date)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_price_changes(current_date: str, previous_date: str) -> list[dict]:
    """Find listings with price changes between snapshots."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT
            curr.mls_number,
            curr.street_address,
            curr.city,
            curr.area_name,
            curr.property_type,
            curr.bedroom_count,
            curr.bathroom_count,
            curr.house_size,
            prev.listing_price as old_price,
            curr.listing_price as new_price,
            ROUND((curr.listing_price - prev.listing_price) * 100.0 / prev.listing_price, 1) as change_pct
        FROM active_listings_snapshot curr
        JOIN active_listings_snapshot prev
            ON curr.mls_number = prev.mls_number AND prev.snapshot_date = ?
        WHERE curr.snapshot_date = ? AND curr.listing_price != prev.listing_price
        ORDER BY ABS(curr.listing_price - prev.listing_price) DESC
    """, (previous_date, current_date)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_new_listings(current_date: str, previous_date: str) -> list[dict]:
    """Find listings that appeared today but weren't there yesterday."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT curr.* FROM active_listings_snapshot curr
        LEFT JOIN active_listings_snapshot prev
            ON curr.mls_number = prev.mls_number AND prev.snapshot_date = ?
        WHERE curr.snapshot_date = ? AND prev.mls_number IS NULL
        ORDER BY curr.listing_price DESC
    """, (previous_date, current_date)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_recent_news(days: int = 1) -> list[dict]:
    """Get news articles from the last N days."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM news_articles
        WHERE first_seen_date >= date('now', ?)
        ORDER BY relevance_score DESC, published_date DESC
    """, (f"-{days} days",)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# Initialize on import
init_db()
