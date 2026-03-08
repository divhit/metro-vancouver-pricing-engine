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
            status TEXT,
            address TEXT,
            sub_area TEXT,
            sub_area_name TEXT,
            city TEXT,
            postal_code TEXT,
            sold_price INTEGER,
            list_price INTEGER,
            price_diff INTEGER,
            price_diff_pct REAL,
            list_date TEXT,
            sold_date TEXT,
            dom INTEGER,
            days_on_mls INTEGER,
            bedrooms INTEGER,
            bathrooms INTEGER,
            floor_area INTEGER,
            year_built INTEGER,
            age INTEGER,
            property_type TEXT,
            style TEXT,
            zoning TEXT,
            view TEXT,
            -- Attached-specific
            locker TEXT,
            parking INTEGER,
            maint_fee REAL,
            bylaw_restrictions TEXT,
            -- Detached-specific
            frontage REAL,
            depth REAL,
            kitchens INTEGER,
            -- Metadata
            pic_count INTEGER,
            pic_url TEXT,
            source_file TEXT,
            source_format TEXT,
            first_seen_date TEXT NOT NULL,
            UNIQUE(mls_number)
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

        CREATE TABLE IF NOT EXISTS ingested_files (
            file_path TEXT PRIMARY KEY,
            ingested_at TEXT NOT NULL,
            row_count INTEGER,
            stats_json TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_sold_date ON sold_listings(sold_date);
        CREATE INDEX IF NOT EXISTS idx_sold_sub_area ON sold_listings(sub_area);
        CREATE INDEX IF NOT EXISTS idx_sold_city ON sold_listings(city);
        CREATE INDEX IF NOT EXISTS idx_sold_mls ON sold_listings(mls_number);
        CREATE INDEX IF NOT EXISTS idx_sold_first_seen ON sold_listings(first_seen_date);
        CREATE INDEX IF NOT EXISTS idx_sold_type ON sold_listings(property_type);
        CREATE INDEX IF NOT EXISTS idx_news_date ON news_articles(published_date);
    """)
    conn.close()


def store_news_articles(articles: list[dict]) -> int:
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


def get_sold_listings_for_date(target_date: str) -> list[dict]:
    """Get all sold listings first seen on a specific date."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM sold_listings WHERE first_seen_date = ? ORDER BY sold_price DESC",
        (target_date,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_sold_listings() -> list[dict]:
    """Get all sold listings ordered by sold price descending."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM sold_listings ORDER BY sold_price DESC"
    ).fetchall()
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


def get_summary_stats() -> dict:
    """Get summary statistics across all stored data."""
    conn = get_connection()

    total = conn.execute("SELECT COUNT(*) FROM sold_listings").fetchone()[0]

    by_type = {dict(r)["property_type"]: dict(r)["cnt"] for r in conn.execute(
        "SELECT property_type, COUNT(*) as cnt FROM sold_listings GROUP BY property_type ORDER BY cnt DESC"
    ).fetchall()}

    by_area = {dict(r)["city"]: dict(r)["cnt"] for r in conn.execute(
        "SELECT city, COUNT(*) as cnt FROM sold_listings GROUP BY city ORDER BY cnt DESC"
    ).fetchall()}

    price_stats = dict(conn.execute("""
        SELECT
            AVG(sold_price) as avg_price,
            MIN(sold_price) as min_price,
            MAX(sold_price) as max_price,
            AVG(dom) as avg_dom,
            AVG(price_diff_pct) as avg_price_diff_pct
        FROM sold_listings WHERE sold_price > 0
    """).fetchone())

    conn.close()
    return {
        "total_listings": total,
        "by_type": by_type,
        "by_area": by_area,
        "price_stats": price_stats,
    }


# Initialize on import
init_db()
