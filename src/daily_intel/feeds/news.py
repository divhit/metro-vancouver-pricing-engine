"""RSS feed aggregator for Vancouver real estate news."""
import logging
import re
from datetime import datetime, timezone
from typing import Optional

import feedparser
import httpx

from ..config import NEWS_FEEDS, RE_KEYWORDS

logger = logging.getLogger(__name__)


def _compute_relevance(title: str, summary: str) -> float:
    """Score article relevance to Vancouver real estate (0-1)."""
    text = f"{title} {summary}".lower()
    hits = sum(1 for kw in RE_KEYWORDS if kw.lower() in text)
    # Normalize: 3+ keyword hits = score 1.0
    return min(hits / 3.0, 1.0)


def _parse_date(entry) -> Optional[str]:
    """Extract published date from feed entry."""
    for field in ("published_parsed", "updated_parsed"):
        parsed = getattr(entry, field, None)
        if parsed:
            try:
                dt = datetime(*parsed[:6], tzinfo=timezone.utc)
                return dt.isoformat()
            except Exception:
                pass
    for field in ("published", "updated"):
        val = getattr(entry, field, None)
        if val:
            return val
    return None


def _clean_html(text: str) -> str:
    """Strip HTML tags from summary text."""
    return re.sub(r"<[^>]+>", "", text or "").strip()


def fetch_feed(feed_config: dict) -> list[dict]:
    """Fetch and parse a single RSS feed."""
    name = feed_config["name"]
    url = feed_config["url"]
    category = feed_config.get("category", "news")

    logger.info(f"Fetching RSS feed: {name}")
    try:
        # Some feeds need a browser-like User-Agent
        with httpx.Client(timeout=15, follow_redirects=True) as client:
            resp = client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; MarketIntelBot/1.0)",
            })
            resp.raise_for_status()
            content = resp.text
    except Exception as e:
        logger.warning(f"Failed to fetch {name}: {e}")
        return []

    feed = feedparser.parse(content)
    articles = []

    for entry in feed.entries[:20]:  # Cap at 20 per feed
        title = entry.get("title", "")
        summary = _clean_html(entry.get("summary", "") or entry.get("description", ""))
        link = entry.get("link", "")
        pub_date = _parse_date(entry)
        relevance = _compute_relevance(title, summary)

        articles.append({
            "feed_name": name,
            "title": title,
            "url": link,
            "published_date": pub_date,
            "summary": summary[:500] if summary else None,
            "category": category,
            "relevance_score": round(relevance, 2),
        })

    logger.info(f"  Parsed {len(articles)} articles from {name}")
    return articles


def fetch_all_feeds() -> list[dict]:
    """Fetch all configured RSS feeds and return combined articles."""
    all_articles = []
    for feed_config in NEWS_FEEDS:
        articles = fetch_feed(feed_config)
        all_articles.extend(articles)

    # Sort by relevance, then date
    all_articles.sort(key=lambda a: (-a["relevance_score"], a.get("published_date") or ""))
    logger.info(f"Total articles fetched: {len(all_articles)}")
    return all_articles


def get_todays_headlines(articles: list[dict], min_relevance: float = 0.3) -> list[dict]:
    """Filter to just today's relevant headlines."""
    today = datetime.now(timezone.utc).date().isoformat()
    relevant = []
    for a in articles:
        if a["relevance_score"] < min_relevance:
            continue
        # Check if published today (approximate — some feeds have timezone issues)
        pub = a.get("published_date", "")
        if pub and pub[:10] >= today:
            relevant.append(a)
        elif not pub:
            # Include articles without dates (they're from today's fetch)
            relevant.append(a)
    return relevant
