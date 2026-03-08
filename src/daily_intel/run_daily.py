#!/usr/bin/env python3
"""
Daily Market Intelligence Report — Main Orchestrator

Run this daily at 6 AM to generate Aparna's morning briefing.

Usage:
    python -m src.daily_intel.run_daily                  # Full daily run
    python -m src.daily_intel.run_daily --ingest-csv     # Just ingest new CSVs
    python -m src.daily_intel.run_daily --report-only    # Just generate report
    python -m src.daily_intel.run_daily --no-email       # Generate but don't email
"""
import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("daily_intel")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.daily_intel.config import REPORTS_DIR, INTEL_DIR
from src.daily_intel.storage.database import (
    init_db,
    store_sold_listings,
    store_news_articles,
    get_sold_listings_for_date,
    get_recent_news,
    get_connection,
)
from src.daily_intel.scrapers.mls_csv_ingester import watch_directory, ingest_csv
from src.daily_intel.scrapers.permits import (
    fetch_vancouver_new_buildings,
    fetch_vancouver_demolitions,
    fetch_vancouver_permits,
    get_permit_summary,
)
from src.daily_intel.feeds.news import fetch_all_feeds, get_todays_headlines
from src.daily_intel.report.generator import generate_report
from src.daily_intel.report.email_sender import send_report


def step_ingest_mls_csvs():
    """Step 1: Check for new MLS CSV exports in Downloads."""
    logger.info("=" * 60)
    logger.info("STEP 1: Ingesting MLS CSV exports")
    logger.info("=" * 60)

    # Watch both Downloads and a dedicated drop folder
    watch_dirs = [
        Path.home() / "Downloads",
        INTEL_DIR / "csv_drops",
    ]

    # Ensure drop folder exists
    (INTEL_DIR / "csv_drops").mkdir(exist_ok=True)

    total_ingested = 0
    for d in watch_dirs:
        if d.exists():
            results = watch_directory(str(d), pattern="ML_*.csv")
            for r in results:
                logger.info(f"  Ingested {r['inserted']} from {r['file']}")
                total_ingested += r["inserted"]

    if total_ingested == 0:
        logger.info("  No new CSV files found")
    else:
        logger.info(f"  Total new listings ingested: {total_ingested}")

    return total_ingested


def step_fetch_permits():
    """Step 2: Fetch building permits from Vancouver Open Data."""
    logger.info("=" * 60)
    logger.info("STEP 2: Fetching building permits")
    logger.info("=" * 60)

    # New buildings (significant projects)
    new_buildings = fetch_vancouver_new_buildings(days_back=7, min_value=500000)
    logger.info(f"  New building permits: {len(new_buildings)}")

    # Demolitions (signals redevelopment)
    demolitions = fetch_vancouver_demolitions(days_back=7)
    logger.info(f"  Demolition permits: {len(demolitions)}")

    # All permits for summary
    all_permits = fetch_vancouver_permits(days_back=7)
    summary = get_permit_summary(all_permits)
    logger.info(f"  Total permits this week: {summary['total']} (${summary['total_value']:,.0f} total value)")

    return new_buildings + demolitions


def step_fetch_news():
    """Step 3: Fetch RSS news feeds."""
    logger.info("=" * 60)
    logger.info("STEP 3: Fetching news feeds")
    logger.info("=" * 60)

    articles = fetch_all_feeds()
    stored = store_news_articles(articles)
    logger.info(f"  Fetched {len(articles)} articles, {stored} new")

    # Filter to relevant headlines
    headlines = get_todays_headlines(articles, min_relevance=0.0)
    logger.info(f"  Today's relevant headlines: {len(headlines)}")

    return articles


def step_generate_report(
    sold_listings: list[dict],
    permits: list[dict],
    news: list[dict],
) -> str:
    """Step 4: Generate HTML report."""
    logger.info("=" * 60)
    logger.info("STEP 4: Generating report")
    logger.info("=" * 60)

    html = generate_report(
        sold_listings=sold_listings,
        new_permits=permits,
        news_articles=news,
    )

    # Save to file
    today = date.today().isoformat()
    report_path = REPORTS_DIR / f"daily_intel_{today}.html"
    report_path.write_text(html)
    logger.info(f"  Report saved: {report_path}")

    # Log to database
    conn = get_connection()
    conn.execute("""
        INSERT OR REPLACE INTO report_log (report_date, generated_at, report_path, stats_json)
        VALUES (?, ?, ?, ?)
    """, (today, datetime.now().isoformat(), str(report_path), f'{{"sold": {len(sold_listings)}, "permits": {len(permits)}, "news": {len(news)}}}'))
    conn.commit()
    conn.close()

    return html


def step_send_email(html: str, no_email: bool = False) -> bool:
    """Step 5: Email the report."""
    logger.info("=" * 60)
    logger.info("STEP 5: Sending email")
    logger.info("=" * 60)

    if no_email:
        logger.info("  --no-email flag set, skipping")
        return False

    sent = send_report(html)
    if sent:
        # Update report log
        today = date.today().isoformat()
        conn = get_connection()
        conn.execute("UPDATE report_log SET email_sent = 1 WHERE report_date = ?", (today,))
        conn.commit()
        conn.close()
        logger.info("  Email sent successfully")
    else:
        logger.info("  Email not sent (not configured or failed)")

    return sent


def run_full_pipeline(no_email: bool = False):
    """Run the complete daily intelligence pipeline."""
    logger.info(f"Starting daily intelligence pipeline — {date.today().isoformat()}")
    start = datetime.now()

    # Initialize database
    init_db()

    # Step 1: Ingest any new MLS CSVs
    step_ingest_mls_csvs()

    # Step 2: Fetch permits
    permits = step_fetch_permits()

    # Step 3: Fetch news
    news = step_fetch_news()

    # Get today's sold listings from database
    today = date.today().isoformat()
    sold = get_sold_listings_for_date(today)
    if not sold:
        # If nothing new today, show recent
        conn = get_connection()
        sold = [dict(r) for r in conn.execute(
            "SELECT * FROM sold_listings ORDER BY first_seen_date DESC LIMIT 20"
        ).fetchall()]
        conn.close()

    # Step 4: Generate report
    html = step_generate_report(sold, permits, news)

    # Step 5: Send email
    step_send_email(html, no_email=no_email)

    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    logger.info(f"Report: {REPORTS_DIR / f'daily_intel_{today}.html'}")

    return html


def main():
    parser = argparse.ArgumentParser(description="Daily Market Intelligence Report")
    parser.add_argument("--ingest-csv", action="store_true", help="Only ingest new CSV files")
    parser.add_argument("--ingest-file", type=str, help="Ingest a specific CSV file")
    parser.add_argument("--report-only", action="store_true", help="Only generate report from existing data")
    parser.add_argument("--no-email", action="store_true", help="Don't send email")
    args = parser.parse_args()

    init_db()

    if args.ingest_file:
        stats = ingest_csv(args.ingest_file)
        logger.info(f"Ingested: {stats}")
        return

    if args.ingest_csv:
        step_ingest_mls_csvs()
        return

    if args.report_only:
        today = date.today().isoformat()
        sold = get_sold_listings_for_date(today)
        if not sold:
            conn = get_connection()
            sold = [dict(r) for r in conn.execute(
                "SELECT * FROM sold_listings ORDER BY first_seen_date DESC LIMIT 20"
            ).fetchall()]
            conn.close()
        news = get_recent_news(days=1)
        permits = fetch_vancouver_new_buildings(days_back=7) + fetch_vancouver_demolitions(days_back=7)
        html = step_generate_report(sold, permits, news)
        step_send_email(html, no_email=args.no_email)
        return

    run_full_pipeline(no_email=args.no_email)


if __name__ == "__main__":
    main()
