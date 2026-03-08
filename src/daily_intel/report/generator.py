"""Generate HTML daily intelligence report for Aparna."""
import logging
from datetime import date, datetime, timedelta
from typing import Optional

from ..storage.database import (
    get_sold_listings_for_date,
    get_recent_news,
    get_connection,
)

logger = logging.getLogger(__name__)


def _fmt_price(val) -> str:
    """Format price as $1,350,000."""
    if val is None:
        return "N/A"
    return f"${int(val):,}"


def _fmt_pct(val) -> str:
    """Format percentage."""
    if val is None:
        return "N/A"
    return f"{val:+.1f}%"


def _sold_vs_list_pct(sold_price, list_price) -> Optional[float]:
    """Calculate sold vs list price percentage."""
    if sold_price and list_price and list_price > 0:
        return round((sold_price - list_price) / list_price * 100, 1)
    return None


def generate_report(
    sold_listings: list[dict],
    new_permits: list[dict],
    news_articles: list[dict],
    market_stats: Optional[dict] = None,
    active_listing_changes: Optional[dict] = None,
    report_date: Optional[str] = None,
) -> str:
    """Generate the full HTML morning briefing."""
    today = report_date or date.today().strftime("%A, %B %d, %Y")

    # Categorize sold listings
    attached_sold = [l for l in sold_listings if l.get("property_type") in
                     ("Apartment/Condo", "Townhouse", "APT", "TWN")]
    detached_sold = [l for l in sold_listings if l.get("property_type") in
                     ("House", "HOUSE", "Single Family Residence", "HSE", "OTHER")]
    other_sold = [l for l in sold_listings if l not in attached_sold and l not in detached_sold]

    # If we can't categorize by type, just show all
    if not attached_sold and not detached_sold:
        all_sold = sold_listings
    else:
        all_sold = None

    # Categorize news
    market_news = [a for a in news_articles if a.get("category") in ("market_stats", "market_analysis")]
    dev_news = [a for a in news_articles if a.get("category") == "municipal"]
    general_news = [a for a in news_articles if a.get("category") == "news"]

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f8f9fa; color: #1a1a1a; }}
  .header {{ background: linear-gradient(135deg, #1a365d, #2d5a87); color: white; padding: 30px; border-radius: 12px; margin-bottom: 24px; }}
  .header h1 {{ margin: 0 0 8px 0; font-size: 24px; }}
  .header .date {{ opacity: 0.85; font-size: 14px; }}
  .section {{ background: white; border-radius: 10px; padding: 24px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .section h2 {{ margin: 0 0 16px 0; font-size: 18px; color: #1a365d; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 16px; }}
  .stat-card {{ background: #f0f4f8; border-radius: 8px; padding: 16px; text-align: center; }}
  .stat-card .number {{ font-size: 28px; font-weight: 700; color: #1a365d; }}
  .stat-card .label {{ font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }}
  .listing {{ border-left: 3px solid #3182ce; padding: 12px 16px; margin-bottom: 12px; background: #f7fafc; border-radius: 0 6px 6px 0; }}
  .listing .address {{ font-weight: 600; font-size: 15px; }}
  .listing .details {{ font-size: 13px; color: #4a5568; margin-top: 4px; }}
  .listing .price {{ font-size: 16px; font-weight: 700; color: #2d5a87; }}
  .listing .price-change {{ font-size: 12px; padding: 2px 6px; border-radius: 4px; }}
  .price-up {{ background: #c6f6d5; color: #22543d; }}
  .price-down {{ background: #fed7d7; color: #742a2a; }}
  .permit {{ padding: 10px 16px; margin-bottom: 8px; background: #fffbeb; border-left: 3px solid #d69e2e; border-radius: 0 6px 6px 0; }}
  .permit .type {{ font-weight: 600; font-size: 13px; color: #744210; }}
  .permit .address {{ font-size: 14px; }}
  .permit .value {{ font-size: 13px; color: #4a5568; }}
  .news-item {{ padding: 10px 0; border-bottom: 1px solid #e2e8f0; }}
  .news-item:last-child {{ border-bottom: none; }}
  .news-item a {{ color: #2d5a87; text-decoration: none; font-weight: 500; }}
  .news-item a:hover {{ text-decoration: underline; }}
  .news-item .source {{ font-size: 12px; color: #718096; }}
  .news-item .summary {{ font-size: 13px; color: #4a5568; margin-top: 4px; }}
  .footer {{ text-align: center; padding: 20px; font-size: 12px; color: #a0aec0; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }}
  .badge-sold {{ background: #e2e8f0; color: #2d3748; }}
  .badge-new {{ background: #c6f6d5; color: #22543d; }}
  .badge-demo {{ background: #fed7d7; color: #742a2a; }}
</style>
</head>
<body>

<div class="header">
  <h1>Daily Market Intelligence</h1>
  <div class="date">{today} | Vancouver East & West</div>
</div>
"""

    # --- MARKET SNAPSHOT ---
    total_sold = len(sold_listings)
    avg_price = sum(l.get("sold_price") or l.get("listing_price") or 0 for l in sold_listings) / max(total_sold, 1)

    html += f"""
<div class="section">
  <h2>Market Snapshot</h2>
  <div class="stat-grid">
    <div class="stat-card">
      <div class="number">{total_sold}</div>
      <div class="label">New Solds Recorded</div>
    </div>
    <div class="stat-card">
      <div class="number">{len(attached_sold)}</div>
      <div class="label">Attached</div>
    </div>
    <div class="stat-card">
      <div class="number">{len(detached_sold)}</div>
      <div class="label">Detached</div>
    </div>
    <div class="stat-card">
      <div class="number">{_fmt_price(avg_price)}</div>
      <div class="label">Avg Sale Price</div>
    </div>
  </div>
</div>
"""

    # --- SOLD LISTINGS ---
    def _render_listings(listings, title, limit=10):
        if not listings:
            return ""
        section = f'<div class="section"><h2>{title} ({len(listings)} total)</h2>'
        for l in listings[:limit]:
            price = l.get("sold_price") or l.get("listing_price") or 0
            address = l.get("street_address", "Unknown")
            area = l.get("area_name", "")
            beds = l.get("bedroom_count") or "?"
            baths = l.get("bathroom_count") or "?"
            sqft = l.get("house_size")
            sqft_str = f" | {sqft:,} sqft" if sqft else ""
            ptype = l.get("property_type", "")

            section += f"""
  <div class="listing">
    <div class="address">{address}</div>
    <div class="details">{area} | {beds} bed / {baths} bath{sqft_str} | {ptype}</div>
    <div class="price">{_fmt_price(price)}</div>
  </div>"""

        if len(listings) > limit:
            section += f'<p style="color:#718096;font-size:13px;">...and {len(listings) - limit} more</p>'
        section += "</div>"
        return section

    if all_sold:
        html += _render_listings(all_sold, "Sold Listings")
    else:
        if detached_sold:
            html += _render_listings(
                sorted(detached_sold, key=lambda l: l.get("sold_price") or l.get("listing_price") or 0, reverse=True),
                "Detached Homes Sold"
            )
        if attached_sold:
            html += _render_listings(
                sorted(attached_sold, key=lambda l: l.get("sold_price") or l.get("listing_price") or 0, reverse=True),
                "Attached (Condos/Townhomes) Sold"
            )

    # --- NEW DEVELOPMENTS & PERMITS ---
    if new_permits:
        html += '<div class="section"><h2>New Building Permits & Developments</h2>'
        for p in new_permits[:10]:
            ptype = p.get("type_of_work", "Permit")
            badge_class = "badge-new" if "New" in ptype else "badge-demo" if "Demol" in ptype else "badge-sold"
            value = _fmt_price(p.get("project_value")) if p.get("project_value") else ""

            html += f"""
  <div class="permit">
    <div class="type"><span class="badge {badge_class}">{ptype}</span></div>
    <div class="address">{p.get("address", "N/A")}</div>
    <div class="value">{p.get("project_description", "")[:120]} {f"| Value: {value}" if value else ""}</div>
  </div>"""
        html += "</div>"

    # --- NEWS ---
    all_news = market_news + dev_news + general_news
    if all_news:
        html += '<div class="section"><h2>Vancouver Real Estate News</h2>'
        for a in all_news[:8]:
            title = a.get("title", "")
            url = a.get("url", "#")
            source = a.get("feed_name", "")
            summary = a.get("summary", "")[:200]

            html += f"""
  <div class="news-item">
    <a href="{url}">{title}</a>
    <div class="source">{source}</div>
    {"<div class='summary'>" + summary + "...</div>" if summary else ""}
  </div>"""
        html += "</div>"

    # --- FOOTER ---
    html += f"""
<div class="footer">
  Generated {datetime.now().strftime("%Y-%m-%d %H:%M")} | Metro Vancouver Pricing Engine<br>
  Data sources: MLS Paragon, City of Vancouver Open Data, STOREYS, Daily Hive, REBGV
</div>
</body>
</html>"""

    return html
