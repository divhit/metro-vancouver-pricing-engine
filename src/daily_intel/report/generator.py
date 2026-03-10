"""Generate HTML daily intelligence report for Aparna — Vancouver East & West."""
import logging
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)


def _fmt_price(val) -> str:
    if val is None or val == 0:
        return "—"
    return f"${int(val):,}"


def _fmt_pct(val) -> str:
    if val is None:
        return "—"
    color = "#22543d" if val >= 0 else "#c53030"
    arrow = "▲" if val > 0 else "▼" if val < 0 else "—"
    return f'<span style="color:{color};font-weight:600">{arrow} {abs(val):.1f}%</span>'


def _fmt_num(val) -> str:
    if val is None:
        return "—"
    return f"{int(val):,}"


def _fmt_sqft(val) -> str:
    if val is None:
        return "—"
    return f"{int(val):,} sqft"


def _fmt_money(val) -> str:
    if val is None:
        return "—"
    return f"${val:,.2f}"


def _price_per_sqft(price, sqft) -> str:
    if price and sqft and sqft > 0:
        return f"${int(price / sqft):,}/sqft"
    return "—"


def generate_report(
    sold_listings: list[dict],
    new_permits: list[dict],
    news_articles: list[dict],
    report_date: Optional[str] = None,
    market_summary: Optional[dict] = None,
) -> str:
    """Generate the full HTML morning briefing with ALL listing details."""
    today_str = report_date or date.today().strftime("%A, %B %d, %Y")

    # Split by type
    detached = [l for l in sold_listings if l.get("source_format") == "detached"
                or l.get("property_type") in ("HOUSE", "House", "Single Family Residence")]
    attached = [l for l in sold_listings if l not in detached]

    # Split by side
    van_east = [l for l in sold_listings if l.get("city") == "Vancouver East"]
    van_west = [l for l in sold_listings if l.get("city") == "Vancouver West"]

    # Stats
    total = len(sold_listings)
    prices = [l["sold_price"] for l in sold_listings if l.get("sold_price")]
    avg_price = sum(prices) / len(prices) if prices else 0
    median_price = sorted(prices)[len(prices) // 2] if prices else 0
    doms = [l["dom"] for l in sold_listings if l.get("dom") is not None]
    avg_dom = sum(doms) / len(doms) if doms else 0
    diffs = [l["price_diff_pct"] for l in sold_listings if l.get("price_diff_pct") is not None]
    avg_diff = sum(diffs) / len(diffs) if diffs else 0
    over_ask = len([d for d in diffs if d > 0])
    under_ask = len([d for d in diffs if d < 0])
    at_ask = len([d for d in diffs if d == 0])

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: 0 auto; padding: 16px; background: #f5f5f5; color: #1a1a1a; font-size: 14px; }}
  .header {{ background: linear-gradient(135deg, #1a365d, #2d5a87); color: white; padding: 28px 24px; border-radius: 10px; margin-bottom: 20px; }}
  .header h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .header .sub {{ opacity: 0.8; font-size: 13px; }}
  .section {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.06); }}
  .section h2 {{ font-size: 16px; color: #1a365d; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; margin-bottom: 14px; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin-bottom: 14px; }}
  .stat {{ background: #f7fafc; border-radius: 6px; padding: 12px; text-align: center; }}
  .stat .n {{ font-size: 22px; font-weight: 700; color: #1a365d; }}
  .stat .l {{ font-size: 11px; color: #718096; text-transform: uppercase; letter-spacing: 0.3px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
  th {{ background: #f7fafc; color: #4a5568; font-weight: 600; text-align: left; padding: 8px 6px; border-bottom: 2px solid #e2e8f0; font-size: 11px; text-transform: uppercase; letter-spacing: 0.3px; white-space: nowrap; }}
  td {{ padding: 7px 6px; border-bottom: 1px solid #edf2f7; vertical-align: top; }}
  tr:hover {{ background: #f7fafc; }}
  .addr {{ font-weight: 600; color: #2d3748; }}
  .area {{ font-size: 11px; color: #718096; }}
  .price {{ font-weight: 700; color: #1a365d; white-space: nowrap; }}
  .over {{ color: #22543d; font-weight: 600; }}
  .under {{ color: #c53030; font-weight: 600; }}
  .at {{ color: #718096; }}
  .tag {{ display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 10px; font-weight: 600; }}
  .tag-det {{ background: #ebf8ff; color: #2b6cb0; }}
  .tag-apt {{ background: #faf5ff; color: #6b46c1; }}
  .tag-twn {{ background: #f0fff4; color: #276749; }}
  .tag-dup {{ background: #fffaf0; color: #c05621; }}
  .permit {{ padding: 8px 12px; margin-bottom: 6px; background: #fffbeb; border-left: 3px solid #d69e2e; border-radius: 0 4px 4px 0; font-size: 13px; }}
  .permit .ptype {{ font-weight: 600; font-size: 11px; color: #744210; text-transform: uppercase; }}
  .news-item {{ padding: 8px 0; border-bottom: 1px solid #edf2f7; }}
  .news-item:last-child {{ border-bottom: none; }}
  .news-item a {{ color: #2d5a87; text-decoration: none; font-weight: 500; font-size: 13px; }}
  .news-item .src {{ font-size: 11px; color: #a0aec0; }}
  .footer {{ text-align: center; padding: 16px; font-size: 11px; color: #a0aec0; }}
</style>
</head>
<body>

<div class="header">
  <h1>Daily Market Intelligence</h1>
  <div class="sub">{today_str} — Vancouver East & West — Sold Listings, Permits & News</div>
</div>

<div class="section">
  <h2>Daily Market Update</h2>
  <div class="stats">
    <div class="stat"><div class="n">{total}</div><div class="l">Total Sold</div></div>
    <div class="stat"><div class="n">{len(van_east)}</div><div class="l">Van East</div></div>
    <div class="stat"><div class="n">{len(van_west)}</div><div class="l">Van West</div></div>
    <div class="stat"><div class="n">{len(detached)}</div><div class="l">Detached</div></div>
    <div class="stat"><div class="n">{len(attached)}</div><div class="l">Attached</div></div>
    <div class="stat"><div class="n">{_fmt_price(avg_price)}</div><div class="l">Avg Sold Price</div></div>
    <div class="stat"><div class="n">{_fmt_price(median_price)}</div><div class="l">Median Price</div></div>
    <div class="stat"><div class="n">{avg_dom:.0f}</div><div class="l">Avg DOM</div></div>
    <div class="stat"><div class="n">{avg_diff:+.1f}%</div><div class="l">Avg vs Ask</div></div>
    <div class="stat"><div class="n">{over_ask}</div><div class="l">Over Ask</div></div>
    <div class="stat"><div class="n">{under_ask}</div><div class="l">Under Ask</div></div>
    <div class="stat"><div class="n">{at_ask}</div><div class="l">At Ask</div></div>
  </div>
</div>
"""

    # --- MARKET vs ASSESSMENT ---
    if market_summary and market_summary.get("matched", 0) > 0:
        ms = market_summary
        sar = ms["overall_sar"]
        sar_pct = (sar - 1) * 100
        sar_color = "#22543d" if sar_pct >= 0 else "#c53030"
        sar_arrow = "▲" if sar_pct > 0 else "▼"

        html += f"""<div class="section">
  <h2>Market vs BC Assessment</h2>
  <p style="margin-bottom:12px;color:#4a5568;font-size:13px">
    Matched {ms['matched']}/{ms['total_sales']} sold listings ({ms['match_rate']}%) to BC Assessment records.
    Shows where the market actually trades relative to assessed values.
  </p>
  <div class="stats">
    <div class="stat"><div class="n" style="color:{sar_color}">{sar:.3f}</div><div class="l">Median SAR</div></div>
    <div class="stat"><div class="n">{_fmt_price(ms['avg_sold_price'])}</div><div class="l">Avg Sold Price</div></div>
    <div class="stat"><div class="n">{_fmt_price(ms['avg_assessed'])}</div><div class="l">Avg Assessed</div></div>
    <div class="stat"><div class="n" style="color:#22543d">{ms['pct_above_assessment']}%</div><div class="l">Above Assessment</div></div>
    <div class="stat"><div class="n" style="color:#c53030">{ms['pct_below_assessment']}%</div><div class="l">Below Assessment</div></div>
  </div>"""

        # Monthly trend
        if ms.get("by_month"):
            html += """<h3 style="font-size:13px;color:#4a5568;margin:12px 0 8px">Monthly Trend (SAR)</h3>
  <div style="overflow-x:auto"><table>
  <tr><th>Month</th><th>Sales</th><th>SAR</th><th>vs Assessed</th><th>Avg Sold</th></tr>"""
            for m in ms["by_month"]:
                pct = (m["median_sar"] - 1) * 100
                clr = "#22543d" if pct >= 0 else "#c53030"
                arrow = "▲" if pct > 0 else "▼"
                html += f"""<tr>
  <td><strong>{m['month']}</strong></td><td>{m['count']}</td>
  <td style="color:{clr};font-weight:700">{m['median_sar']:.3f}</td>
  <td style="color:{clr}">{arrow} {abs(pct):.1f}%</td>
  <td>{_fmt_price(m['avg_sold'])}</td></tr>"""
            html += "</table></div>"

        # By area
        if ms.get("by_area"):
            html += """<h3 style="font-size:13px;color:#4a5568;margin:12px 0 8px">By Neighbourhood</h3>
  <div style="overflow-x:auto"><table>
  <tr><th>Area</th><th>Sales</th><th>SAR</th><th>vs Assessed</th><th>Avg Sold</th><th>Avg Assessed</th></tr>"""
            for a in ms["by_area"][:20]:
                pct = (a["median_sar"] - 1) * 100
                clr = "#22543d" if pct >= 0 else "#c53030"
                arrow = "▲" if pct > 0 else "▼"
                html += f"""<tr>
  <td><strong>{a['area']}</strong></td><td>{a['count']}</td>
  <td style="color:{clr};font-weight:700">{a['median_sar']:.3f}</td>
  <td style="color:{clr}">{arrow} {abs(pct):.1f}%</td>
  <td>{_fmt_price(a['avg_sold'])}</td><td>{_fmt_price(a['avg_assessed'])}</td></tr>"""
            html += "</table></div>"

        html += "</div>"

    # --- DETACHED SOLD TABLE ---
    if detached:
        det_sorted = sorted(detached, key=lambda l: l.get("sold_price") or 0, reverse=True)
        html += _render_table(det_sorted, "Detached Homes Sold", is_detached=True)

    # --- ATTACHED SOLD TABLE ---
    if attached:
        att_sorted = sorted(attached, key=lambda l: l.get("sold_price") or 0, reverse=True)
        html += _render_table(att_sorted, "Attached (Condos/Townhomes/Duplexes) Sold", is_detached=False)

    # --- PERMITS ---
    if new_permits:
        html += '<div class="section"><h2>New Building Permits This Week</h2>'
        for p in new_permits[:15]:
            ptype = p.get("type_of_work", "Permit")
            value = _fmt_price(p.get("project_value")) if p.get("project_value") else ""
            desc = (p.get("project_description") or "")[:150]
            html += f"""<div class="permit">
  <div class="ptype">{ptype}</div>
  <div><strong>{p.get("address", "")}</strong> {f"— {value}" if value else ""}</div>
  <div style="color:#718096;font-size:12px">{desc}</div>
</div>"""
        html += "</div>"

    # --- NEWS ---
    if news_articles:
        html += '<div class="section"><h2>Vancouver Real Estate News</h2>'
        for a in news_articles[:10]:
            html += f"""<div class="news-item">
  <a href="{a.get("url", "#")}">{a.get("title", "")}</a>
  <span class="src"> — {a.get("feed_name", "")}</span>
</div>"""
        html += "</div>"

    html += f"""<div class="footer">
  Generated {datetime.now().strftime("%Y-%m-%d %H:%M")} | Metro Vancouver Pricing Engine<br>
  Sources: MLS Paragon (Van East & West), City of Vancouver Open Data, STOREYS, Daily Hive, GVR
</div>
</body></html>"""

    return html


def _render_table(listings: list[dict], title: str, is_detached: bool) -> str:
    """Render a full-detail listing table with ALL fields."""
    count = len(listings)

    html = f'<div class="section"><h2>{title} ({count})</h2>'
    html += '<div style="overflow-x:auto"><table>'

    if is_detached:
        html += """<tr>
  <th>MLS#</th><th>Address</th><th>Area</th><th>Sold $</th><th>List $</th>
  <th>Diff</th><th>DOM</th><th>Bed</th><th>Bath</th><th>SqFt</th><th>$/SqFt</th>
  <th>Yr Blt</th><th>Style</th><th>Frontage</th><th>Depth</th><th>Zoning</th>
  <th>Postal</th>
</tr>"""
    else:
        html += """<tr>
  <th>MLS#</th><th>Address</th><th>Area</th><th>Sold $</th><th>List $</th>
  <th>Diff</th><th>DOM</th><th>Bed</th><th>Bath</th><th>SqFt</th><th>$/SqFt</th>
  <th>Yr Blt</th><th>Type</th><th>Maint Fee</th><th>Parking</th><th>Locker</th>
</tr>"""

    for l in listings:
        mls = l.get("mls_number", "")
        addr = l.get("address", "")
        area = l.get("sub_area_name", "")
        sold = l.get("sold_price")
        ask = l.get("list_price")
        diff_pct = l.get("price_diff_pct")
        dom = l.get("dom")
        beds = l.get("bedrooms")
        baths = l.get("bathrooms")
        sqft = l.get("floor_area")
        yr = l.get("year_built")
        ppsqft = _price_per_sqft(sold, sqft)

        # Diff styling
        if diff_pct is not None:
            if diff_pct > 0:
                diff_html = f'<span class="over">+{diff_pct:.1f}%</span>'
            elif diff_pct < 0:
                diff_html = f'<span class="under">{diff_pct:.1f}%</span>'
            else:
                diff_html = '<span class="at">0%</span>'
        else:
            diff_html = "—"

        # Property type tag
        ptype = l.get("property_type", "")
        if "Apartment" in ptype or "Condo" in ptype:
            tag = '<span class="tag tag-apt">APT</span>'
        elif "Townhouse" in ptype:
            tag = '<span class="tag tag-twn">TWN</span>'
        elif "Duplex" in ptype or "1/2" in ptype:
            tag = '<span class="tag tag-dup">DUP</span>'
        elif "HOUSE" in ptype or "House" in ptype:
            tag = '<span class="tag tag-det">DET</span>'
        else:
            tag = f'<span class="tag">{ptype[:6]}</span>'

        html += "<tr>"
        html += f'<td>{mls}</td>'
        html += f'<td><span class="addr">{addr}</span><br><span class="area">{area}</span></td>'
        html += f'<td>{l.get("city", "")[:6]}</td>'
        html += f'<td class="price">{_fmt_price(sold)}</td>'
        html += f'<td>{_fmt_price(ask)}</td>'
        html += f'<td>{diff_html}</td>'
        html += f'<td>{_fmt_num(dom)}</td>'
        html += f'<td>{_fmt_num(beds)}</td>'
        html += f'<td>{_fmt_num(baths)}</td>'
        html += f'<td>{_fmt_sqft(sqft) if sqft else "—"}</td>'
        html += f'<td>{ppsqft}</td>'
        html += f'<td>{_fmt_num(yr)}</td>'

        if is_detached:
            html += f'<td>{l.get("style") or "—"}</td>'
            html += f'<td>{l.get("frontage") or "—"}</td>'
            html += f'<td>{l.get("depth") or "—"}</td>'
            html += f'<td>{l.get("zoning") or "—"}</td>'
            html += f'<td>{l.get("postal_code") or "—"}</td>'
        else:
            html += f'<td>{tag}</td>'
            html += f'<td>{_fmt_money(l.get("maint_fee")) if l.get("maint_fee") else "—"}</td>'
            html += f'<td>{_fmt_num(l.get("parking"))}</td>'
            html += f'<td>{l.get("locker") or "—"}</td>'

        html += "</tr>"

    html += "</table></div></div>"
    return html
