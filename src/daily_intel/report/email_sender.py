"""Send daily intelligence report via email."""
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import date
from typing import Optional

from ..config import EMAIL_FROM, EMAIL_TO, SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS

logger = logging.getLogger(__name__)


def send_report(html_content: str, subject: Optional[str] = None) -> bool:
    """Send the HTML report via email."""
    if not all([EMAIL_FROM, EMAIL_TO, SMTP_USER, SMTP_PASS]):
        logger.warning("Email not configured — skipping send. Set INTEL_EMAIL_* env vars.")
        return False

    today = date.today().strftime("%b %d, %Y")
    subject = subject or f"Daily Market Intel - {today}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO

    # Plain text fallback
    text_part = MIMEText("Your daily market intelligence report is attached as HTML.", "plain")
    html_part = MIMEText(html_content, "html")

    msg.attach(text_part)
    msg.attach(html_part)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        logger.info(f"Report emailed to {EMAIL_TO}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False
