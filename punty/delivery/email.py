"""Email delivery for notifications."""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

logger = logging.getLogger(__name__)


async def send_email(
    to_email: str,
    subject: str,
    body_html: str,
    body_text: Optional[str] = None,
) -> dict:
    """Send an email notification.

    Uses SMTP settings from app_settings table:
    - smtp_host: SMTP server hostname
    - smtp_port: SMTP server port (usually 587 for TLS)
    - smtp_user: SMTP username/email
    - smtp_password: SMTP password or app password
    - smtp_from: From email address (optional, defaults to smtp_user)

    Returns dict with status and any error message.
    """
    from punty.models.database import async_session
    from punty.models.settings import AppSettings
    from sqlalchemy import select

    # Load SMTP settings from database
    async with async_session() as db:
        settings = {}
        for key in ["smtp_host", "smtp_port", "smtp_user", "smtp_password", "smtp_from"]:
            result = await db.execute(select(AppSettings).where(AppSettings.key == key))
            setting = result.scalar_one_or_none()
            if setting:
                settings[key] = setting.value

    # Validate required settings
    required = ["smtp_host", "smtp_port", "smtp_user", "smtp_password"]
    missing = [k for k in required if not settings.get(k)]
    if missing:
        logger.warning(f"Email not sent - missing SMTP settings: {missing}")
        return {"status": "error", "message": f"Missing SMTP settings: {missing}"}

    smtp_host = settings["smtp_host"]
    smtp_port = int(settings["smtp_port"])
    smtp_user = settings["smtp_user"]
    smtp_password = settings["smtp_password"]
    smtp_from = settings.get("smtp_from", smtp_user)

    # Build email message
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = smtp_from
    msg["To"] = to_email

    # Add plain text part
    if body_text:
        msg.attach(MIMEText(body_text, "plain"))

    # Add HTML part
    msg.attach(MIMEText(body_html, "html"))

    try:
        # Connect and send
        logger.info(f"Sending email to {to_email}: {subject}")
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_from, to_email, msg.as_string())

        logger.info(f"Email sent successfully to {to_email}")
        return {"status": "sent", "to": to_email}

    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return {"status": "error", "message": str(e)}


def format_morning_prep_email(results: dict) -> tuple[str, str, str]:
    """Format the morning prep results into an email.

    Returns (subject, body_html, body_text) tuple.
    """
    # Determine overall status
    has_errors = bool(results.get("errors"))
    meetings_scraped = results.get("meetings_scraped", [])
    early_mail_generated = results.get("early_mail_generated", [])

    if has_errors:
        status_emoji = "⚠️"
        status_text = "Completed with errors"
    else:
        status_emoji = "✅"
        status_text = "Completed successfully"

    subject = f"{status_emoji} PuntyAI Morning Prep - {status_text}"

    # Build HTML body
    html_parts = [
        "<html><body style='font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;'>",
        f"<h1 style='color: #e91e63;'>{status_emoji} Morning Prep Report</h1>",
        f"<p style='color: #666;'>Started: {results.get('started_at', 'N/A')}<br>",
        f"Completed: {results.get('completed_at', 'N/A')}</p>",
        "<hr style='border: 1px solid #eee;'>",
    ]

    # Calendar summary
    html_parts.append(f"<h2>📅 Calendar</h2>")
    if results.get("calendar_scraped"):
        html_parts.append(f"<p style='color: green;'>✓ Scraped successfully - {results.get('meetings_found', 0)} meetings found</p>")
    else:
        html_parts.append("<p style='color: red;'>✗ Calendar scrape failed</p>")

    # Meetings scraped
    html_parts.append("<h2>🏇 Data Scraped</h2>")
    if meetings_scraped:
        html_parts.append("<ul style='color: green;'>")
        for venue in meetings_scraped:
            html_parts.append(f"<li>✓ {venue}</li>")
        html_parts.append("</ul>")
    else:
        html_parts.append("<p style='color: orange;'>No meetings scraped</p>")

    # Early mail generated
    html_parts.append("<h2>📧 Early Mail Generated</h2>")
    if early_mail_generated:
        html_parts.append("<ul style='color: green;'>")
        for venue in early_mail_generated:
            html_parts.append(f"<li>✓ {venue}</li>")
        html_parts.append("</ul>")
    else:
        html_parts.append("<p style='color: orange;'>No early mail generated</p>")

    # Errors
    errors = results.get("errors", [])
    if errors:
        html_parts.append("<h2 style='color: red;'>❌ Errors</h2>")
        html_parts.append("<ul style='color: red;'>")
        for error in errors:
            html_parts.append(f"<li>{error}</li>")
        html_parts.append("</ul>")

    # Footer
    html_parts.append("<hr style='border: 1px solid #eee;'>")
    html_parts.append("<p style='color: #999; font-size: 12px;'>")
    html_parts.append("<a href='https://app.punty.ai/review'>Review Early Mail</a> | ")
    html_parts.append("<a href='https://app.punty.ai/meets'>View Meetings</a>")
    html_parts.append("</p>")
    html_parts.append("</body></html>")

    body_html = "".join(html_parts)

    # Build plain text body
    text_parts = [
        f"Morning Prep Report - {status_text}",
        f"Started: {results.get('started_at', 'N/A')}",
        f"Completed: {results.get('completed_at', 'N/A')}",
        "",
        "CALENDAR",
        f"{'Scraped successfully' if results.get('calendar_scraped') else 'Failed'} - {results.get('meetings_found', 0)} meetings found",
        "",
        "DATA SCRAPED",
    ]
    for venue in meetings_scraped:
        text_parts.append(f"  - {venue}")
    if not meetings_scraped:
        text_parts.append("  No meetings scraped")

    text_parts.append("")
    text_parts.append("EARLY MAIL GENERATED")
    for venue in early_mail_generated:
        text_parts.append(f"  - {venue}")
    if not early_mail_generated:
        text_parts.append("  No early mail generated")

    if errors:
        text_parts.append("")
        text_parts.append("ERRORS")
        for error in errors:
            text_parts.append(f"  - {error}")

    text_parts.append("")
    text_parts.append("Review: https://app.punty.ai/review")

    body_text = "\n".join(text_parts)

    return subject, body_html, body_text
