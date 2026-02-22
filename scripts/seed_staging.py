"""Seed staging database from production.

Copies the production DB to staging, then sanitizes all sensitive API keys.
Run on server: cd /opt/puntyai && source venv/bin/activate && python scripts/seed_staging.py
"""

import shutil
import sqlite3
import sys
from pathlib import Path

PROD_DB = Path("/opt/puntyai/data/punty.db")
STAGING_DB = Path("/opt/puntyai-staging/data/punty.db")

# Keys to blank out in staging (prevent real API calls)
SENSITIVE_KEYS = [
    "anthropic_api_key",
    "openai_api_key",
    "twitter_api_key",
    "twitter_api_secret",
    "twitter_access_token",
    "twitter_access_secret",
    "resend_api_key",
    "telegram_bot_token",
    "telegram_owner_id",
    "betfair_username",
    "betfair_password",
    "betfair_app_key",
    "facebook_page_access_token",
]


def main():
    if not PROD_DB.exists():
        print(f"ERROR: Production DB not found at {PROD_DB}")
        sys.exit(1)

    # Backup existing staging DB
    if STAGING_DB.exists():
        backup = STAGING_DB.with_suffix(".db.bak")
        shutil.copy2(STAGING_DB, backup)
        print(f"Backed up existing staging DB to {backup}")

    # Ensure staging data directory exists
    STAGING_DB.parent.mkdir(parents=True, exist_ok=True)

    # Copy production DB to staging
    shutil.copy2(PROD_DB, STAGING_DB)
    print(f"Copied {PROD_DB} -> {STAGING_DB}")

    # Sanitize sensitive keys
    conn = sqlite3.connect(STAGING_DB)
    cursor = conn.cursor()

    blanked = 0
    for key in SENSITIVE_KEYS:
        cursor.execute(
            "UPDATE app_settings SET value = '' WHERE key = ?", (key,)
        )
        if cursor.rowcount > 0:
            blanked += 1
            print(f"  Blanked: {key}")

    conn.commit()
    print(f"\nSanitized {blanked} sensitive keys")

    # Print row counts for verification
    tables = ["meetings", "races", "runners", "content", "picks", "app_settings"]
    print("\nRow counts:")
    for table in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"  {table}: {count:,}")
        except sqlite3.OperationalError:
            print(f"  {table}: (table not found)")

    conn.close()
    print(f"\nDone! Restart staging: systemctl restart puntyai-staging")


if __name__ == "__main__":
    main()
