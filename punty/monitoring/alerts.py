"""Telegram alert composition and dispatch for performance monitoring."""

import logging
from datetime import date

from punty.config import melb_today
from punty.models.database import async_session
from punty.monitoring.performance import (
    compute_daily_digest,
    compute_rolling_comparison,
    check_regressions,
    check_calibration_drift,
)

logger = logging.getLogger(__name__)


def _fmt_pnl(value: float) -> str:
    """Format a P&L value with sign before dollar sign: +$10.00 or -$20.00."""
    if value >= 0:
        return f"+${value:.2f}"
    return f"-${abs(value):.2f}"


def format_daily_digest(
    digest: dict,
    comparison: dict,
    regressions: list[str],
    calibration: str | None,
) -> str:
    """Format the daily P&L digest as a Telegram message."""
    lines = []

    # Header
    d = digest.get("date", melb_today().isoformat())
    lines.append(f"Daily P&L Report — {d}")
    lines.append("")

    # Today's totals
    total_pnl = digest["total_pnl"]
    lines.append(
        f"Today: {_fmt_pnl(total_pnl)} "
        f"({digest['total_bets']} bets, {digest['total_winners']} winners, "
        f"{digest['total_strike_rate']}% SR)"
    )
    lines.append("")

    # By product type
    by_product = digest.get("by_product", {})
    if by_product:
        lines.append("By Type:")
        type_labels = {
            "selection": "Selections",
            "exotic": "Exotics",
            "sequence": "Sequences",
            "big3_multi": "Big3 Multi",
        }
        for ptype, label in type_labels.items():
            if ptype in by_product:
                p = by_product[ptype]
                lines.append(
                    f"  {label}: {_fmt_pnl(p['pnl'])} "
                    f"({p['bets']}/{p['winners']}, {p['strike_rate']}%)"
                )

    # Punty's Pick
    pp = digest.get("puntys_pick")
    if pp:
        lines.append(
            f"  Punty's Pick: {_fmt_pnl(pp['pnl'])} "
            f"({pp['bets']}/{pp['winners']}, {pp['strike_rate']}%)"
        )
    lines.append("")

    # Rolling comparison
    c = comparison
    if c["current_7d_bets"] > 0 or c["previous_7d_bets"] > 0:
        lines.append(
            f"Rolling 7-Day: {_fmt_pnl(c['current_7d_pnl'])} "
            f"vs prev {_fmt_pnl(c['previous_7d_pnl'])} "
            f"({_fmt_pnl(c['pnl_delta'])})"
        )
        sds = "+" if c["sr_delta"] >= 0 else ""
        lines.append(
            f"Strike Rate: {c['current_strike_rate']}% "
            f"vs prev {c['previous_strike_rate']}% "
            f"({sds}{c['sr_delta']}pp)"
        )
        lines.append("")

    # Regressions
    if regressions:
        lines.append("\u26a0\ufe0f REGRESSIONS:")
        for r in regressions:
            lines.append(f"  \u2022 {r}")
        lines.append("")
    else:
        lines.append("No regressions detected.")
        lines.append("")

    # Calibration
    if calibration:
        lines.append(calibration)

    return "\n".join(lines).strip()


async def send_daily_digest(app) -> None:
    """Compose and send daily P&L digest via Telegram.

    Called by APScheduler at 23:00 AEDT.
    """
    telegram_bot = getattr(app.state, "telegram_bot", None)
    if not telegram_bot or not telegram_bot.is_running():
        logger.info("Telegram bot not running, skipping daily digest")
        return

    try:
        async with async_session() as db:
            today = melb_today()
            digest = await compute_daily_digest(db, today)

            # Skip if no bets today
            if digest["total_bets"] == 0:
                logger.info("No bets today, skipping daily digest")
                return

            comparison = await compute_rolling_comparison(db)
            regressions = await check_regressions(db)
            calibration = await check_calibration_drift(db)

        msg = format_daily_digest(digest, comparison, regressions, calibration)
        await telegram_bot.send_alert(msg)
        logger.info(f"Daily digest sent: {digest['total_bets']} bets, ${digest['total_pnl']:.2f}")

    except Exception as e:
        logger.error(f"Failed to send daily digest: {e}", exc_info=True)


async def send_regression_alert(app, message: str) -> None:
    """Send an immediate regression alert via Telegram."""
    telegram_bot = getattr(app.state, "telegram_bot", None)
    if not telegram_bot or not telegram_bot.is_running():
        return

    try:
        await telegram_bot.send_alert(message)
        logger.info(f"Regression alert sent: {message[:80]}...")
    except Exception as e:
        logger.error(f"Failed to send regression alert: {e}")
