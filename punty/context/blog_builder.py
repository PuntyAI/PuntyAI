"""Blog context builder — assembles all weekly data into an AI prompt context string."""

import json
import logging
from datetime import date, timedelta
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from punty.config import melb_today
from punty.models.settings import AppSettings

logger = logging.getLogger(__name__)


async def build_blog_context(db: AsyncSession) -> str:
    """Assemble all blog data into a context string for AI prompt injection.

    Loads weekly patterns, awards, ledger, future races, and news from
    AppSettings (stored by the Thursday night refresh job) or computes fresh
    if stale.
    """
    sections: list[str] = []
    today = melb_today()
    week_start = today - timedelta(days=7)

    sections.append(f"WEEK: {week_start.isoformat()} to {today.isoformat()}")
    sections.append("")

    # ── Weekly Awards ──────────────────────────────────────────────────
    awards = await _load_json_setting(db, "weekly_awards")
    if awards:
        sections.append("=== WEEKLY AWARDS DATA ===")
        if "jockey_of_the_week" in awards:
            j = awards["jockey_of_the_week"]
            sections.append(f"Jockey of the Week: {j['name']} — {j['wins']}/{j['bets']} winners, P&L ${j['pnl']:+.2f}, ROI {j['roi']}%")
        if "roughie_of_the_week" in awards:
            r = awards["roughie_of_the_week"]
            sections.append(f"Roughie of the Week: {r['horse']} at {r['venue']} R{r['race_number']} — ${r['odds']:.2f}, P&L ${r['pnl']:+.2f}")
        if "value_bomb" in awards:
            v = awards["value_bomb"]
            sections.append(f"Value Bomb: {v['horse']} at {v['venue']} R{v['race_number']} — ${v['odds']:.2f} {v['bet_type']}, P&L ${v['pnl']:+.2f}")
        if "track_to_watch" in awards:
            t = awards["track_to_watch"]
            sections.append(f"Track to Watch: {t['venue']} — {t['wins']}/{t['bets']} ({t['strike_rate']}% SR), P&L ${t['pnl']:+.2f}")
        if "wooden_spoon" in awards:
            w = awards["wooden_spoon"]
            sections.append(f"Wooden Spoon: {w['venue']} — {w['wins']}/{w['bets']} winners, P&L ${w['pnl']:+.2f}")
        if "power_rankings" in awards:
            pr = awards["power_rankings"]
            if pr.get("jockeys"):
                sections.append("\nJockey Power Rankings (Last 30 Days):")
                for i, j in enumerate(pr["jockeys"], 1):
                    sections.append(f"  {i}. {j['name']} — {j['wins']}/{j['bets']} ({j['strike_rate']}% SR), P&L ${j['pnl']:+.2f}")
            if pr.get("trainers"):
                sections.append("\nTrainer Power Rankings (Last 30 Days):")
                for i, t in enumerate(pr["trainers"], 1):
                    sections.append(f"  {i}. {t['name']} — {t['wins']}/{t['bets']} ({t['strike_rate']}% SR), P&L ${t['pnl']:+.2f}")
        sections.append("")

    # ── Weekly Ledger ──────────────────────────────────────────────────
    ledger = await _load_json_setting(db, "weekly_ledger")
    if ledger:
        sections.append("=== WEEKLY LEDGER DATA ===")
        tw = ledger.get("this_week", {})
        lw = ledger.get("last_week", {})
        sections.append(f"This Week: {tw.get('total_bets', 0)} bets, {tw.get('winners', 0)} winners ({tw.get('strike_rate', 0)}% SR)")
        sections.append(f"  Staked: ${tw.get('total_staked', 0):.2f}, P&L: ${tw.get('total_pnl', 0):+.2f}, ROI: {tw.get('roi', 0)}%")
        sections.append(f"Last Week: {lw.get('total_bets', 0)} bets, P&L: ${lw.get('total_pnl', 0):+.2f}, ROI: {lw.get('roi', 0)}%")
        sections.append(f"Trend: {ledger.get('trend', 'flat')} (change: ${ledger.get('pnl_change', 0):+.2f})")

        best = ledger.get("best_bet")
        if best:
            sections.append(f"Best Bet: {best['horse']} at {best['venue']} R{best['race_number']} — ${best['odds']:.2f} {best['bet_type']}, P&L ${best['pnl']:+.2f}")
        worst = ledger.get("worst_bet")
        if worst:
            sections.append(f"Worst Bet: {worst['horse']} at {worst['venue']} R{worst['race_number']} — P&L ${worst['pnl']:+.2f}")

        breakdown = ledger.get("bet_type_breakdown", [])
        if breakdown:
            sections.append("\nBet Type Breakdown This Week:")
            for bt in breakdown:
                sections.append(f"  {bt['bet_type']}: {bt['wins']}/{bt['bets']} ({bt['strike_rate']}% SR), P&L ${bt['pnl']:+.2f}, ROI {bt['roi']}%")

        streak = ledger.get("streak", {})
        if streak.get("count", 0) > 0:
            sections.append(f"\nCurrent Streak: {streak['count']} {streak['type']} day(s) in a row")
        sections.append("")

    # ── Pattern Insights ───────────────────────────────────────────────
    patterns = await _load_json_setting(db, "weekly_patterns")
    if patterns:
        sections.append("=== DEEP PATTERN INSIGHTS ===")
        sections.append("(Pick the most interesting/surprising finding for Pattern Spotlight)")
        for dim_name, entries in patterns.items():
            if not isinstance(entries, list) or not entries:
                continue
            sections.append(f"\n{dim_name.upper().replace('_', ' ')}:")
            # Show top 5 and bottom 2 for each dimension
            for entry in entries[:5]:
                sections.append(f"  {entry['insight_text']}")
            if len(entries) > 7:
                sections.append("  ...")
                for entry in entries[-2:]:
                    sections.append(f"  {entry['insight_text']}")
        sections.append("")

    # ── Future Races (Crystal Ball) ─────────────────────────────────────
    try:
        from punty.models.future_race import FutureRace, FutureNomination
        from sqlalchemy.orm import selectinload
        future_q = (
            select(FutureRace)
            .options(selectinload(FutureRace.nominations))
            .where(FutureRace.date > today)
            .where(FutureRace.group_level.in_(["Group 1", "Group 2"]))
            .order_by(FutureRace.date)
            .limit(5)
        )
        future_result = await db.execute(future_q)
        future_races = future_result.scalars().all()
        if future_races:
            sections.append("=== UPCOMING GROUP RACES (CRYSTAL BALL) ===")
            for fr in future_races:
                # Races within 2 days have declared fields (barriers, jockeys, weights)
                days_away = (fr.date - today).days
                is_declared = days_away <= 2
                field_label = "Declared Field" if is_declared else "Nominations"

                sections.append(f"\n{fr.race_name} — {fr.venue} ({fr.date.isoformat()}) {'[FIELDS DECLARED]' if is_declared else ''}")
                if fr.distance:
                    sections.append(f"  Distance: {fr.distance}m")
                if fr.prize_money:
                    sections.append(f"  Prize: ${fr.prize_money:,}")
                if fr.nominations:
                    sections.append(f"  {field_label} ({len(fr.nominations)}):")
                    for nom in fr.nominations[:15]:
                        parts = [f"    {nom.horse_name}"]
                        if nom.barrier and is_declared:
                            parts.append(f"(Bar {nom.barrier})")
                        if nom.jockey:
                            parts.append(f"(J: {nom.jockey})")
                        if nom.trainer:
                            parts.append(f"(T: {nom.trainer})")
                        if nom.weight and is_declared:
                            parts.append(f"{nom.weight}kg")
                        if nom.career_record:
                            parts.append(f"Record: {nom.career_record}")
                        sections.append(" ".join(parts))
                    if len(fr.nominations) > 15:
                        sections.append(f"    ... and {len(fr.nominations) - 15} more")
            sections.append("")
    except Exception as e:
        logger.debug(f"Future races context failed: {e}")

    # ── News Headlines ─────────────────────────────────────────────────
    news = await _load_json_setting(db, "recent_news_headlines")
    if news and isinstance(news, list):
        sections.append("=== RECENT RACING NEWS (AROUND THE TRAPS) ===")
        sections.append("(Use 2-3 of these as topical hooks. Add your own spin.)")
        for headline in news[:15]:
            title = headline.get("title", "")
            source = headline.get("source", "")
            date_str = headline.get("date", "")
            snippet = headline.get("snippet", "")
            line = f"- [{source}] {title}"
            if date_str:
                line += f" ({date_str})"
            sections.append(line)
            if snippet:
                sections.append(f"  {snippet[:150]}")
        sections.append("")

    context = "\n".join(sections)
    logger.info(f"Blog context built: {len(context)} chars, {len(sections)} lines")
    return context


async def _load_json_setting(db: AsyncSession, key: str) -> Optional[dict | list]:
    """Load a JSON value from AppSettings."""
    result = await db.execute(
        select(AppSettings).where(AppSettings.key == key)
    )
    setting = result.scalar_one_or_none()
    if setting and setting.value:
        try:
            return json.loads(setting.value)
        except (json.JSONDecodeError, TypeError):
            pass
    return None
