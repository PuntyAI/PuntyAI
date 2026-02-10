"""Strategy performance aggregator for RAG betting intelligence.

Computes per-bet-type performance stats from settled picks,
populates PatternInsight table, and builds formatted context
for injection into Early Mail generation prompts.
"""

import json
import logging
from datetime import timedelta
from typing import Any, Optional

from sqlalchemy import select, func, case, and_, or_, delete
from sqlalchemy.ext.asyncio import AsyncSession

from punty.config import melb_today, melb_now_naive
from punty.memory.models import PatternInsight
from punty.models.pick import Pick
from punty.models.meeting import Meeting, Race

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Aggregation queries
# ──────────────────────────────────────────────

async def aggregate_bet_type_performance(
    db: AsyncSession,
    window_days: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Aggregate performance by bet type.

    Returns list of dicts with: category, sub_type, bets, winners,
    strike_rate, staked, returned, pnl, roi, avg_odds, best_win_pnl.
    """
    date_filter = []
    if window_days:
        cutoff = melb_today() - timedelta(days=window_days)
        date_filter.append(Pick.settled_at >= cutoff)

    results: list[dict[str, Any]] = []

    # --- SELECTIONS by bet_type ---
    sel_q = (
        select(
            Pick.bet_type,
            func.count(Pick.id).label("bets"),
            func.sum(case((Pick.hit == True, 1), else_=0)).label("winners"),
            func.sum(Pick.bet_stake).label("staked"),
            func.sum(Pick.pnl).label("pnl"),
            func.avg(Pick.odds_at_tip).label("avg_odds"),
            func.max(Pick.pnl).label("best_win"),
        )
        .where(
            Pick.settled == True,
            Pick.pick_type == "selection",
            Pick.bet_type.isnot(None),
            Pick.bet_type != "exotics_only",
            *date_filter,
        )
        .group_by(Pick.bet_type)
    )
    for row in (await db.execute(sel_q)).all():
        bt, bets, winners, staked, pnl, avg_odds, best = row
        staked = float(staked or 0)
        pnl = float(pnl or 0)
        winners = int(winners or 0)
        results.append(_make_stat(
            "selection", (bt or "win").replace("_", " ").title(),
            bets, winners, staked, pnl, float(avg_odds or 0), float(best or 0),
        ))

    # --- EXOTICS by exotic_type ---
    ex_q = (
        select(
            Pick.exotic_type,
            func.count(Pick.id),
            func.sum(case((Pick.hit == True, 1), else_=0)),
            func.sum(Pick.exotic_stake),
            func.sum(Pick.pnl),
            func.max(Pick.pnl),
        )
        .where(
            Pick.settled == True,
            Pick.pick_type == "exotic",
            *date_filter,
        )
        .group_by(Pick.exotic_type)
    )
    # Merge rows that normalise to the same exotic name
    exotic_merged: dict[str, list] = {}
    for row in (await db.execute(ex_q)).all():
        etype, bets, winners, staked, pnl, best = row
        name = _normalise_exotic(etype or "Exotic")
        if name in exotic_merged:
            m = exotic_merged[name]
            m[0] += bets; m[1] += int(winners or 0)
            m[2] += float(staked or 0); m[3] += float(pnl or 0)
            m[4] = max(m[4], float(best or 0))
        else:
            exotic_merged[name] = [bets, int(winners or 0), float(staked or 0), float(pnl or 0), float(best or 0)]
    for name, (bets, winners, staked, pnl, best) in exotic_merged.items():
        results.append(_make_stat("exotic", name, bets, winners, staked, pnl, 0, best))

    # --- SEQUENCES by type + variant ---
    seq_q = (
        select(
            Pick.sequence_type,
            Pick.sequence_variant,
            func.count(Pick.id),
            func.sum(case((Pick.hit == True, 1), else_=0)),
            func.sum(Pick.exotic_stake),
            func.sum(Pick.pnl),
            func.max(Pick.pnl),
        )
        .where(
            Pick.settled == True,
            Pick.pick_type == "sequence",
            *date_filter,
        )
        .group_by(Pick.sequence_type, Pick.sequence_variant)
    )
    for row in (await db.execute(seq_q)).all():
        stype, svar, bets, winners, staked, pnl, best = row
        staked = float(staked or 0)
        pnl = float(pnl or 0)
        winners = int(winners or 0)
        label = f"{(stype or 'Sequence').replace('_', ' ').title()} ({(svar or 'all').title()})"
        results.append(_make_stat(
            "sequence", label,
            bets, winners, staked, pnl, 0, float(best or 0),
        ))

    # --- BIG3 MULTI ---
    b3_q = (
        select(
            func.count(Pick.id),
            func.sum(case((Pick.hit == True, 1), else_=0)),
            func.sum(Pick.exotic_stake),
            func.sum(Pick.pnl),
            func.max(Pick.pnl),
        )
        .where(
            Pick.settled == True,
            Pick.pick_type == "big3_multi",
            *date_filter,
        )
    )
    row = (await db.execute(b3_q)).one()
    if row[0] and row[0] > 0:
        bets, winners, staked, pnl, best = row
        staked = float(staked or 0)
        pnl = float(pnl or 0)
        winners = int(winners or 0)
        results.append(_make_stat(
            "big3_multi", "Big 3 Multi",
            bets, winners, staked, pnl, 0, float(best or 0),
        ))

    return results


async def aggregate_tip_rank_performance(
    db: AsyncSession,
    window_days: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Aggregate selection performance by tip rank (1=top pick ... 4=roughie)."""
    date_filter = []
    if window_days:
        cutoff = melb_today() - timedelta(days=window_days)
        date_filter.append(Pick.settled_at >= cutoff)

    q = (
        select(
            Pick.tip_rank,
            func.count(Pick.id),
            func.sum(case((Pick.hit == True, 1), else_=0)),
            func.sum(Pick.bet_stake),
            func.sum(Pick.pnl),
            func.avg(Pick.odds_at_tip),
        )
        .where(
            Pick.settled == True,
            Pick.pick_type == "selection",
            Pick.tip_rank.isnot(None),
            *date_filter,
        )
        .group_by(Pick.tip_rank)
        .order_by(Pick.tip_rank)
    )

    rank_labels = {1: "Top Pick", 2: "2nd Pick", 3: "3rd Pick", 4: "Roughie"}
    results = []
    for row in (await db.execute(q)).all():
        rank, bets, winners, staked, pnl, avg_odds = row
        staked = float(staked or 0)
        pnl = float(pnl or 0)
        winners = int(winners or 0)
        results.append({
            "rank": rank,
            "label": rank_labels.get(rank, f"Rank {rank}"),
            "bets": bets,
            "winners": winners,
            "strike_rate": round(winners / bets * 100, 1) if bets else 0,
            "staked": round(staked, 2),
            "pnl": round(pnl, 2),
            "roi": round(pnl / staked * 100, 1) if staked else 0,
            "avg_odds": round(float(avg_odds or 0), 2),
        })
    return results


# ──────────────────────────────────────────────
# PatternInsight population
# ──────────────────────────────────────────────

async def populate_pattern_insights(db: AsyncSession) -> int:
    """Refresh pattern_insights table with current aggregated stats.

    Called after settlement. Populates the previously-empty table with
    bet_type_overall, bet_type_30d, and tip_rank pattern rows.
    """
    await db.execute(delete(PatternInsight))

    now = melb_now_naive()
    count = 0

    for window, ptype in [(None, "bet_type_overall"), (30, "bet_type_30d")]:
        stats = await aggregate_bet_type_performance(db, window_days=window)
        for s in stats:
            db.add(PatternInsight(
                pattern_type=ptype,
                pattern_key=f"{s['category']}_{s['sub_type'].lower().replace(' ', '_')}",
                sample_count=s["bets"],
                hit_rate=s["strike_rate"],
                avg_pnl=s["pnl"] / s["bets"] if s["bets"] else 0,
                avg_odds=s["avg_odds"],
                insight_text=_insight_text(s, "all-time" if window is None else "last 30 days"),
                conditions_json=json.dumps({"window": ptype, **s}),
                created_at=now,
                updated_at=now,
            ))
            count += 1

    ranks = await aggregate_tip_rank_performance(db)
    for r in ranks:
        db.add(PatternInsight(
            pattern_type="tip_rank",
            pattern_key=f"rank_{r['rank']}",
            sample_count=r["bets"],
            hit_rate=r["strike_rate"],
            avg_pnl=r["pnl"] / r["bets"] if r["bets"] else 0,
            avg_odds=r["avg_odds"],
            insight_text=f"{r['label']}: {r['strike_rate']}% SR at avg ${r['avg_odds']:.2f}, {r['roi']:+.1f}% ROI",
            conditions_json=json.dumps(r),
            created_at=now,
            updated_at=now,
        ))
        count += 1

    await db.flush()
    logger.info(f"Populated {count} pattern insights")
    return count


# ──────────────────────────────────────────────
# Strategy context builder (for prompt injection)
# ──────────────────────────────────────────────

async def build_strategy_context(db: AsyncSession, **_kw: Any) -> str:
    """Build the 'Your Betting Track Record' block for AI prompt injection.

    Returns formatted markdown string with actual $ figures, ROI per bet type,
    strategy directives, and ROI targets.
    """
    overall = await aggregate_bet_type_performance(db)
    if not overall:
        return ""

    parts: list[str] = [
        "## YOUR BETTING TRACK RECORD",
        "These are your ACTUAL results. Real money. Real returns. Use this to adjust your strategy.",
        "",
    ]

    # Grand totals
    total_staked = sum(s["staked"] for s in overall)
    total_pnl = sum(s["pnl"] for s in overall)
    total_bets = sum(s["bets"] for s in overall)
    total_winners = sum(s["winners"] for s in overall)
    overall_roi = round(total_pnl / total_staked * 100, 1) if total_staked else 0
    overall_sr = round(total_winners / total_bets * 100, 1) if total_bets else 0

    parts.append(
        f"**ALL-TIME**: {total_bets} bets, {total_winners} winners ({overall_sr}%), "
        f"${total_staked:.2f} staked, {_pnl_str(total_pnl)} P&L ({overall_roi:+.1f}% ROI)"
    )
    parts.append("")

    # Per-bet-type scorecard
    parts.append("### Bet Type Scorecard (All-Time)")
    for cat, label in [
        ("selection", "Selections"), ("exotic", "Exotics"),
        ("sequence", "Sequences"), ("big3_multi", "Multi"),
    ]:
        items = sorted(
            [s for s in overall if s["category"] == cat],
            key=lambda x: x["pnl"], reverse=True,
        )
        if not items:
            continue
        parts.append(f"**{label}:**")
        for s in items:
            flag = "PROFITABLE" if s["roi"] > 0 else "LOSING" if s["roi"] < -10 else "BREAKEVEN"
            parts.append(
                f"- {s['sub_type']}: {s['bets']} bets, {s['strike_rate']}% SR, "
                f"${s['staked']:.0f} staked, {_pnl_str(s['pnl'])} P&L "
                f"({s['roi']:+.1f}% ROI) [{flag}]"
            )
    parts.append("")

    # 30-day rolling
    recent = await aggregate_bet_type_performance(db, window_days=30)
    if recent:
        r_pnl = sum(s["pnl"] for s in recent)
        r_staked = sum(s["staked"] for s in recent)
        r_bets = sum(s["bets"] for s in recent)
        r_roi = round(r_pnl / r_staked * 100, 1) if r_staked else 0
        parts.append(f"### Last 30 Days: {r_bets} bets, {_pnl_str(r_pnl)} P&L ({r_roi:+.1f}% ROI)")
        improving = [s["sub_type"] for s in recent if s["roi"] > 0]
        declining = [s["sub_type"] for s in recent if s["roi"] < -15 and s["bets"] >= 5]
        if improving:
            parts.append(f"Trending UP: {', '.join(improving)}")
        if declining:
            parts.append(f"Trending DOWN: {', '.join(declining)}")
        parts.append("")

    # Tip rank calibration
    ranks = await aggregate_tip_rank_performance(db)
    if ranks:
        parts.append("### Pick Rank Performance")
        for r in ranks:
            parts.append(
                f"- {r['label']}: {r['strike_rate']}% SR at avg ${r['avg_odds']:.2f}, "
                f"{r['roi']:+.1f}% ROI ({r['bets']} bets)"
            )
        parts.append("")

    # Strategy directives
    parts.append("### STRATEGY DIRECTIVES (Based on Actual Results)")
    for d in _generate_directives(overall, recent, ranks):
        parts.append(f"- {d}")
    parts.append("")

    # ROI targets
    parts.append("### ROI TARGETS")
    parts.append("Your job is to get positive ROI on EVERY bet type. Targets:")
    parts.append("- Selections: Target +5% ROI minimum")
    parts.append("- Exotics: Target breakeven (0% ROI)")
    parts.append("- Sequences: Fewer, higher-confidence combos")
    parts.append("- Big 3 Multi: Target 10%+ strike rate")
    parts.append("")

    return "\n".join(parts)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _make_stat(
    category: str, sub_type: str, bets: int, winners: int,
    staked: float, pnl: float, avg_odds: float, best_win: float,
) -> dict[str, Any]:
    return {
        "category": category,
        "sub_type": sub_type,
        "bets": bets,
        "winners": winners,
        "strike_rate": round(winners / bets * 100, 1) if bets else 0,
        "staked": round(staked, 2),
        "returned": round(staked + pnl, 2),
        "pnl": round(pnl, 2),
        "roi": round(pnl / staked * 100, 1) if staked else 0,
        "avg_odds": round(avg_odds, 2),
        "best_win_pnl": round(best_win, 2),
    }


def _pnl_str(pnl: float) -> str:
    return f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"


def _insight_text(stat: dict, window: str) -> str:
    roi = stat["roi"]
    if roi > 5:
        tone = "PROFITABLE"
    elif roi > -5:
        tone = "BREAKEVEN"
    elif roi > -20:
        tone = "slightly unprofitable"
    else:
        tone = "LOSING"
    return (
        f"{stat['sub_type']} bets are {tone} ({window}): "
        f"{stat['strike_rate']}% SR over {stat['bets']} bets, "
        f"{_pnl_str(stat['pnl'])} P&L, {roi:+.1f}% ROI"
    )


def _normalise_exotic(raw: str) -> str:
    low = raw.lower().strip()
    if "trifecta" in low and "box" in low:
        return "Trifecta Box"
    if "exacta" in low and "standout" in low:
        return "Exacta Standout"
    if "trifecta" in low and "standout" in low:
        return "Trifecta Standout"
    if "first" in low and ("four" in low or "4" in low):
        return "First Four"
    if "quinella" in low:
        return "Quinella"
    if "exacta" in low:
        return "Exacta"
    if "trifecta" in low:
        return "Trifecta"
    return raw.title()


def _generate_directives(
    overall: list[dict], recent: list[dict], ranks: list[dict],
) -> list[str]:
    """Generate actionable strategy directives from performance data."""
    directives: list[str] = []

    profitable = sorted([s for s in overall if s["roi"] > 0], key=lambda x: x["roi"], reverse=True)
    losing = sorted([s for s in overall if s["roi"] < -10 and s["bets"] >= 10], key=lambda x: x["roi"])

    if profitable:
        b = profitable[0]
        directives.append(
            f"LEAN INTO {b['sub_type'].upper()} bets — your best at {b['roi']:+.1f}% ROI over {b['bets']} bets"
        )
    if losing:
        w = losing[0]
        directives.append(
            f"REDUCE {w['sub_type'].upper()} bets — losing at {w['roi']:+.1f}% ROI. "
            "Consider switching to Place or Each Way instead"
        )

    # Tip rank insights
    if ranks:
        top = next((r for r in ranks if r["rank"] == 1), None)
        roughie = next((r for r in ranks if r["rank"] == 4), None)
        if top and top["roi"] < 0:
            directives.append(
                f"YOUR TOP PICKS return {top['roi']:+.1f}% ROI — be more selective with #1 confidence calls"
            )
        if roughie and roughie["roi"] > 0:
            directives.append(
                f"ROUGHIES are profitable at {roughie['roi']:+.1f}% ROI — keep hunting value at big odds"
            )
        elif roughie and roughie["strike_rate"] < 5:
            directives.append(
                f"ROUGHIES hitting only {roughie['strike_rate']}% — consider 'Exotics only' more often"
            )

    # Place vs Win comparison
    win_s = next((s for s in overall if s["sub_type"] == "Win"), None)
    place_s = next((s for s in overall if s["sub_type"] == "Place"), None)
    if win_s and place_s and place_s["roi"] > win_s["roi"] + 10:
        directives.append(
            f"PLACE outperforming WIN ({place_s['roi']:+.1f}% vs {win_s['roi']:+.1f}% ROI) — "
            "use Place as your default for close races"
        )

    # Each Way
    ew_s = next((s for s in overall if s["sub_type"] == "Each Way"), None)
    if ew_s:
        if ew_s["roi"] > 0:
            directives.append(f"EACH WAY working at {ew_s['roi']:+.1f}% ROI — keep using on $6-$15 value plays")
        elif ew_s["roi"] < -20:
            directives.append(f"EACH WAY bleeding at {ew_s['roi']:+.1f}% ROI — reserve for genuine value only ($8+)")

    # Trend comparison
    if recent and overall:
        r_pnl = sum(s["pnl"] for s in recent)
        o_pnl = sum(s["pnl"] for s in overall)
        if r_pnl > 0 and o_pnl < 0:
            directives.append("MOMENTUM POSITIVE — recent form improving, keep it up")
        elif r_pnl < 0 and o_pnl > 0:
            directives.append("RECENT SLUMP — tighten selections and reduce exotic exposure")

    if not directives:
        directives.append("Keep building the sample — need more data for clear patterns")

    return directives
