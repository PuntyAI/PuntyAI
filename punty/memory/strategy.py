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
from punty.models.meeting import Meeting, Race, Runner

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


async def aggregate_puntys_pick_performance(
    db: AsyncSession,
    window_days: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    """Aggregate performance of ALL Punty's Picks (is_puntys_pick=True).

    Includes both selection and exotic Punty's Picks.
    Uses bet_stake for selections, exotic_stake for exotics.
    """
    date_filter = []
    if window_days:
        cutoff = melb_today() - timedelta(days=window_days)
        date_filter.append(Pick.settled_at >= cutoff)

    # Selection Punty's Picks
    sel_q = (
        select(
            func.count(Pick.id),
            func.sum(case((Pick.hit == True, 1), else_=0)),
            func.sum(Pick.bet_stake),
            func.sum(Pick.pnl),
            func.avg(Pick.odds_at_tip),
        )
        .where(
            Pick.settled == True,
            Pick.pick_type == "selection",
            Pick.is_puntys_pick == True,
            Pick.bet_type != "exotics_only",
            *date_filter,
        )
    )
    sel_row = (await db.execute(sel_q)).one()

    # Exotic Punty's Picks
    exo_q = (
        select(
            func.count(Pick.id),
            func.sum(case((Pick.hit == True, 1), else_=0)),
            func.sum(Pick.exotic_stake),
            func.sum(Pick.pnl),
        )
        .where(
            Pick.settled == True,
            Pick.pick_type == "exotic",
            Pick.is_puntys_pick == True,
            *date_filter,
        )
    )
    exo_row = (await db.execute(exo_q)).one()

    sel_bets = sel_row[0] or 0
    exo_bets = exo_row[0] or 0
    total_bets = sel_bets + exo_bets

    if total_bets == 0:
        return None

    sel_winners = int(sel_row[1] or 0)
    exo_winners = int(exo_row[1] or 0)
    sel_staked = float(sel_row[2] or 0)
    exo_staked = float(exo_row[2] or 0)
    sel_pnl = float(sel_row[3] or 0)
    exo_pnl = float(exo_row[3] or 0)
    sel_avg_odds = float(sel_row[4] or 0)

    total_winners = sel_winners + exo_winners
    total_staked = sel_staked + exo_staked
    total_pnl = sel_pnl + exo_pnl

    return {
        "bets": total_bets,
        "winners": total_winners,
        "strike_rate": round(total_winners / total_bets * 100, 1) if total_bets else 0,
        "staked": round(total_staked, 2),
        "pnl": round(total_pnl, 2),
        "roi": round(total_pnl / total_staked * 100, 1) if total_staked else 0,
        "avg_odds": round(sel_avg_odds, 2),
        # Breakdown
        "selection_bets": sel_bets,
        "selection_winners": sel_winners,
        "exotic_bets": exo_bets,
        "exotic_winners": exo_winners,
    }


async def get_recent_results_with_context(
    db: AsyncSession, limit: int = 15,
) -> list[str]:
    """Get recent settled picks across ALL types with win/loss context.

    Returns formatted strings showing what won, what lost, and key details.
    """
    from sqlalchemy.orm import selectinload

    q = (
        select(Pick)
        .where(Pick.settled == True)
        .order_by(Pick.settled_at.desc())
        .limit(limit)
    )
    result = await db.execute(q)
    picks = result.scalars().all()
    if not picks:
        return []

    lines = []
    for p in picks:
        hit = "WIN" if p.hit else "LOSS"
        pnl = p.pnl or 0
        pnl_str = f"+${pnl:.2f}" if pnl > 0 else f"${pnl:.2f}"

        if p.pick_type == "selection":
            rank_label = {1: "Top", 2: "2nd", 3: "3rd", 4: "Roughie"}.get(p.tip_rank, "?")
            bet = (p.bet_type or "win").replace("_", " ").title()
            odds = p.odds_at_tip or 0
            pp = " [PP]" if p.is_puntys_pick else ""
            # Get runner finish position
            runner_result = await db.execute(
                select(Runner.finish_position, Runner.result_margin)
                .where(
                    Runner.race_id == f"{p.meeting_id}-r{p.race_number}",
                    Runner.saddlecloth == p.saddlecloth,
                )
            )
            rr = runner_result.one_or_none()
            pos = f"Finished {rr[0]}" if rr and rr[0] else "?"
            margin = f" ({rr[1]})" if rr and rr[1] else ""
            lines.append(
                f"- [{hit}] {rank_label}{pp}: {p.horse_name} @ ${odds:.2f} {bet} "
                f"→ {pos}{margin} | {pnl_str}"
            )

        elif p.pick_type == "exotic":
            etype = p.exotic_type or "Exotic"
            stake = p.exotic_stake or 20
            lines.append(
                f"- [{hit}] {etype}: runners {p.exotic_runners} — ${stake:.0f} stake | {pnl_str}"
            )

        elif p.pick_type == "sequence":
            stype = (p.sequence_type or "sequence").replace("_", " ").title()
            svar = (p.sequence_variant or "").title()
            stake = p.exotic_stake or 0
            lines.append(
                f"- [{hit}] {stype} ({svar}): ${stake:.0f} stake | {pnl_str}"
            )

        elif p.pick_type == "big3":
            lines.append(
                f"- [{hit}] Big 3: {p.horse_name} @ ${p.odds_at_tip or 0:.2f} | {pnl_str}"
            )

        elif p.pick_type == "big3_multi":
            lines.append(
                f"- [{hit}] Big 3 Multi: ${p.exotic_stake or 0:.0f} stake | {pnl_str}"
            )

    return lines


# ──────────────────────────────────────────────
# PatternInsight population
# ──────────────────────────────────────────────

async def populate_pattern_insights(db: AsyncSession) -> int:
    """Refresh pattern_insights table with current aggregated stats.

    Called after settlement. Populates the previously-empty table with
    bet_type_overall, bet_type_30d, and tip_rank pattern rows.
    """
    # Only delete non-deep-learning patterns (preserve deep_learning_* rows)
    await db.execute(
        delete(PatternInsight).where(
            ~PatternInsight.pattern_type.like("deep_learning_%")
        )
    )

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

    # Punty's Pick insight
    pp = await aggregate_puntys_pick_performance(db)
    if pp:
        db.add(PatternInsight(
            pattern_type="puntys_pick",
            pattern_key="puntys_pick_overall",
            sample_count=pp["bets"],
            hit_rate=pp["strike_rate"],
            avg_pnl=pp["pnl"] / pp["bets"] if pp["bets"] else 0,
            avg_odds=pp["avg_odds"],
            insight_text=f"Punty's Pick: {pp['strike_rate']}% SR, {pp['roi']:+.1f}% ROI over {pp['bets']} bets",
            conditions_json=json.dumps(pp),
            created_at=now,
            updated_at=now,
        ))
        count += 1

    await db.flush()
    logger.info(f"Populated {count} pattern insights")
    return count


async def _get_deep_learning_patterns(db: AsyncSession) -> list[dict]:
    """Query PatternInsight rows from deep learning analyses.

    Returns HIGH and MEDIUM confidence patterns, sorted by edge size.
    """
    result = await db.execute(
        select(PatternInsight)
        .where(PatternInsight.pattern_type.like("deep_learning_%"))
        .order_by(PatternInsight.avg_pnl.desc())  # avg_pnl stores edge
    )
    rows = result.scalars().all()
    patterns = []
    for r in rows:
        try:
            conds = json.loads(r.conditions_json) if r.conditions_json else {}
        except (json.JSONDecodeError, TypeError):
            conds = {}
        confidence = conds.get("confidence", "LOW")
        if confidence in ("HIGH", "MEDIUM"):
            patterns.append({
                "type": r.pattern_type,
                "key": r.pattern_key,
                "insight": r.insight_text,
                "confidence": confidence,
                "sample_size": r.sample_count,
                "win_rate": r.hit_rate,
                "edge": r.avg_pnl,
            })
    return patterns


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

    # Punty's Pick performance (the highlighted best-bet per race)
    pp_stats = await aggregate_puntys_pick_performance(db)
    if pp_stats:
        pp = pp_stats
        parts.append("### PUNTY'S PICK Performance (Your Best-Bet Recommendation)")
        parts.append(
            f"- **All-Time**: {pp['bets']} picks, {pp['winners']} winners "
            f"({pp['strike_rate']}% SR), {_pnl_str(pp['pnl'])} P&L ({pp['roi']:+.1f}% ROI)"
        )
        if pp.get("selection_bets") and pp.get("exotic_bets"):
            parts.append(
                f"  - Selections: {pp['selection_bets']} picks, {pp['selection_winners']} winners"
                f"  |  Exotics: {pp['exotic_bets']} picks, {pp['exotic_winners']} winners"
            )
        if pp["strike_rate"] < 30:
            parts.append(
                f"- **WARNING**: Your best-bet picks are only hitting {pp['strike_rate']}%. "
                "The public sees this stat. You need to do BETTER."
            )
        if pp["roi"] < 0:
            parts.append(
                f"- **LOSING MONEY**: Punty's Pick ROI is {pp['roi']:+.1f}%. "
                "Pick higher-probability plays, not long shots."
            )
        pp_30 = await aggregate_puntys_pick_performance(db, window_days=30)
        if pp_30 and pp_30["bets"] >= 3:
            parts.append(
                f"- **Last 30 Days**: {pp_30['bets']} picks, {pp_30['winners']} winners "
                f"({pp_30['strike_rate']}% SR), {pp_30['roi']:+.1f}% ROI"
            )
        parts.append("")

    # Strategy directives
    parts.append("### STRATEGY DIRECTIVES (Based on Actual Results)")
    for d in _generate_directives(overall, recent, ranks):
        parts.append(f"- {d}")
    parts.append("")

    # Historical patterns from deep learning analysis
    try:
        dl_patterns = await _get_deep_learning_patterns(db)
    except Exception:
        dl_patterns = []
    if dl_patterns:
        parts.append("### HISTORICAL PATTERNS (From Deep Learning Analysis)")
        parts.append("Statistical patterns discovered from 280K+ historical runners:")
        for p in dl_patterns[:15]:  # Cap at 15 most relevant
            parts.append(f"- [{p['confidence']}] {p['insight']}")
        parts.append("")

    # ROI targets
    parts.append("### ROI TARGETS")
    parts.append("Your job is to get positive ROI on EVERY bet type. Targets:")
    parts.append("- Selections: Target +5% ROI minimum")
    parts.append("- Exotics: Target breakeven (0% ROI)")
    parts.append("- Sequences: Fewer, higher-confidence combos")
    parts.append("- Big 3 Multi: Target 10%+ strike rate")
    parts.append("")

    # Recent results with context (all pick types)
    recent_results = await get_recent_results_with_context(db, limit=15)
    if recent_results:
        parts.append("### RECENT RESULTS (Last 15 Settled Bets — All Types)")
        parts.append("Learn from these. What patterns do you see in winners vs losers?")
        parts.append("")
        for r in recent_results:
            parts.append(r)
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
    """Normalize exotic type names to canonical forms for consistent tracking."""
    low = raw.lower().strip()
    if "first" in low and ("four" in low or "4" in low):
        if "box" in low:
            return "First4 Box"
        return "First4"
    if "trifecta" in low:
        if "standout" in low:
            return "Trifecta Standout"
        return "Trifecta Box"
    if "quinella" in low:
        return "Quinella"
    if "exacta" in low:
        return "Exacta"
    return raw.title()


def _generate_directives(
    overall: list[dict], recent: list[dict], ranks: list[dict],
) -> list[str]:
    """Generate actionable strategy directives from performance data."""
    directives: list[str] = []

    profitable = sorted([s for s in overall if s["roi"] > 0 and s["bets"] >= 30], key=lambda x: x["roi"], reverse=True)
    losing = sorted([s for s in overall if s["roi"] < -10 and s["bets"] >= 30], key=lambda x: x["roi"])

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

    # Tip rank insights (need >= 30 bets for reliable signal)
    if ranks:
        top = next((r for r in ranks if r["rank"] == 1), None)
        roughie = next((r for r in ranks if r["rank"] == 4), None)
        if top and top["bets"] >= 30 and top["roi"] < 0:
            directives.append(
                f"YOUR TOP PICKS return {top['roi']:+.1f}% ROI — be more selective with #1 confidence calls"
            )
        if roughie and roughie["bets"] >= 30 and roughie["roi"] > 0:
            directives.append(
                f"ROUGHIES are profitable at {roughie['roi']:+.1f}% ROI — keep hunting value at big odds"
            )
        elif roughie and roughie["bets"] >= 30 and roughie["strike_rate"] < 5:
            directives.append(
                f"ROUGHIES hitting only {roughie['strike_rate']}% — consider 'Exotics only' more often"
            )

    # Place vs Win comparison
    win_s = next((s for s in overall if s["sub_type"] == "Win"), None)
    place_s = next((s for s in overall if s["sub_type"] == "Place"), None)
    if win_s and place_s and win_s["bets"] >= 30 and place_s["bets"] >= 30 and place_s["roi"] > win_s["roi"] + 10:
        directives.append(
            f"PLACE outperforming WIN ({place_s['roi']:+.1f}% vs {win_s['roi']:+.1f}% ROI) — "
            "use Place as your default for close races"
        )

    # Each Way
    ew_s = next((s for s in overall if s["sub_type"] == "Each Way"), None)
    if ew_s and ew_s["bets"] >= 30:
        if ew_s["roi"] > 0:
            directives.append(f"EACH WAY working at {ew_s['roi']:+.1f}% ROI — keep using on $6-$15 value plays")
        elif ew_s["roi"] < -20:
            directives.append(f"EACH WAY bleeding at {ew_s['roi']:+.1f}% ROI — reserve for genuine value only ($8+)")

    # Exotic-specific directives
    exotic_items = [s for s in overall if s["category"] == "exotic"]
    if exotic_items:
        # Find best and worst exotic types
        profitable_exotics = [s for s in exotic_items if s["roi"] > 0 and s["bets"] >= 30]
        losing_exotics = [s for s in exotic_items if s["roi"] < -20 and s["bets"] >= 30]
        if profitable_exotics:
            best = max(profitable_exotics, key=lambda x: x["roi"])
            directives.append(
                f"EXOTIC WINNER: {best['sub_type']} at {best['roi']:+.1f}% ROI over {best['bets']} bets — "
                "favour this format"
            )
        if losing_exotics:
            worst = min(losing_exotics, key=lambda x: x["roi"])
            directives.append(
                f"EXOTIC LOSER: {worst['sub_type']} bleeding at {worst['roi']:+.1f}% ROI — "
                "only use when pre-calculated value ≥ 2.0x"
            )
        total_exotic_roi = (
            sum(s["pnl"] for s in exotic_items) / sum(s["staked"] for s in exotic_items) * 100
            if sum(s["staked"] for s in exotic_items) > 0 else 0
        )
        if total_exotic_roi < -20:
            directives.append(
                f"EXOTIC ALERT: Overall exotic ROI is {total_exotic_roi:+.1f}% — "
                "only pick exotics from the pre-calculated value table (value ≥ 1.2x)"
            )

    # Sequence-specific directives
    seq_items = [s for s in overall if s["category"] == "sequence"]
    if seq_items:
        profitable_seqs = [s for s in seq_items if s["roi"] > 0]
        losing_seqs = [s for s in seq_items if s["roi"] < -20 and s["bets"] >= 30]
        if profitable_seqs:
            for ps in profitable_seqs:
                directives.append(
                    f"SEQUENCE WINNER: {ps['sub_type']} at {ps['roi']:+.1f}% ROI — keep running this variant"
                )
        if losing_seqs:
            for ls in losing_seqs:
                directives.append(
                    f"SEQUENCE LOSER: {ls['sub_type']} at {ls['roi']:+.1f}% ROI — "
                    "drop this variant or tighten leg selections"
                )
        # Check if Big6 or Early Quaddie are consistently losing
        big6_items = [s for s in seq_items if "big6" in s.get("sub_type", "").lower() or "big 6" in s.get("sub_type", "").lower()]
        early_q_items = [s for s in seq_items if "early" in s.get("sub_type", "").lower()]
        if big6_items and all(s["roi"] < 0 for s in big6_items) and sum(s["bets"] for s in big6_items) >= 50:
            directives.append("DROP BIG 6 — consistently unprofitable across all variants")
        if early_q_items and all(s["roi"] < 0 for s in early_q_items) and sum(s["bets"] for s in early_q_items) >= 50:
            directives.append("DROP EARLY QUADDIE — consistently unprofitable across all variants")

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
