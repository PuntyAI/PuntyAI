"""Picks tracker — loads picks from Pick model and compares against results."""

import json
import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from punty.models.pick import Pick

logger = logging.getLogger(__name__)


async def _load_meeting_tips(db: AsyncSession, meeting_id: str) -> list[dict]:
    """Load tips from Pick model for a meeting."""
    result = await db.execute(
        select(Pick).where(
            Pick.meeting_id == meeting_id,
            Pick.pick_type == "selection",
        )
    )
    picks = result.scalars().all()

    tips = []
    for p in picks:
        tips.append({
            "horse_name": p.horse_name or "",
            "race_number": p.race_number,
            "saddlecloth": p.saddlecloth,
            "odds_at_tip": p.odds_at_tip,
            "place_odds_at_tip": p.place_odds_at_tip,
            "confidence": "best_bet" if p.tip_rank == 1 else "standard",
            "is_big3": False,
            "is_roughie": (p.odds_at_tip or 0) >= 10.0,
            "exotic_type": None,
            # Settlement data from Pick model
            "_hit": p.hit,
            "_pnl": p.pnl,
            "_settled": p.settled,
        })

    # Mark big3 picks
    b3_result = await db.execute(
        select(Pick).where(
            Pick.meeting_id == meeting_id,
            Pick.pick_type == "big3",
        )
    )
    big3_picks = b3_result.scalars().all()
    big3_names = {(p.horse_name or "").upper() for p in big3_picks}
    for tip in tips:
        if tip["horse_name"].upper() in big3_names:
            tip["is_big3"] = True

    return tips


def _compare_tips_to_runners(tips: list[dict], runners) -> list[dict]:
    """Pure function: compare tip dicts against Runner ORM objects."""
    runners_by_name = {r.horse_name.upper(): r for r in runners}
    tip_results = []
    for tip in tips:
        runner = runners_by_name.get(tip["horse_name"].upper())
        if not runner:
            continue

        won = runner.finish_position == 1
        placed = runner.finish_position is not None and runner.finish_position <= 3

        # Use settled P&L from Pick model if available (uses fixed odds)
        # Otherwise fall back to tote dividend calculation for unsettled
        if tip.get("_settled") and tip.get("_pnl") is not None:
            pnl = tip["_pnl"]
        else:
            pnl = 0.0
            if won and runner.win_dividend:
                pnl = runner.win_dividend - 1.0  # 1 unit stake
            elif placed and runner.place_dividend:
                pnl = runner.place_dividend - 1.0
            else:
                pnl = -1.0 if runner.finish_position is not None else 0.0

        tip_results.append({
            "horse": tip["horse_name"],
            "saddlecloth": tip.get("saddlecloth") or runner.saddlecloth,
            "tip_odds": tip["odds_at_tip"],
            "place_odds": tip.get("place_odds_at_tip"),
            "starting_price": runner.starting_price,
            "finish_pos": runner.finish_position,
            "margin": runner.result_margin,
            "won": won,
            "placed": placed,
            "pnl": round(pnl, 2),
            "is_big3": tip.get("is_big3", False),
            "is_roughie": tip.get("is_roughie", False),
            "confidence": tip.get("confidence", "standard"),
        })
    return tip_results


async def build_race_comparison(db: AsyncSession, meeting_id: str, race_number: int, all_tips: list[dict] | None = None) -> dict:
    """Compare early mail tips against actual race results.

    Args:
        all_tips: Pre-loaded tips for the meeting. If None, will be loaded from DB.
                  Pass this to avoid repeated queries in batch operations.

    Returns dict with tips comparison, exotic hits, and running P&L.
    """
    from punty.models.meeting import Runner

    if all_tips is None:
        all_tips = await _load_meeting_tips(db, meeting_id)

    race_tips = [t for t in all_tips if t["race_number"] == race_number]

    # Get runners with results
    race_id = f"{meeting_id}-r{race_number}"
    result = await db.execute(
        select(Runner).where(Runner.race_id == race_id)
    )
    runners = result.scalars().all()

    tip_results = _compare_tips_to_runners(race_tips, runners)

    # Calculate running P&L from Pick model directly
    running_pnl = await _calculate_running_pnl(db, meeting_id, race_number)

    return {
        "race_number": race_number,
        "tips": tip_results,
        "exotic_hit": False,
        "running_day_pnl": running_pnl,
    }


async def build_meeting_summary(db: AsyncSession, meeting_id: str) -> dict:
    """Aggregate all race comparisons for a meeting summary."""
    from punty.models.meeting import Meeting, Race

    result = await db.execute(
        select(Race).where(Race.meeting_id == meeting_id).order_by(Race.race_number)
    )
    races = result.scalars().all()

    # Load tips once for the entire meeting
    all_tips = await _load_meeting_tips(db, meeting_id)

    total_tips = 0
    winners = 0
    placers = 0
    total_pnl = 0.0
    best_result = None
    worst_beat = None
    big3_results = []
    race_summaries = []

    for race in races:
        if not race.results_status or race.results_status == "Open":
            continue

        comparison = await build_race_comparison(db, meeting_id, race.race_number, all_tips=all_tips)
        race_summaries.append(comparison)

        for tip in comparison["tips"]:
            total_tips += 1
            if tip["won"]:
                winners += 1
                if best_result is None or tip["pnl"] > best_result["pnl"]:
                    best_result = tip
            if tip["placed"]:
                placers += 1
            total_pnl += tip["pnl"]

            if tip["is_big3"]:
                big3_results.append(tip)

            # Worst beat: close loss at good odds
            if not tip["won"] and tip["finish_pos"] == 2 and tip["tip_odds"] and tip["tip_odds"] >= 3.0:
                if worst_beat is None or (tip["tip_odds"] > worst_beat.get("tip_odds", 0)):
                    worst_beat = tip

    strike_rate = (winners / total_tips * 100) if total_tips > 0 else 0.0

    return {
        "meeting_id": meeting_id,
        "total_tips": total_tips,
        "winners": winners,
        "placers": placers,
        "strike_rate": round(strike_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "best_result": best_result,
        "worst_beat": worst_beat,
        "big3_results": big3_results,
        "race_summaries": race_summaries,
    }


async def build_pick_ledger(db: AsyncSession, meeting_id: str) -> dict:
    """Build per-tier P&L ledger for the punt review prompt.

    Returns structured ledger with buckets: top1, top2, top3, roughie,
    exotics, sequences, big3_multi — each with staked/returned/net.
    """
    result = await db.execute(
        select(Pick).where(
            Pick.meeting_id == meeting_id,
            Pick.settled == True,
        )
    )
    picks = result.scalars().all()

    buckets = {
        "top1": {"label": "Wins (Top 1)", "staked": 0.0, "returned": 0.0, "picks": []},
        "top2": {"label": "Wins (Top 2)", "staked": 0.0, "returned": 0.0, "picks": []},
        "top3": {"label": "Wins (Top 3)", "staked": 0.0, "returned": 0.0, "picks": []},
        "roughie": {"label": "Wins (Roughie)", "staked": 0.0, "returned": 0.0, "picks": []},
        "exotics": {"label": "Exotics", "staked": 0.0, "returned": 0.0, "picks": []},
        "sequences": {"label": "Sequences", "staked": 0.0, "returned": 0.0, "picks": []},
        "big3_multi": {"label": "Big 3 Multi", "staked": 0.0, "returned": 0.0, "picks": []},
    }

    for p in picks:
        pick_info = {
            "race": p.race_number,
            "horse": p.horse_name,
            "hit": p.hit,
            "pnl": p.pnl or 0.0,
        }

        if p.pick_type == "selection":
            rank = p.tip_rank
            if rank == 1:
                bucket = buckets["top1"]
            elif rank == 2:
                bucket = buckets["top2"]
            elif rank == 3:
                bucket = buckets["top3"]
            elif rank == 4:
                bucket = buckets["roughie"]
            else:
                bucket = buckets["top1"]  # fallback
            stake = p.bet_stake or 1.0
            bucket["staked"] += stake
            bucket["returned"] += max(0.0, (p.pnl or 0.0) + stake) if p.hit else 0.0
            bucket["picks"].append(pick_info)

        elif p.pick_type == "exotic":
            # exotic_stake is the total outlay for this bet
            cost = p.exotic_stake or 1.0
            buckets["exotics"]["staked"] += cost
            buckets["exotics"]["returned"] += max(0.0, (p.pnl or 0.0) + cost) if p.hit else 0.0
            pick_info["exotic_type"] = p.exotic_type
            pick_info["exotic_runners"] = p.exotic_runners
            pick_info["cost"] = cost
            buckets["exotics"]["picks"].append(pick_info)

        elif p.pick_type == "sequence":
            legs = json.loads(p.sequence_legs) if p.sequence_legs else []
            num_combos = 1
            for leg in legs:
                num_combos *= len(leg)
            base = p.exotic_stake or 1.0
            cost = num_combos * base
            buckets["sequences"]["staked"] += cost
            buckets["sequences"]["returned"] += max(0.0, (p.pnl or 0.0) + cost) if p.hit else 0.0
            pick_info["sequence_type"] = p.sequence_type
            pick_info["sequence_variant"] = p.sequence_variant
            pick_info["combos"] = num_combos
            pick_info["cost"] = cost
            buckets["sequences"]["picks"].append(pick_info)

        elif p.pick_type == "big3_multi":
            stake = p.exotic_stake or 10.0
            buckets["big3_multi"]["staked"] += stake
            buckets["big3_multi"]["returned"] += max(0.0, (p.pnl or 0.0) + stake) if p.hit else 0.0
            buckets["big3_multi"]["picks"].append(pick_info)

        # big3 individual picks — skip (P&L on multi row)

    # Calculate net per bucket and totals
    total_staked = 0.0
    total_returned = 0.0
    for b in buckets.values():
        b["net"] = round(b["returned"] - b["staked"], 2)
        b["staked"] = round(b["staked"], 2)
        b["returned"] = round(b["returned"], 2)
        total_staked += b["staked"]
        total_returned += b["returned"]

    return {
        "buckets": buckets,
        "total_staked": round(total_staked, 2),
        "total_returned": round(total_returned, 2),
        "total_net": round(total_returned - total_staked, 2),
    }


async def _calculate_running_pnl(db: AsyncSession, meeting_id: str, up_to_race: int) -> float:
    """Calculate running P&L from settled Pick rows up to a given race number."""
    from sqlalchemy import func

    result = await db.execute(
        select(func.sum(Pick.pnl)).where(
            Pick.meeting_id == meeting_id,
            Pick.settled == True,
            Pick.race_number <= up_to_race,
            Pick.pick_type != "big3",  # P&L tracked on multi row
        )
    )
    total = result.scalar()
    return round(float(total or 0), 2)
