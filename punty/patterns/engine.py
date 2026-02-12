"""Deep pattern analysis engine — mines all settled picks across 12 dimensions.

Produces structured JSON summaries for blog generation and upserts into
the PatternInsight table for RAG context injection.
"""

import json
import logging
from datetime import timedelta
from typing import Any, Optional

from sqlalchemy import select, func, case, and_, extract, cast, Integer as SAInt
from sqlalchemy.ext.asyncio import AsyncSession

from punty.config import melb_today, melb_now_naive
from punty.memory.models import PatternInsight
from punty.models.pick import Pick
from punty.models.meeting import Meeting, Race, Runner

logger = logging.getLogger(__name__)

MIN_SAMPLE = 5  # minimum picks per group to include in results


def _date_filter(window_days: Optional[int] = None) -> list:
    """Build date filter clauses for settled picks."""
    if window_days:
        cutoff = melb_today() - timedelta(days=window_days)
        return [Pick.settled_at >= cutoff]
    return []


def _make_result(dimension: str, key: str, bets: int, winners: int,
                 staked: float, pnl: float, avg_odds: float = 0) -> dict:
    """Build a standardised result dict."""
    sr = round(winners / bets * 100, 1) if bets > 0 else 0
    roi = round(pnl / staked * 100, 1) if staked > 0 else 0
    return {
        "dimension": dimension,
        "key": key,
        "sample_count": bets,
        "winners": winners,
        "hit_rate": sr,
        "pnl": round(pnl, 2),
        "roi": roi,
        "avg_odds": round(avg_odds, 2),
        "insight_text": f"{key}: {sr}% SR ({winners}/{bets}), ROI {roi}%, P&L ${pnl:+.2f}",
    }


# ── Individual dimension analyses ──────────────────────────────────────────


async def analyse_venue_performance(
    db: AsyncSession, window_days: Optional[int] = None,
) -> list[dict]:
    """Performance grouped by venue."""
    df = _date_filter(window_days)
    q = (
        select(
            Meeting.venue,
            func.count(Pick.id),
            func.sum(case((Pick.hit == True, 1), else_=0)),
            func.sum(Pick.bet_stake),
            func.sum(Pick.pnl),
            func.avg(Pick.odds_at_tip),
        )
        .join(Meeting, Pick.meeting_id == Meeting.id)
        .where(Pick.settled == True, Pick.pick_type == "selection", *df)
        .group_by(Meeting.venue)
        .having(func.count(Pick.id) >= MIN_SAMPLE)
        .order_by(func.sum(Pick.pnl).desc())
    )
    results = []
    for venue, bets, wins, staked, pnl, odds in (await db.execute(q)).all():
        results.append(_make_result(
            "venue", venue or "Unknown", bets, int(wins or 0),
            float(staked or 0), float(pnl or 0), float(odds or 0),
        ))
    return results


async def analyse_distance_bands(
    db: AsyncSession, window_days: Optional[int] = None,
) -> list[dict]:
    """Performance by distance band: sprint/middle/staying."""
    df = _date_filter(window_days)

    # Use a CASE expression to bucket distances
    band = case(
        (Race.distance < 1300, "Sprint (< 1300m)"),
        (Race.distance < 1900, "Middle (1300-1900m)"),
        else_="Staying (1900m+)",
    )

    q = (
        select(
            band.label("band"),
            func.count(Pick.id),
            func.sum(case((Pick.hit == True, 1), else_=0)),
            func.sum(Pick.bet_stake),
            func.sum(Pick.pnl),
            func.avg(Pick.odds_at_tip),
        )
        .join(Race, and_(
            Pick.meeting_id == Race.meeting_id,
            Pick.race_number == Race.race_number,
        ))
        .where(Pick.settled == True, Pick.pick_type == "selection",
               Race.distance.isnot(None), *df)
        .group_by("band")
        .having(func.count(Pick.id) >= MIN_SAMPLE)
    )
    results = []
    for band_name, bets, wins, staked, pnl, odds in (await db.execute(q)).all():
        results.append(_make_result(
            "distance_band", band_name, bets, int(wins or 0),
            float(staked or 0), float(pnl or 0), float(odds or 0),
        ))
    return results


async def analyse_track_conditions(
    db: AsyncSession, window_days: Optional[int] = None,
) -> list[dict]:
    """Performance by base track condition: Firm/Good/Soft/Heavy."""
    df = _date_filter(window_days)

    # Extract base condition word (strip numeric rating)
    import re

    q = (
        select(
            Meeting.track_condition,
            func.count(Pick.id),
            func.sum(case((Pick.hit == True, 1), else_=0)),
            func.sum(Pick.bet_stake),
            func.sum(Pick.pnl),
            func.avg(Pick.odds_at_tip),
        )
        .join(Meeting, Pick.meeting_id == Meeting.id)
        .where(Pick.settled == True, Pick.pick_type == "selection",
               Meeting.track_condition.isnot(None), *df)
        .group_by(Meeting.track_condition)
    )

    # Merge by base condition in Python (SQLite lacks regex functions)
    merged: dict[str, list] = {}
    for cond, bets, wins, staked, pnl, odds in (await db.execute(q)).all():
        base = re.sub(r"[\s\d()]+", " ", (cond or "").strip()).strip().title() or "Unknown"
        if base not in merged:
            merged[base] = [0, 0, 0.0, 0.0, 0.0, 0.0]
        m = merged[base]
        m[0] += bets
        m[1] += int(wins or 0)
        m[2] += float(staked or 0)
        m[3] += float(pnl or 0)
        m[4] += float(odds or 0) * bets  # weighted sum for avg
        m[5] += bets  # weight denominator

    results = []
    for base_cond, (bets, wins, staked, pnl, odds_sum, weight) in merged.items():
        if bets >= MIN_SAMPLE:
            avg_odds = odds_sum / weight if weight > 0 else 0
            results.append(_make_result("track_condition", base_cond, bets, wins, staked, pnl, avg_odds))
    return sorted(results, key=lambda r: r["pnl"], reverse=True)


async def analyse_barriers(
    db: AsyncSession, window_days: Optional[int] = None,
) -> list[dict]:
    """Barrier analysis — group by barrier range across all venues."""
    df = _date_filter(window_days)

    barrier_band = case(
        (Runner.barrier <= 4, "Barrier 1-4 (Inside)"),
        (Runner.barrier <= 8, "Barrier 5-8 (Middle)"),
        (Runner.barrier <= 12, "Barrier 9-12 (Wide)"),
        else_="Barrier 13+ (Very Wide)",
    )

    q = (
        select(
            barrier_band.label("barrier_band"),
            func.count(Pick.id),
            func.sum(case((Pick.hit == True, 1), else_=0)),
            func.sum(Pick.bet_stake),
            func.sum(Pick.pnl),
            func.avg(Pick.odds_at_tip),
        )
        .join(Race, and_(
            Pick.meeting_id == Race.meeting_id,
            Pick.race_number == Race.race_number,
        ))
        .join(Runner, and_(
            Runner.race_id == Race.id,
            Runner.saddlecloth == Pick.saddlecloth,
        ))
        .where(Pick.settled == True, Pick.pick_type == "selection",
               Runner.barrier.isnot(None), *df)
        .group_by("barrier_band")
        .having(func.count(Pick.id) >= MIN_SAMPLE)
    )
    results = []
    for band, bets, wins, staked, pnl, odds in (await db.execute(q)).all():
        results.append(_make_result(
            "barrier", band, bets, int(wins or 0),
            float(staked or 0), float(pnl or 0), float(odds or 0),
        ))
    return results


async def analyse_jockey_trainer(
    db: AsyncSession, window_days: Optional[int] = None,
) -> dict[str, list[dict]]:
    """Top/bottom jockeys and trainers by ROI."""
    df = _date_filter(window_days)

    result = {"jockeys": [], "trainers": []}

    for field, label in [(Runner.jockey, "jockeys"), (Runner.trainer, "trainers")]:
        q = (
            select(
                field,
                func.count(Pick.id),
                func.sum(case((Pick.hit == True, 1), else_=0)),
                func.sum(Pick.bet_stake),
                func.sum(Pick.pnl),
                func.avg(Pick.odds_at_tip),
            )
            .join(Race, and_(
                Pick.meeting_id == Race.meeting_id,
                Pick.race_number == Race.race_number,
            ))
            .join(Runner, and_(
                Runner.race_id == Race.id,
                Runner.saddlecloth == Pick.saddlecloth,
            ))
            .where(Pick.settled == True, Pick.pick_type == "selection",
                   field.isnot(None), *df)
            .group_by(field)
            .having(func.count(Pick.id) >= MIN_SAMPLE)
            .order_by(func.sum(Pick.pnl).desc())
            .limit(20)
        )
        for name, bets, wins, staked, pnl, odds in (await db.execute(q)).all():
            result[label].append(_make_result(
                label.rstrip("s"), name or "Unknown", bets, int(wins or 0),
                float(staked or 0), float(pnl or 0), float(odds or 0),
            ))

    return result


async def analyse_odds_ranges(
    db: AsyncSession, window_days: Optional[int] = None,
) -> list[dict]:
    """Performance by odds range: short/mid/roughie."""
    df = _date_filter(window_days)

    odds_band = case(
        (Pick.odds_at_tip < 3.0, "Short-priced (< $3)"),
        (Pick.odds_at_tip < 10.0, "Mid-range ($3-$10)"),
        else_="Roughie ($10+)",
    )

    q = (
        select(
            odds_band.label("odds_band"),
            func.count(Pick.id),
            func.sum(case((Pick.hit == True, 1), else_=0)),
            func.sum(Pick.bet_stake),
            func.sum(Pick.pnl),
            func.avg(Pick.odds_at_tip),
        )
        .where(Pick.settled == True, Pick.pick_type == "selection",
               Pick.odds_at_tip.isnot(None), *df)
        .group_by("odds_band")
        .having(func.count(Pick.id) >= MIN_SAMPLE)
    )
    results = []
    for band, bets, wins, staked, pnl, odds in (await db.execute(q)).all():
        results.append(_make_result(
            "odds_range", band, bets, int(wins or 0),
            float(staked or 0), float(pnl or 0), float(odds or 0),
        ))
    return results


async def analyse_bet_types(
    db: AsyncSession, window_days: Optional[int] = None,
) -> list[dict]:
    """Performance by bet type (delegates to existing strategy.py)."""
    from punty.memory.strategy import aggregate_bet_type_performance
    stats = await aggregate_bet_type_performance(db, window_days=window_days)
    results = []
    for s in stats:
        results.append(_make_result(
            "bet_type", f"{s['category']} - {s['sub_type']}", s["bets"], s["winners"],
            s["staked"], s["pnl"], s.get("avg_odds", 0),
        ))
    return results


async def analyse_speed_map_positions(
    db: AsyncSession, window_days: Optional[int] = None,
) -> list[dict]:
    """Performance by predicted speed map position."""
    df = _date_filter(window_days)

    q = (
        select(
            Runner.speed_map_position,
            func.count(Pick.id),
            func.sum(case((Pick.hit == True, 1), else_=0)),
            func.sum(Pick.bet_stake),
            func.sum(Pick.pnl),
            func.avg(Pick.odds_at_tip),
        )
        .join(Race, and_(
            Pick.meeting_id == Race.meeting_id,
            Pick.race_number == Race.race_number,
        ))
        .join(Runner, and_(
            Runner.race_id == Race.id,
            Runner.saddlecloth == Pick.saddlecloth,
        ))
        .where(Pick.settled == True, Pick.pick_type == "selection",
               Runner.speed_map_position.isnot(None),
               Runner.speed_map_position != "", *df)
        .group_by(Runner.speed_map_position)
        .having(func.count(Pick.id) >= MIN_SAMPLE)
    )
    results = []
    for pos, bets, wins, staked, pnl, odds in (await db.execute(q)).all():
        results.append(_make_result(
            "speed_map", (pos or "unknown").replace("_", " ").title(),
            bets, int(wins or 0),
            float(staked or 0), float(pnl or 0), float(odds or 0),
        ))
    return results


async def analyse_day_of_week(
    db: AsyncSession, window_days: Optional[int] = None,
) -> list[dict]:
    """Performance by day of week."""
    df = _date_filter(window_days)

    # SQLite: strftime('%w', date) returns 0=Sunday, 1=Monday, ..., 6=Saturday
    day_num = func.cast(func.strftime("%w", Meeting.date), SAInt)

    q = (
        select(
            day_num.label("day_num"),
            func.count(Pick.id),
            func.sum(case((Pick.hit == True, 1), else_=0)),
            func.sum(Pick.bet_stake),
            func.sum(Pick.pnl),
            func.avg(Pick.odds_at_tip),
        )
        .join(Meeting, Pick.meeting_id == Meeting.id)
        .where(Pick.settled == True, Pick.pick_type == "selection", *df)
        .group_by("day_num")
        .having(func.count(Pick.id) >= MIN_SAMPLE)
    )

    day_names = {0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday",
                 4: "Thursday", 5: "Friday", 6: "Saturday"}
    results = []
    for day, bets, wins, staked, pnl, odds in (await db.execute(q)).all():
        results.append(_make_result(
            "day_of_week", day_names.get(day, f"Day {day}"), bets, int(wins or 0),
            float(staked or 0), float(pnl or 0), float(odds or 0),
        ))
    return results


async def analyse_seasonal_trends(db: AsyncSession) -> list[dict]:
    """Performance by month (all time)."""
    month_num = func.cast(func.strftime("%m", Meeting.date), SAInt)
    year_num = func.strftime("%Y", Meeting.date)

    q = (
        select(
            year_num.label("year"),
            month_num.label("month"),
            func.count(Pick.id),
            func.sum(case((Pick.hit == True, 1), else_=0)),
            func.sum(Pick.bet_stake),
            func.sum(Pick.pnl),
        )
        .join(Meeting, Pick.meeting_id == Meeting.id)
        .where(Pick.settled == True, Pick.pick_type == "selection")
        .group_by("year", "month")
        .having(func.count(Pick.id) >= MIN_SAMPLE)
        .order_by("year", "month")
    )

    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    results = []
    for year, month, bets, wins, staked, pnl in (await db.execute(q)).all():
        mn = month_names.get(month, f"M{month}")
        results.append(_make_result(
            "seasonal", f"{mn} {year}", bets, int(wins or 0),
            float(staked or 0), float(pnl or 0),
        ))
    return results


async def analyse_weather_impact(
    db: AsyncSession, window_days: Optional[int] = None,
) -> list[dict]:
    """Performance by weather condition."""
    df = _date_filter(window_days)

    q = (
        select(
            Meeting.weather_condition,
            func.count(Pick.id),
            func.sum(case((Pick.hit == True, 1), else_=0)),
            func.sum(Pick.bet_stake),
            func.sum(Pick.pnl),
            func.avg(Pick.odds_at_tip),
        )
        .join(Meeting, Pick.meeting_id == Meeting.id)
        .where(Pick.settled == True, Pick.pick_type == "selection",
               Meeting.weather_condition.isnot(None), *df)
        .group_by(Meeting.weather_condition)
        .having(func.count(Pick.id) >= MIN_SAMPLE)
    )
    results = []
    for weather, bets, wins, staked, pnl, odds in (await db.execute(q)).all():
        results.append(_make_result(
            "weather", weather or "Unknown", bets, int(wins or 0),
            float(staked or 0), float(pnl or 0), float(odds or 0),
        ))
    return results


async def analyse_field_size(
    db: AsyncSession, window_days: Optional[int] = None,
) -> list[dict]:
    """Performance by field size: small/medium/large."""
    df = _date_filter(window_days)

    field_band = case(
        (Race.field_size <= 8, "Small (1-8 runners)"),
        (Race.field_size <= 12, "Medium (9-12 runners)"),
        else_="Large (13+ runners)",
    )

    q = (
        select(
            field_band.label("field_band"),
            func.count(Pick.id),
            func.sum(case((Pick.hit == True, 1), else_=0)),
            func.sum(Pick.bet_stake),
            func.sum(Pick.pnl),
            func.avg(Pick.odds_at_tip),
        )
        .join(Race, and_(
            Pick.meeting_id == Race.meeting_id,
            Pick.race_number == Race.race_number,
        ))
        .where(Pick.settled == True, Pick.pick_type == "selection",
               Race.field_size.isnot(None), Race.field_size > 0, *df)
        .group_by("field_band")
        .having(func.count(Pick.id) >= MIN_SAMPLE)
    )
    results = []
    for band, bets, wins, staked, pnl, odds in (await db.execute(q)).all():
        results.append(_make_result(
            "field_size", band, bets, int(wins or 0),
            float(staked or 0), float(pnl or 0), float(odds or 0),
        ))
    return results


# ── Master orchestrator ────────────────────────────────────────────────────


async def run_deep_pattern_analysis(
    db: AsyncSession, window_days: Optional[int] = None,
) -> dict[str, Any]:
    """Run all 12 dimension analyses and return structured JSON summary.

    Also upserts results into PatternInsight table for RAG context.
    """
    logger.info("Starting deep pattern analysis...")

    summary: dict[str, Any] = {}

    # Run all analyses
    analyses = {
        "venue": analyse_venue_performance,
        "distance_band": analyse_distance_bands,
        "track_condition": analyse_track_conditions,
        "barrier": analyse_barriers,
        "odds_range": analyse_odds_ranges,
        "bet_type": analyse_bet_types,
        "speed_map": analyse_speed_map_positions,
        "day_of_week": analyse_day_of_week,
        "seasonal": analyse_seasonal_trends,
        "weather": analyse_weather_impact,
        "field_size": analyse_field_size,
    }

    for dim_name, fn in analyses.items():
        try:
            if dim_name == "seasonal":
                result = await fn(db)
            else:
                result = await fn(db, window_days=window_days)
            summary[dim_name] = result
            logger.info(f"  {dim_name}: {len(result)} groups")
        except Exception as e:
            logger.warning(f"  {dim_name} analysis failed: {e}")
            summary[dim_name] = []

    # Jockey/trainer is a special case (returns dict not list)
    try:
        jt = await analyse_jockey_trainer(db, window_days=window_days)
        summary["jockey"] = jt.get("jockeys", [])
        summary["trainer"] = jt.get("trainers", [])
        logger.info(f"  jockey: {len(summary['jockey'])} entries, trainer: {len(summary['trainer'])} entries")
    except Exception as e:
        logger.warning(f"  jockey/trainer analysis failed: {e}")
        summary["jockey"] = []
        summary["trainer"] = []

    # Upsert into PatternInsight table
    upserted = 0
    for dim_name, entries in summary.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            try:
                await _upsert_pattern_insight(db, dim_name, entry)
                upserted += 1
            except Exception as e:
                logger.debug(f"PatternInsight upsert failed for {dim_name}/{entry.get('key')}: {e}")

    await db.commit()
    logger.info(f"Deep pattern analysis complete: {upserted} insights upserted")

    return summary


async def _upsert_pattern_insight(db: AsyncSession, pattern_type: str, entry: dict):
    """Upsert a single pattern insight row."""
    key = entry.get("key", "")
    existing = await db.execute(
        select(PatternInsight).where(
            PatternInsight.pattern_type == pattern_type,
            PatternInsight.pattern_key == key,
        )
    )
    row = existing.scalar_one_or_none()

    if row:
        row.sample_count = entry["sample_count"]
        row.hit_rate = entry["hit_rate"]
        row.avg_pnl = entry["pnl"]
        row.avg_odds = entry["avg_odds"]
        row.insight_text = entry["insight_text"]
        row.updated_at = melb_now_naive()
    else:
        db.add(PatternInsight(
            pattern_type=pattern_type,
            pattern_key=key,
            sample_count=entry["sample_count"],
            hit_rate=entry["hit_rate"],
            avg_pnl=entry["pnl"],
            avg_odds=entry["avg_odds"],
            insight_text=entry["insight_text"],
            conditions_json=json.dumps({"dimension": entry["dimension"]}),
        ))
