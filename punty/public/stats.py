"""Stats page route."""

from datetime import date

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import select, func, and_, or_, case

from punty.config import melb_today
from punty.models.database import async_session
from punty.models.pick import Pick
from punty.models.content import Content
from punty.models.meeting import Meeting, Race, Runner

from punty.public.deps import templates

router = APIRouter()


async def get_bet_type_stats(
    venue: str | None = None,
    state: str | None = None,
    distance_min: int | None = None,
    distance_max: int | None = None,
    track_condition: str | None = None,
    race_class: str | None = None,
    jockey: str | None = None,
    trainer: str | None = None,
    horse_sex: str | None = None,
    tip_rank: int | None = None,
    odds_min: float | None = None,
    odds_max: float | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    field_size_min: int | None = None,
    field_size_max: int | None = None,
    weather: str | None = None,
    barrier_min: int | None = None,
    barrier_max: int | None = None,
    today: bool = False,
) -> list[dict]:
    """Get strike rate, won/total, and P&L for every bet type with optional filters."""

    # Desired display order
    _SEL_ORDER = ["Win", "Saver Win", "Place", "Each Way"]
    _EXOTIC_ORDER = ["Quinella", "Exacta", "Trifecta", "First 4"]
    _SEQ_ORDER = ["Early Quaddie", "Quaddie", "Big6"]
    _VARIANT_ORDER = ["Skinny", "Balanced", "Wide"]

    def _normalise_exotic(raw: str) -> str:
        low = raw.lower().strip()
        if low in ("box trifecta", "trifecta box", "trifecta (box)", "trifecta (boxed)", "trifecta boxed",
                    "trifecta standout", "trifecta (standout)"):
            return "Trifecta"
        if low in ("exacta standout", "exacta (standout)"):
            return "Exacta"
        if low in ("first four", "first 4", "first four (boxed)", "first four box",
                    "first4", "first4 box", "first 4 standout", "first four standout"):
            return "First 4"
        return raw

    def _esc(s: str) -> str:
        """Escape ILIKE wildcards in user input."""
        return s.replace("%", "\\%").replace("_", "\\_")

    # Build filter condition lists by table
    # Hard floor: exclude pre-audit data
    meeting_conds = [Meeting.date >= date(2026, 2, 17)]
    if venue:
        meeting_conds.append(Meeting.venue.ilike(f"%{_esc(venue)}%"))
    if state:
        from punty.venues import get_venues_for_state
        state_tracks = get_venues_for_state(state)
        if state_tracks:
            meeting_conds.append(or_(*[Meeting.venue.ilike(f"%{t}%") for t in state_tracks]))
    if date_from:
        meeting_conds.append(Meeting.date >= date_from)
    if date_to:
        meeting_conds.append(Meeting.date <= date_to)
    if track_condition:
        meeting_conds.append(Meeting.track_condition.ilike(f"%{_esc(track_condition)}%"))
    if weather:
        meeting_conds.append(Meeting.weather_condition.ilike(f"%{_esc(weather)}%"))
    if today:
        meeting_conds.append(Meeting.date == melb_today())

    race_conds = []
    if distance_min:
        race_conds.append(Race.distance >= distance_min)
    if distance_max:
        race_conds.append(Race.distance <= distance_max)
    if race_class:
        race_conds.append(Race.class_.ilike(f"%{_esc(race_class)}%"))
    if field_size_min:
        race_conds.append(Race.field_size >= field_size_min)
    if field_size_max:
        race_conds.append(Race.field_size <= field_size_max)

    runner_conds = []
    if jockey:
        runner_conds.append(Runner.jockey.ilike(f"%{_esc(jockey)}%"))
    if trainer:
        runner_conds.append(Runner.trainer.ilike(f"%{_esc(trainer)}%"))
    if horse_sex:
        runner_conds.append(Runner.horse_sex == horse_sex)
    if barrier_min:
        runner_conds.append(Runner.barrier >= barrier_min)
    if barrier_max:
        runner_conds.append(Runner.barrier <= barrier_max)

    pick_conds = []
    if tip_rank:
        pick_conds.append(Pick.tip_rank == tip_rank)
    if odds_min:
        pick_conds.append(Pick.odds_at_tip >= odds_min)
    if odds_max:
        pick_conds.append(Pick.odds_at_tip <= odds_max)

    needs_meeting = bool(meeting_conds)
    needs_race = bool(race_conds) or bool(runner_conds)
    needs_runner = bool(runner_conds)
    has_runner_or_pick_filters = needs_runner or bool(pick_conds)

    def _apply_joins(query, include_runner=False):
        """Add JOINs to query based on active filters."""
        if needs_meeting:
            query = query.join(Meeting, Pick.meeting_id == Meeting.id)
        if needs_race or (include_runner and needs_runner):
            query = query.join(Race, and_(
                Race.meeting_id == Pick.meeting_id,
                Race.race_number == Pick.race_number,
            ))
        if include_runner and needs_runner:
            query = query.join(Runner, and_(
                Runner.race_id == Race.id,
                Runner.saddlecloth == Pick.saddlecloth,
            ))
        return query

    def _extra_conds(include_runner=False, include_pick=False):
        """Collect filter conditions for the query."""
        conds = list(meeting_conds) + list(race_conds)
        if include_runner:
            conds.extend(runner_conds)
        if include_pick:
            conds.extend(pick_conds)
        return conds

    async with async_session() as db:
        # --- Selections by bet_type (exclude exotics_only) ---
        sel_query = select(
            Pick.bet_type,
            func.count(Pick.id),
            func.sum(case((Pick.hit == True, 1), else_=0)),
            func.sum(Pick.pnl),
            func.sum(Pick.bet_stake),
        )
        sel_query = _apply_joins(sel_query, include_runner=True)
        sel_conds = [
            Pick.settled == True,
            Pick.pick_type == "selection",
            Pick.bet_type.isnot(None),
            Pick.bet_type != "exotics_only",
        ] + _extra_conds(include_runner=True, include_pick=True)
        sel_result = await db.execute(
            sel_query.where(and_(*sel_conds)).group_by(Pick.bet_type)
        )

        sel_stats = {}
        for bet_type, total, hits, pnl, staked in sel_result.all():
            hits = int(hits or 0)
            pnl = float(pnl or 0)
            staked = float(staked or 0)
            rate = round(hits / total * 100, 1) if total > 0 else 0
            label = (bet_type or "unknown").replace("_", " ").title()
            sel_stats[label] = {
                "category": "Selections",
                "type": label,
                "won": hits,
                "total": total,
                "rate": rate,
                "pnl": round(pnl, 2),
                "staked": round(staked, 2),
            }

        # --- Exotics (skip when runner/pick filters active) ---
        exotic_stats: dict[str, dict] = {}
        if not has_runner_or_pick_filters:
            exotic_query = select(
                Pick.exotic_type,
                func.count(Pick.id),
                func.sum(case((Pick.hit == True, 1), else_=0)),
                func.sum(Pick.pnl),
                func.sum(Pick.exotic_stake),
            )
            exotic_query = _apply_joins(exotic_query, include_runner=False)
            exotic_conds = [
                Pick.settled == True,
                Pick.pick_type == "exotic",
                Pick.exotic_type.isnot(None),
            ] + _extra_conds()
            exotic_result = await db.execute(
                exotic_query.where(and_(*exotic_conds)).group_by(Pick.exotic_type)
            )

            for exotic_type, total, hits, pnl, staked in exotic_result.all():
                label = _normalise_exotic(exotic_type)
                hits = int(hits or 0)
                pnl = float(pnl or 0)
                staked = float(staked or 0)
                if label in exotic_stats:
                    exotic_stats[label]["won"] += hits
                    exotic_stats[label]["total"] += total
                    exotic_stats[label]["pnl"] = round(exotic_stats[label]["pnl"] + pnl, 2)
                    exotic_stats[label]["staked"] = round(exotic_stats[label]["staked"] + staked, 2)
                else:
                    exotic_stats[label] = {
                        "category": "Exotics",
                        "type": label,
                        "won": hits,
                        "total": total,
                        "rate": 0,
                        "pnl": round(pnl, 2),
                        "staked": round(staked, 2),
                    }
            for s in exotic_stats.values():
                s["rate"] = round(s["won"] / s["total"] * 100, 1) if s["total"] > 0 else 0

        # --- Sequences (skip when runner/pick filters active) ---
        seq_stats = {}
        if not has_runner_or_pick_filters:
            seq_query = select(
                Pick.sequence_type,
                Pick.sequence_variant,
                func.count(Pick.id),
                func.sum(case((Pick.hit == True, 1), else_=0)),
                func.sum(Pick.pnl),
                func.sum(Pick.exotic_stake),
            )
            seq_query = _apply_joins(seq_query, include_runner=False)
            seq_conds = [
                Pick.settled == True,
                Pick.pick_type == "sequence",
            ] + _extra_conds()
            seq_result = await db.execute(
                seq_query.where(and_(*seq_conds)).group_by(Pick.sequence_type, Pick.sequence_variant)
            )

            for seq_type, seq_variant, total, hits, pnl, staked in seq_result.all():
                hits = int(hits or 0)
                pnl = float(pnl or 0)
                staked = float(staked or 0)
                rate = round(hits / total * 100, 1) if total > 0 else 0
                label = (seq_type or "Sequence").replace("_", " ").title()
                seq_stats[label] = {
                    "category": "Sequences",
                    "type": label,
                    "won": hits,
                    "total": total,
                    "rate": rate,
                    "pnl": round(pnl, 2),
                    "staked": round(staked, 2),
                }

        # --- Big 3 Multi (skip when runner/pick filters active) ---
        big3_stat = None
        if not has_runner_or_pick_filters:
            big3_query = select(
                func.count(Pick.id),
                func.sum(case((Pick.hit == True, 1), else_=0)),
                func.sum(Pick.pnl),
                func.sum(Pick.exotic_stake),
            )
            big3_query = _apply_joins(big3_query, include_runner=False)
            big3_conds = [
                Pick.settled == True,
                Pick.pick_type == "big3_multi",
            ] + _extra_conds()
            big3_result = await db.execute(
                big3_query.where(and_(*big3_conds))
            )
            row = big3_result.one()
            if row[0] and row[0] > 0:
                hits = int(row[1] or 0)
                pnl = float(row[2] or 0)
                staked = float(row[3] or 0)
                rate = round(hits / row[0] * 100, 1) if row[0] > 0 else 0
                big3_stat = {
                    "category": "Multi",
                    "type": "Big 3 Multi",
                    "won": hits,
                    "total": row[0],
                    "rate": rate,
                    "pnl": round(pnl, 2),
                    "staked": round(staked, 2),
                }

        # Build ordered output
        stats = []
        for key in _SEL_ORDER:
            if key in sel_stats:
                stats.append(sel_stats[key])
        for key in _EXOTIC_ORDER:
            if key in exotic_stats:
                stats.append(exotic_stats[key])
        for key, val in exotic_stats.items():
            if key not in _EXOTIC_ORDER:
                stats.append(val)
        for prefix in _SEQ_ORDER:
            for variant in _VARIANT_ORDER:
                key = f"{prefix} ({variant})"
                if key in seq_stats:
                    stats.append(seq_stats[key])
            # Catch any variants not in _VARIANT_ORDER
            for key in sorted(seq_stats.keys()):
                if key.startswith(prefix) and key not in [f"{prefix} ({v})" for v in _VARIANT_ORDER]:
                    stats.append(seq_stats[key])
        if big3_stat:
            stats.append(big3_stat)

        return stats


@router.get("/stats", response_class=HTMLResponse)
async def stats_page(request: Request):
    """Today's daily dashboard — celebrating wins, insights, and performance."""
    from punty.public.dashboard import get_daily_dashboard
    dashboard = await get_daily_dashboard()
    return templates.TemplateResponse(
        "stats.html",
        {"request": request, **dashboard},
    )
