"""Tips dashboard and HTMX partial routes."""

from datetime import date, datetime

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import select, func, and_, or_, case

from punty.config import melb_today, melb_now, MELB_TZ
from punty.models.database import async_session
from punty.models.pick import Pick
from punty.models.content import Content
from punty.models.meeting import Meeting, Race, Runner

from punty.public.deps import templates, _norm_exotic, _build_win_card

router = APIRouter()


async def get_next_race(exclude_venues: list[str] | None = None, offset: int = 0) -> dict:
    """Get the next upcoming race for countdown display.

    Args:
        exclude_venues: Optional list of venue names to skip (for venue filtering).
        offset: Number of races to skip forward from the next one (0 = very next race).
    """
    async with async_session() as db:
        today = melb_today()
        now = melb_now()
        now_naive = now.replace(tzinfo=None)

        # Get today's selected meetings
        meetings_result = await db.execute(
            select(Meeting).where(
                and_(
                    Meeting.date == today,
                    Meeting.selected == True,
                    Meeting.meeting_type.in_(["race", None]),
                )
            )
        )
        meetings = {m.id: m for m in meetings_result.scalars().all()}

        # Filter out excluded venues
        if exclude_venues:
            meetings = {mid: m for mid, m in meetings.items() if m.venue not in exclude_venues}

        if not meetings:
            return {"has_next": False}

        # Get all races for today's meetings that haven't finished yet
        # Note: notin_ doesn't handle NULL correctly, so we need explicit OR for NULL values
        races_result = await db.execute(
            select(Race).where(
                and_(
                    Race.meeting_id.in_(list(meetings.keys())),
                    Race.start_time.isnot(None),
                    or_(
                        Race.results_status.is_(None),
                        Race.results_status.notin_(["Paying", "Closed", "Final"]),
                    ),
                )
            ).order_by(Race.start_time)
        )
        races = races_result.scalars().all()

        # Find all upcoming races (include races that started within the last 60s
        # so the slider stays on a race for 1 minute after jump before advancing)
        from datetime import timedelta as _td
        cutoff = now_naive - _td(seconds=60)
        upcoming_races = [r for r in races if r.start_time and r.start_time > cutoff]

        if not upcoming_races:
            return {"has_next": False, "all_done": True}

        # Apply offset (clamp to valid range)
        idx = min(offset, len(upcoming_races) - 1)
        idx = max(0, idx)
        next_race = upcoming_races[idx]

        meeting = meetings.get(next_race.meeting_id)
        # Add timezone info for JavaScript
        start_time_aware = next_race.start_time.replace(tzinfo=MELB_TZ)

        return {
            "has_next": True,
            "venue": meeting.venue if meeting else "Unknown",
            "meeting_id": next_race.meeting_id,
            "race_number": next_race.race_number,
            "race_name": next_race.name,
            "distance": next_race.distance,
            "class": next_race.class_,
            "start_time_iso": start_time_aware.isoformat(),
            "start_time_formatted": next_race.start_time.strftime("%H:%M"),
            "race_offset": idx,
            "total_upcoming": len(upcoming_races),
        }


async def _get_picks_for_race(meeting_id: str, race_number: int) -> list[dict]:
    """Direct query for picks in a specific race — fallback when upcoming filter misses them."""
    import json as _json_fb

    async with async_session() as db:
        active_content = select(Content.id).where(
            Content.status.notin_(["superseded", "rejected"])
        ).scalar_subquery()

        result = await db.execute(
            select(Pick)
            .where(
                Pick.meeting_id == meeting_id,
                Pick.race_number == race_number,
                Pick.content_id.in_(active_content),
                Pick.settled == False,
            )
            .order_by(Pick.tip_rank.nullslast())
        )
        picks = result.scalars().all()

    out = []
    for pick in picks:
        if pick.pick_type not in ("selection", "exotic"):
            continue

        if pick.pick_type == "selection":
            name = pick.horse_name or "Runner"
        else:
            name = f"{pick.exotic_type or 'Exotic'} R{pick.race_number}"

        exotic_runners = None
        if pick.pick_type == "exotic" and pick.exotic_runners:
            try:
                exotic_runners = _json_fb.loads(pick.exotic_runners) if isinstance(pick.exotic_runners, str) else pick.exotic_runners
            except (ValueError, TypeError):
                exotic_runners = None
            if exotic_runners and isinstance(exotic_runners, list) and any(isinstance(r, list) for r in exotic_runners):
                exotic_runners = [item for sub in exotic_runners for item in (sub if isinstance(sub, list) else [sub])]

        # Normalise to percentage (0-100). DB stores some as decimal (0.306),
        # some as percentage (30.6) depending on source (prob engine vs JSON parser).
        raw_wp = pick.win_probability or 0
        raw_pp = pick.place_probability or 0
        wp = round((raw_wp * 100 if raw_wp <= 1 else raw_wp), 1) if raw_wp else None
        pp = round((raw_pp * 100 if raw_pp <= 1 else raw_pp), 1) if raw_pp else None
        bt_lower = (pick.bet_type or "").lower()
        show_prob = pp if bt_lower == "place" and pp else wp

        out.append({
            "name": name,
            "saddlecloth": pick.saddlecloth,
            "venue": "",
            "meeting_id": meeting_id,
            "race_number": race_number,
            "pick_type": pick.pick_type,
            "bet_type": (pick.bet_type or "").replace("_", " ").title(),
            "exotic_type": pick.exotic_type,
            "exotic_runners": exotic_runners,
            "odds": pick.odds_at_tip,
            "stake": round(pick.bet_stake or pick.exotic_stake or 0, 2),
            "tip_rank": pick.tip_rank,
            "value_rating": round(pick.value_rating, 2) if pick.value_rating else None,
            "win_prob": wp,
            "place_prob": pp,
            "show_prob": show_prob,
            "confidence": pick.confidence,
            "is_puntys_pick": pick.is_puntys_pick or False,
            "start_time": None,
            "is_edge": (pick.value_rating or 0) >= 1.1 and pick.pick_type == "selection",
        })
    return out


def _compute_pick_data(all_picks: list) -> dict:
    """Compute winners, sequences, selections lookup, and stats from a single picks list.

    Returns dict with keys: winners_map, winning_exotics, winning_sequences,
    sequence_results, picks_lookup, meeting_stats.
    """
    winners_map = {}       # {race_number: [saddlecloth, ...]}
    winning_exotics = {}   # {race_number: exotic_type}
    losing_exotics = {}    # {race_number: exotic_type}  — settled but NOT hit
    winning_sequences = [] # [{type, variant}]
    sequence_results = []
    picks_lookup = {}      # {race_number: {saddlecloth: {...}}}
    pp_picks = {}          # {race_number: saddlecloth}  — Punty's Pick per race

    # Accumulators for stats
    sel_stats = {}   # {bet_type: {total, hits, pnl, staked}}
    ex_stats = {}    # {norm_type: {total, hits, pnl, staked}}
    seq_stats = {}   # {(seq_type, variant): {total, hits, pnl, staked}}
    b3_stats = {"total": 0, "hits": 0, "pnl": 0.0, "staked": 0.0}

    for pick in all_picks:
        # --- Selection picks lookup (all, not just settled) ---
        if pick.pick_type == "selection" and pick.race_number and pick.saddlecloth:
            # Edge/confidence calculation
            bt_lower = (pick.bet_type or "").lower()
            model_prob = None
            if "place" in bt_lower and pick.place_probability:
                raw_pp = pick.place_probability
                model_prob = round(raw_pp * 100, 1) if raw_pp <= 1 else round(raw_pp, 1)
            elif pick.win_probability:
                raw_wp = pick.win_probability
                model_prob = round(raw_wp * 100, 1) if raw_wp <= 1 else round(raw_wp, 1)
            market_implied = round(100 / pick.odds_at_tip, 1) if pick.odds_at_tip else None
            edge_pct = round(model_prob - market_implied, 1) if model_prob and market_implied else None
            if edge_pct is not None:
                if edge_pct >= 10:
                    confidence = "HIGH EDGE"
                elif edge_pct >= 5:
                    confidence = "VALUE"
                elif edge_pct >= 3:
                    confidence = "EDGE"
                else:
                    confidence = "SPECULATIVE"
            else:
                confidence = None

            picks_lookup.setdefault(pick.race_number, {})[pick.saddlecloth] = {
                "tip_rank": pick.tip_rank,
                "bet_type": pick.bet_type,
                "hit": pick.hit,
                "pnl": float(pick.pnl) if pick.pnl is not None else None,
                "odds": float(pick.odds_at_tip) if pick.odds_at_tip else None,
                "stake": float(pick.bet_stake) if pick.bet_stake else None,
                "model_prob": model_prob,
                "market_implied": market_implied,
                "edge_pct": edge_pct,
                "confidence": confidence,
                "is_puntys_pick": bool(pick.is_puntys_pick),
                "is_roughie": bool(pick.is_roughie) if hasattr(pick, 'is_roughie') else (pick.tip_rank == 4),
            }
            # Track Punty's Pick per race (only if horse has an actual bet)
            if pick.is_puntys_pick:
                bt = (pick.bet_type or "").lower()
                if bt and bt not in ("no_bet", "exotics_only") and pick.bet_stake and pick.bet_stake > 0:
                    pp_picks[pick.race_number] = pick.saddlecloth
                else:
                    # PP on a no-bet pick — defer, will reassign below
                    pp_picks.setdefault(pick.race_number, None)

        # --- Winners (settled + hit) ---
        if pick.settled and pick.hit:
            if pick.pick_type in ("selection", "big3") and pick.race_number and pick.saddlecloth:
                winners_map.setdefault(pick.race_number, []).append(pick.saddlecloth)
            elif pick.pick_type == "exotic" and pick.race_number and pick.exotic_type:
                winning_exotics[pick.race_number] = pick.exotic_type
            elif pick.pick_type == "sequence" and pick.sequence_type:
                winning_sequences.append({
                    "type": pick.sequence_type,
                    "variant": pick.sequence_variant,
                })
            elif pick.pick_type == "big3_multi":
                winning_sequences.append({"type": "big3_multi", "variant": None})

        # --- Losing exotics (settled but NOT hit) ---
        if pick.settled and not pick.hit:
            if pick.pick_type == "exotic" and pick.race_number and pick.exotic_type:
                losing_exotics[pick.race_number] = pick.exotic_type

        # --- Settled sequence/multi results ---
        if pick.settled and pick.pick_type in ("sequence", "big3_multi"):
            seq_type = pick.sequence_type or pick.pick_type
            label = (pick.sequence_type or "multi").replace("_", " ").title()
            sequence_results.append({
                "type": seq_type,
                "variant": pick.sequence_variant,
                "label": label,
                "hit": bool(pick.hit),
                "pnl": float(pick.pnl) if pick.pnl is not None else 0.0,
                "stake": float(pick.exotic_stake or pick.bet_stake or 0),
                "start_race": pick.sequence_start_race,
            })

        # --- Stats accumulators (settled only) ---
        if not pick.settled:
            continue

        if pick.pick_type == "selection" and pick.bet_type and pick.bet_type != "exotics_only":
            bt = pick.bet_type
            if bt not in sel_stats:
                sel_stats[bt] = {"total": 0, "hits": 0, "pnl": 0.0, "staked": 0.0}
            sel_stats[bt]["total"] += 1
            if pick.hit:
                sel_stats[bt]["hits"] += 1
            sel_stats[bt]["pnl"] += float(pick.pnl or 0)
            sel_stats[bt]["staked"] += float(pick.bet_stake or 0)

        elif pick.pick_type == "exotic" and pick.exotic_type:
            label = _norm_exotic(pick.exotic_type)
            if label not in ex_stats:
                ex_stats[label] = {"total": 0, "hits": 0, "pnl": 0.0, "staked": 0.0}
            ex_stats[label]["total"] += 1
            if pick.hit:
                ex_stats[label]["hits"] += 1
            ex_stats[label]["pnl"] += float(pick.pnl or 0)
            ex_stats[label]["staked"] += float(pick.exotic_stake or 0)

        elif pick.pick_type == "sequence":
            key = (pick.sequence_type, pick.sequence_variant)
            if key not in seq_stats:
                seq_stats[key] = {"total": 0, "hits": 0, "pnl": 0.0, "staked": 0.0}
            seq_stats[key]["total"] += 1
            if pick.hit:
                seq_stats[key]["hits"] += 1
            seq_stats[key]["pnl"] += float(pick.pnl or 0)
            seq_stats[key]["staked"] += float(pick.exotic_stake or 0)

        elif pick.pick_type == "big3_multi":
            b3_stats["total"] += 1
            if pick.hit:
                b3_stats["hits"] += 1
            b3_stats["pnl"] += float(pick.pnl or 0)
            b3_stats["staked"] += float(pick.exotic_stake or 0)

    # --- Reassign Punty's Pick from no-bet picks to best staked pick ---
    for rn, sc in list(pp_picks.items()):
        if sc is None:
            # Find highest-ranked staked selection in this race
            race_picks = picks_lookup.get(rn, {})
            best_sc, best_rank = None, 999
            for s, info in race_picks.items():
                bt = (info.get("bet_type") or "").lower()
                if bt and bt not in ("no_bet", "exotics_only") and info.get("stake") and info["stake"] > 0:
                    rank = info.get("tip_rank") or 999
                    if rank < best_rank:
                        best_rank = rank
                        best_sc = s
            if best_sc:
                pp_picks[rn] = best_sc
            else:
                del pp_picks[rn]

    # --- Build meeting_stats list ---
    meeting_stats = []

    # Selections by bet_type (ordered)
    for k in ["win", "saver_win", "place", "each_way"]:
        if k in sel_stats:
            s = sel_stats[k]
            label = k.replace("_", " ").title()
            meeting_stats.append({
                "category": "Selections", "type": label,
                "won": s["hits"], "total": s["total"],
                "rate": round(s["hits"] / s["total"] * 100, 1) if s["total"] else 0,
                "pnl": round(s["pnl"], 2), "staked": round(s["staked"], 2),
            })

    # Exotics (ordered)
    ex_order = ["Quinella", "Exacta", "Trifecta", "First 4"]
    for k in ex_order:
        if k in ex_stats:
            s = ex_stats[k]
            meeting_stats.append({
                "category": "Exotics", "type": k,
                "won": s["hits"], "total": s["total"],
                "rate": round(s["hits"] / s["total"] * 100, 1) if s["total"] else 0,
                "pnl": round(s["pnl"], 2), "staked": round(s["staked"], 2),
            })
    for k, s in ex_stats.items():
        if k not in ex_order:
            meeting_stats.append({
                "category": "Exotics", "type": k,
                "won": s["hits"], "total": s["total"],
                "rate": round(s["hits"] / s["total"] * 100, 1) if s["total"] else 0,
                "pnl": round(s["pnl"], 2), "staked": round(s["staked"], 2),
            })

    # Sequences
    for (st, sv), s in seq_stats.items():
        label = (st or "Sequence").replace("_", " ").title()
        meeting_stats.append({
            "category": "Sequences", "type": label,
            "won": s["hits"], "total": s["total"],
            "rate": round(s["hits"] / s["total"] * 100, 1) if s["total"] else 0,
            "pnl": round(s["pnl"], 2), "staked": round(s["staked"], 2),
        })

    # Big3 Multi
    if b3_stats["total"] > 0:
        meeting_stats.append({
            "category": "Multi", "type": "Big 3 Multi",
            "won": b3_stats["hits"], "total": b3_stats["total"],
            "rate": round(b3_stats["hits"] / b3_stats["total"] * 100, 1) if b3_stats["total"] else 0,
            "pnl": round(b3_stats["pnl"], 2), "staked": round(b3_stats["staked"], 2),
        })

    return {
        "winners_map": winners_map,
        "winning_exotics": winning_exotics,
        "losing_exotics": losing_exotics,
        "winning_sequences": winning_sequences,
        "sequence_results": sequence_results,
        "picks_lookup": picks_lookup,
        "pp_picks": pp_picks,
        "meeting_stats": meeting_stats,
    }


async def get_daily_dashboard() -> dict:
    """Build today's daily scoreboard — results, upcoming, timeline, insights."""
    from sqlalchemy import desc, func as sa_func
    from collections import Counter, defaultdict
    from punty.results.celebrations import get_celebration
    from punty.results.picks import get_performance_summary

    today = melb_today()
    now = melb_now()
    now_naive = now.replace(tzinfo=None)

    async with async_session() as db:
        # ── Performance summary (reuse existing) ──
        performance = await get_performance_summary(db, today)

        # ── Subquery: active content only ──
        active_content = select(Content.id).where(
            Content.status.notin_(["superseded", "rejected"])
        ).scalar_subquery()

        # ── ALL picks today (settled + unsettled) with runner/race details ──
        # Use coalesce so sequences (race_number=NULL) join via sequence_start_race
        pick_race_num = sa_func.coalesce(Pick.race_number, Pick.sequence_start_race)
        result = await db.execute(
            select(Pick, Runner, Race, Meeting)
            .join(Meeting, Pick.meeting_id == Meeting.id)
            .outerjoin(Race, and_(
                Race.meeting_id == Pick.meeting_id,
                Race.race_number == pick_race_num,
            ))
            .outerjoin(Runner, and_(
                Runner.race_id == Race.id,
                Runner.saddlecloth == Pick.saddlecloth,
            ))
            .where(
                Meeting.date == today,
                Meeting.selected == True,
                Pick.content_id.in_(active_content),
            )
            .order_by(Meeting.venue, Race.race_number, Pick.tip_rank.nullslast())
        )
        all_rows = result.all()

        # Split into settled and unsettled
        settled_rows = [(p, r, rc, m) for p, r, rc, m in all_rows if p.settled]
        unsettled_rows = [(p, r, rc, m) for p, r, rc, m in all_rows if not p.settled]

        # Sort settled by PNL desc for big wins
        settled_by_pnl = sorted(settled_rows, key=lambda x: x[0].pnl or 0, reverse=True)

        # ── Big Wins (top 10 by PNL) ──
        big_wins = []
        for pick, runner, race, meeting in settled_by_pnl:
            if not pick.hit or not pick.pnl or pick.pnl <= 0:
                continue
            big_wins.append(_build_win_card(pick, runner, race, meeting, get_celebration))
            if len(big_wins) >= 10:
                break

        # ── P&L Timeline (settled picks ordered by settled_at for chart) ──
        timeline = []
        settled_chrono = sorted(settled_rows, key=lambda x: x[0].settled_at or x[0].created_at)
        running_pnl = 0.0
        for pick, runner, race, meeting in settled_chrono:
            pnl = pick.pnl or 0
            running_pnl += pnl
            # Use race start time or settled_at for x-axis
            ts = (race.start_time if race else None) or pick.settled_at or pick.created_at
            rn = pick.race_number or pick.sequence_start_race or "?"
            timeline.append({
                "time": ts.strftime("%H:%M") if ts else "?",
                "pnl": round(pnl, 2),
                "cumulative": round(running_pnl, 2),
                "label": f"{meeting.venue} R{rn}",
                "hit": bool(pick.hit),
            })

        # ── Upcoming bets (unsettled, race not finished) ──
        upcoming = []
        for pick, runner, race, meeting in unsettled_rows:
            if pick.pick_type not in ("selection", "exotic", "sequence", "big3_multi"):
                continue
            # Race status
            rs = ((race.results_status if race else None) or "").lower()
            if rs in ("paying", "closed", "final"):
                continue  # Already finished, just not settled yet — skip
            # Display name
            if pick.pick_type == "selection":
                name = pick.horse_name or "Runner"
            elif pick.pick_type == "exotic":
                name = f"{pick.exotic_type or 'Exotic'} R{pick.race_number}"
            elif pick.pick_type == "sequence":
                name = f"{(pick.sequence_variant or '').title()} {pick.sequence_type or 'Sequence'}"
            else:
                name = pick.horse_name or f"R{pick.race_number}"

            stake = pick.bet_stake or pick.exotic_stake or 0
            # Probability: use place_prob for Place bets, win_prob otherwise
            raw_wp = pick.win_probability or 0
            raw_pp = pick.place_probability or 0
            wp = round((raw_wp * 100 if raw_wp <= 1 else raw_wp), 1) if raw_wp else None
            pp = round((raw_pp * 100 if raw_pp <= 1 else raw_pp), 1) if raw_pp else None
            bt_lower = (pick.bet_type or "").lower()
            show_prob = pp if bt_lower == "place" and pp else wp

            # Exotic runners as flat list
            exotic_runners = None
            if pick.pick_type == "exotic" and pick.exotic_runners:
                import json as _json2
                try:
                    exotic_runners = _json2.loads(pick.exotic_runners) if isinstance(pick.exotic_runners, str) else pick.exotic_runners
                except (ValueError, TypeError):
                    exotic_runners = None
                # Flatten nested arrays (e.g. [[5], [8, 2, 1]] → [5, 8, 2, 1])
                if exotic_runners and isinstance(exotic_runners, list) and any(isinstance(r, list) for r in exotic_runners):
                    exotic_runners = [item for sub in exotic_runners for item in (sub if isinstance(sub, list) else [sub])]

            upcoming.append({
                "name": name,
                "saddlecloth": pick.saddlecloth,
                "venue": meeting.venue,
                "meeting_id": meeting.id,
                "race_number": pick.race_number,
                "pick_type": pick.pick_type,
                "bet_type": (pick.bet_type or "").replace("_", " ").title(),
                "exotic_type": pick.exotic_type,
                "exotic_runners": exotic_runners,
                "odds": pick.odds_at_tip,
                "stake": round(stake, 2),
                "tip_rank": pick.tip_rank,
                "value_rating": round(pick.value_rating, 2) if pick.value_rating else None,
                "win_prob": wp,
                "place_prob": pp,
                "show_prob": show_prob,
                "confidence": pick.confidence,
                "is_puntys_pick": pick.is_puntys_pick or False,
                "start_time": race.start_time.strftime("%H:%M") if race and race.start_time else None,
                "is_edge": (pick.value_rating or 0) >= 1.1 and pick.pick_type == "selection",
            })

        # ── Edge picks (upcoming selections with value_rating >= 1.1) ──
        edge_picks = [u for u in upcoming if u.get("is_edge")]
        edge_picks.sort(key=lambda x: x.get("value_rating") or 0, reverse=True)

        # ── Bet type breakdown (P&L from settled only; unsettled in count + staked) ──
        bt_stats = defaultdict(lambda: {"bets": 0, "winners": 0, "staked": 0.0, "pnl": 0.0})

        def _classify_bt(pick):
            """Classify pick into normalized bet type label, or None to skip."""
            if pick.pick_type == "big3":
                return None
            if pick.pick_type == "selection":
                raw_bt = str(pick.bet_type or "unknown").lower().replace("_", " ")
                if raw_bt in ("no bet", "no_bet", "exotics only", "exotics_only"):
                    return None
                return "Win" if raw_bt == "saver win" else raw_bt.title()
            if pick.pick_type == "exotic":
                raw_et = str(pick.exotic_type or "Exotic").lower()
                for suffix in (" standout", " box", " boxed"):
                    raw_et = raw_et.replace(suffix, "")
                return raw_et.strip().title()
            if pick.pick_type in ("sequence", "big3_multi"):
                return str(pick.sequence_type or pick.pick_type or "Sequence").replace("_", " ").title()
            return str(pick.pick_type or "unknown").replace("_", " ").title()

        # Settled picks: actual P&L and wins
        for pick, runner, race, meeting in settled_rows:
            bt = _classify_bt(pick)
            if not bt:
                continue
            bt_stats[bt]["bets"] += 1
            stake = pick.exotic_stake if pick.pick_type == "exotic" else pick.bet_stake
            bt_stats[bt]["staked"] += stake or 0
            bt_stats[bt]["pnl"] += pick.pnl or 0
            if pick.hit:
                bt_stats[bt]["winners"] += 1

        # Unsettled picks: include in bet count + staked, but NOT P&L
        for pick, runner, race, meeting in unsettled_rows:
            bt = _classify_bt(pick)
            if not bt:
                continue
            if pick.tracked_only:
                continue
            stake = pick.exotic_stake if pick.pick_type == "exotic" else pick.bet_stake
            if not stake or stake <= 0:
                continue
            bt_stats[bt]["bets"] += 1
            bt_stats[bt]["staked"] += stake
        bet_types = []
        for bt, data in sorted(bt_stats.items()):
            if bt in ("Skip", "None", "skip", "none", "", None):
                continue
            data["strike_rate"] = round(data["winners"] / data["bets"] * 100, 1) if data["bets"] else 0
            data["roi"] = round(data["pnl"] / data["staked"] * 100, 1) if data["staked"] else 0
            data["name"] = bt
            data["staked"] = round(data["staked"], 2)
            data["pnl"] = round(data["pnl"], 2)
            bet_types.append(data)

        # ── Rank Strike Rates ──
        # Rank 1: win strike (1st place), Rank 2-4: place strike (top 3)
        rank_stats = {}
        for pick, runner, race, meeting in settled_rows:
            if pick.pick_type != "selection" or not pick.tip_rank:
                continue
            if (pick.bet_type or "").lower() == "no_bet":
                continue
            rank = pick.tip_rank
            if rank not in rank_stats:
                rank_stats[rank] = {"bets": 0, "hits": 0}
            rank_stats[rank]["bets"] += 1
            fp = runner.finish_position if runner else None
            if fp:
                if rank == 1 and fp == 1:
                    rank_stats[rank]["hits"] += 1
                elif rank >= 2 and fp <= 3:
                    rank_stats[rank]["hits"] += 1
        rank_strike = []
        for rank in sorted(rank_stats.keys()):
            rs = rank_stats[rank]
            strike = round(rs["hits"] / rs["bets"] * 100, 1) if rs["bets"] else 0
            rank_strike.append({
                "rank": rank,
                "bets": rs["bets"],
                "hits": rs["hits"],
                "strike": strike,
                "label": "Win" if rank == 1 else "Place",
            })

        # ── Insights ──
        insights = []
        winning_selections = [
            (p, r, rc, m) for p, r, rc, m in settled_rows
            if p.hit and p.pnl and p.pnl > 0 and p.pick_type == "selection"
        ]
        all_selections = [
            (p, r, rc, m) for p, r, rc, m in settled_rows if p.pick_type == "selection"
        ]

        # Hot Jockey
        jockey_wins = Counter()
        jockey_rides = Counter()
        for pick, runner, race, meeting in all_selections:
            j = runner.jockey if runner else None
            if j:
                jockey_rides[j] += 1
                if pick.hit:
                    jockey_wins[j] += 1
        if jockey_wins:
            hot_j, hot_j_wins = jockey_wins.most_common(1)[0]
            hot_j_rides = jockey_rides[hot_j]
            parts = hot_j.split()
            short_j = f"{parts[0][0]} {' '.join(parts[1:])}" if len(parts) > 1 else hot_j
            insights.append({
                "icon": "fire", "label": "Hot Jockey",
                "text": f"{short_j} ({hot_j_wins}/{hot_j_rides} winners)",
            })

        # Hot Trainer
        trainer_wins = Counter()
        trainer_runs = Counter()
        for pick, runner, race, meeting in all_selections:
            t = runner.trainer if runner else None
            if t:
                trainer_runs[t] += 1
                if pick.hit:
                    trainer_wins[t] += 1
        if trainer_wins:
            hot_t, hot_t_wins = trainer_wins.most_common(1)[0]
            hot_t_runs = trainer_runs[hot_t]
            parts = hot_t.split()
            short_t = f"{parts[0][0]} {' '.join(parts[1:])}" if len(parts) > 1 else hot_t
            insights.append({
                "icon": "trophy", "label": "Hot Trainer",
                "text": f"{short_t} ({hot_t_wins}/{hot_t_runs} winners)",
            })

        # Best Odds Win
        if winning_selections:
            best_odds_win = max(winning_selections, key=lambda x: x[0].odds_at_tip or 0)
            bo_pick = best_odds_win[0]
            if bo_pick.odds_at_tip and bo_pick.odds_at_tip >= 4.0:
                insights.append({
                    "icon": "zap", "label": "Best Odds",
                    "text": f"{bo_pick.horse_name} at ${bo_pick.odds_at_tip:.2f} saluted",
                })

        # Frontrunners vs closers
        pace_wins = Counter()
        for pick, runner, race, meeting in winning_selections:
            smp = (runner.speed_map_position if runner else "") or ""
            if smp.lower() in ("leader", "on_pace"):
                pace_wins["front"] += 1
            elif smp.lower() in ("midfield", "backmarker"):
                pace_wins["back"] += 1
        total_pace = pace_wins["front"] + pace_wins["back"]
        if total_pace >= 3:
            insights.append({
                "icon": "horse", "label": "Pace",
                "text": f"{pace_wins['front']}/{total_pace} winners led or sat on pace",
            })

        # Best Venue
        venue_pnl = Counter()
        venue_bets = Counter()
        for pick, runner, race, meeting in settled_rows:
            venue_bets[meeting.venue] += 1
            venue_pnl[meeting.venue] += pick.pnl or 0
        if venue_pnl:
            best_venue = max(venue_pnl, key=venue_pnl.get)
            bv_pnl = venue_pnl[best_venue]
            bv_bets = venue_bets[best_venue]
            if bv_pnl > 0:
                insights.append({
                    "icon": "pin", "label": "Best Venue",
                    "text": f"{best_venue} +${bv_pnl:.0f} from {bv_bets} bets",
                })

        # Best bet type
        if bet_types:
            best_bt = max(bet_types, key=lambda x: x["pnl"])
            if best_bt["pnl"] > 0:
                insights.append({
                    "icon": "chart", "label": "Bet Type",
                    "text": f"{best_bt['name']} bets leading today ({best_bt['roi']:+.0f}% ROI)",
                })

        # ── Venue breakdown ──
        venues = []
        # Include unsettled counts too
        venue_upcoming = Counter()
        for pick, runner, race, meeting in unsettled_rows:
            venue_upcoming[meeting.venue] += 1
        all_venue_names = set(list(venue_bets.keys()) + list(venue_upcoming.keys()))
        for v in sorted(all_venue_names):
            v_wins = sum(1 for p, r, rc, m in settled_rows if m.venue == v and p.hit and p.pnl and p.pnl > 0)
            v_total = venue_bets.get(v, 0)
            v_pnl = venue_pnl.get(v, 0)
            v_up = venue_upcoming.get(v, 0)
            # Get meeting_id for cross-link
            v_mid = None
            for p, r, rc, m in all_rows:
                if m.venue == v:
                    v_mid = m.id
                    break
            venues.append({
                "name": v,
                "meeting_id": v_mid,
                "bets": v_total,
                "winners": v_wins,
                "upcoming": v_up,
                "pnl": round(v_pnl, 2),
            })

        # ── Recent Race Results (last 3 settled races, all picks per race) ──
        race_groups: dict[str, dict] = {}  # key = "venue-rN"
        for pick, runner, race_obj, meeting in settled_rows:
            if pick.pick_type not in ("selection", "exotic"):
                continue
            rn = pick.race_number
            if not rn:
                continue
            key = f"{meeting.id}-r{rn}"
            if key not in race_groups:
                race_groups[key] = {
                    "venue": meeting.venue,
                    "meeting_id": meeting.id,
                    "race_number": rn,
                    "race_name": race_obj.name if race_obj else None,
                    "settled_at": pick.settled_at or pick.created_at,
                    "picks": [],
                    "exotics": [],
                    "total_pnl": 0.0,
                    "total_staked": 0.0,
                }
            rg = race_groups[key]
            raw_pnl = pick.pnl or 0

            if pick.pick_type == "selection":
                bt_lower = (pick.bet_type or "").lower()
                is_no_bet = bt_lower == "no_bet"
                # no_bet picks: zero out phantom PNL from settlement bug
                pnl = 0.0 if is_no_bet else raw_pnl
                stake = pick.bet_stake or 0
                if not is_no_bet:
                    rg["total_staked"] += stake
                    rg["total_pnl"] += pnl
                ts = pick.settled_at or pick.created_at
                if ts and (not rg["settled_at"] or ts > rg["settled_at"]):
                    rg["settled_at"] = ts
                rg["picks"].append({
                    "name": pick.horse_name or "Runner",
                    "saddlecloth": pick.saddlecloth,
                    "tip_rank": pick.tip_rank,
                    "bet_type": (pick.bet_type or "").replace("_", " ").title(),
                    "odds": pick.odds_at_tip,
                    "hit": bool(pick.hit),
                    "pnl": round(pnl, 2),
                    "is_no_bet": is_no_bet,
                    "is_puntys_pick": pick.is_puntys_pick or False,
                    "finish_pos": runner.finish_position if runner else None,
                })
            elif pick.pick_type == "exotic":
                stake = pick.exotic_stake or 0
                rg["total_staked"] += stake
                rg["total_pnl"] += raw_pnl
                ts = pick.settled_at or pick.created_at
                if ts and (not rg["settled_at"] or ts > rg["settled_at"]):
                    rg["settled_at"] = ts
                rg["exotics"].append({
                    "exotic_type": (pick.exotic_type or "Exotic").replace("_", " "),
                    "hit": bool(pick.hit),
                    "pnl": round(raw_pnl, 2),
                    "stake": round(stake, 2),
                })

        # Sort picks within each race by tip_rank; exotics by type
        for rg in race_groups.values():
            rg["picks"].sort(key=lambda x: x.get("tip_rank") or 99)
            rg["total_pnl"] = round(rg["total_pnl"], 2)
            rg["total_staked"] = round(rg["total_staked"], 2)
        # Take most recently settled 3 races
        recent_race_results = sorted(
            race_groups.values(),
            key=lambda x: x["settled_at"] or now_naive,
            reverse=True,
        )[:3]

        # ── Meeting P&L timeline (per-meeting cumulative P&L by race) ──
        # Aggregate all picks per (venue, race_number) into one net P&L
        meeting_race_pnl: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        for pick, runner, race, meeting in settled_chrono:
            pnl = pick.pnl or 0
            venue = meeting.venue
            rn = pick.race_number or pick.sequence_start_race or 0
            if rn:
                meeting_race_pnl[venue][rn] += pnl

        # Build cumulative timeline with one point per race
        meeting_pnl = []
        for venue in sorted(meeting_race_pnl.keys()):
            races = meeting_race_pnl[venue]
            entries = []
            cum = 0.0
            for rn in sorted(races.keys()):
                cum += races[rn]
                entries.append({
                    "race": rn,
                    "pnl": round(races[rn], 2),
                    "cumulative": round(cum, 2),
                })
            if entries:
                meeting_pnl.append({"venue": venue, "data": entries})

        # ── Summary counts for has_data logic ──
        total_picks = len(all_rows)
        total_settled = len(settled_rows)
        total_upcoming = len(unsettled_rows)

        return {
            "date": today.strftime("%d %B %Y").lstrip("0"),
            "date_iso": today.isoformat(),
            "performance": performance,
            "big_wins": big_wins,
            "timeline": timeline,
            "meeting_pnl": meeting_pnl,
            "upcoming": upcoming[:30],  # Cap at 30 for page size
            "edge_picks": edge_picks[:8],
            "bet_types": bet_types,
            "rank_strike": rank_strike,
            "insights": insights,
            "venues": venues,
            "total_picks": total_picks,
            "total_settled": total_settled,
            "total_upcoming": total_upcoming,
            "has_data": total_picks > 0,
            "has_settled": total_settled > 0,
            "recent_race_results": recent_race_results,
        }


async def get_best_of_meets() -> dict:
    """Get best winner, roughie, and exotic pick per today's meeting."""
    today = melb_today()

    async with async_session() as db:
        active_content = select(Content.id).where(
            Content.status.notin_(["superseded", "rejected"])
        ).scalar_subquery()

        result = await db.execute(
            select(Pick, Meeting)
            .join(Meeting, Pick.meeting_id == Meeting.id)
            .where(
                Meeting.date == today,
                Meeting.selected == True,
                Pick.content_id.in_(active_content),
            )
        )
        rows = result.all()

    from punty.venues import guess_state as get_state_for_track

    meets = {}
    for pick, meeting in rows:
        mid = meeting.id
        if mid not in meets:
            meets[mid] = {
                "venue": meeting.venue,
                "meeting_id": mid,
                "state": get_state_for_track(meeting.venue) or "AUS",
                "best_winner": None,
                "roughie": None,
                "exotic": None,
            }

        # Best Winner: selection, tip_rank <= 3, Win/Saver Win only, highest win_prob
        bt_lower = (pick.bet_type or "").lower()
        if (pick.pick_type == "selection"
                and (pick.tip_rank or 99) <= 3
                and bt_lower in ("win", "saver_win", "saver win")
                and (pick.win_probability or 0) >= 0.22):
            current = meets[mid]["best_winner"]
            if not current or (pick.win_probability or 0) > (current.get("win_prob") or 0):
                meets[mid]["best_winner"] = {
                    "horse": pick.horse_name,
                    "race": pick.race_number,
                    "odds": pick.odds_at_tip,
                    "bet_type": (pick.bet_type or "").replace("_", " ").title(),
                    "win_prob": pick.win_probability,
                    "tip_rank": pick.tip_rank,
                    "is_puntys_pick": pick.is_puntys_pick or False,
                }

        # Best Roughie: tip_rank == 4, highest value_rating, odds >= $8
        if (pick.pick_type == "selection"
                and pick.tip_rank == 4
                and (pick.odds_at_tip or 0) >= 8
                and (pick.win_probability or 0) >= 0.08):
            current = meets[mid]["roughie"]
            if not current or (pick.value_rating or 0) > (current.get("value_rating") or 0):
                meets[mid]["roughie"] = {
                    "horse": pick.horse_name,
                    "race": pick.race_number,
                    "odds": pick.odds_at_tip,
                    "bet_type": (pick.bet_type or "").replace("_", " ").title(),
                    "value_rating": pick.value_rating,
                }

        # Best Exotic: exotic with highest stake per meeting (prefer higher race)
        if pick.pick_type == "exotic":
            import json as _json3
            runners = pick.exotic_runners
            if isinstance(runners, str):
                try:
                    runners = _json3.loads(runners)
                except (ValueError, TypeError):
                    runners = []
            # Flatten nested arrays (e.g. [[5], [8, 2, 1]] → [5, 8, 2, 1])
            if runners and isinstance(runners, list) and any(isinstance(r, list) for r in runners):
                runners = [item for sub in runners for item in (sub if isinstance(sub, list) else [sub])]
            new_exotic = {
                "type": pick.exotic_type,
                "race": pick.race_number,
                "runners": runners or [],
                "stake": pick.exotic_stake or 0,
            }
            current = meets[mid]["exotic"]
            # Replace if no current, higher stake, or same stake but later race
            if (not current
                    or (new_exotic["stake"] or 0) > (current.get("stake") or 0)
                    or ((new_exotic["stake"] or 0) == (current.get("stake") or 0)
                        and (new_exotic["race"] or 0) > (current.get("race") or 0))):
                meets[mid]["exotic"] = new_exotic

    return meets


async def get_all_sequences() -> list:
    """Get all sequence bets (quaddies, big6, etc.) for today's meetings.
    Excludes sequences whose first leg race has already started."""
    today = melb_today()
    import json as _json
    from datetime import datetime, timezone

    async with async_session() as db:
        active_content = select(Content.id).where(
            Content.status.notin_(["superseded", "rejected"])
        ).scalar_subquery()

        result = await db.execute(
            select(Pick, Meeting)
            .join(Meeting, Pick.meeting_id == Meeting.id)
            .where(
                Meeting.date == today,
                Meeting.selected == True,
                Pick.content_id.in_(active_content),
                Pick.pick_type == "sequence",
            )
        )
        rows = result.all()

        # Fetch race start times for filtering started sequences
        race_times = {}
        if rows:
            meeting_ids = list({m.id for _, m in rows})
            races_result = await db.execute(
                select(Race.meeting_id, Race.race_number, Race.start_time)
                .where(Race.meeting_id.in_(meeting_ids))
            )
            for mid, rn, st in races_result.all():
                race_times[(mid, rn)] = st

    if not rows:
        return []

    from zoneinfo import ZoneInfo
    MELB_TZ_LOCAL = ZoneInfo("Australia/Melbourne")
    now = datetime.now(MELB_TZ_LOCAL)
    sequences = []
    for pick, meeting in rows:
        # Skip sequences whose last leg has already started (fully underway)
        start_race = pick.sequence_start_race or 1
        legs_data = pick.sequence_legs
        if isinstance(legs_data, str):
            try:
                legs_data = _json.loads(legs_data)
            except (ValueError, TypeError):
                legs_data = []
        # Handle string format "1,3/4,2/2,7/1,3,5" (legs separated by /)
        if isinstance(legs_data, str) and "/" in legs_data:
            legs_data = [
                [int(r) for r in leg.strip().split(",") if r.strip().isdigit()]
                for leg in legs_data.split("/")
            ]
        elif isinstance(legs_data, str):
            legs_data = []
        num_legs = len(legs_data) if legs_data else 4
        last_leg_race = start_race + num_legs - 1
        last_leg_time = race_times.get((meeting.id, last_leg_race))
        if last_leg_time:
            if last_leg_time.tzinfo is None:
                last_leg_time = last_leg_time.replace(tzinfo=MELB_TZ_LOCAL)
            if last_leg_time <= now:
                continue
        first_leg_time = race_times.get((meeting.id, start_race))
        legs = legs_data

        combo_count = 1
        for leg in (legs or []):
            if isinstance(leg, list):
                combo_count *= len(leg)
            elif isinstance(leg, dict):
                combo_count *= len(leg.get("runners", [1]))

        # Build compact leg summaries (e.g. "R5: 3,9,6")
        leg_summaries = []
        start = pick.sequence_start_race or 1
        for i, leg in enumerate(legs or []):
            if isinstance(leg, list):
                runners = leg
            elif isinstance(leg, dict):
                runners = leg.get("runners", [])
            else:
                runners = [leg]
            leg_summaries.append({
                "race": start + i,
                "runners": [str(r) for r in runners],
            })

        # Store first leg time for sorting and display
        flt = first_leg_time
        if flt and flt.tzinfo is None:
            from zoneinfo import ZoneInfo
            flt = flt.replace(tzinfo=ZoneInfo("Australia/Melbourne"))

        sequences.append({
            "venue": meeting.venue,
            "meeting_id": meeting.id,
            "type": (pick.sequence_type or "quaddie").title(),
            "variant": (pick.sequence_variant or "").title(),
            "legs": leg_summaries,
            "combo_count": combo_count,
            "stake": pick.exotic_stake or pick.bet_stake,
            "first_leg_time": flt.isoformat() if flt else None,
            "first_leg_race": start_race,
        })

    # Sort by first leg time (soonest first = "up next")
    sequences.sort(key=lambda s: s.get("first_leg_time") or "9999")

    return sequences


@router.get("/tips", response_class=HTMLResponse)
async def tips_dashboard(request: Request):
    """Racing dashboard — one-stop page for today's racing."""
    import asyncio
    from datetime import timedelta
    from punty.results.picks import get_performance_history
    from punty.public.pages import get_winner_stats, get_recent_wins_public

    today = melb_today()
    week_ago = today - timedelta(days=7)

    # Run all data fetches in parallel
    dashboard, stats, next_race_data, recent_wins = await asyncio.gather(
        get_daily_dashboard(),
        get_winner_stats(today=True),
        get_next_race(),
        get_recent_wins_public(15),
    )

    # 7-day performance history
    async with async_session() as db:
        perf_history = await get_performance_history(db, week_ago, today)

    seven_day_pnl = sum(d["pnl"] for d in perf_history)
    seven_day_bets = sum(d["bets"] for d in perf_history)
    seven_day_roi = round(seven_day_pnl / (seven_day_bets * 10) * 100, 1) if seven_day_bets else 0

    # Always fetch today's selected meetings with next-race jump times
    from punty.venues import guess_state as get_state_for_track, is_metro
    tips_meeting_ids = {t["meeting_id"] for t in stats.get("todays_tips", [])}
    async with async_session() as db:
        # Get meetings with their next un-run race start_time for sorting
        # Two subqueries: one for min start_time, one for the race_number
        # that corresponds to that start_time (can't get both in one GROUP BY)
        next_jump_sq = (
            select(
                Race.meeting_id,
                func.min(Race.start_time).label("next_jump"),
            )
            .where(
                or_(
                    Race.results_status.is_(None),
                    Race.results_status.in_(["Open", "scheduled"]),
                )
            )
            .group_by(Race.meeting_id)
            .subquery()
        )
        meetings_result = await db.execute(
            select(Meeting, next_jump_sq.c.next_jump)
            .outerjoin(next_jump_sq, Meeting.id == next_jump_sq.c.meeting_id)
            .where(
                and_(
                    Meeting.date == today,
                    Meeting.selected == True,
                    Meeting.meeting_type.in_(["race", None]),
                )
            )
            .order_by(next_jump_sq.c.next_jump.asc().nullslast())
        )
        today_meetings = meetings_result.all()

        # Resolve race_number for each meeting's next jump time
        next_race_nums: dict[str, int] = {}
        for m, next_jump in today_meetings:
            if next_jump:
                rn_result = await db.execute(
                    select(Race.race_number)
                    .where(
                        Race.meeting_id == m.id,
                        Race.start_time == next_jump,
                    )
                    .limit(1)
                )
                rn = rn_result.scalar_one_or_none()
                if rn:
                    next_race_nums[m.id] = rn

    meetings_list = []
    for m, next_jump in today_meetings:
        meetings_list.append({
            "venue": m.venue,
            "meeting_id": m.id,
            "state": get_state_for_track(m.venue) or "AUS",
            "track_condition": m.track_condition,
            "next_jump": next_jump.isoformat() if next_jump else None,
            "next_race_num": next_race_nums.get(m.id),
            "is_metro": is_metro(m.venue),
            "has_tips": m.id in tips_meeting_ids,
        })

    # Enrich todays_tips with track_condition
    for tip in stats.get("todays_tips", []):
        tip["track_condition"] = None
        match = next((m for m in meetings_list if m["meeting_id"] == tip.get("meeting_id")), None)
        if match:
            tip["track_condition"] = match["track_condition"]

    # Best of meets + sequence bets
    best_of, sequences = await asyncio.gather(
        get_best_of_meets(),
        get_all_sequences(),
    )

    # Match next race to upcoming picks
    next_race_picks = []
    if next_race_data.get("has_next"):
        nr_mid = next_race_data.get("meeting_id")
        nr_rn = next_race_data.get("race_number")
        for u in dashboard.get("upcoming", []):
            if u.get("meeting_id") == nr_mid and u.get("race_number") == nr_rn:
                next_race_picks.append(u)

        # Fallback: direct query if upcoming filter returned empty
        if not next_race_picks:
            next_race_picks = await _get_picks_for_race(nr_mid, nr_rn)

    # Build scratched saddlecloths for next race
    next_race_scratched = set()
    if next_race_data.get("has_next"):
        nr_mid = next_race_data.get("meeting_id")
        nr_rn = next_race_data.get("race_number")
        race_id = f"{nr_mid}-r{nr_rn}"
        async with async_session() as db:
            runners_res = await db.execute(
                select(Runner).where(Runner.race_id == race_id, Runner.scratched == True)
            )
            next_race_scratched = {r.saddlecloth for r in runners_res.scalars().all() if r.saddlecloth}

    # Build primary_play from next race picks (Punty's Pick or rank 1)
    primary_play = None
    if next_race_picks and next_race_data.get("has_next"):
        # Prefer Punty's Pick, else rank 1
        pp = next((p for p in next_race_picks if p.get("is_puntys_pick")), None)
        if not pp:
            pp = next((p for p in next_race_picks if p.get("tip_rank") == 1), None)
        if not pp and next_race_picks:
            pp = next_race_picks[0]

        if pp and pp.get("odds"):
            bt_lower = (pp.get("bet_type") or "").lower()
            if "place" in bt_lower and pp.get("place_prob"):
                model_prob = pp["place_prob"] / 100
            elif pp.get("win_prob"):
                model_prob = pp["win_prob"] / 100
            else:
                model_prob = (pp.get("place_prob") or pp.get("win_prob") or 0) / 100
            market_implied = 1 / pp["odds"] if pp["odds"] else 0
            edge_pct = round((model_prob - market_implied) * 100, 1)
            if edge_pct >= 10:
                confidence = "HIGH EDGE"
            elif edge_pct >= 5:
                confidence = "VALUE"
            elif edge_pct >= 3:
                confidence = "EDGE"
            else:
                confidence = "SMALL EDGE"
            primary_play = {
                **pp,
                "model_prob": round(model_prob * 100, 1),
                "market_implied": round(market_implied * 100, 1),
                "edge_pct": edge_pct,
                "confidence": confidence,
            }

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "dashboard": dashboard,
            "stats": stats,
            "next_race": next_race_data,
            "next_race_picks": next_race_picks,
            "next_race_scratched": next_race_scratched,
            "recent_wins": recent_wins,
            "perf_history": perf_history,
            "seven_day_pnl": round(seven_day_pnl, 2),
            "seven_day_roi": seven_day_roi,
            "best_of": best_of,
            "sequences": sequences,
            "meetings_list": meetings_list,
            "primary_play": primary_play,
        }
    )


@router.get("/tips/_glance", response_class=HTMLResponse)
async def tips_glance(request: Request):
    """HTMX partial: at-a-glance strip refresh."""
    import asyncio
    from datetime import timedelta
    from punty.results.picks import get_performance_history
    from punty.public.pages import get_winner_stats

    today = melb_today()
    week_ago = today - timedelta(days=7)

    dashboard, stats, next_race_data = await asyncio.gather(
        get_daily_dashboard(),
        get_winner_stats(today=True),
        get_next_race(),
    )

    async with async_session() as db:
        perf_history = await get_performance_history(db, week_ago, today)

    seven_day_pnl = sum(d["pnl"] for d in perf_history)
    seven_day_bets = sum(d["bets"] for d in perf_history)
    seven_day_roi = round(seven_day_pnl / (seven_day_bets * 10) * 100, 1) if seven_day_bets else 0

    # Quick meetings count (independent of content approval)
    async with async_session() as db:
        meetings_count_result = await db.execute(
            select(func.count()).select_from(Meeting).where(
                and_(
                    Meeting.date == today,
                    Meeting.selected == True,
                    Meeting.meeting_type.in_(["race", None]),
                )
            )
        )
        meetings_count = meetings_count_result.scalar() or 0

    return templates.TemplateResponse(
        "partials/glance_strip.html",
        {
            "request": request,
            "dashboard": dashboard,
            "stats": stats,
            "next_race": next_race_data,
            "seven_day_pnl": round(seven_day_pnl, 2),
            "seven_day_roi": seven_day_roi,
            "meetings_count": meetings_count,
        }
    )


@router.get("/tips/_next-race", response_class=HTMLResponse)
async def tips_next_race(request: Request):
    """HTMX partial: next race hero refresh."""
    import asyncio

    dashboard, next_race_data = await asyncio.gather(
        get_daily_dashboard(),
        get_next_race(),
    )

    next_race_picks = []
    if next_race_data.get("has_next"):
        nr_mid = next_race_data.get("meeting_id")
        nr_rn = next_race_data.get("race_number")
        for u in dashboard.get("upcoming", []):
            if u.get("meeting_id") == nr_mid and u.get("race_number") == nr_rn:
                next_race_picks.append(u)

        # Fallback: direct query if upcoming filter returned empty
        if not next_race_picks:
            next_race_picks = await _get_picks_for_race(nr_mid, nr_rn)

    # Build scratched saddlecloths for next race
    next_race_scratched = set()
    if next_race_data.get("has_next"):
        nr_mid = next_race_data.get("meeting_id")
        nr_rn = next_race_data.get("race_number")
        race_id = f"{nr_mid}-r{nr_rn}"
        async with async_session() as db:
            runners_res = await db.execute(
                select(Runner).where(Runner.race_id == race_id, Runner.scratched == True)
            )
            next_race_scratched = {r.saddlecloth for r in runners_res.scalars().all() if r.saddlecloth}

    return templates.TemplateResponse(
        "partials/next_race.html",
        {
            "request": request,
            "next_race": next_race_data,
            "next_race_picks": next_race_picks,
            "next_race_scratched": next_race_scratched,
        }
    )


@router.get("/tips/_live-edge", response_class=HTMLResponse)
async def tips_live_edge(request: Request, exclude: str = "", offset: int = 0):
    """HTMX partial: Live Edge Zone refresh.

    Args:
        exclude: Comma-separated venue names to skip (for venue filter).
        offset: Race offset from next upcoming (0 = very next race).
    """
    import asyncio

    exclude_venues = [v.strip() for v in exclude.split(",") if v.strip()] if exclude else None

    dashboard, next_race_data = await asyncio.gather(
        get_daily_dashboard(),
        get_next_race(exclude_venues=exclude_venues, offset=max(0, offset)),
    )

    next_race_picks = []
    if next_race_data.get("has_next"):
        nr_mid = next_race_data.get("meeting_id")
        nr_rn = next_race_data.get("race_number")
        for u in dashboard.get("upcoming", []):
            if u.get("meeting_id") == nr_mid and u.get("race_number") == nr_rn:
                next_race_picks.append(u)
        if not next_race_picks:
            next_race_picks = await _get_picks_for_race(nr_mid, nr_rn)

    # Build primary_play
    primary_play = None
    if next_race_picks and next_race_data.get("has_next"):
        pp = next((p for p in next_race_picks if p.get("is_puntys_pick")), None)
        if not pp:
            pp = next((p for p in next_race_picks if p.get("tip_rank") == 1), None)
        if not pp and next_race_picks:
            pp = next_race_picks[0]

        if pp and pp.get("odds"):
            bt_lower = (pp.get("bet_type") or "").lower()
            if "place" in bt_lower and pp.get("place_prob"):
                model_prob = pp["place_prob"] / 100
            elif pp.get("win_prob"):
                model_prob = pp["win_prob"] / 100
            else:
                model_prob = (pp.get("place_prob") or pp.get("win_prob") or 0) / 100
            market_implied = 1 / pp["odds"] if pp["odds"] else 0
            edge_pct = round((model_prob - market_implied) * 100, 1)
            if edge_pct >= 10:
                confidence = "HIGH EDGE"
            elif edge_pct >= 5:
                confidence = "VALUE"
            elif edge_pct >= 3:
                confidence = "EDGE"
            else:
                confidence = "SMALL EDGE"
            primary_play = {
                **pp,
                "model_prob": round(model_prob * 100, 1),
                "market_implied": round(market_implied * 100, 1),
                "edge_pct": edge_pct,
                "confidence": confidence,
            }

    return templates.TemplateResponse(
        "partials/live_edge.html",
        {
            "request": request,
            "next_race": next_race_data,
            "next_race_picks": next_race_picks,
            "primary_play": primary_play,
        }
    )


@router.get("/tips/api/dashboard/recent-results", response_class=HTMLResponse)
async def tips_recent_results(request: Request):
    """HTMX partial: Recent Results auto-refresh (last 3 settled races)."""
    dashboard = await get_daily_dashboard()
    return templates.TemplateResponse(
        "partials/recent_results.html",
        {
            "request": request,
            "recent_race_results": dashboard.get("recent_race_results", []),
        }
    )


@router.get("/tips/_results-feed", response_class=HTMLResponse)
async def tips_results_feed(request: Request):
    """HTMX partial: results feed refresh."""
    import asyncio
    from punty.public.pages import get_recent_wins_public

    dashboard, recent_wins = await asyncio.gather(
        get_daily_dashboard(),
        get_recent_wins_public(15),
    )

    return templates.TemplateResponse(
        "partials/results_feed.html",
        {
            "request": request,
            "dashboard": dashboard,
            "recent_wins": recent_wins,
        }
    )


@router.get("/tips/archive", response_class=HTMLResponse)
async def tips_archive(
    request: Request,
    page: int = 1,
    venue: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
):
    """Public tips calendar page showing meetings by date."""
    from datetime import date as date_type
    from punty.public.pages import get_tips_calendar

    # Parse date strings
    parsed_from = None
    parsed_to = None
    if date_from:
        try:
            parsed_from = date_type.fromisoformat(date_from)
        except ValueError:
            pass
    if date_to:
        try:
            parsed_to = date_type.fromisoformat(date_to)
        except ValueError:
            pass

    calendar_data = await get_tips_calendar(
        page=page, per_page=30, venue=venue,
        date_from=parsed_from, date_to=parsed_to,
    )

    # Build filter query string for pagination links
    filter_params = ""
    if venue:
        filter_params += f"&venue={venue}"
    if date_from:
        filter_params += f"&date_from={date_from}"
    if date_to:
        filter_params += f"&date_to={date_to}"

    return templates.TemplateResponse(
        "tips.html",
        {
            "request": request,
            "calendar": calendar_data["calendar"],
            "page": calendar_data["page"],
            "total_pages": calendar_data["total_pages"],
            "total_dates": calendar_data["total_dates"],
            "has_next": calendar_data["has_next"],
            "has_prev": calendar_data["has_prev"],
            "venue": venue or "",
            "date_from": date_from or "",
            "date_to": date_to or "",
            "filter_params": filter_params,
        }
    )
