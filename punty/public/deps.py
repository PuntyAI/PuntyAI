"""Shared utilities for public website routes."""

from datetime import date, datetime
from pathlib import Path

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, FileResponse, Response, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, func, and_, or_, text

from punty.config import melb_today, melb_now, MELB_TZ
from punty.models.database import async_session
from punty.models.pick import Pick
from punty.models.content import Content
from punty.models.meeting import Meeting, Race, Runner
from punty.models.live_update import LiveUpdate

# Templates directory
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=templates_dir)

# Static files directory
static_dir = Path(__file__).parent / "static"

from punty.config import settings as _app_settings
templates.env.globals["is_staging"] = _app_settings.is_staging


def _norm_exotic(raw: str) -> str:
    """Normalize exotic type names for display."""
    low = raw.lower().strip()
    return {
        "box trifecta": "Trifecta", "trifecta box": "Trifecta",
        "trifecta (box)": "Trifecta", "trifecta (boxed)": "Trifecta",
        "trifecta boxed": "Trifecta", "exacta standout": "Exacta",
        "exacta (standout)": "Exacta",
        "trifecta standout": "Trifecta",
        "trifecta (standout)": "Trifecta",
        "first four": "First 4", "first 4": "First 4",
        "first four (boxed)": "First 4", "first four box": "First 4",
        "first4": "First 4", "first4 box": "First 4",
        "first 4 standout": "First 4", "first four standout": "First 4",
    }.get(low, raw)


def _build_win_card(pick, runner, race, meeting, get_celebration) -> dict:
    """Build a win card dict for the big wins section."""
    stake = pick.bet_stake or pick.exotic_stake or 1.0
    returned = stake + (pick.pnl or 0)

    if pick.pick_type == "selection":
        display_name = pick.horse_name or "Runner"
    elif pick.pick_type == "exotic":
        display_name = f"{pick.exotic_type or 'Exotic'} R{pick.race_number}"
    elif pick.pick_type == "sequence":
        display_name = f"{(pick.sequence_variant or '').title()} {pick.sequence_type or 'Sequence'}"
    elif pick.pick_type == "big3_multi":
        display_name = "Big 3 Multi"
    else:
        display_name = pick.horse_name or f"R{pick.race_number}"

    smp = (runner.speed_map_position if runner else None) or ""
    running_style = {
        "leader": "Led all the way", "on_pace": "Sat on pace",
        "midfield": "Settled midfield", "backmarker": "Came from behind", "": None,
    }.get(smp.lower(), smp.title() if smp else None)

    margin = runner.result_margin if runner else None
    if margin and pick.pick_type == "selection":
        margin_text = f"Won by {margin}"
    elif pick.pick_type == "exotic":
        margin_text = f"Paid ${returned:.2f}"
    else:
        margin_text = None

    return {
        "display_name": display_name,
        "venue": meeting.venue,
        "meeting_id": meeting.id,
        "race_number": pick.race_number,
        "race_name": race.name if race else None,
        "jockey": runner.jockey if runner else None,
        "trainer": runner.trainer if runner else None,
        "odds": pick.odds_at_tip,
        "bet_type": (pick.bet_type or "").replace("_", " ").title(),
        "stake": round(stake, 2),
        "returned": round(returned, 2),
        "pnl": round(pick.pnl, 2),
        "tip_rank": pick.tip_rank,
        "pick_type": pick.pick_type,
        "running_style": running_style,
        "margin_text": margin_text,
        "celebration": get_celebration(pick.pnl, pick.pick_type),
        "is_puntys_pick": pick.is_puntys_pick or False,
    }


# ---------------------------------------------------------------------------
# Betfair Tracker auth helpers
# ---------------------------------------------------------------------------

_BF_TRACKER_PASSWORD = "Spanks31+"
_BF_TRACKER_COOKIE = "bf_tracker_auth"


def _bf_make_token() -> str:
    """Create HMAC token for the betfair tracker cookie."""
    import hmac
    import hashlib
    from punty.config import settings
    return hmac.new(settings.secret_key.encode(), _BF_TRACKER_PASSWORD.encode(), hashlib.sha256).hexdigest()[:32]


def _bf_check_auth(request: Request) -> bool:
    """Check if the betfair tracker cookie is valid."""
    token = request.cookies.get(_BF_TRACKER_COOKIE, "")
    return token == _bf_make_token()
