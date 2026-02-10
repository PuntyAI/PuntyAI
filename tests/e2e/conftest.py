"""E2E test infrastructure — real server, auth bypass, DB seeding, screenshots."""

import base64
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
from itsdangerous import TimestampSigner

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
E2E_PORT = 18765
E2E_SECRET = "e2e-test-secret-key-do-not-use-in-prod"
BASE_URL = f"http://127.0.0.1:{E2E_PORT}"
TODAY = date.today()
YESTERDAY = TODAY - timedelta(days=1)
SCREENSHOTS_DIR = Path(__file__).parent / "screenshots"

# Host header that triggers public-site routing (must be in PUBLIC_SITE_HOSTS)
PUBLIC_HOST = "127.0.0.1:8000"


# ---------------------------------------------------------------------------
# Session cookie forging
# ---------------------------------------------------------------------------
def _forge_cookie(session_data: dict) -> str:
    """Create a signed Starlette SessionMiddleware cookie."""
    payload = base64.b64encode(json.dumps(session_data).encode("utf-8"))
    signer = TimestampSigner(E2E_SECRET)
    return signer.sign(payload).decode("utf-8")


_AUTH_SESSION = {
    "user": {"email": "rochey@punty.ai", "name": "Rochey", "picture": ""},
}
_AUTH_COOKIE = _forge_cookie(_AUTH_SESSION)


def _auth_cookies() -> list[dict]:
    return [{"name": "session", "value": _AUTH_COOKIE,
             "domain": "127.0.0.1", "path": "/"}]


def get_csrf_token(page) -> str:
    """Fetch a CSRF token using a Playwright Page's request context."""
    resp = page.request.get("/api/csrf-token")
    return resp.json()["csrf_token"]


# ---------------------------------------------------------------------------
# DB seeding (runs once, before the server starts) — sync to avoid event loop
# ---------------------------------------------------------------------------
def _seed_database(db_path: Path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from punty.models.database import Base
    from punty.models.meeting import Meeting, Race, Runner
    from punty.models.content import Content
    from punty.models.pick import Pick
    from punty.models.settings import AppSettings

    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)

    now = datetime.now()

    with Session(engine) as db:
        # ── Meetings ──────────────────────────────────────────────
        m1 = Meeting(
            id=f"flemington-{TODAY}", venue="Flemington", date=TODAY,
            track_condition="Good 3", weather="Fine 22°",
            rail_position="True", selected=True,
            penetrometer=3.5, weather_condition="Fine", weather_temp=22,
            created_at=now, updated_at=now,
        )
        m2 = Meeting(
            id=f"randwick-{YESTERDAY}", venue="Randwick", date=YESTERDAY,
            track_condition="Soft 5", weather="Overcast 18°",
            rail_position="+2m 1000-400", selected=True,
            penetrometer=5.2, weather_condition="Overcast", weather_temp=18,
            created_at=now, updated_at=now,
        )
        db.add_all([m1, m2])

        # ── Races & Runners ──────────────────────────────────────
        jockeys = ["J. McDonald", "J. Bowman", "D. Lane",
                    "C. Williams", "H. Coffey", "J. Kah"]
        trainers = ["C. Maher", "G. Waterhouse", "J. Cummings",
                     "L. Freedman", "P. Moody", "A. Neasham"]
        positions = ["leader", "on_pace", "midfield",
                     "backmarker", "midfield", "on_pace"]

        for meeting in (m1, m2):
            prefix = "Flem" if meeting == m1 else "Rand"
            for rn in range(1, 5):
                race = Race(
                    id=f"{meeting.id}-r{rn}",
                    meeting_id=meeting.id, race_number=rn,
                    name=f"{'BM78' if rn < 3 else 'Group 3'} Race {rn}",
                    distance=1200 + (rn - 1) * 200,
                    class_="BM78" if rn < 3 else "Group 3",
                    prize_money=75000 + rn * 25000,
                    start_time=datetime.combine(
                        meeting.date,
                        datetime.min.time().replace(hour=12 + rn),
                    ),
                    status="Final", results_status="Paying",
                    track_condition=meeting.track_condition,
                    race_type="Flat", age_restriction="3YO+",
                    weight_type="Handicap", field_size=6,
                    created_at=now, updated_at=now,
                )
                db.add(race)

                for sc in range(1, 7):
                    odds = round(2.0 + sc * 1.5, 2)
                    db.add(Runner(
                        id=f"{race.id}-{sc}", race_id=race.id,
                        horse_name=f"{prefix} Star {rn}-{sc}",
                        saddlecloth=sc, barrier=sc,
                        weight=round(58.0 - sc * 0.5, 1),
                        jockey=jockeys[sc - 1], trainer=trainers[sc - 1],
                        form="1-2-3",
                        current_odds=odds, opening_odds=odds + 1.0,
                        place_odds=round(1.2 + sc * 0.3, 2),
                        horse_sex="Gelding" if sc % 2 == 0 else "Mare",
                        horse_age=4 + (sc % 3),
                        finish_position=sc,
                        win_dividend=odds if sc == 1 else None,
                        place_dividend=round(1.2 + sc * 0.3, 2) if sc <= 3 else None,
                        speed_map_position=positions[sc - 1],
                        created_at=now, updated_at=now,
                    ))

        # ── Content ───────────────────────────────────────────────
        c1 = Content(
            id="e2e-content-flem", meeting_id=m1.id,
            content_type="early_mail", status="sent",
            raw_content="*PUNTY EARLY MAIL* — Flemington\nTest content.",
            sent_at=now, created_at=now, updated_at=now,
        )
        c2 = Content(
            id="e2e-content-rand", meeting_id=m2.id,
            content_type="early_mail", status="pending_review",
            raw_content="*PUNTY EARLY MAIL* — Randwick\nAwaiting review.",
            created_at=now, updated_at=now,
        )
        db.add_all([c1, c2])

        # ── Picks ─────────────────────────────────────────────────
        pick_n = 0
        for content, meeting in [(c1, m1), (c2, m2)]:
            prefix = "Flem" if meeting == m1 else "Rand"
            for rn in range(1, 5):
                # 4 selections per race (tip_rank 1-4)
                for rank in range(1, 5):
                    pick_n += 1
                    sc = rank
                    odds = round(2.0 + rank * 1.5, 2)
                    p_odds = round(1.2 + rank * 0.3, 2)
                    is_hit = rank <= 2 and rn % 2 == 1
                    bet = ["win", "place", "each_way", "win"][rank - 1]
                    stake = [10.0, 5.0, 5.0, 0.0][rank - 1]

                    if is_hit and bet == "win":
                        pnl = odds * stake - stake
                    elif is_hit and bet == "place":
                        pnl = p_odds * stake - stake
                    elif is_hit and bet == "each_way":
                        pnl = (odds * stake / 2) + (p_odds * stake / 2) - stake
                    else:
                        pnl = -stake

                    db.add(Pick(
                        id=f"e2epk{pick_n:04d}",
                        content_id=content.id, meeting_id=meeting.id,
                        race_number=rn,
                        horse_name=f"{prefix} Star {rn}-{sc}",
                        saddlecloth=sc, tip_rank=rank,
                        odds_at_tip=odds, place_odds_at_tip=p_odds,
                        pick_type="selection", bet_type=bet, bet_stake=stake,
                        hit=is_hit, pnl=round(pnl, 2),
                        settled=True, settled_at=now, created_at=now,
                    ))

                # One exacta per race
                pick_n += 1
                hit = rn == 1
                db.add(Pick(
                    id=f"e2epk{pick_n:04d}",
                    content_id=content.id, meeting_id=meeting.id,
                    race_number=rn, pick_type="exotic",
                    exotic_type="exacta", exotic_runners="[1, 2]",
                    exotic_stake=20.0,
                    hit=hit, pnl=80.0 if hit else -20.0,
                    settled=True, settled_at=now, created_at=now,
                ))

            # One quaddie per meeting (races 1-4)
            pick_n += 1
            db.add(Pick(
                id=f"e2epk{pick_n:04d}",
                content_id=content.id, meeting_id=meeting.id,
                pick_type="sequence", sequence_type="quaddie",
                sequence_variant="balanced",
                sequence_legs='[[1,2],[1,3],[1,2],[1]]',
                sequence_start_race=1,
                hit=False, pnl=-50.0,
                settled=True, settled_at=now, created_at=now,
            ))

        # ── AppSettings defaults ──────────────────────────────────
        for key, info in AppSettings.DEFAULTS.items():
            db.add(AppSettings(
                key=key, value=info["value"], description=info["description"],
            ))

        db.commit()
    engine.dispose()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def e2e_db_path(tmp_path_factory):
    """Seed a temporary SQLite database and return its path."""
    db_path = tmp_path_factory.mktemp("e2e") / "punty_e2e.db"
    _seed_database(db_path)
    return db_path


@pytest.fixture(scope="session")
def e2e_server(e2e_db_path):
    """Start a real uvicorn server and yield the base URL."""
    env = os.environ.copy()
    env["PUNTY_DB_PATH"] = str(e2e_db_path)
    env["PUNTY_SECRET_KEY"] = E2E_SECRET
    env["PUNTY_DEBUG"] = "false"  # Keeps SQLAlchemy echo off (avoids log flood)
    env["PUNTY_LOG_LEVEL"] = "WARNING"
    # Clear OAuth to prevent init errors
    env.pop("PUNTY_GOOGLE_CLIENT_ID", None)
    env.pop("PUNTY_GOOGLE_CLIENT_SECRET", None)

    # Redirect output to files to avoid PIPE buffer deadlock
    log_dir = e2e_db_path.parent
    stdout_log = open(log_dir / "server_stdout.log", "w")
    stderr_log = open(log_dir / "server_stderr.log", "w")

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "punty.main:app",
         "--host", "127.0.0.1", "--port", str(E2E_PORT),
         "--log-level", "warning"],
        env=env,
        stdout=stdout_log,
        stderr=stderr_log,
    )

    # Wait for server readiness
    import httpx
    ready = False
    for _ in range(90):
        try:
            resp = httpx.get(f"{BASE_URL}/health", timeout=2)
            if resp.status_code == 200:
                ready = True
                break
        except Exception:
            pass
        # Check if process died
        if proc.poll() is not None:
            break
        time.sleep(0.5)

    if not ready:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        stdout_log.close()
        stderr_log.close()
        out = (log_dir / "server_stdout.log").read_text()[-2000:]
        err = (log_dir / "server_stderr.log").read_text()[-2000:]
        raise RuntimeError(
            f"E2E server did not start within 45s.\n"
            f"stdout (last 2000 chars):\n{out}\n"
            f"stderr (last 2000 chars):\n{err}"
        )

    yield BASE_URL

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    stdout_log.close()
    stderr_log.close()


# -- Playwright integration ------------------------------------------------

@pytest.fixture(scope="session")
def browser_context_args(e2e_server):
    """Provide base_url to all Playwright contexts."""
    return {"base_url": e2e_server}


@pytest.fixture
def auth_context(browser, e2e_server):
    """Browser context with forged admin session cookie."""
    ctx = browser.new_context(base_url=e2e_server)
    ctx.add_cookies(_auth_cookies())
    yield ctx
    ctx.close()


@pytest.fixture
def auth_page(auth_context):
    """Authenticated Playwright page (admin site)."""
    page = auth_context.new_page()
    yield page
    page.close()


@pytest.fixture
def public_page(browser, e2e_server):
    """Page for public site tests.

    Chromium blocks overriding the Host header, so for public page navigation
    we route through /public prefix directly (same as what the hostname
    middleware rewrites to). For API calls, /api/public/* is always accessible.
    """
    ctx = browser.new_context(base_url=e2e_server)
    page = ctx.new_page()
    yield page
    ctx.close()


# -- Viewport fixtures -----------------------------------------------------

@pytest.fixture
def mobile_page(browser, e2e_server):
    """Mobile viewport (375x812) with auth."""
    ctx = browser.new_context(
        base_url=e2e_server,
        viewport={"width": 375, "height": 812},
        user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)",
    )
    ctx.add_cookies(_auth_cookies())
    page = ctx.new_page()
    yield page
    ctx.close()


@pytest.fixture
def tablet_page(browser, e2e_server):
    """Tablet viewport (768x1024) with auth."""
    ctx = browser.new_context(
        base_url=e2e_server,
        viewport={"width": 768, "height": 1024},
    )
    ctx.add_cookies(_auth_cookies())
    page = ctx.new_page()
    yield page
    ctx.close()


@pytest.fixture
def desktop_page(browser, e2e_server):
    """Desktop viewport (1440x900) with auth."""
    ctx = browser.new_context(
        base_url=e2e_server,
        viewport={"width": 1440, "height": 900},
    )
    ctx.add_cookies(_auth_cookies())
    page = ctx.new_page()
    yield page
    ctx.close()


@pytest.fixture
def mobile_public_page(browser, e2e_server):
    """Mobile viewport for public site (uses /public prefix directly)."""
    ctx = browser.new_context(
        base_url=e2e_server,
        viewport={"width": 375, "height": 812},
        user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)",
    )
    page = ctx.new_page()
    yield page
    ctx.close()


# -- Screenshot on failure -------------------------------------------------

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Store test outcome on the item for screenshot capture."""
    outcome = yield
    setattr(item, f"rep_{outcome.get_result().when}", outcome.get_result())


@pytest.fixture(autouse=True)
def _screenshot_on_failure(request):
    """Auto-capture a screenshot when a test fails."""
    yield
    rep = getattr(request.node, "rep_call", None)
    if not rep or not rep.failed:
        return
    # Find any Playwright Page in the test fixtures
    for val in request.node.funcargs.values():
        if hasattr(val, "screenshot"):
            SCREENSHOTS_DIR.mkdir(exist_ok=True)
            safe = request.node.nodeid.replace("::", "__").replace("/", "_")
            try:
                val.screenshot(path=str(SCREENSHOTS_DIR / f"{safe}.png"))
            except Exception:
                pass
            break
