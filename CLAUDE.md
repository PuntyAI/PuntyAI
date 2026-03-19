# CLAUDE.md — PuntyAI Project Guide

## What is PuntyAI?
AI-powered Australian horse racing content generator and betting tracker. Scrapes race data, generates humorous "Punty" early mail tips via Claude/OpenAI, tracks bet settlement and P&L.

## Tech Stack
- **Backend:** Python 3.11+, FastAPI (async), SQLAlchemy 2.0 (async ORM), SQLite (aiosqlite)
- **Frontend:** Jinja2 templates, Tailwind CSS (CDN), HTMX, Alpine.js
- **AI:** Anthropic Claude (primary) / OpenAI GPT-4 (fallback) via `punty/ai/`
- **Scraping:** httpx (async HTTP), Playwright (browser automation for speed maps)
- **Delivery:** WhatsApp Business API, Twitter/X API (Tweepy)

## Directory Structure
```
punty/
├── ai/              # AI content generation
│   └── generator.py # Main generation logic, SSE streaming
├── api/             # FastAPI route handlers
│   ├── meets.py     # Meeting CRUD, scrape/speed-map SSE streams
│   ├── content.py   # Content generation, review, approval
│   ├── results.py   # Results checking, settlement, monitor
│   ├── delivery.py  # WhatsApp/Twitter send endpoints
│   └── settings.py  # App settings CRUD
├── auth.py          # Google OAuth, session middleware, CSRF
├── config.py        # Pydantic settings (env: PUNTY_* prefix)
├── context/
│   ├── builder.py   # Builds AI prompt context from DB data
│   └── versioning.py # Context snapshots for change detection
├── delivery/
│   ├── whatsapp.py  # WhatsApp Business API client
│   └── twitter.py   # Twitter/X posting (Tweepy)
├── formatters/
│   ├── whatsapp.py  # Markdown → WhatsApp formatting
│   └── twitter.py   # Content → tweet-sized chunks
├── models/
│   ├── database.py  # Engine, session factory, init_db()
│   ├── meeting.py   # Meeting, Race, Runner models
│   ├── content.py   # Content model (early_mail, meeting_wrapup, etc.)
│   ├── pick.py      # Pick model (selections, exotics, sequences)
│   └── settings.py  # AppSettings key-value store
├── results/
│   ├── parser.py    # Regex parser: AI text → Pick dicts
│   ├── picks.py     # Pick storage, settlement, P&L summary
│   └── monitor.py   # Background results polling loop
├── scrapers/
│   ├── base.py      # BaseScraper (httpx client, ID generation)
│   ├── playwright_base.py # Browser singleton for JS-heavy sites
│   ├── racing_scraper.py  # Race fields, runners, form from racing.com
│   ├── tab_scraper.py     # Odds, results, dividends from TAB
│   ├── speed_map_scraper.py # Predicted running positions
│   └── orchestrator.py    # Coordinates all scrapers, SSE progress
└── web/
    ├── routes.py    # Page routes (dashboard, meets, review, etc.)
    └── templates/   # Jinja2 HTML templates
prompts/
├── early_mail.md    # Main early mail generation prompt
└── *.md             # Other prompt templates
```

## Data Flow
1. **Scrape** → `orchestrator.py` fetches meeting/race/runner data from racing.com + TAB
2. **Speed Maps** → Playwright scrapes predicted running positions
3. **Generate** → `generator.py` builds context via `builder.py`, sends to AI with prompt from `prompts/`
4. **Review** → Content lands in review queue (status: `pending_review`)
5. **Approve** → `parser.py` extracts picks from AI text → stored as `Pick` rows
6. **Results** → `tab_scraper.py` fetches finish positions + dividends
7. **Settle** → `picks.py` calculates P&L per pick based on bet type and dividends
8. **Deliver** → WhatsApp/Twitter formatting and send

## Key Models

### Meeting (`meetings` table)
- `id`: e.g., `sale-2026-02-01`
- `venue`, `date`, `track_condition`, `weather_*`, `rail_position`
- Has many `Race` objects

### Race (`races` table)
- `id`: e.g., `sale-2026-02-01-r1`
- `race_number`, `name`, `distance`, `class_`, `prize_money`
- `results_status`: Open → Interim → Paying → Closed
- `exotic_results`: JSON dict of dividend payouts
- Has many `Runner` objects

### Runner (`runners` table)
- `race_id` + `saddlecloth` identify a runner
- `horse_name`, `jockey`, `trainer`, `weight`, `form`, `last_five`
- `current_odds`, `opening_odds`, odds from multiple bookmakers
- `finish_position`, `win_dividend`, `place_dividend`, `result_margin`
- `speed_map_position`: leader/on_pace/midfield/backmarker

### Content (`content` table)
- `id`, `meeting_id`, `content_type` (early_mail, meeting_wrapup, race_preview)
- `status`: draft → pending_review → approved → sent (or rejected/regenerating)
- `raw_content`: AI-generated text
- `whatsapp_formatted`, `twitter_formatted`: platform-specific versions

### Pick (`picks` table)
- `pick_type`: selection | big3 | big3_multi | exotic | sequence
- **Selections**: `horse_name`, `saddlecloth`, `tip_rank` (1-4), `odds_at_tip`, `bet_type` (win/saver_win/place/each_way), `bet_stake`
- **Exotics**: `exotic_type`, `exotic_runners` (JSON array), `exotic_stake`
- **Sequences**: `sequence_type` (quaddie/big6), `sequence_variant` (skinny/balanced/wide), `sequence_legs` (JSON), `sequence_start_race`
- **Settlement**: `hit` (bool), `pnl` (float), `settled` (bool), `settled_at`

## Settlement Logic (picks.py)

### Selections ($20 pool per race, AI allocates across 4 picks)
- **Win / Saver Win**: `pnl = win_dividend × stake - stake` if 1st, else `-stake`
- **Place**: `pnl = place_dividend × stake - stake` if top 3, else `-stake`
- **Each Way**: Half on win, half on place. If wins: both halves pay. If places: place half pays, win half lost.

### Exotics ($20 fixed per race)
- Trifecta/Exacta/Quinella/First4 checked against finish positions
- `pnl = dividend × stake - stake` if hit

### Sequences (Quaddie/Big6)
- Each leg checked: did the winner's saddlecloth appear in our leg selections?
- All legs must hit. Cost = combos × unit price.

### Big3 Multi
- 3 horses across different races, all must win. P&L on multi row only.

## Parser (parser.py)
Regex-based extraction from AI-generated early mail text:
- `_BIG3_HORSE`: `1) *HORSE NAME* (Race X, No.Y) — $Z.ZZ`
- `_SELECTION`: `1. *HORSE NAME* (No.X) — $Y.YY`
- `_BET_LINE`: `Bet: $X Win|Place|Each Way|Saver Win`
- `_ROUGHIE`: `Roughie: *HORSE* (No.X) — $Y.YY`
- `_EXOTIC`: `Trifecta|Exacta|Quinella: runners — $20`
- `_SEQ_VARIANT`: `Skinny|Balanced|Wide: legs (combos × $unit = $total)`

## Config (env vars, PUNTY_ prefix)
- `PUNTY_SECRET_KEY` — Session/CSRF signing key
- `PUNTY_DATABASE_URL` — SQLite path (default: `sqlite+aiosqlite:///data/punty.db`)
- `PUNTY_DEBUG` — Debug mode
- `PUNTY_GOOGLE_CLIENT_ID` / `PUNTY_GOOGLE_CLIENT_SECRET` — OAuth
- `PUNTY_ALLOWED_EMAILS` — Comma-separated authorized emails
- AI keys stored in DB `app_settings` table (not env)

## Server / Deployment
- **Server**: Ubuntu VM at `app.punty.ai`, SSH as `root`
- **App location**: `/opt/puntyai/`
- **Database**: `/opt/puntyai/data/punty.db`
- **Service**: `systemd` unit `puntyai.service`
- **Deploy**: `git push` → SSH fetch/checkout specific files → `systemctl restart puntyai`
- **Process**: uvicorn on 127.0.0.1:8000, reverse proxied by Caddy

## Common Operations
```bash
# Deploy all files
ssh -i ~/.ssh/id_ed25519 root@app.punty.ai "cd /opt/puntyai && git fetch origin master && git checkout -f origin/master && chmod 666 prompts/*.md && systemctl restart puntyai"

# Deploy specific files (add chmod if deploying prompts)
ssh -i ~/.ssh/id_ed25519 root@app.punty.ai "cd /opt/puntyai && git fetch origin master && git checkout -f origin/master -- <files> && chmod 666 prompts/*.md && systemctl restart puntyai"

# Re-parse picks from approved content
ssh root@app.punty.ai "cd /opt/puntyai && source venv/bin/activate && python3 -c '...asyncio script...'"

# Check logs
ssh root@app.punty.ai "journalctl -u puntyai --no-pager -n 50"

# DB is at /opt/puntyai/data/punty.db
```

**Note:** `chmod 666 prompts/*.md` is required after git checkout because git resets permissions, and the `punty` user needs write access to edit personality from the Settings UI.

## Testing

**IMPORTANT:** Always run tests before committing code changes.

```bash
# Run all unit tests (~1,540 tests, ~5s)
pytest tests/unit -v

# Run specific test file (fastest for iterating)
pytest tests/unit/test_pre_selections.py -v

# Run tests matching keyword
pytest tests/unit -k "exotic" -v

# Run with coverage
pytest tests/ --cov=punty --cov-report=term-missing
```

### Test Structure (1,540+ unit tests)
- `tests/unit/test_probability.py` — 278 tests, probability engine core
- `tests/unit/test_pre_selections.py` — 143 tests, pick ranking/selection
- `tests/unit/test_parser.py` — 30 tests, pick extraction from AI text
- `tests/unit/test_picks.py` — 60 tests, settlement calculations
- `tests/unit/test_pre_sequences.py` — 48 tests, quaddie/sequence logic
- `tests/unit/test_content_validator.py` — 53 tests, content validation
- `tests/unit/test_generator.py` — 10 tests, AI content generation
- `tests/unit/test_bet_optimizer.py` — 59 tests, bet type decisions
- `tests/e2e/` — 102 E2E browser tests (Playwright, non-blocking in CI)

### CI/CD Pipeline
- GitHub Actions: push → unit tests → e2e tests → staging → production
- Unit tests gate deployment (must pass); E2E tests are non-blocking
- Branch protection: PRs required for master, 2 status checks must pass
- Pre-commit hook runs unit tests with `--last-failed --fail-first` for speed

### Pre-commit Hook Setup
```bash
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
```

## Change Management Process

### For every code change:
1. **Make changes** to the codebase
2. **Run affected tests** — `pytest tests/unit/test_<affected>.py -v`
3. **Run full unit suite** — `pytest tests/unit -q` (pre-commit hook does this)
4. **Commit** — branch off master, commit with descriptive message
5. **Push branch** — `git push -u origin <branch-name>`
6. **Create PR** — `gh pr create` with summary + test plan
7. **Wait for CI** — `gh pr checks <PR#> --watch`
8. **Merge** — `gh pr merge <PR#> --squash --delete-branch`
9. **Deploy** — SSH deploy command (see Common Operations)
10. **Verify** — Check logs: `journalctl -u puntyai --no-pager -n 50`

### Key files to test by area:
| Change area | Test file(s) to run |
|---|---|
| Probability engine | `test_probability.py` |
| Pick ranking/selection | `test_pre_selections.py` |
| Bet types/staking | `test_bet_optimizer.py`, `test_bet_type_tuning.py` |
| AI output parsing | `test_parser.py` |
| Settlement/P&L | `test_picks.py` |
| Content generation | `test_generator.py` |
| Content validation | `test_content_validator.py` |
| Quaddie/sequences | `test_pre_sequences.py` |
| Prompt templates | `test_generator.py`, `test_parser.py` |
| Betfair queue | `test_betfair_betting.py` |
| HTML formatting | Manual check on tips page |
| Settings UI | Manual check on settings page |

### Ad-hoc scripts convention:
- Prefix with `_` (e.g. `scripts/_check_pnl.py`) — gitignored, throwaway
- No prefix (e.g. `scripts/build_calibration.py`) — tracked, reusable utilities

## Design Theme
Cyberpunk neon: dark bg (#0a0a0f), Orbitron display font, Rajdhani headings, Source Sans Pro body. Magenta/cyan/orange gradient accents. Sunset gradient race number pills. Pick badges: magenta (top), cyan (2nd), yellow (3rd), purple (roughie).
