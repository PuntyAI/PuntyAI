# PuntyAI — Technical Architecture

## System Overview

PuntyAI is an AI-powered Australian horse racing content generator and betting tracker. It scrapes multi-source racing data, runs a 10-factor probability engine, generates humorous tips via LLM, tracks bet settlement and P&L, and delivers content across multiple platforms.

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION LAYER                          │
│  PuntingForm API · Racing.com GraphQL · TAB · WillyWeather     │
│  Calendar · News · Track Conditions · Future Races              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    Orchestrator (merge + upsert)
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                      DATA LAYER (SQLite)                        │
│  Meetings · Races · Runners (70+ fields) · Content · Picks     │
│  AppSettings · LiveUpdates · PatternInsights · RaceAssessments  │
│  ContextSnapshots · FutureRaces · TuningLog                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
┌─────────▼──────┐ ┌──────▼──────┐ ┌───────▼──────┐
│  PROBABILITY   │ │   CONTEXT   │ │     RAG      │
│    ENGINE      │ │   BUILDER   │ │   PIPELINE   │
│ 10-factor model│ │ 60+ sources │ │ Strategy +   │
│ Harville exotic│ │ per meeting │ │ Assessments +│
│ Kelly staking  │ │ Weather/Pace│ │ Deep Learning│
│ Self-tuning    │ │ Enrichment  │ │ + Knowledge  │
└────────┬───────┘ └──────┬──────┘ └───────┬──────┘
         │                │                │
         └────────────────┼────────────────┘
                          │
               ┌──────────▼──────────┐
               │   AI GENERATION     │
               │  OpenAI GPT-5.2     │
               │  8 prompt templates │
               │  60-110K tokens/gen │
               └──────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │   CONTENT PIPELINE  │
               │  Parse → Review →   │
               │  Approve → Format   │
               └──────────┬──────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
    ┌─────▼─────┐  ┌─────▼─────┐  ┌──────▼─────┐
    │  Twitter   │  │  Facebook │  │   Email    │
    │  (25K chr) │  │  (63K chr)│  │ (Resend/   │
    │  Tweepy    │  │  Graph API│  │  SMTP)     │
    └───────────┘  └───────────┘  └────────────┘

               ┌──────────────────────┐
               │   SETTLEMENT LOOP    │
               │  ResultsMonitor      │
               │  (every 30-90s)      │
               │  Odds · Scratchings  │
               │  Results · Settlement│
               │  Celebrations · Pace │
               │  Weather · Wrapups   │
               └──────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Runtime | Python 3.11+, async/await throughout |
| Web Framework | FastAPI (Starlette) |
| ORM | SQLAlchemy 2.0 (async) |
| Database | SQLite + aiosqlite (WAL mode) |
| Templates | Jinja2 |
| Frontend | Tailwind CSS (CDN), HTMX, Alpine.js, Chart.js |
| AI | OpenAI GPT-5.2 (Responses API, Reasoning mode) |
| Scraping | httpx (async HTTP), Playwright (browser automation) |
| Scheduling | APScheduler |
| Delivery | Tweepy (Twitter), httpx (Facebook Graph API), Resend/SMTP (Email) |
| Remote Mgmt | Telegram Bot → Claude API (Sonnet 4.5) |
| Testing | pytest, pytest-asyncio, Playwright (e2e) |
| CI/CD | GitHub Actions |
| Hosting | Ubuntu VM, uvicorn behind Caddy reverse proxy |

## Data Flow

### 1. Daily Pipeline

```
00:05  CalendarScraper → Find meetings → Auto-select → Initial scrape
05:00  DailyMorningScrape → Full PF API scrape + speed maps
R1-2h  MeetingPreRaceJob → TAB odds → Weather → Speed maps → Generate → Approve → Post
Racing ResultsMonitor → Poll results → Settle → Celebrate → Pace analysis → Weather
R(last)+30m  MeetingPostRaceJob → Settlement check → Wrapup → Post → Probability tuning
```

### 2. Scraper Orchestration

```
PuntingForm API (primary)
  ├── /form/meetingslist → Venue resolution
  ├── /form/fields → All races + runners
  ├── /form/form → 10-run form history per horse
  ├── /User/Speedmaps → Speed maps + pace factors
  ├── /Ratings/MeetingRatings → Run style + ranks
  ├── /Updates/Conditions → Track conditions (all states)
  └── /Updates/Scratchings → Scratchings (all states)

Racing.com GraphQL (supplement)
  ├── Race fields → Comments, stewards, odds (5 bookmakers)
  ├── Speed maps → speedValue 1-12 (fallback)
  ├── Results → Positions, margins, dividends
  └── Sectionals → GraphQL or CSV fallback

TAB (optional)          → Live odds
WillyWeather (weather)  → Wind, temp, humidity, rain, radar, cameras
```

### 3. Content Generation

```
Meeting Context (40-80K tokens)
  ├── Meeting header (track, rail, weather, wind impact)
  ├── Per-race breakdown:
  │   ├── Markdown table (barrier, odds, form, speed, probability)
  │   ├── Runner details (stats, gear, comments, pedigree)
  │   ├── Stewards excuses, time ratings, weight analysis
  │   ├── Extended form (last 5 starts with sectionals)
  │   ├── Probability rankings + VALUE plays
  │   └── Pre-calculated exotic combinations (Harville)
  └── Sequence leg confidence (HIGH/MED/LOW per race)

RAG Context (10-25K tokens)
  ├── Strategy track record (per-bet-type P&L + directives)
  ├── Race assessments (2-5 similar races via hybrid similarity)
  ├── Deep learning patterns (top 15 HIGH/MED confidence)
  └── Runner memories (top 3 runners × 2 similar situations)

Personality Prompt + Early Mail Prompt (8K tokens)

→ OpenAI GPT-5.2 (Reasoning, 10-min timeout)
→ 10-20K output tokens
→ Parser extracts picks via regex
→ Content enters review queue
```

### 4. Settlement Pipeline

```
Race completes (Paying/Closed status)
  → Scrape results (positions, dividends, exotics)
  → Upsert to Runner model
  → settle_picks_for_race()
     ├── Selections: win/place/each_way/saver_win P&L
     ├── Exotics: trifecta/exacta/quinella/first4 flexi P&L
     ├── Sequences: leg-by-leg winner check, flexi P&L
     └── Big3 Multi: all 3 legs must win (waits for all races)
  → Check celebrations (5x+ payout → Facebook/Twitter post)
  → Pace analysis (leader/closer bias → Facebook post, max 3/meeting)
  → Record pattern insights for self-tuning
```

## Database Schema

### Core Tables

| Table | PK Format | Key Fields |
|-------|-----------|------------|
| `meetings` | `{venue-slug}-{date}` | venue, date, track_condition, weather_*, selected |
| `races` | `{meeting-id}-r{num}` | distance, class_, prize_money, start_time, results_status, exotic_results |
| `runners` | `{race-id}-{sc}-{horse}` | 70+ fields: horse_name, odds (5 bookmakers), form_history, speed_map_position, pf_* |
| `content` | UUID | meeting_id, content_type, status, raw_content, twitter_formatted, facebook_formatted |
| `picks` | UUID(16) | pick_type, bet_type, bet_stake, hit, pnl, settled, win_probability, factors_json |

### Supporting Tables

| Table | Purpose |
|-------|---------|
| `app_settings` | Key-value store for API keys, weights, configuration |
| `live_updates` | Race-day celebrations, pace analysis posts |
| `context_snapshots` | Versioned meeting context for change detection |
| `race_assessments` | LLM-generated post-race learnings with embeddings |
| `pattern_insights` | Aggregated betting patterns (844+ from deep learning) |
| `probability_tuning_log` | Weight adjustment history for calibration dashboard |
| `future_races` | Upcoming Group races with nominations |
| `settings_audit_log` | Settings change tracking |

## Probability Engine

### 10-Factor Model

| Factor | Default Weight | Signal |
|--------|---------------|--------|
| Market | 22% | Median odds across 6 bookmakers (overround-adjusted) |
| Form | 15% | Last 5, track/dist/condition stats, recency-weighted |
| Deep Learning | 10% | 844+ historical patterns matched to runner attributes |
| Pace | 11% | PF map_factor + speed_rank + pace scenario interaction |
| Barrier | 9% | Relative position × distance multiplier |
| Jockey/Trainer | 11% | Jockey 60% / Trainer 40% weighted strike rate |
| Movement | 7% | Odds direction + magnitude from flucs/opening comparison |
| Class | 5% | Class suitability + spell fitness curve |
| Weight | 5% | Weight vs race average (±3kg = ±10% score) |
| Horse Profile | 5% | Age peak curve (4-5yo optimal) + sex consistency |

### Safeguards

- **Dynamic Market Boost**: When >40% of factors are near-neutral, market weight increases to ~50%
- **Market Floor**: Model probability cannot be <50% of market-implied (50/50 blend)
- **Self-Tuning**: Point-biserial correlation, 70/30 smoothing, bounds 2%-35%, 24h cooldown
- **Quarter-Kelly Staking**: Conservative 0.25 fraction, $1 minimum, $20 cap

### Exotic Calculations

- **Harville model**: Conditional probabilities for exacta/trifecta/first4
- **Quinella**: Sum of both orderings
- **Box/Standout**: Full permutation enumeration
- **Value threshold**: >= 1.2x for regular exotics, >= 1.5x for Punty's Pick

## Authentication & Security

- **Google OAuth** with email whitelist (`PUNTY_ALLOWED_EMAILS`)
- **CSRF protection**: HMAC-SHA256 tokens with per-session secrets
- **Hostname routing**: `punty.ai` (public, no auth) vs `app.punty.ai` (admin, auth required)
- **SQLAlchemy parameterized queries** (no SQL injection surface)
- **Path traversal protection** in Telegram tools
- **API keys in DB** (not source code), redacted in audit logs

## Delivery Channels

| Channel | Max Length | Format | Auth |
|---------|-----------|--------|------|
| Twitter/X | 25K chars | Unicode bold, hashtags, profanity filter | OAuth 1.0a |
| Facebook | 63K chars | Plain text, profanity rotation | Page Access Token |
| Email | Unlimited | HTML + plain text | Resend API / SMTP |
| Telegram | 4096 chars | Markdown, chunked | Bot Token + Owner ID |

## Hosting & Deployment

```
Production: Ubuntu VM at app.punty.ai
  ├── App: uvicorn on 127.0.0.1:8000
  ├── Proxy: Caddy (automatic HTTPS)
  ├── Service: systemd unit puntyai.service
  ├── Database: /opt/puntyai/data/punty.db
  └── Deploy: git push → SSH fetch/checkout → systemctl restart

CI/CD: GitHub Actions
  ├── unit-test job (759 tests)
  ├── e2e-test job (103 tests, Playwright)
  └── Manual deploy via SSH
```
