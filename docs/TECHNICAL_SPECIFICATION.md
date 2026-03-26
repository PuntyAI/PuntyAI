# PuntyAI — Technical Specification & Code Review

> Prepared for full redesign. Covers every page, data flow, mathematical model, and known issue in the system.

---

## Table of Contents

1. [What is PuntyAI](#1-what-is-puntyai)
2. [Architecture Overview](#2-architecture-overview)
3. [Design System](#3-design-system)
4. [Public Website (punty.ai)](#4-public-website-puntyai)
5. [Admin Application (app.punty.ai)](#5-admin-application-apppuntyai)
6. [Data Pipeline](#6-data-pipeline)
7. [AI Content Generation](#7-ai-content-generation)
8. [Probability Engine](#8-probability-engine)
9. [Betting & Settlement](#9-betting--settlement)
10. [Delivery](#10-delivery)
11. [Infrastructure](#11-infrastructure)
12. [Bugs & Math Issues](#12-bugs--math-issues)
13. [Dead Code](#13-dead-code)
14. [Technical Debt](#14-technical-debt)

---

## 1. What is PuntyAI

AI-powered Australian horse racing tipping platform. The system:

1. **Scrapes** race data from 5+ sources (PuntingForm, Racing.com, Betfair, PointsBet, TAB)
2. **Calculates** win/place probabilities using an LGBM ML model + market calibration
3. **Generates** humorous daily tips ("early mail") via GPT-5.4 with a character persona ("Punty")
4. **Tracks** bet settlement and P&L across selections, exotics (quinella/exacta/first4), and sequences (quaddie)
5. **Auto-bets** on Betfair exchange using Kelly staking
6. **Delivers** content to Twitter/X, Facebook, Telegram, and email

Two separate frontends serve different audiences:
- **punty.ai** — public marketing + daily tips dashboard (no auth)
- **app.punty.ai** — admin dashboard for managing the full pipeline (Google OAuth)

---

## 2. Architecture Overview

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11+, FastAPI (async), uvicorn |
| Database | SQLite (aiosqlite, WAL mode), DuckDB (offline analytics) |
| ORM | SQLAlchemy 2.0 (async) |
| Frontend | Jinja2 templates, Tailwind CSS (CDN), HTMX 1.9, Alpine.js 3.x |
| AI | OpenAI GPT-5.4 (primary), Anthropic Claude (Telegram agent) |
| ML | LightGBM (probability ranking), scikit-learn |
| Scraping | httpx (async HTTP), Playwright (browser automation) |
| Scheduling | APScheduler (async) |
| Delivery | Tweepy (Twitter/X), Resend (email), python-telegram-bot |
| Charts | Chart.js 4.x |

### Dual-Host Architecture

A single FastAPI process serves both sites via `HostnameRoutingMiddleware`:
- Requests to `punty.ai` have their path rewritten to `/public/*`
- Requests to `app.punty.ai` route to admin endpoints directly
- Auth middleware exempts all `/public/*` paths

### Middleware Stack (outermost → innermost)

```
SessionMiddleware → HostnameRoutingMiddleware → RateLimitMiddleware → AuthMiddleware → CSRFMiddleware → GZipMiddleware → CacheControlMiddleware
```

### Database Schema (6 core tables)

```
meetings (1) ──→ (N) races (1) ──→ (N) runners
                      │
                      └──→ (N) picks

content (1) ──→ (N) picks

betfair_bets (standalone)
app_settings (key-value store)
```

Plus supporting tables: `context_snapshots`, `race_memories`, `pattern_insights`, `race_assessments`, `settings_audit`, `token_usage`, `future_races`, `scheduled_jobs`, `analysis_weights`.

---

## 3. Design System

### Colour Palette

| Token | Hex | Usage |
|-------|-----|-------|
| `--bg-dark` | `#0a0a0f` | Page background |
| `--bg-card` | `#12121a` | Card background |
| `--bg-card-alt` | `#1a1a25` | Alternate card |
| `--magenta` | `#e91e8c` | Primary accent, rank 1 badge, CTAs |
| `--cyan` | `#00d4ff` | Secondary accent, rank 2 badge, winners count |
| `--orange` | `#ff6b35` | Tertiary accent |
| `--yellow` | `#ffc107` | Warnings, rank 3 badge, strike rate |
| `--purple` | `#9d4edd` | Roughie badge, sequences |
| `--gradient-sunset` | magenta→orange→yellow | Logo, buttons, race number pills |
| `--gradient-header` | magenta→purple→cyan | Nav border accent |

### Typography

| Role | Font | Usage |
|------|------|-------|
| Display | Orbitron | Race number pills, balance figures, page titles |
| Headings | Rajdhani | Nav links, section headers, buttons, stats labels |
| Body | Source Sans Pro | All body text |

### Key Components

- **Race number pills** — 40px sunset-gradient rounded squares, Orbitron font
- **Pick badges** — Rank 1 (magenta), Rank 2 (cyan), Rank 3 (yellow), Roughie (purple)
- **Runner highlight** — Magenta left-border gradient on top pick's row
- **Scanlines overlay** — Subtle CRT-style full-screen pseudo-element
- **Toast notifications** — Fixed bottom-right, 3s auto-dismiss (cyan/green/magenta/yellow)
- **Sidebar panels** — `--bg-card-alt` with subtle border, compact stat displays
- **Accordion headers** — Click-to-expand with chevron rotation, Alpine.js `x-collapse`

### Responsive Breakpoints

| Breakpoint | Behaviour |
|------------|-----------|
| Default (mobile) | Single column, bottom nav (public), hamburger (admin) |
| `sm:` 640px | Minor layout adjustments |
| `md:` 768px | 2-column starts |
| `lg:` 1024px | Full desktop layout (65/35 split on dashboard) |

### External Dependencies (CDN)

- Tailwind CSS (precompiled `/static/tailwind.css`)
- HTMX 1.9.10
- Alpine.js 3.x + `@alpinejs/collapse`
- Chart.js 4.x
- Google Fonts (Orbitron, Rajdhani, Source Sans Pro)
- BugHerd (feedback sidebar)
- Microsoft Clarity (admin), Google Tag Manager (public)

---

## 4. Public Website (punty.ai)

No authentication required. Age verification modal on first visit (localStorage).

### Navigation

**Desktop:** Horizontal nav — Home / Tips (dropdown: Dashboard, All Tips, Scorecard) / Toolbox (dropdown: How It Works, Blog, Glossary, Bet Calculator) / About / Contact

**Mobile:** Fixed bottom nav (5 items: Home, Tips, Scorecard, How It Works, More) + slide-out panel

### Pages

#### 4.1 Homepage (`GET /`)
Marketing landing page. Hero with Punty character image, stats strip (all-time winners, 30-day SR, 7-day ROI), today's meetings grid with state badges and next jump countdowns, social CTA.

**Data:** `stats` (winner counts, ROI), `meetings` (today's selected), `cta_images` (social promo)

#### 4.2 Tips Dashboard (`GET /tips`) — **Most Complex Page**
Two-column layout (65/35 desktop). The main daily racing hub.

**Left column:**
- Live Edge Zone — next race picks with probability/edge data, venue filter checkboxes, race offset navigation (HTMX partial, refreshes on interaction)
- Today's Meetings grid — state badges, track condition, metro indicator, next jump countdown
- Best of Each Meet — per-meeting: best winner, roughie, exotic
- Sequence Bets — quaddie/early quaddie legs display
- Stats metrics row (bets, winners, strike %, P&L, 7-day)
- P&L by Meeting chart (Chart.js, collapsible)

**Right column (sidebar):**
- Recent Race Results — last 3 settled races (HTMX auto-refresh 60s)
- 7-Day Trend — bar chart
- By Bet Type — win/place/exotic/sequence P&L breakdown
- By Tip Rank — rank 1-4 strike rates
- Quick Links

**HTMX partials (auto-refreshing):**
- `GET /tips/_glance` → at-a-glance strip (every 30s)
- `GET /tips/_live-edge?exclude=&offset=N` → live edge zone
- `GET /tips/api/dashboard/recent-results` → recent results (every 60s)
- `GET /tips/_results-feed` → combined results feed

**Data (heavy):** `dashboard` dict (big_wins, timeline, upcoming, edge_picks, bet_types, rank_strike, insights, venues, recent_race_results, meeting_pnl, performance), `stats`, `next_race`, `next_race_picks`, `recent_wins`, `perf_history`, `seven_day_pnl/roi`, `best_of`, `sequences`, `meetings_list`, `primary_play`

#### 4.3 Meeting Tips Detail (`GET /tips/{meeting_id}`)
Individual meeting view. Cached 5 minutes (`Cache-Control: public, max-age=300`).

**Sections:**
- Venue switcher pills (same-day meetings)
- Meeting header (date, track condition, weather, rail position)
- AI early mail content (formatted HTML)
- Race-by-race accordion: runners table with pick badges, odds, form, speed map position, finish positions, dividends
- Win celebrations for settled winners
- Scratched pick warnings with alternatives
- Venue historical stats
- Meeting P&L stats table
- Live updates feed
- Sequence results summary
- Wrap-up section (post-meeting AI analysis)

**Data:** `meeting`, `early_mail` (HTML), `wrapup`, `winners`, `winning_exotics/sequences`, `pp_picks`, `races` (full runner data), `venue_stats`, `same_day_meetings`, `scratched_picks`, `alternatives`

#### 4.4 Tips Archive (`GET /tips/archive`)
Calendar-style paginated archive (30 dates/page). Filter by venue and date range.

#### 4.5 Stats / Daily Scorecard (`GET /stats`)
Full scoreboard: win celebration cards, P&L timeline chart, upcoming bets, edge picks, auto-scrolling insights marquee, bet type breakdown, venue P&L table.

#### 4.6 Blog (`GET /blog`, `GET /blog/{slug}`)
Weekly AI-generated blog posts. Paginated listing, individual post with prev/next navigation.

#### 4.7 Static Pages
- How It Works (`/how-it-works`)
- About (`/about`)
- Contact (`/contact`)
- Glossary (`/glossary`)
- Betting Calculator (`/calculator`) — client-side JS calculator
- Terms (`/terms`), Privacy (`/privacy`), Data Deletion (`/data-deletion`)

#### 4.8 Betfair Tracker (`GET /betfair-tracker`)
Password-protected public page (HMAC cookie, 90-day expiry). Read-only Betfair P&L tracking. Alpine.js app with period tabs, compound growth chart, bet history table.

#### 4.9 SEO
- Sitemaps: `/sitemap.xml`, `/sitemap-static.xml`, `/sitemap-blog.xml`, `/sitemap-tips.xml`
- `/robots.txt`, `/llms.txt`
- Schema.org `WebSite` structured data
- Open Graph + Twitter Card meta on all pages

---

## 5. Admin Application (app.punty.ai)

Google OAuth required. Allowed emails whitelist via `PUNTY_ALLOWED_EMAILS` env var. Sessions expire after 7 days absolute or 24 hours idle.

### Navigation
Desktop horizontal: Dashboard / Meets / Content / Review (badge) / Betfair / Calibration / Backtest / Settings

### Pages

#### 5.1 Dashboard (`GET /`)
Command center: today's meetings with countdown timers, pending review count (HTMX 60s refresh), recent content list, active scheduled jobs, results monitor status, today's P&L + cumulative chart, token usage stats, recent wins.

#### 5.2 Meets (`GET /meets`)
Meeting management with pipeline status indicator (4-step: scraped → generated → approved → sent). Bulk actions via SSE streams: scrape all, speed maps all, generate all. Paginated archive (20/page). Per-meeting P&L.

#### 5.3 Meeting Detail (`GET /meets/{meeting_id}`)
Deep dive: all races with full runner fields, picks overlaid on runners, exotics and sequences per race. SSE streams for scraping and generation. Content management (approve/reject/regenerate). Probability breakdown per race. Odds refresh.

#### 5.4 Content (`GET /content`)
Content management table (last 100 items). Status filters with counts. Bulk reject pending.

#### 5.5 Review Queue (`GET /review`)
Pending review items, oldest-first. Links to review detail.

#### 5.6 Review Detail (`GET /review/{content_id}`)
Full review: raw AI text textarea with live preview, approve/reject/edit, re-generate, format for platforms, prev/next navigation, extracted picks preview.

#### 5.7 Betfair Auto-Bet (`GET /betfair`)
Alpine.js app. Today's bet queue with enable/disable toggles, bet type cycling (Place/Win/BSP), individual and bulk stake controls. Live balance, P&L chart, settled history with period tabs. Scheduler start/stop. Calibration data display. $1M Mission progress bar.

#### 5.8 Analytics / Backtest (`GET /analytics`)
DuckDB-powered historical analytics. Filters: date range, state, venue, class, distance, odds range, track condition. Multiple Chart.js charts: strike rate, P&L by dimension, calibration curve. Requires `data/analytics.duckdb` (built offline).

#### 5.9 Settings (`GET /settings`)
- API key management (masked display, per-provider update)
- Scheduler controls (morning prep toggle, run now, reschedule 7:30-8:30 random)
- Probability weight sliders
- Analysis framework weight sliders (26 factors)
- Personality prompt textarea (editable, saved to DB)
- Feature toggles (app_settings key-value)
- Results monitor controls
- Token usage summary

#### 5.10 Learnings (`GET /settings/learnings`)
Post-race AI assessment history. Last 20 meetings with assessment counts, P&L, hit rates. Overall statistics.

#### 5.11 Probability / Calibration (`GET /probability`)
Probability engine dashboard: calibration curve (predicted vs actual), per-factor performance, Brier scores (model vs market), value performance, weight history, bet type threshold tuning, context multiplier insights.

---

## 6. Data Pipeline

### 6.1 Scraping Sources (priority order)

| Source | Auth | Primary Data |
|--------|------|-------------|
| PuntingForm API | API key | Fields, form, speed maps, AI ratings, conditions |
| Racing.com | Playwright + GraphQL interception | Multi-bookie odds, form comments, expert tips |
| Betfair Exchange | Cert-based API | Exchange WIN + PLACE odds |
| PointsBet | No auth (public JSON API) | Primary corporate odds + place odds |
| TAB (Playwright) | No auth | International (HK) odds, fallback |
| Racing Australia | No auth (HTML scrape) | Fallback fields, cross-check barrier/weight/jockey |
| WillyWeather | API key | Live weather for 16 tracked AU venues |
| Open-Meteo | No auth | NZ weather fallback |
| loveracing.nz | HTML scrape | NZ track conditions |

### 6.2 Full Scrape Pipeline (per meeting)

```
Step 1: PuntingForm API
  ├── /form/fields → runner base data, stats, career
  ├── /form/form → 10 past starts per runner (form_history)
  ├── /User/Speedmaps → pace position, speed rank, map factor
  └── /Ratings/MeetingRatings → PF AI scores and prices
  FALLBACK 1: RA HTML fields
  FALLBACK 2: Racing.com Playwright GraphQL

Step 2: Track conditions
  ├── PuntingForm conditions API
  ├── Racing Australia HTML (authoritative for AU)
  └── HKJC (Hong Kong only)

Step 2d: RA cross-check (barrier, weight, jockey corrections)

Step 3: Racing.com supplement (odds, form comments, stewards)

Step 4: Betfair exchange odds (WIN + PLACE markets)

Step 4b: PointsBet corporate odds

Step 5: International only (TAB Playwright / HKJC)

Post-scrape: Derived fields (days_since_last_run, class_stats)
             3-attempt SQLite commit with retry
```

### 6.3 Odds Priority (for `current_odds`)

Trusted tier: PointsBet (primary) + Betfair (exchange, live backs only)
Supplementary: Sportsbet → TAB → Bet365 → Ladbrokes

The context builder computes `_get_median_odds()` per runner, anchored to the trusted tier. Supplementary sources only included if within 50% of trusted median.

### 6.4 Data Models

**Meeting** (19 columns): venue, date, track_condition, weather fields, rail_position, penetrometer, rainfall, irrigation, going_stick. No `state` column — derived at runtime from `venues.py`.

**Race** (18 columns): number, name, distance, class, prize_money, start_time, results_status (Open→Interim→Paying→Closed), exotic_results (JSON dividends), sectional_times, expert_tips.

**Runner** (51 columns): identity (name, saddlecloth, barrier, weight, jockey, trainer), form (last10, career, form_history JSON), lineage (age, sex, sire, dam), stats (9 JSON stat columns for track/distance/condition/jockey/trainer), speed map (position, PF rank/settle/map_factor), odds (6 bookmakers + flucs + opening), results (finish_position, dividends, margins), PF AI ratings, gear.

**Pick** (38 columns): pick_type (selection/exotic/sequence/big3), horse/race identification, bet_type (win/place/saver_win/each_way), stake, odds_at_tip, tip_rank, exotic fields (type, runners JSON, structure), sequence fields (type, variant, legs JSON, start_race), settlement (hit, pnl, settled, settled_at), Punty's Pick dual-settlement (pp_bet_type, pp_odds, pp_hit, pp_pnl), factors_json.

**Content** (status lifecycle): `draft → pending_review → approved → sent` (or `rejected` / `superseded`). Stores raw_content, twitter_formatted, facebook_formatted, context_snapshot_id.

**BetfairBet**: meeting_id, race_number, horse_name, bet_type, odds, stake, place_probability, kelly_fraction, status (pending/queued/placed/settled/cancelled), pnl, balance tracking.

---

## 7. AI Content Generation

### 7.1 Pipeline

```
Trigger (UI SSE stream or scheduler)
  → Pre-generate odds refresh
  → ContextBuilder.build_meeting_context()
    → Load probability weights + LGBM settings from DB
    → Load meeting + races + runners from DB
    → Fetch live weather (WillyWeather/Open-Meteo)
    → Fetch standard times for time ratings
    → Fetch jockey/trainer strike rates (PuntingForm)
    → Per race: _build_race_context()
      → Assemble full runner data dict
      → Calculate odds movement (heavy_support/firming/drifting/big_drift)
      → Classify pace scenario (hot/slow/genuine/moderate)
      → Run probability engine → win/place probs, value ratings
      → Run pre_selections → bet types, stakes, edge gate
    → Build sequence_leg_analysis + pre_built_sequences + pre_big3
  → Context snapshot (SHA-256 hash, change detection)
  → Learning context (optional RAG, 30s timeout)
    → Strategy context (historical P&L at track/conditions)
    → Batch embeddings for similarity search
    → Runner pattern memories (if ≥10 settled memories)
  → Format context → structured markdown prompt
  → AI call (GPT-5.4, temperature 0.5, 600s timeout)
  → Post-process:
    → _correct_sequence_legs() — fix AI's saddlecloth numbers
    → _correct_exotic_runners() — replace non-pick exotics
  → Save as Content (status: pending_review)
```

### 7.2 Prompt Structure

**System:** personality.md (Punty's voice, rules, glossary) + Analysis Framework Weights

**User prompt contains:**
- Meeting header (weather, rail, rainfall, wind impact)
- Per-race markdown table (No/Horse/Barrier/Jockey/Trainer/Odds/Form/Speed Map/Move/Win%/Place%/Value)
- Per-race runner details (career, gear, first/second up, track/distance stats, sectionals)
- Form comments (top 3, truncated 200 chars)
- Speed ratings (FAST/SLOW only)
- Weight analysis (carrier warnings, change direction)
- Context multipliers (shown as STRONG/weak labels, never raw numbers)
- Probability rankings (top 6)
- Value plays (>0.90 threshold) and place value plays (>1.05)
- Pattern insights from deep learning
- PF AI overlay/underlay signals (10%+ discrepancies)
- Pre-calculated selections block (AI is told to use these)
- Pre-built sequences (AI copies verbatim)
- Pre-calculated Big 3 Multi
- Exotic combo table (filtered: must anchor on picks)

### 7.3 Review Workflow

On **approve**: supersedes previous approved content for same meeting, deletes old picks, calls `store_picks_from_content()` (regex extraction) + `store_picks_as_memories()` (RAG), then `populate_bet_queue()` (Betfair auto-bet).

On **unapprove**: reverts to pending_review, deletes picks, removes Twitter/Facebook posts.

### 7.4 Content Types & Temperatures

| Type | Temperature | Review Required |
|------|------------|-----------------|
| early_mail | 0.5 | Yes |
| results | 0.8 | Yes |
| meeting_wrapup | 0.8 | Yes |
| race_preview | 0.8 | No (auto-approved) |
| update_alert | 0.8 | Yes |
| weekly_blog | 0.9 | Yes |

---

## 8. Probability Engine

### 8.1 Three-Layer Architecture

```
Layer 1 (PRIMARY): LGBM-rank + market-calibrate
  ├── LGBM model ranks runners (45.6% top-1 accuracy)
  ├── Market odds provide calibrated probabilities
  ├── Tissue used as tiebreaker only (5% gap + 8% disagreement)
  └── LGBM_RANK_INFLUENCE = 0.55 (55% LGBM, 45% market)

Layer 2 (FALLBACK): Tissue engine
  ├── Multiplicative condition-lookup model
  └── 29% top-1 accuracy (worse than market favourite at 33.5%)

Layer 3 (FULL FALLBACK): Weighted factor engine
  ├── 10 factor scores (market, form, class, pace, barrier, etc.)
  ├── Weighted sum → normalized → power-sharpened (^1.25)
  └── Market floor blend (max 40% market when model diverges >60%)
```

### 8.2 LGBM Probability Pipeline

1. **LGBM prediction** → raw scores normalized to sum=1.0 (ordering only, raw probs are inverted)
2. **Tissue tiebreaker** → single-pass: swap adjacent runners within 5% LGBM gap if tissue disagrees by 8%+
3. **Rank-weight assignment** → `weight[i] = ((n-i)/n)^1.5` (exponential decay)
4. **Blend** → `blended_win = 0.45 × market_implied + 0.55 × rank_weight` (after normalization)
5. **Place probability** → Harville (1973) conditional model on blended win probs, top_n=8 runners, capped at 0.75

### 8.3 Harville Place Probability

Standard conditional probability formulation:
```
P(i places) = P(i=1st) + P(i=2nd) + P(i=3rd)
P(i=2nd) = Σ_{j≠i} P(j=1st) × P(i=1st | j removed)
P(i=3rd) = Σ_{j≠i} Σ_{k≠i,j} P(j=1st) × P(k=2nd|j removed) × P(i=1st|j,k removed)
```
Computed over top 8 runners by win probability (O(n³) → O(512) with cutoff).

### 8.4 Key Thresholds

| Location | Value | Meaning |
|----------|-------|---------|
| LGBM influence | 0.55 | 55% LGBM rank, 45% market |
| Rank exponent | 1.5 | Sharpness of rank-weight decay |
| Tissue tiebreaker gap | 5% LGBM, 8% tissue | Swap threshold |
| Place probability cap | 0.75 | Max single-runner place prob |
| Harville top_n | 8 | Runners included in conditional calc |
| Value plays threshold | 0.90 | Win value_rating for prompt inclusion |
| Place value threshold | 1.05 | Place value_rating for prompt inclusion |
| Power sharpening (fallback) | 1.25 | Concentrates probability mass |
| Market floor trigger | model/market < 0.60 | Blend toward market |
| Market floor max blend | 40% | Maximum market contribution |

### 8.5 Pre-Selections (Bet Type Rules)

```
If field_size ≤ 4: Win for all ranks (no place market)
Rank 1 (not roughie): Win
Rank 2+: Place
Roughies: always Place
Rank 3+: tracked_only (zero stake, MAX_STAKED_PICKS=2)
```

**Edge gate thresholds:**
- Win: `win_prob ≥ 0.15`
- Place at <$3: `place_prob ≥ 0.55`
- Place at $3-$6: `place_prob ≥ 0.40`
- Place at $6+: `place_prob ≥ 0.35`
- High-prob override: `place_prob ≥ 0.70` (always passes)

**Pool tiers:** $25 (high EV or high PP), $20 (standard), $12 (low confidence)

### 8.6 Exotic Routing (Race-Shape Decision Tree)

```
DOMINANT (win prob spread > 12%):
  → Exacta Standout, Exacta 2-box, Quinella 2-runner

STRUCTURED (spread 5-12%):
  → Quinella 3-box (default), First4 4-box

OPEN (spread < 5%):
  → Box structures with wider coverage
```

Budget: $15 per race. Flexi % calculated per structure. Minimum 10% flexi floor.

### 8.7 Sequence Construction (Quaddie/Early Quaddie)

- Legs built from top runners by win probability
- Chaos leg detection: field ≥14, or fav ≥$5, or top1_prob ≤0.22, or top1-top3 gap ≤6%
- Budget optimization: trim/add runners per leg to fit budget
- Combo cost: product of runners per leg × unit price
- Big6 killed (1/59 hit rate, -95.7% ROI)

---

## 9. Betting & Settlement

### 9.1 Betfair Auto-Bet (Kelly Staking)

On content approval → `populate_bet_queue()` creates `BetfairBet` rows for rank 1-4 picks.

Kelly formula (as implemented — see bugs section):
```
edge = place_probability - (1/odds)
kelly = edge / (odds - 1)
if half_kelly: kelly *= 0.5
kelly = min(kelly, 0.06)  # 6% cap
stake = kelly * balance
```

Min stake: $5.00 (Betfair minimum). Max 4 bets per meeting.

Scheduler polls every 30s, places bets 10 minutes before race start.

### 9.2 Calibration

Bins settled bets by predicted place probability (10 bins, 0.10 width). Computes actual strike rate per bin. Requires 15 samples minimum per bin. Linear interpolation between bin centers. 1-hour TTL cache.

### 9.3 Settlement Math

**Win/Saver Win:**
```
PNL = (win_dividend × stake) - stake   if 1st
PNL = -stake                           otherwise
```

**Place:**
```
PNL = (place_dividend × stake) - stake   if top 3
PNL = -stake                             otherwise
NTD (≤4 runners): voided
```

**Each Way (half win, half place):**
```
Won:     PNL = (win_dividend × half) + (place_dividend × half) - stake
Placed:  PNL = (place_dividend × half) - stake
Lost:    PNL = -stake
```

**Exotics (flexi):**
```
flexi_pct = stake / num_combos
PNL = (dividend × flexi_pct) - stake
```

**Sequences (quaddie):**
```
All legs must hit (winner's saddlecloth in our leg selections)
Scratched legs: free pass (TAB rule)
PNL = (dividend × flexi_pct) - stake
```

**Dead heat:** Divide dividend by number of dead heat participants.

**Sanity guard:** 10× ratio check between tip-time odds and tote dividend.

---

## 10. Delivery

### 10.1 Twitter/X

- Tweepy v2 for posting, v1.1 for media upload
- Long-form posts (X Premium)
- Markdown → Unicode Mathematical Bold Sans-Serif conversion
- Profanity substitution (`cunts` → `legends`, `fuck*` → `bloody`)
- Hashtags: `#AusRacing #HorseRacing` + 2 venue-specific
- Early mail truncated to R1-R2 with punty.ai CTA
- 2 retries (5s, 15s delays)

### 10.2 Email (Resend)

Primary: Resend API. Fallback: SMTP. Used for morning prep reports and content delivery.

### 10.3 Telegram

Bot with Claude Sonnet agent. Owner-only authorization. Can execute bash commands on server (admin interface). Sends daily P&L digest at 23:00 AEDT.

### 10.4 Facebook

OAuth token exchange flow (3-step for permanent page token). Posts via Graph API.

### 10.5 WhatsApp

Referenced in CLAUDE.md but **does not exist** in the codebase. No `whatsapp_formatted` column on Content model, no `punty/delivery/whatsapp.py`, no `punty/formatters/whatsapp.py`.

---

## 11. Infrastructure

### 11.1 Deployment

```
git push origin master
  → GitHub Actions CI
    → unit-test (pytest)
    → e2e-test (Playwright, continue-on-error)
    → deploy-staging (SSH to /opt/puntyai-staging, health check port 8001, 3-retry rollback)
    → deploy-production (SSH to /opt/puntyai, health check port 8000, 3-retry rollback)
```

No manual approval gate between staging and production.

### 11.2 Server

- Ubuntu VM at `app.punty.ai` (209.38.86.195)
- uvicorn on 127.0.0.1:8000, reverse proxied by Caddy
- systemd unit `puntyai.service`
- SQLite at `/opt/puntyai/data/punty.db`

### 11.3 Startup Sequence

1. Validate secret key
2. `init_db()` — create tables + run migrations
3. Load personality prompt cache
4. Start APScheduler
5. Setup daily automation (per-meeting scrape/generate jobs)
6. Schedule 23:00 daily P&L digest
7. Settlement pass (catch picks missed by restarts)
8. Start Telegram bot
9. Start Results Monitor
10. Conditionally start Betfair Bet Scheduler

### 11.4 Scheduled Jobs

- **00:05** — Calendar scrape (meeting discovery)
- **07:30-08:30** (random) — Morning prep (full scrape + generation)
- **Every 30s** — Betfair bet execution check
- **23:00** — Daily P&L digest via Telegram

### 11.5 Background Processes

- **Results Monitor** — polls for race results, triggers settlement
- **Betfair Bet Scheduler** — 30s poll loop, executes due bets, refreshes selections
- **Telegram Bot** — async message handler with Claude agent

---

## 12. Bugs & Math Issues

### CRITICAL

#### 12.1 Kelly Formula Bug (`punty/betting/queue.py:121`)

The `calculate_kelly_stake` function uses an incorrect Kelly formula:

```python
# Implementation (WRONG):
edge = place_probability - (1/odds)
kelly = edge / (odds - 1)

# Correct Kelly:
kelly = (place_probability * odds - 1) / (odds - 1)
```

The implementation divides by an extra factor of `odds`. At $2.00: understates Kelly by 50%. At $1.50: understates by 33%. The `_recommended_stake` function in `probability.py` uses the correct formula — two different Kelly calculations exist in the codebase giving different results.

**Impact:** Systematically under-bets, especially at short place odds. Partially masked by the 6% cap but affects all bets where correct Kelly > implemented Kelly and both are below cap.

#### 12.2 Edge Gate Checks Win Odds Instead of Place Odds (`pre_selections.py:882-888`)

```python
elif pick.bet_type == "Place" and pick.odds:
    place_cap = 10.0 if pick.rank == 1 else 8.0
    if pick.odds > place_cap:  # THIS IS WIN ODDS, NOT PLACE ODDS
        pick.tracked_only = True
```

A horse at $12 win / $3.50 place gets excluded from Place bets despite strong place value. Should check `pick.place_odds`.

### HIGH

#### 12.3 Calibration Includes Win Bets for Place Probability (`calibration.py:57`)

```sql
AND bet_type IN ('place', 'win', 'saver_win')
```

Win bets that placed-but-didn't-win are recorded as `hit=False`, artificially lowering calibrated place probability. Should only include `bet_type = 'place'` or use a separate hit criterion for place.

#### 12.4 Sequence Type Check Logic Inverted (`picks.py:939`)

```python
elif seq_type not in "early":  # Tests if seq_type is substring of "early", not the reverse
```

Works by coincidence because "quaddie" is not a substring of "early". Should be `"early" not in seq_type`.

#### 12.5 Context Double-Build in Snapshot (`versioning.py:20`)

`create_context_snapshot()` calls `builder.build_meeting_context()` internally, but the caller (`generate_early_mail_stream`) already built the full context. The snapshot step rebuilds the entire context a second time — all DB queries, weather fetches, probability calculations. Doubles compute cost.

### MEDIUM

#### 12.6 Place Probability Normalization (`probability.py:738-740`)

The `min(0.75, ...)` cap is applied before renormalization, so when the cap fires for multiple runners, the sum of all place probabilities can be less than `place_count`. Minor mathematical inconsistency.

#### 12.7 Delivery Status Raw String (`twitter.py:177`)

```python
content.status = "delivery_failed"  # Raw string, not ContentStatus enum
```

Bypasses the `ContentStatus` enum. Could cause issues with any code doing enum comparison.

#### 12.8 `deep_learning` Weight Inconsistency (`probability.py:376`)

Calibrated path hardcodes `deep_learning = 0.0` while DEFAULT_WEIGHTS has `0.03`. Creates different behaviour depending on whether calibration file exists.

#### 12.9 Place Odds Sanity Threshold Too Loose (`probability.py:669`)

`expected_place * 2.5` allows $3.70 place on a $2.00 win shot through uncorrected. Should be tighter (1.5× or 2.0×).

#### 12.10 rainfall Column Type Mismatch

ORM declares `Text` in the model, but the ALTER TABLE migration in `database.py:249` types it as `REAL`. SQLite's loose typing allows both to coexist but any numeric comparison queries would silently fail.

---

## 13. Dead Code

### 13.1 Scrapers

| File | Status | Notes |
|------|--------|-------|
| `punty/scrapers/tab.py` (`TabScraper`) | Dead | Placeholder CSS selectors that never match. Superseded by Betfair/PointsBet/tab_playwright. Called from scheduler but returns empty. |
| `punty/scrapers/punters.py` (`PuntersScraper`) | Dead | Placeholder CSS selectors. Exported from `__init__.py`, called in scheduler, always returns empty. |

### 13.2 Models

| Item | Notes |
|------|-------|
| `Result` model (`models/meeting.py:340`) | Table defined, `Race.results` relationship exists, but nothing writes to it. All result data lands on Runner columns. |
| `Runner.speed_value` | Set from PF but never surfaced in context builder or templates. |
| `Runner.sectional_400/800` | Set during result collection but unused downstream. |
| `Runner.starting_price` | Stored as string but `win_dividend` is used for settlement. |
| `Content.whatsapp_formatted` | Referenced in CLAUDE.md but column does not exist on model. |

### 13.3 Functions

| Function | Location | Notes |
|----------|----------|-------|
| `_place_probability()` | `probability.py:2761` | Defined but only `_harville_place_probability()` is used. |
| `generate_meeting_wrapup()` | `generator.py:939` | Duplicate non-streaming wrapper; streaming version is the one used. |
| `send_thread()` naming | `twitter.py` | Misleadingly named — posts single long-form, not a thread. Thread-splitting logic in `format_as_thread()` is unused. |
| `_get_sequence_lanes()` | `generator.py:1033` | Duplicates logic also in `pre_sequences.py`. |
| `initialise.md` prompt | `prompts/` | Not in `REQUIRED_PROMPTS`, not referenced in generator. |

### 13.4 Dependencies

| Package | Notes |
|---------|-------|
| `bcrypt>=4.1.0` | In `pyproject.toml` but no password hashing in codebase (all auth is Google OAuth). |

---

## 14. Technical Debt

### 14.1 Architecture

| Issue | Impact | Effort |
|-------|--------|--------|
| `public/routes.py` is 3,200+ lines | Hard to navigate, high merge conflict risk | High — split into multiple route modules |
| Public API endpoints live in `main.py` (511 lines) | Inconsistent placement, mixed concerns | Medium — move to `api/public.py` |
| Privacy page route lives in `auth.py` | Misplaced | Low |
| Dual schema definition for `betfair_bets` (ORM + raw DDL) | Maintenance confusion | Low |
| Dual schema for `future_races`/`future_nominations` | Same | Low |
| No `state` column on meetings | Every state lookup is a runtime venue→state derivation | Medium — add column, backfill |
| No FK enforcement in SQLite | Stale runners possible | Low risk (app logic handles) |
| Single-process in-memory rate limiting | Resets on restart, no multi-worker support | Low risk (single process currently) |

### 14.2 Scraping

| Issue | Impact |
|-------|--------|
| Playwright scrape lock acquired manually (not via context manager) in streaming version | Risk of lock non-release on exception |
| Racing.com GraphQL parse failures silently discarded (`except Exception: return`) | Entire race data dropped without warning |
| WillyWeather venue locations hardcoded (16 venues) | Adding venues requires code changes |
| Betfair `_strip_venue_prefix` duplicates `venues.py::SPONSOR_PREFIXES` | Two sources of truth |
| PointsBet `MAX_VALID_ODDS = 501.0` hardcoded in two files | Should be shared constant |

### 14.3 AI Pipeline

| Issue | Impact |
|-------|--------|
| Context snapshot rebuilds entire context (double compute) | Doubles generation latency (~10s wasted) |
| Form comments fallback checks `comment_long` on already-processed dict (never set) | Dead fallback path |
| Two different Kelly formulas in codebase (queue.py vs probability.py) | Inconsistent staking calculations |
| `send_thread()` name vs actual behaviour | Confusing for maintainers |
| CLAUDE.md references WhatsApp infrastructure that doesn't exist | Misleading documentation |

### 14.4 CI/CD

| Issue | Impact |
|-------|--------|
| No manual approval gate between staging and production | Live betting system deploys automatically |
| E2E tests set `continue-on-error: true` | Failures don't block deploy |

### 14.5 Hardcoded Magic Numbers

Over 40 hardcoded thresholds across the codebase. Key ones that should be configurable:

| Value | Location | Purpose |
|-------|----------|---------|
| 1.5 | probability.py | Rank weight exponent |
| 0.75 | probability.py | Place probability cap |
| 8 | probability.py | Harville top_n |
| 0.90 / 1.05 | builder.py | Value play thresholds |
| 50% / 30% | orchestrator.py | Speed map completeness thresholds |
| 10.0 / 8.0 | pre_selections.py | Place odds caps (checks wrong field) |
| 0.15 | pre_selections.py | Win EV threshold for high pool |
| 0.85 | pre_sequences.py | TAB takeout rate |
| 2.5 | probability.py | Place odds sanity multiplier |
| 300s | playwright_base.py | Browser idle timeout |

---

*Document generated 2026-03-13 from full codebase review across all modules.*
