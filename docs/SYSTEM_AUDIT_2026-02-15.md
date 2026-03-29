# PuntyAI Comprehensive System Audit
**Date:** 2026-02-15 | **Scope:** End-to-end data flow, probability engine, AI context, bet settlement, website/UX

---

## Executive Summary

Five parallel audits were conducted across the entire PuntyAI stack. The system is **sophisticated and mostly functional** with strong fundamentals in probability modeling, AI context generation, and settlement logic. However, critical gaps exist in data validation, DL pattern verification, and UX hardening.

**Overall System Health: 75/100**

| Component | Score | Status |
|-----------|-------|--------|
| Scraper Data Flow | 7/10 | Working, gaps in coverage |
| Probability Engine | 8/10 | Sophisticated, needs DL validation |
| AI Context & Prompts | 8/10 | Rich context, missing ROI segmentation |
| Bet Settlement | 8.5/10 | Solid, 3 edge-case fixes needed |
| Website & UX | 6.5/10 | Good design, needs error handling |

---

## 1. SCRAPER DATA FLOW

### Data Sources & Coverage

| Source | Fields | Coverage | Role |
|--------|--------|----------|------|
| Racing.com GraphQL | ~50 fields (form, stats, gear, stewards) | 90%+ | Primary race data |
| TAB API | Odds, results, dividends | 99% (odds), variable (place_odds) | Odds + settlement |
| Sportsbet | Odds | 95%+ | Now primary odds (24% error vs TAB's 334%) |
| Punting Form | Speed maps, A2E stats, track conditions | 40-70% | Supplementary intelligence |
| WillyWeather | Hourly forecasts, live observations | 95%+ | Weather context |

### Recent Fixes (This Session)
- **SB-primary odds** — Switched from TAB (334% error) to Sportsbet (24% error)
- **days_since_last_run** — Was 0% populated, now 99.5% (derived from form_history)
- **class_stats** — Was 0% populated, now 81.8% (derived from form_history)
- **place_odds** — Was 43%, now 85.7% (TAB-only, no fabrication)
- **Track condition flip-flop** — Fixed monitor overwriting specific conditions with generic
- **PF date filtering** — Fixed conditions API returning wrong-date entries

### Remaining Data Gaps

| Field | Coverage | Impact | Priority |
|-------|----------|--------|----------|
| form_history (JSON) | 50-60% in early races | Loses 6 combo form signals | HIGH |
| speed_map_position | 40-50% | Pace factor near-useless without it | MEDIUM |
| A2E jockey/trainer JSON | 70% | Falls back to string parsing | MEDIUM |
| bet365/ladbrokes/betfair odds | 0% | Only 2 of 5 bookmakers populated | LOW |
| track_dist_stats | 40% | Falls back to track_stats (65%) | LOW |

---

## 2. PROBABILITY ENGINE

### Factor Weights & Health

| Factor | Weight | Coverage | Reliability | Status |
|--------|--------|----------|-------------|--------|
| Market Consensus | 40% | 99% | HIGH | ✅ Working well |
| Form Rating | 32% | 95% base, 50% combo | HIGH | ✅ Mostly working |
| Deep Learning | 8% | 20-50% match | **UNVALIDATED** | ⚠️ CRITICAL |
| Jockey/Trainer | 7% | 85% | HIGH | ✅ Working |
| Weight Carried | 4% | 95% | MEDIUM | ⚠️ Could improve |
| Horse Profile | 3% | 90% | MEDIUM | ⚠️ Could improve |
| Class/Fitness | 3% | 50-80% | MEDIUM | ⚠️ Gaps |
| Barrier Draw | 2% | 98% | MEDIUM | ⚠️ Needs per-venue calibration |
| Pace | 1% | 40-50% | ANTI-PREDICTIVE | ⚠️ Near-useless |
| Movement | 0% | 80% | UNKNOWN | Deliberately disabled |

### Post-Processing Pipeline
1. **Context multipliers** — Venue/distance/class-specific adjustments (from context_profiles.json)
2. **Dynamic market boost** — When 40%+ factors are neutral, market weight increases to 65%
3. **Sharpening** — Power 1.45 to fix favorite underestimation
4. **Market floor** — Prevents >65% disagreement with market implied probability
5. **Place probability** — Field-size-adjusted multiplier (2.0x to 3.3x)
6. **Quarter-Kelly staking** — `stake = kelly_fraction × 0.25 × pool`

### Critical Issues

**1. Deep Learning Patterns UNVALIDATED (8% weight)**
- 844 patterns loaded from `dl_patterns.json` (discovered from 280K runners)
- **No hold-out test validation** — patterns may be overfitted noise
- No sample_size weighting (50-sample pattern treated same as 5,000-sample)
- 5 pattern types skipped as "non-discriminative" but rationale undocumented
- **Recommendation:** Validate on hold-out set. Filter: edge >1%, confidence ≥0.7, sample ≥100. Disable if <50% valid.

**2. Calibration Files Not In Repo**
- `calibrated_params.json` — Engine defaults to hand-coded piecewise linear fallback
- `context_profiles.json` — Built from one-off Proform dump, now stale
- **Recommendation:** Commit files, build rebuild pipeline from production data

**3. Hardcoded Thresholds Without Justification**
- SHARPEN = 1.45 (empirical but not re-validated)
- Barrier multipliers (1.3, 1.0, 0.7) — no per-venue calibration
- Handicap neutral = 70 (regional variation unaccounted)
- Prize money benchmarks static (no inflation adjustment)
- Pace: 3+ leaders = hot_pace (arbitrary)

---

## 3. AI CONTEXT & PROMPT SYSTEM

### What's Sent to the AI

The system sends **exceptionally rich context** — 230+ fields per runner organized into:

| Category | Data Sent | Quality |
|----------|-----------|---------|
| Meeting & Environment | Venue, track condition, weather (hourly), rail, penetrometer | ✅ Complete |
| Runner Form | Form string, last_five, career, stats, stewards comments | ✅ Complete |
| Odds & Market | Current, opening, movement direction/magnitude, movers summary | ✅ Complete |
| Speed Maps | Position, pf_map_factor, pf_speed_rank, pf_jockey_factor | ⚠️ 40-50% coverage |
| Probability Rankings | Win/place prob, value rating, recommended stake per runner | ✅ Complete |
| Pre-Calculated Selections | Top 3 + roughie, bet types, stakes, expected return | ✅ Complete |
| Pre-Calculated Exotics | Harville model combos, value ≥1.2x, consistency-enforced | ✅ Complete |
| Sequence Lanes | Skinny/Balanced/Wide with leg confidence and combo counts | ✅ Complete |
| Context Multipliers | Per-factor multipliers for win + place separately | ✅ Complete |
| Strategy Performance | All-time ROI, 30-day rolling, per-rank breakdown, directives | ✅ Complete |
| Deep Learning Patterns | Max 15 HIGH/MED patterns matching today's runners | ⚠️ Passive info |

### Critical Gaps in AI Context

| Missing Data | Why It Matters | Priority |
|--------------|----------------|----------|
| **Venue-type ROI** | Metro -0.7% vs Provincial +9.0% — AI never told | HIGH |
| **Distance-specific ROI** | Sprint vs staying performance differs | HIGH |
| **Roughie allocation rules** | $8-$20: +53% ROI, $50+: -100% ROI — AI picks blindly | HIGH |
| **ROI conflict resolution** | Racing says "Win", ROI says "Place" — no guidance | MEDIUM |
| **Market validation signal** | "Is this horse's support consistent with form?" | MEDIUM |
| **Sequence/selection consistency** | Lanes can diverge from Top 3 without warning | LOW |

### Context Multipliers NOT Wired Into Selection Logic

This is a **major integration gap**: context multipliers are calculated for both win and place, shown in the prompt, but `pre_selections.py` uses **fixed thresholds** (WIN_MIN_PROB=0.18, PLACE_MIN_PROB=0.35) that never receive the multipliers. When place context is strong (multiplier >1.2x), the system should automatically lower the place threshold and recommend more Place bets.

### Parser Fragility

The regex parser expects exact formatting:
- "Degenerate Exotic" header (fails on "Exotic of the Race")
- `No.X` format (fails on `No X` with space)
- `$X.XX` odds format (fails without `$`)
- No validation that ALL races have picks (missing R3 = silent failure)

---

## 4. BET SETTLEMENT

### Settlement Accuracy

| Bet Type | Formula | Verified | Issues |
|----------|---------|----------|--------|
| Win/Saver Win | `(odds × stake) - stake` | ✅ | None |
| Place | `(place_odds × stake) - stake` | ✅ | Small field rules correct |
| Each Way | `(win × half) + (place × half) - stake` | ⚠️ | **Issue #1 below** |
| Exotic (box/flexi) | `(dividend × unit) - stake` | ✅ | Scratching rules correct |
| Sequence (quaddie/big6) | `(dividend × unit) - total_stake` | ⚠️ | **Issue #2 below** |
| Big3 Multi | `(multi_odds × stake) - stake` | ✅ | Timing check correct |

### Issues Found

**Issue #1: Each Way Win Without Place Odds = Loss (MEDIUM)**
- If horse WINS but `place_odds` is None, both halves treated as loss
- Should: pay win half at win odds, log warning for missing place half
- Location: `picks.py` line 226

**Issue #2: Hit Without Dividend = Loss Instead of Refund (MEDIUM)**
- Exotic/sequence hits but dividend missing → treated as loss ($-20)
- TAB rules would typically refund
- Location: `picks.py` lines 565, 697

**Issue #3: No Dead Heat Handling (LOW)**
- Dead heats not adjusted (dividends typically halved)
- Depends on scraper pre-adjusting — if not, P&L wrong
- Missing `dead_heat_count` field on Runner model

**Issue #4: Cumulative P&L Excludes Losing Sequences (LOW)**
- `picks.py` line 915 filters out losing sequences from cumulative metric
- May be intentional but undocumented

### What's Working Well
- Australian place rules (2 places for 5-7 runners, 3 for 8+) ✅
- Scratching free-pass logic (all selections scratched = auto-hit) ✅
- Dividend backfill (re-settles when TabTouch provides late dividends) ✅
- Big3 timing (waits for all 3 races to resolve before settling) ✅
- Proper rounding (`round(..., 2)` on all P&L) ✅

---

## 5. WEBSITE & UX

### Route Coverage

| Route | Quality | Critical Issues |
|-------|---------|-----------------|
| Dashboard | Good | N+1 queries, exceptions swallowed silently |
| Meetings List | Good | Archive shows no status info |
| Meeting Detail (1,438 lines) | Excellent | Speed map warning too late, no per-race scrape status |
| Review Queue | Basic | No pagination, no filtering, "Reject All" has no confirmation |
| Review Detail | Good | Prev/next navigation broken across dates |
| Probability Tuning | Good | Read-only, good visualizations |
| Settings | Excellent | Comprehensive, good validation |

### Critical UX Gaps

| Issue | Severity | Fix |
|-------|----------|-----|
| No global error display | HIGH | Toast notification middleware |
| Missing confirmation dialogs (Reject All, Delete) | HIGH | Add "Are you sure?" modals |
| Review prev/next broken across dates | HIGH | Add date filter to query |
| No input validation on 30%+ of routes | HIGH | Return 404 not 500 |
| Race conditions (double-generate, double-settle) | MEDIUM | Idempotency keys, mutexes |
| No unsaved changes warning | MEDIUM | Dirty flag on settings |
| Mobile tables broken | LOW | Horizontal scroll containers |
| Accessibility (keyboard nav, ARIA) | LOW | WCAG AA compliance |

### Design System
- Cyberpunk theme consistent and polished (9/10 visual design)
- Good use of CSS variables, gradients, typography
- HTMX for partial updates (good performance)
- SSE streaming for long operations (good UX)

---

## 6. PRIORITY RECOMMENDATIONS

### CRITICAL (Do This Week)

| # | Action | Component | Impact |
|---|--------|-----------|--------|
| 1 | **Validate DL patterns on hold-out test set** | Probability | 8% of model weight is unverified |
| 2 | **Fix Each Way win logic** (missing place_odds) | Settlement | Incorrect P&L on E/W winners |
| 3 | **Change hit-without-dividend to refund** | Settlement | Prevents false losses |
| 4 | **Add venue/distance/condition ROI to AI context** | AI Context | AI doesn't know where it performs well |
| 5 | **Add roughie allocation rules** ($8-$20 vs $50+) | AI Context | -100% ROI on $50+ roughies preventable |

### HIGH (Do This Month)

| # | Action | Component | Impact |
|---|--------|-----------|--------|
| 6 | **Wire context multipliers into pre_selections.py** | AI/Probability | Place bets (+13.8% ROI) underutilized |
| 7 | **Rebuild context profiles from live DB** | Probability | Current profiles stale (Proform one-off) |
| 8 | **Stabilize form_history pipeline** (backfill post-race) | Scraper | 50% missing = 6 combo signals lost |
| 9 | **Add input validation to all API routes** | Website | 30%+ of routes can crash on bad input |
| 10 | **Add confirmation dialogs on destructive actions** | Website | "Reject All" can destroy 50 items |

### MEDIUM (Do This Quarter)

| # | Action | Component | Impact |
|---|--------|-----------|--------|
| 11 | Calibrate barrier factor per venue from historical data | Probability | Replace hardcoded multipliers |
| 12 | Add dead heat detection to settlement | Settlement | Prevent incorrect P&L on ties |
| 13 | Add global error display (toast notifications) | Website | Errors only visible in console |
| 14 | Fix review navigation across dates | Website | Core workflow broken |
| 15 | Validate pace factor (remove if no edge) | Probability | 1% weight, anti-predictive |
| 16 | Add market validation signal to context | AI Context | "Is support consistent with form?" |
| 17 | Add ROI conflict resolution to prompt | AI Prompt | Racing vs ROI guidance missing |

### LOW (Ongoing)

| # | Action | Component |
|---|--------|-----------|
| 18 | Monthly DL pattern re-validation on new data | Probability |
| 19 | Quarterly calibration curve rebuild | Probability |
| 20 | Mobile responsiveness fixes | Website |
| 21 | WCAG AA accessibility compliance | Website |
| 22 | Idempotency keys on social delivery | Website |
| 23 | Document all hardcoded thresholds with rationale | Probability |

---

## 7. ROI IMPROVEMENT OPPORTUNITIES

Based on historical performance data and this audit:

### Highest-Impact Changes for Strike Rate & ROI

**1. Lean Into Place Bets Programmatically**
- Place bets: +13.8% ROI (profit engine)
- Win bets: -7.6% ROI (losing overall)
- **Action:** Wire context multipliers so place-strong races automatically get lower PLACE_MIN_PROB threshold

**2. Roughie Band Management**
- $8-$20 roughies: +53% ROI
- $50+ roughies: 0/33 hits, -100% ROI
- **Action:** Add explicit band rules to AI context; cap roughie odds at $30

**3. Provincial Focus**
- Provincial venues: +9.0% ROI
- Metro venues: -0.7% ROI
- **Action:** Add venue-type ROI to context; more aggressive staking at provincials

**4. Win Bet Sweet Spot**
- $4-$6 odds on Win: +60.8% ROI
- Under $2 on Win: -38.9% ROI
- **Action:** Add odds-band guidance to prompt; avoid short-priced Win bets

**5. Validate DL Patterns**
- 844 patterns at 8% weight — if even 50% are valid, this is the biggest untapped edge
- If mostly noise, removing them improves model by reducing confusion
- **Action:** Hold-out validation is the single highest-value engineering task

**6. Form History Completeness**
- 50% missing = 6 combo form signals lost
- Combo signals capture venue specialists, distance+condition experts, jockey partnerships
- **Action:** Post-race backfill pipeline to build form_history from results

---

## 8. SYSTEM ARCHITECTURE STRENGTHS

Things working exceptionally well:

1. **Multi-source data merge with conflict resolution** — Racing.com → TAB → PF → Playwright, with quality hierarchy
2. **Pre-calculated selections pipeline** — Harville model exotics, value-weighted picks, Kelly staking
3. **Rich AI context** — 230+ fields per runner, strategy learning, deep learning patterns
4. **Dividend backfill** — Smart re-settlement when TabTouch provides late dividends
5. **Track condition intelligence** — Specificity checking prevents data regression
6. **Probability normalization** — Sum-to-1, sharpening, market floor — mathematically sound
7. **Australian racing rules** — Small field place rules, flexi betting, scratching free passes
8. **Real-time UX** — SSE streaming for scraping/generation, countdown timers, live odds

---

## Appendix: Files Audited

| File | Lines | Role |
|------|-------|------|
| punty/probability.py | 2,629 | Probability engine (10 factors + calibration) |
| punty/ai/generator.py | ~1,400 | Context formatting + AI prompt construction |
| punty/context/builder.py | ~800 | Data assembly from DB to context dict |
| punty/ai/pre_selections.py | ~700 | Pre-calculated picks, exotics, Punty's Pick |
| punty/ai/pre_sequences.py | ~250 | Sequence lane construction |
| punty/results/picks.py | ~1,120 | Settlement logic + P&L calculations |
| punty/results/parser.py | ~600 | Regex extraction from AI text |
| punty/results/monitor.py | ~1,200 | Background results polling + settlement |
| punty/scrapers/orchestrator.py | ~500 | Multi-source data merge pipeline |
| punty/scrapers/racing_com.py | ~800 | Racing.com GraphQL scraper |
| punty/web/routes.py | ~1,400 | Page routes + dashboard |
| punty/api/*.py | ~3,000 | 70+ API endpoints |
| templates/*.html | ~6,700 | 13 Jinja2 templates |
| prompts/*.md | ~600 | AI personality + instruction prompts |
| punty/memory/strategy.py | ~600 | Strategy performance + learning context |
