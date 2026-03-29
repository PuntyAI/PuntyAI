# PuntyAI Probability Engine — Complete Manual

## Table of Contents
1. [What Is the Probability Engine?](#1-what-is-the-probability-engine)
2. [Dashboard Sections Explained](#2-dashboard-sections-explained)
3. [The 10-Factor Model](#3-the-10-factor-model)
4. [Data Flow: Source to Signal](#4-data-flow-source-to-signal)
5. [Context-Aware Multipliers](#5-context-aware-multipliers)
6. [Self-Tuning System](#6-self-tuning-system)
7. [How It Improves Over Time](#7-how-it-improves-over-time)
8. [Key Indicators to Watch](#8-key-indicators-to-watch)
9. [Known Limitations & Improvement Opportunities](#9-known-limitations--improvement-opportunities)

---

## 1. What Is the Probability Engine?

The probability engine is the mathematical brain behind PuntyAI's betting recommendations. For every runner in every race, it calculates:

- **Win probability** — our estimated chance of winning (0-100%)
- **Place probability** — estimated chance of finishing top 2/3
- **Value rating** — our probability / market probability (>1.0 = underpriced by the market)
- **Recommended stake** — quarter-Kelly criterion sizing from a $20/race pool
- **Factor breakdown** — individual scores from each of the 10 analysis factors

These outputs directly drive:
- Which horses Punty recommends (highest value ratings)
- What bet types are assigned (win/place/each way based on probability thresholds)
- How much to stake on each pick
- Pre-selections for exotics and sequence bets

**File:** `punty/probability.py` (~2,058 lines)

---

## 2. Dashboard Sections Explained

The `/probability` page has 7 sections. Here's what each shows and what to look for.

### 2.1 Summary Stats (Top Row)

| Stat | What It Means | Good Value | Warning |
|------|--------------|------------|---------|
| **Brier Score** | Model accuracy (lower = better). Perfect = 0.000, coin flip = 0.250 | < 0.200 | > 0.250 |
| **Market Brier** | Market accuracy baseline — what you'd get just using odds | Typically 0.180-0.220 | N/A |
| **Picks Analyzed** | Total settled selection picks with probability data | Growing over time | Stagnant = not storing probs |
| **Last Tune** | How many picks the most recent auto-tune analyzed | > 50 | 0 or missing |

**Key indicator:** If Model Brier < Market Brier, the engine is outperforming raw market odds. If Model Brier > Market Brier, the engine is adding noise rather than signal — factor weights likely need adjustment.

### 2.2 Calibration Chart

Shows predicted probability vs actual win rate across probability buckets (0-5%, 5-10%, ..., 50%+).

- **Perfect model:** The blue "Actual Win Rate" line sits exactly on the dashed "Perfect Calibration" line
- **Overconfident model:** Actual rate < predicted (line below diagonal) — we're overestimating our picks
- **Underconfident model:** Actual rate > predicted (line above diagonal) — we're underestimating winners
- **Pink "Market Baseline":** Shows where the market is right

**What to look for:**
- The actual line should track the diagonal. Systematic deviation in one bucket means the model is miscalibrated there
- If the 30-50% bucket shows 20% actual but 35% predicted, the engine is overconfident on favourites
- Small sample sizes (hover to see count) make individual points unreliable — need 20+ picks per bucket minimum

### 2.3 Factor Performance

**Bar chart** showing each factor's "edge" — the difference between the factor's average score for winners vs losers.

**Table columns:**
| Column | Meaning |
|--------|---------|
| **Weight** | Current weight (green if above default, red if below) |
| **Default** | The starting weight before any tuning |
| **Edge** | Winner avg score - Loser avg score. Positive = factor predicts winners |
| **Accuracy** | Point-biserial correlation with outcomes. Higher = more predictive |

**What to look for:**
- Factors with **positive edge AND high weight** are working well — they're predictive and the model relies on them
- Factors with **negative edge AND high weight** are actively hurting predictions — they'll get downweighted at next tune
- Factors with **zero edge** contribute nothing — their weight is effectively wasted
- **Market consensus** should always have the highest edge (it embeds all public information)
- If a factor's weight diverges far from default, check if the edge supports it

### 2.4 Weight History

Line chart showing how factor weights evolve through auto-tuning cycles. Each point is a tuning event.

**What to look for:**
- **Convergence:** Weights stabilizing means the engine has found a good configuration
- **Oscillation:** Weights bouncing up/down means not enough data or conflicting signals
- **Market weight trending up** suggests other factors lack predictive power with current data
- The table below shows each tuning event's date, picks analyzed, and what changed

### 2.5 Value Performance

ROI by value rating bucket. This is the most important section for profitability.

| Bucket | Meaning | Target |
|--------|---------|--------|
| **<0.80 (Unders)** | Market thinks they're better than we do | Should show negative ROI |
| **0.80-1.00** | Near fair value | Slightly negative ROI expected |
| **1.00-1.10** | Mild value | Break-even to slight positive |
| **1.10-1.30 (Value)** | Clear value picks | Positive ROI = model works |
| **1.30+ (Strong Value)** | Strongest disagreement with market | Best ROI if model is correct |

**Key indicator:** If the 1.30+ bucket shows positive ROI, the engine genuinely finds value the market misses. If all buckets are negative, the value detection isn't working — probabilities are miscalibrated relative to the market.

### 2.6 Category Breakdown

Performance by race type (distance bucket × track condition). Shows where the model excels and struggles.

**What to look for:**
- **Low Brier score** in a category = good predictions there
- **Positive ROI** in a category = profitable picks
- **Negative ROI + High Brier** in a category = the model struggles there — context multipliers should help

### 2.7 Context-Aware Multipliers

Interactive explorer showing how factor weights are dynamically adjusted based on racing context (venue type × distance × class).

Use the 3 dropdowns to explore different contexts. Each factor shows a multiplier:

| Multiplier | Meaning | Visual |
|------------|---------|--------|
| **> 1.5** | Factor is very strong in this context | Green, "amplified" |
| **1.1 - 1.5** | Factor gets a boost | Light green, "boosted" |
| **0.9 - 1.1** | No significant difference | Grey, "neutral" |
| **0.5 - 0.9** | Factor is weaker here | Light red, "dampened" |
| **< 0.5** | Factor has almost no predictive power here | Red, "suppressed" |

**Example insights:**
- Barrier draw is **amplified** in metro sprints (tight tracks, less recovery time) but **dampened** in country staying races
- Market consensus is **dampened** in provincial/country races (thinner markets, less informed money)
- Jockey/trainer stats are **amplified** in metro races (quality spread is wider)

The **Notable Context Effects** at the bottom highlights the most extreme multipliers across all contexts.

### 2.8 Bet Type Performance

Shows P&L by bet type (win, place, each way, saver win) plus exotic and sequence performance. Includes current thresholds and optimal thresholds calculated from historical data.

---

## 3. The 10-Factor Model

Every runner gets scored 0.0-1.0 on each factor, where 0.5 = neutral. Scores above 0.5 favour the runner; below 0.5 penalize them.

### 3.1 Market Consensus (Default 22%)

**Source:** Multi-bookmaker median odds (TAB, Sportsbet, Bet365, Ladbrokes, Betfair)

**How it works:**
1. Collects odds from all available bookmakers
2. Takes the **median** (not average — resistant to outliers/errors)
3. Converts to implied probability: `1.0 / odds`
4. Normalizes by dividing by the total overround (removes bookmaker margin)

**Why it matters:** Market odds embed the aggregate wisdom of millions of dollars in betting activity. This is the strongest single predictor of race outcomes, typically explaining ~60-70% of the variance in race results. However, it's not perfect — systematic biases (favourite-longshot bias, overreaction to recent form) create value opportunities.

**Data path:** TAB scraper → `runner.current_odds`, `runner.odds_*` fields → `_get_median_odds()` → `_market_consensus()`

### 3.2 Market Movement (Default 7%)

**Source:** Opening odds vs current odds, plus odds fluctuation history

**How it works:**
1. Checks `odds_flucs` JSON for full fluctuation history (preferred)
2. Falls back to `opening_odds` vs `current_odds` comparison
3. Heavy firming (>20% price drop) → strong positive signal (+0.10)
4. Big drift (>30% price increase) → negative signal (-0.08)

**Why it matters:** "Smart money" from professional punters and syndicates often moves odds before the public catches on. A horse firming from $8 to $5 is being backed by people who know something.

**Data path:** TAB scraper → `runner.odds_flucs`, `runner.opening_odds`, `runner.current_odds` → `_market_movement_factor()`

### 3.3 Form Rating (Default 15%)

**Source:** Race history, track/distance/condition stats, career record, first-up/second-up stats

**Sub-signals (7 total):**

| Signal | Weight | Source | What It Measures |
|--------|--------|--------|-----------------|
| Last five results | 1.0 | `runner.last_five` | Recent finishing positions (recency-weighted) |
| Track+distance stats | 1.5 | `runner.track_dist_stats` | Win rate at this track AND distance (strongest form signal) |
| Condition stats | 1.0 | `runner.good/soft/heavy_track_stats` | Win rate on today's track condition |
| Distance stats | 0.8 | `runner.distance_stats` | Win rate at this distance (broader than track+dist) |
| First-up/second-up | 0.5-0.7 | `runner.first_up_stats` / `runner.second_up_stats` | Performance returning from a spell |
| Career win/place % | 0.6 | `runner.career_record` | Overall career winning and placing rate |
| Average condition score | 0.4 | All condition stat fields aggregated | Overall adaptability across track conditions |

**Career stats enrichment (new):** Q5-Q1 analysis of 267K runners showed career win percentage has a 26.6% predictive spread and career place percentage has 18.8%. These are blended 70/30 (win/place) and added as a weighted signal. Requires minimum 10 career starts.

**Data path:** Racing.com scraper OR PF API → `runner.*_stats` fields → `_form_rating()` → form factor score

### 3.4 Class & Fitness (Default 5%)

**Source:** Class-level stats, handicap rating, prize money, form margins, days since last run

**Sub-signals (5 total):**

| Signal | Source | What It Measures |
|--------|--------|-----------------|
| Class stats | `runner.class_stats` | Win rate at this class level |
| Handicap rating | `runner.handicap_rating` | Handicapper's quality assessment (proxy for class) |
| Prize money per start | `runner.career_prize_money` / career starts | Quality of opposition faced (benchmarked by race class) |
| Average margin | `runner.form_history` JSON | How close the horse finishes to the winner (last 5 starts) |
| Days since last run | `runner.days_since_last_run` | Fitness curve: 14-28 days optimal, >90 days concerning |

**Prize money benchmark logic:** The engine uses class-appropriate benchmarks — $50K/start for Stakes, BM rating × $500 for BM races, $15K for Maidens/Restricted. A horse earning $40K/start in a BM64 (benchmark $32K) gets a positive adjustment.

**Data path:** PF API + racing.com → runner fields → `_class_factor()` → class/fitness score

### 3.5 Pace Factor (Default 11%)

**Source:** Speed map positions, PF map factor, jockey factor, pace scenario

**How it works:**
1. Determines **pace scenario** from speed map: hot, genuine, slow, or moderate
2. Cross-references each runner's predicted position with the scenario
3. Leaders benefit in slow/genuine pace, closers benefit in hot pace
4. PF **map factor** (>1.0 = advantage) is the primary signal
5. PF **jockey factor** provides a small supporting signal

**Pace scenarios:**
- **Hot pace** (3+ leaders): Speed horses will tire, closers get their chance
- **Genuine pace** (1 leader, some on-pace): Fair race, slight leader advantage
- **Slow pace** (no leaders): On-pace runners dominate, backmarkers struggle
- **Moderate** (default): No significant bias

**Data path:** Speed map scraper (Playwright) → `runner.speed_map_position` + PF API → `runner.pf_map_factor`, `runner.pf_speed_rank` → `_pace_factor()`

### 3.6 Barrier Draw (Default 9%)

**Source:** Barrier number, field size, race distance

**How it works:**
1. Calculates **relative position**: `(barrier - 1) / (field_size - 1)` → 0.0 (rail) to 1.0 (widest)
2. Inside (< 0.15): +0.06 boost
3. Outside (> 0.85): -0.08 penalty
4. **Distance multiplier**: Sprints (≤1200m) × 1.3, Middle × 1.0, Staying × 0.7
5. Barriers matter most in sprints (less time to cross and recover)

**Data path:** Racing.com scraper → `runner.barrier` + `race.distance` → `_barrier_draw_factor()`

### 3.7 Jockey & Trainer (Default 11%)

**Source:** PF A2E stats (career, last 100, combo), or racing.com string stats

**How it works:**
1. **Jockey (60% weight):** Career strike rate, A2E ratio, Profit on Turnover, last-100 trend
2. **Trainer (40% weight):** Same metrics but lower weight (less direct race impact)
3. **Combo career bonus (20%):** Jockey-trainer combination strike rate (synergy signal)
4. **Combo last 100 (25%):** Recent J/T combo form — captures current partnerships (Q5-Q1: 13.1%)

**A2E (Actual to Expected):** A PF metric where >1.0 means the jockey/trainer earns more prize money than expected given the quality of horses they ride/train. High A2E = consistently extracts more from their horses.

**Data path:** PF API → `runner.jockey_stats`, `runner.trainer_stats` (JSON with career, last100, combo blocks) → `_jockey_trainer_factor()`

### 3.8 Weight Carried (Default 5%)

**Source:** Race weight, race average weight, race distance, race class

**How it works:**
1. Calculates **weight differential** from race average
2. **Class proxy effect** (dominant): Heavier weight = handicapper rates horse higher = positive signal
3. **Physical burden** (secondary): Only penalizes at extremes (>3kg above average)
4. **Distance scaling**: Weight matters more in sprints
5. **Class amplification (new)**: Effect is 1.5× stronger in low-class races (Maidens/Class 1/Restricted) — backtester showed 9.2% Q5-Q1 spread in low-class vs 4.1% in open

**Data path:** Racing.com scraper → `runner.weight` → `_weight_factor()`

### 3.9 Horse Profile (Default 5%)

**Source:** Horse age, sex, race class context

**How it works:**
1. **Age curve**: Peak 4-5yo (+0.05), 3/6yo (+0.02), 2yo (-0.03), 8+yo (-0.04)
2. **Gelding consistency**: Small +0.01 for geldings
3. **Colt context bonus (new)**: Colts get +0.08 in Maidens/Class 1/Restricted (22% SR vs 8.6% for geldings from backtester data), but -0.02 in open company

**Data path:** Racing.com scraper → `runner.horse_age`, `runner.horse_sex` → `_horse_profile_factor()`

### 3.10 Deep Learning Patterns (Default 10%)

**Source:** 844+ statistical patterns from 280K+ historical runners (PatternInsight table)

**Pattern types (8):**
1. **Pace patterns:** Running style × venue × distance × condition
2. **Barrier bias:** Barrier position × venue × distance
3. **Jockey-trainer combos:** Specific J/T partnerships × state
4. **Track-distance-condition:** Venue × distance × condition combinations
5. **Acceleration:** Market movement type × venue × distance × condition
6. **Pace collapse:** Leader performance when pace is hot
7. **Condition specialist:** Performance on specific track conditions
8. **Market patterns:** Odds range × state

**How it works:**
1. Each pattern has an **edge** (expected advantage) and **confidence** (HIGH/MEDIUM)
2. For each runner, all matching patterns are found
3. Individual contributions are capped at ±0.15
4. Total adjustment is capped at ±0.25
5. HIGH confidence patterns count at 100%, MEDIUM at 60%

**Data path:** `punty/deep_learning/` analysis scripts → `PatternInsight` DB table → `load_dl_patterns_for_probability()` → module cache → `_deep_learning_factor()`

---

## 4. Data Flow: Source to Signal

### 4.1 Complete Data Pipeline

```
EXTERNAL DATA SOURCES
│
├─ Racing.com (httpx)
│  ├─ Meetings, races, runners, fields
│  ├─ Form history, last_five, stats strings
│  ├─ Barrier, weight, age, sex, jockey, trainer
│  └─ Speed map positions (Playwright browser)
│
├─ TAB (httpx)
│  ├─ Current odds from multiple bookmakers
│  ├─ Opening odds, odds fluctuations
│  ├─ Results, dividends, finish positions
│  └─ Exotic dividends (trifecta, exacta, etc.)
│
├─ Punting Form API (httpx)
│  ├─ Jockey/trainer A2E stats (career, last100, combo)
│  ├─ Speed maps (map_factor, speed_rank, jockey_factor)
│  ├─ Meeting ratings
│  └─ Form data (career record, condition stats, prize money)
│
├─ WillyWeather API (httpx)
│  ├─ Wind speed/direction + straight bearings
│  ├─ Rain radar
│  └─ Sky camera imagery
│
└─ Deep Learning Historical Analysis
   └─ 844+ patterns from 280K runners
       (stored in PatternInsight table)
```

### 4.2 Scraping → Database Storage

```
orchestrator.py
├─ racing_scraper.py  → Meeting, Race, Runner models (SQLAlchemy)
├─ tab_scraper.py     → Runner.current_odds, odds_*, results
├─ speed_map_scraper  → Runner.speed_map_position
└─ punting_form.py    → Runner.jockey_stats (JSON), pf_map_factor, etc.
```

All runner data lands in the `runners` table with ~50 columns covering everything from basic form to detailed stats JSON.

### 4.3 Probability Calculation

```
context/builder.py → builds AI context
  │
  ├─ Loads probability_weights from AppSettings
  ├─ Loads DL patterns from PatternInsight table
  └─ For each race:
       probability.py::calculate_race_probabilities()
       │
       ├─ Step 1: Calculate 10 raw factor scores per runner
       │   ├─ _market_consensus()       ← multi-bookie median odds
       │   ├─ _market_movement_factor() ← odds shift detection
       │   ├─ _form_rating()            ← 7 sub-signals
       │   ├─ _class_factor()           ← 5 sub-signals
       │   ├─ _pace_factor()            ← speed map + pace scenario
       │   ├─ _barrier_draw_factor()    ← barrier × distance
       │   ├─ _jockey_trainer_factor()  ← A2E + combo stats
       │   ├─ _weight_factor()          ← weight diff × class
       │   ├─ _horse_profile_factor()   ← age + sex + context
       │   └─ _deep_learning_factor()   ← pattern matching
       │
       ├─ Step 1a: Apply context multipliers
       │   └─ _get_context_multipliers() → amplify/dampen by context
       │
       ├─ Step 1b: Dynamic market weight boost
       │   └─ _boost_market_weight() → increase market weight when
       │       other factors lack information (maiden races, etc.)
       │
       ├─ Step 1c: Weighted sum → raw probability score
       │
       ├─ Step 2: Normalize to sum to 1.0
       │
       └─ Step 2b: Market floor
           └─ _apply_market_floor() → prevent >65% disagreement
               with strong market (blend 60/40 model/market)
```

### 4.4 Probability → Picks → Settlement → Learning

```
PICKS STORAGE (store_picks_from_content)
│
├─ Parse AI-generated early mail → Pick rows
├─ Attach win_probability, value_rating, factors_json
│   from calculate_race_probabilities()
└─ Store in picks table with all metadata

SETTLEMENT (settle_picks_for_race)
│
├─ TAB results come in → finish_position, dividends
├─ Calculate P&L: won_amount - stake
├─ Mark hit=True/False, settled=True
└─ Store pnl, settled_at

LEARNING (post-race job)
│
├─ probability_tuning.py::maybe_tune_weights()
│   ├─ Load last 60 days of settled picks with factors_json
│   ├─ Calculate point-biserial correlation per factor
│   ├─ Derive optimal weights via softmax
│   ├─ Smooth: 70% old + 30% optimal
│   ├─ Apply bounds (2% min, 35% max), normalize to 1.0
│   └─ Save to AppSettings + TuningLog if change > 0.5%
│
├─ Race assessment → strategy insights
│   └─ memory/strategy.py → PatternInsight for future prompts
│
└─ Dashboard data updated automatically
```

---

## 5. Context-Aware Multipliers

### 5.1 What Are They?

Not all factors are equally predictive in all contexts. Barrier draw matters enormously in metro sprints but barely at all in country staying races. The context system adjusts each factor's influence based on the specific race being analyzed.

### 5.2 How They're Built

A builder script (`scripts/build_context_profiles.py`) processes 190K+ historical runners from PF form data:

1. **Classify each runner** into 3 context dimensions:
   - **Venue type:** metro_vic, metro_nsw, metro_qld, metro_other, provincial, country
   - **Distance:** sprint (≤1100), short (1200-1399), middle (1400-1799), classic (1800-2199), staying (2200+)
   - **Class:** maiden, class1, restricted, class2, class3, bm58, bm64, bm72, open

2. **For each context combination** (e.g., "metro_vic|sprint|open"):
   - Sort runners into quintiles (Q1-Q5) by each signal
   - Calculate Q5-Q1 win rate spread within this context
   - Compare to the overall Q5-Q1 spread across ALL runners
   - **Multiplier = context_spread / overall_spread** (capped 0.3 to 2.5)

3. **Output:** 168 full profiles + 73 fallback profiles in `punty/data/context_profiles.json`

### 5.3 How They're Applied at Runtime

In `calculate_race_probabilities()`, after computing raw factor scores:

```
adjusted_score = 0.5 + (raw_score - 0.5) × multiplier
```

- A score of **0.7** with multiplier **1.5** → `0.5 + (0.2 × 1.5)` = **0.80** (amplified)
- A score of **0.7** with multiplier **0.5** → `0.5 + (0.2 × 0.5)` = **0.60** (dampened)
- A score of **0.5** (neutral) stays **0.5** regardless of multiplier
- Results clamped to [0.05, 0.95]

### 5.4 Hierarchical Fallback

When the exact context combo isn't in the profiles:
1. Try exact: `venue_type|distance|class` (168 profiles)
2. Try `venue_type|distance` (fallbacks)
3. Try `distance|class` (fallbacks)
4. Default: no adjustment (multiplier = 1.0)

### 5.5 Context Profiles Not Used For

- **Deep learning factor** — already context-aware by design (patterns are venue/distance/condition specific)
- **Condition dimension** — Not included because historical PF form data doesn't contain TrackCondition field

---

## 6. Self-Tuning System

### 6.1 When Does It Tune?

- **Trigger:** `meeting_post_race_job()` calls `maybe_tune_weights()` after each meeting completes
- **Cooldown:** Minimum 24 hours between tuning runs
- **Minimum data:** Requires 50+ settled picks with factor data
- **Threshold:** Only saves if the biggest weight change exceeds 0.5%

### 6.2 How Does It Tune?

1. **Analyze factor performance:** For each factor, calculate the point-biserial correlation between factor scores and win/loss outcomes
2. **Derive optimal weights:** Use absolute correlation as a quality signal → softmax normalization → produces "ideal" weights
3. **Smooth:** Blend 70% current weights + 30% optimal weights (conservative to prevent overfitting)
4. **Bound:** Enforce minimum 2% and maximum 35% per factor
5. **Normalize:** Scale all weights to sum to 1.0
6. **Log:** Record old weights, new weights, and all metrics in `probability_tuning_log`

### 6.3 Why 70/30 Smoothing?

The 70/30 blend prevents the model from overreacting to short-term noise. A factor that happens to correlate with winners this week might not next week. By only applying 30% of the optimal change, the model makes gradual, stable improvements.

---

## 7. How It Improves Over Time

The system has **three improvement loops**, each operating on different timescales:

### 7.1 Loop 1: Weight Tuning (Daily)

Every meeting completion triggers a potential weight re-tune. As more picks are settled:
- Factors that consistently predict winners get higher weights
- Factors that don't contribute get downweighted
- The model converges toward the optimal weight distribution for current racing conditions

### 7.2 Loop 2: Context Profiles (Monthly rebuild recommended)

Running `build_context_profiles.py` with fresh historical data updates the multiplier profiles. As more data is ingested:
- Context multipliers become more accurate (larger samples → more reliable Q5-Q1 spreads)
- New contexts that previously had insufficient data become usable
- The model adapts to changing racing conditions (e.g., new track configurations, COVID-era field changes)

### 7.3 Loop 3: Signal Enrichment (Manual)

The backtester identified 6 new sub-signals now integrated into the engine. Future analysis of settled data can identify more predictive signals to add. The architecture supports adding new sub-signals to existing factors without changing weights.

### 7.4 The Feedback Data Path

```
Race runs → Results come in → Picks settled with P&L
                                      ↓
                              factors_json stored on each pick
                                      ↓
                              probability_tuning.py analyzes:
                              - Which factors predicted winners?
                              - How well-calibrated are probabilities?
                              - Which value buckets are profitable?
                                      ↓
                              Weights adjusted → Better predictions
                                      ↓
                              Next race uses improved weights
```

---

## 8. Key Indicators to Watch

### 8.1 Health Indicators

| Indicator | Healthy | Concerning | Action |
|-----------|---------|------------|--------|
| Brier Score (Model) | < 0.220 | > 0.250 | Check factor performance, may need more data |
| Brier Score (Model vs Market) | Model < Market | Model > Market | Factor weights hurting, consider resetting to defaults |
| Picks with factors | Growing | Stagnant | Check if `store_picks_from_content` is attaching probabilities |
| Value 1.30+ ROI | Positive | Consistently negative | Value detection miscalibrated |
| Calibration 0-5% bucket | Actual ~3-5% | Actual 15%+ | Low-probability picks winning too often = model is underweighting longshots |
| Calibration 30-50% bucket | Actual ~35-45% | Actual < 20% | High-probability picks losing too much = overconfident on favourites |

### 8.2 What "Good" Looks Like

- **Calibration:** Blue line close to diagonal across all buckets
- **Factor performance:** Market consensus has highest edge, followed by form/JT. All factors have positive edge
- **Value performance:** Monotonically increasing ROI as value rating increases (Unders = negative, Strong Value = positive)
- **Weight history:** Smooth convergence, not wild oscillation
- **Context multipliers:** Sensible patterns (barrier amplified in sprints, JT amplified in metro)

### 8.3 Red Flags

- Market consensus with negative edge → odds data might be stale or wrong
- All factors at edge ~0 → not enough data, or factors are truly unpredictive
- Weight history showing one factor dominating (>30%) → other factors may have data issues
- Calibration wildly off-diagonal → systematic bias in the probability model
- Brier score increasing over time → model getting worse (possible overfitting from tuning)

---

## 9. Known Limitations & Improvement Opportunities

### 9.1 Current Limitations

1. **No track condition in context profiles:** Historical PF data doesn't include TrackCondition, so context multipliers don't vary by going. Track condition only affects the Form factor's condition-specific stat lookup, not the multiplier system.

2. **Context profile sample sizes:** Some profiles are built from as few as 50 runners. Metro/provincial profiles are robust (500-5000 runners), but edge cases (e.g., metro_other|classic|restricted) can be noisy.

3. **Backfill gap:** Picks created before the probability engine was integrated don't have `factors_json`, so they can't contribute to factor performance analysis. Only picks from the integration date onward feed the tuning loop.

4. **Single Brier score:** The dashboard shows overall Brier score but doesn't break it down by factor or time period, making it hard to see which factors are improving/degrading.

5. **No live recalculation:** Probabilities are calculated once at pick creation time. Late scratchings, dramatic odds shifts, or weather changes after that point aren't reflected.

6. **Static context profiles:** The profiles are pre-computed from historical data and loaded from a JSON file. They don't update automatically as new race data comes in.

### 9.2 Improvement Opportunities

1. **Time-series Brier tracking:** Chart Brier score over time (weekly rolling window) to see if the model is trending better or worse. Would immediately reveal if a tuning cycle made things worse.

2. **Odds-at-post vs odds-at-tip:** Currently uses `current_odds` at tip time. Capturing final SP odds would allow better value detection analysis — how much edge erodes between tip time and jump.

3. **Weather-adjusted probabilities:** WillyWeather data is available but not used in the probability engine. Wind direction × track orientation could be a predictive signal, especially for staying races.

4. **Ensemble approach:** Instead of a single weighted sum, use separate models for different contexts (sprints vs stayers, metro vs country) and ensemble their outputs. The context multiplier system is a step toward this.

5. **Recalibration at post time:** Re-run probability calculations just before race jump with final odds, confirmed scratchings, and latest weather. Would require scheduler integration.

6. **Track condition integration for context profiles:** When the PF API's Conditions endpoint provides per-race condition data, rebuild context profiles with a 4th dimension.

7. **Rolling tuning window:** Currently uses fixed 60-day lookback. A rolling window that weights recent results more heavily would adapt faster to changing conditions while retaining historical stability.

8. **Factor interaction terms:** Currently factors are independent (weighted sum). Some factors interact — e.g., barrier draw × pace scenario (wide barrier + hot pace = less penalty). Adding interaction terms could capture non-linear relationships.

### 9.3 Data Quality Watchpoints

- **PF API rate limits:** If the PF API is unavailable during a scrape, runners won't have A2E stats or map factors. This degrades JT and Pace factor accuracy, causing the market boost to kick in.
- **TAB odds freshness:** Odds are scraped during the pre-race job (~2 hours before race 1). By race time, odds may have shifted significantly. Late movers won't be captured.
- **Racing.com stats format changes:** The stats parser handles 4 formats, but any change to racing.com's output could silently produce `None` stats, reducing form factor accuracy.
