# PuntyAI — OpenAI Prompt Documentation

**Complete documentation of every prompt template, injected data, and AI task.**

---

## Prompt Templates

### 1. `prompts/early_mail.md` (388 lines) — Main Product

**Purpose:** Generate pre-race tips with Big 3, race-by-race picks, exotics, and sequences.

**Model:** GPT-5.2 (Reasoning: high, max_output: 32K, timeout: 600s)

**System Prompt:** `personality.md` + `analysis_weights` (from AppSettings)

**Data Injected:**
- Full meeting context (40-80K tokens): track, rail, weather, per-race runner tables, probability data, exotic combinations, sequence leg confidence
- RAG strategy context (5-10K tokens): bet type scorecard, directives, deep learning patterns
- RAG assessment context (2-5K tokens): similar past races with key learnings
- Runner-level memories (0-2K tokens): past situations for top runners

**Output Structure:**
1. Header: `*PUNTY EARLY MAIL — {VENUE}*` with group greeting
2. Meet Snapshot: track/rail/weather/jockeys/trainers + Punty's take
3. Punty's Big 3 + Multi: 3 horses across races + treble bet
4. Race-by-Race: Top 3 + Roughie ($20 pool), Degenerate Exotic, Punty's Pick
5. Sequence Lanes: Early Quaddie, Quaddie, Big 6 (Skinny/Balanced/Wide variants)
6. Nuggets From The Track: 3 smart insights
7. Find Out More: punty.ai link
8. Final Word: rotating closers

**Key Rules:**
- Probability must match bet type (Win prob for Win bets, Place prob for Place)
- At least 1 Win/Each Way bet per race (can't go all-Place)
- Exotic runners from Top 3 + Roughie (only override if 1.5x vs 1.2x value)
- Sequence combos validated: `combos × unit = total_outlay`
- Uses pre-calculated exotic combinations (Harville model)
- References actual betting track record in commentary
- Punty's Pick = best bet per race (publicly tracked)

---

### 2. `prompts/personality.md` (163 lines) — Voice & Style

**Purpose:** Define Punty's personality across all content types.

**Data Injected:** None (static prompt, loaded from DB with file fallback)

**Key Rules:**
- Punty the Loose Unit: expert analyst, degenerate larrikin, sicko mathematician
- Australian English, swearing allowed (never "cunt")
- No emojis, no citations, single `*bold*` for headings
- Numbers: Race 4, No.7, $X.XX, 10U
- Never confuse barrier with saddlecloth
- Reference specific stats ("Place at +11% ROI")
- 86-entry racing slang glossary

---

### 3. `prompts/wrap_up.md` (175 lines) — Post-Race Review

**Purpose:** Generate end-of-meeting punt review.

**Model:** GPT-5.2 (Reasoning: high)

**Data Injected:**
- Meeting context with results (positions, margins, dividends)
- Pick ledger with actual P&L per pick
- Sectional times (if available)
- RAG strategy context

**Output Structure (11 sections):**
1. The Wrap — one-line opener
2. How It Unfolded — 2 paragraphs on track/tempo
3. Stats Board — 4 sharp one-liners (≤22 words each)
4. Winners Board — only winning selections
5. Exotics That Landed — only exotics that returned
6. Sequences That Hit — only sequences that returned
7. Big 3 Multi Result — clear hit/miss
8. Punty's Picks — review each Pick from Early Mail
9. Track Read — pace/speed map accuracy
10. Quick Hits — one line per race (ALL races)
11. Closing — 2-3 sentences (tone varies by result)

---

### 4. `prompts/weekly_blog.md` (99 lines) — Friday Recap

**Purpose:** Generate weekly "From the Horse's Mouth" column.

**Model:** GPT-5.2 (Reasoning: high)

**Data Injected:**
- Weekly performance data (P&L, awards, patterns)
- Blog context: awards (jockey/roughie/value/track/wooden spoon), future races, deep patterns, ledger, news headlines

**Output Structure (7 sections):**
1. Header with funny headline
2. Punty Awards (5 awards)
3. The Crystal Ball (2-3 upcoming Group races)
4. Pattern Spotlight (most interesting deep pattern)
5. The Ledger (honest weekly P&L)
6. Around The Traps (2-3 news takes)
7. Final Word (closing rant)

---

### 5. `prompts/initialise.md` (76 lines) — Pre-Generation Setup

**Purpose:** Create meeting setup before generation.

**Model:** GPT-5.2 (Reasoning: high)

**Data Injected:** Full meeting context

**Output:** Meeting details, race weighting plan, focus races, lean list, sequences posture, closer.

---

### 6. `prompts/results.md` (72 lines) — Post-Race Commentary

**Purpose:** Generate commentary after individual race results.

**Model:** GPT-5.2 (Reasoning: high)

**Data Injected:** Race results with runner details, pick outcomes, sectionals

**Output:** Quick summary, how it unfolded, winner analysis, hard luck stories, dividends, sectionals, Punty's take, running P&L, tee up next race.

---

### 7. `prompts/speed_map_update.md` (40 lines) — Pace Change Alert

**Purpose:** Alert when speed maps change race outlook.

**Model:** GPT-5.2 (Reasoning: high)

**Data Injected:** Previous vs new speed map data, affected races

**Output:** Alert header, what changed, impact assessment, new angles, revised recommendation.

---

### 8. `prompts/race_preview.md` (30 lines) — Single Race Preview

**Purpose:** Focused single race analysis (200-300 words).

**Model:** GPT-5.2 (Reasoning: high)

**Data Injected:** Single race context

**Output:** Race header, the story, pace scenario, key runners (3-4), Punty's pick, each way danger.

---

## Context Injection Detail

### What Each Runner Gets in Context

```markdown
| No. | Horse | Bar | Jockey | Trainer | Odds | Form | Speed Map | Market | Rank | MapF | Win% | Pl% | Val |
```

Plus expanded prose:
- Days since last run, career record
- **GEAR CHANGES** (flagged as important)
- First-up / second-up stats
- Track/distance/condition stats
- Jockey stats (career + L100, A2E, HOT/COLD tags)
- Trainer stats (career + L100, A2E)
- Jockey/trainer combo stats
- Strike rates with actual-to-expected ratios
- Class stats
- Sectionals (L400, L800)
- Pedigree (sire x dam x dam_sire)
- Comments (top 6 per race, 200 char max)
- Stewards comments (150 char max)
- **LAST-START EXCUSES** (legitimate reasons for poor runs)
- **SPEED RATINGS** (FAST/STANDARD/SLOW vs venue standards)
- **WEIGHT FORM ANALYSIS** (warnings when carrying more than optimal)
- Extended form (last 5 starts with venue/dist/pos/margin/track/settled/sectionals)
- Probability rankings (top 6 by win prob)
- VALUE PLAYS (value_rating > 1.05)
- Pre-calculated exotic combinations (Harville, value >= 1.2x)

### RAG Context Components

**Strategy Track Record:**
```markdown
## YOUR BETTING TRACK RECORD
Grand Total: X bets | Y winners | Z% SR | $A staked | $B P&L | C% ROI

### Bet Type Scorecard
[Per-type performance tables]

### PUNTY'S PICK Performance
[All-time + 30-day breakdown — WARNING if SR <30% or ROI <0]

### STRATEGY DIRECTIVES
- LEAN INTO Place bets — your best at +11% ROI
- REDUCE Win bets — losing at -12% ROI
- DROP BIG 6 — consistently unprofitable

### HISTORICAL PATTERNS (Deep Learning)
[Top 15 HIGH/MED confidence patterns]

### RECENT RESULTS
[Last 15 settled bets with outcomes]
```

**Assessment Context:**
```markdown
## Past Learnings from Similar Races
**Flemington 1400m BM78 Good 3 (Rail: +3m)** ✅ +$14.50
  → Leaders dominated on Good 3 with rail out, barrier 1-4 had 60% of winners
```

---

## Token Budget Summary

| Content Type | Input | Output | Total | Cost Est. |
|-------------|-------|--------|-------|-----------|
| Early Mail | 56-105K | 10-20K | 66-125K | $0.27-$0.46 |
| Meeting Wrapup | 30-60K | 8-15K | 38-75K | $0.15-$0.30 |
| Results Commentary | 15-30K | 3-6K | 18-36K | $0.07-$0.14 |
| Race Preview | 10-20K | 1-2K | 11-22K | $0.04-$0.09 |
| Weekly Blog | 20-40K | 8-15K | 28-55K | $0.11-$0.22 |
| Post-Race Assessment | 15-25K | 2-3K | 17-28K | $0.07-$0.11 |
| Content Fix | 3-8K | 1-2K | 4-10K | $0.02-$0.04 |
| Initialise | 30-60K | 5-10K | 35-70K | $0.14-$0.28 |
