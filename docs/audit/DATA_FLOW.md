# PuntyAI — End-to-End Data Flow Diagram

## 1. Data Ingestion Pipeline

```
                          ┌──────────────────┐
                          │   EXTERNAL APIS   │
                          └────────┬─────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │                         │                         │
    PuntingForm API          Racing.com              WillyWeather
    (Primary source)         (Supplement)            (Weather)
         │                         │                         │
    ┌────┴────┐             ┌──────┴──────┐           ┌──────┴──────┐
    │Meetings │             │  GraphQL    │           │ Forecasts   │
    │Fields   │             │  Intercept  │           │ Observations│
    │Form(10) │             │  via        │           │ Wind/Radar  │
    │SpeedMaps│             │  Playwright │           │ Camera      │
    │Ratings  │             └──────┬──────┘           └──────┬──────┘
    │Conditions│                   │                         │
    │Scratchngs│                   │                         │
    └────┬────┘                    │                         │
         │                         │                         │
         └─────────────┬───────────┘                         │
                       │                                     │
              ┌────────▼─────────┐                          │
              │   ORCHESTRATOR    │                          │
              │  (merge + upsert) │◄─────────────────────────┘
              │                   │
              │ • _upsert_meeting │
              │ • _merge_supplement│
              │ • _is_more_specific│
              │ • upsert_results  │
              └────────┬──────────┘
                       │
              ┌────────▼──────────┐
              │    SQLite DB       │
              │  meetings          │
              │  races             │
              │  runners (70+flds) │
              └────────┬──────────┘
                       │
```

## 2. Context Assembly Pipeline

```
              ┌────────────────────┐
              │    SQLite DB        │
              └────────┬───────────┘
                       │
              ┌────────▼───────────┐
              │  CONTEXT BUILDER    │
              │  (builder.py)       │
              │                     │
              │  For each race:     │
              │  ├─ Runner data     │
              │  ├─ Market movement │
              │  ├─ Pace scenario   │
              │  ├─ Stewards excuses│
              │  ├─ Time ratings    │
              │  └─ Weight analysis │
              └────────┬───────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
    ┌────▼────┐  ┌─────▼─────┐ ┌────▼─────┐
    │PROBABIL-│  │  WEATHER   │ │  CONTEXT  │
    │ITY      │  │  (WillyW)  │ │ SNAPSHOT  │
    │ENGINE   │  │            │ │(versioning│
    │         │  │ Forecasts  │ │  .py)     │
    │10-factor│  │ Wind impact│ │           │
    │Harville │  │ Rain prob  │ │ SHA256    │
    │Kelly    │  │ Observations│ │ hash for  │
    │DL ptrns │  └─────┬─────┘ │ change    │
    └────┬────┘        │       │ detection │
         │             │       └────┬──────┘
         └─────────────┼────────────┘
                       │
              ┌────────▼───────────┐
              │ FORMATTED CONTEXT   │
              │ (40-80K tokens)     │
              │                     │
              │ Meeting header      │
              │ Per-race tables     │
              │ Runner details      │
              │ Probability data    │
              │ Exotic combos       │
              │ Sequence confidence │
              │ Aggregations        │
              └────────┬───────────┘
                       │
```

## 3. RAG Pipeline

```
              ┌────────────────────┐
              │    SQLite DB        │
              └────────┬───────────┘
                       │
    ┌──────────────────┼──────────────────┐
    │                  │                  │
┌───▼────┐      ┌─────▼─────┐     ┌─────▼─────┐
│STRATEGY │      │ASSESSMENTS│     │DEEP       │
│CONTEXT  │      │           │     │LEARNING   │
│         │      │ Hybrid    │     │PATTERNS   │
│Per-type │      │ retrieval:│     │           │
│P&L data │      │ SQL filter│     │844+       │
│Directives│     │ + embed   │     │patterns   │
│Recent   │      │ similarity│     │HIGH/MED   │
│results  │      │ + attr    │     │confidence │
│         │      │ bonuses   │     │           │
└───┬────┘      └─────┬─────┘     └─────┬─────┘
    │                  │                  │
    └──────────────────┼──────────────────┘
                       │
              ┌────────▼───────────┐
              │  RAG CONTEXT        │
              │  (10-25K tokens)    │
              │                     │
              │  Strategy scorecard │
              │  Strategy directives│
              │  Similar races      │
              │  Deep patterns      │
              │  Runner memories    │
              └────────┬───────────┘
                       │
```

## 4. AI Generation Pipeline

```
    Formatted Context         RAG Context          Prompts
    (40-80K tokens)           (10-25K tokens)      (8K tokens)
         │                         │                    │
         └─────────────────────────┼────────────────────┘
                                   │
                          ┌────────▼──────────┐
                          │  SYSTEM PROMPT     │
                          │  personality.md    │
                          │  + analysis_weights│
                          └────────┬──────────┘
                                   │
                          ┌────────▼──────────┐
                          │  USER PROMPT       │
                          │  early_mail.md     │
                          │  + formatted context│
                          │  + RAG context     │
                          └────────┬──────────┘
                                   │
                          ┌────────▼──────────┐
                          │  OpenAI GPT-5.2    │
                          │  Reasoning: high   │
                          │  Timeout: 600s     │
                          │  Max output: 32K   │
                          └────────┬──────────┘
                                   │
                          ┌────────▼──────────┐
                          │  RAW CONTENT       │
                          │  (10-20K tokens)   │
                          │                    │
                          │  Big 3 + Multi     │
                          │  Race-by-race tips │
                          │  Exotics           │
                          │  Sequences         │
                          │  Nuggets           │
                          └────────┬──────────┘
                                   │
```

## 5. Content Processing Pipeline

```
              ┌────────────────────┐
              │  RAW AI CONTENT     │
              └────────┬───────────┘
                       │
              ┌────────▼───────────┐
              │  PARSER (regex)     │
              │                     │
              │  _BIG3_HORSE        │
              │  _SELECTION (1-3)   │
              │  _ROUGHIE (rank 4)  │
              │  _BET_LINE          │
              │  _EXOTIC            │
              │  _SEQ_VARIANT       │
              │  _SEQ_COSTING       │
              │  _PUNTYS_PICK       │
              │  _PROBABILITY       │
              │  _VALUE_RATING      │
              └────────┬───────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
    ┌────▼────┐  ┌─────▼─────┐ ┌────▼─────┐
    │CONTENT  │  │   PICKS    │ │FORMATTERS│
    │         │  │            │ │          │
    │status:  │  │selections  │ │Twitter   │
    │draft →  │  │exotics     │ │Facebook  │
    │pending →│  │sequences   │ │HTML      │
    │approved→│  │big3/multi  │ │Email     │
    │sent     │  │            │ │          │
    └────┬────┘  └─────┬─────┘ └────┬─────┘
         │             │             │
```

## 6. Delivery Pipeline

```
              ┌────────────────────┐
              │  APPROVED CONTENT   │
              │  + FORMATTED TEXT   │
              └────────┬───────────┘
                       │
              ┌────────▼───────────┐
              │  AUTO-APPROVE       │
              │  VALIDATION         │
              │                     │
              │  Early Mail:        │
              │  • Min 2000 chars   │
              │  • Big 3 section    │
              │  • Sequence section │
              │  • All races covered│
              │  • 10+ odds, 5+ SCs│
              │                     │
              │  Wrapup:            │
              │  • Min 1000 chars   │
              │  • Venue mentioned  │
              │  • Ledger + Hits    │
              └────────┬───────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
    ┌────▼────┐  ┌─────▼─────┐ ┌────▼─────┐
    │TWITTER  │  │ FACEBOOK  │ │  EMAIL   │
    │         │  │           │ │          │
    │Long-form│  │Page post  │ │HTML body │
    │(25K max)│  │(63K max)  │ │+ plain   │
    │Hashtags │  │Plain text │ │Resend /  │
    │Profanity│  │Profanity  │ │SMTP      │
    │filter   │  │rotation   │ │          │
    │         │  │           │ │          │
    │Tweepy   │  │Graph API  │ │httpx     │
    │OAuth1.0a│  │v21.0      │ │          │
    └─────────┘  └───────────┘ └──────────┘
```

## 7. Settlement Pipeline

```
              ┌────────────────────┐
              │  RESULTS MONITOR    │
              │  (polling loop)     │
              │  every 30-90s       │
              └────────┬───────────┘
                       │
              ┌────────▼───────────┐
              │  CHECK RACE STATUS  │
              │  Racing.com GraphQL │
              │                     │
              │  Open → Interim →   │
              │  Paying → Closed    │
              └────────┬───────────┘
                       │
              ┌────────▼───────────┐
              │  SCRAPE RESULTS     │
              │                     │
              │  Positions          │
              │  Margins            │
              │  Win/Place dividends│
              │  Exotic dividends   │
              └────────┬───────────┘
                       │
              ┌────────▼───────────┐
              │  SETTLE PICKS       │
              │  (picks.py)         │
              │                     │
              │  Selections:        │
              │  Win: odds×stake-stk│
              │  Place: pl_odds×stk │
              │  Each Way: split    │
              │                     │
              │  Exotics:           │
              │  Flexi: div×(stk/   │
              │    combos)-stk      │
              │                     │
              │  Sequences:         │
              │  All legs must hit  │
              │  Flexi from total   │
              │                     │
              │  Big3 Multi:        │
              │  All 3 races must   │
              │  complete first     │
              └────────┬───────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
    ┌────▼────┐  ┌─────▼─────┐ ┌────▼────────┐
    │CELEBRATE│  │PACE BIAS  │ │PATTERN      │
    │         │  │ANALYSIS   │ │INSIGHTS     │
    │≥5x pout│  │After R3/R4│ │             │
    │→ social │  │Max 3/meet │ │Record for   │
    │post     │  │→ social   │ │self-tuning  │
    │         │  │post       │ │+ strategy   │
    └─────────┘  └───────────┘ └─────────────┘
```

## 8. Self-Tuning Pipeline

```
              ┌────────────────────┐
              │  SETTLED PICKS      │
              │  with factors_json  │
              └────────┬───────────┘
                       │
              ┌────────▼───────────┐
              │  maybe_tune_weights │
              │  (post-race job)    │
              │                     │
              │  Guards:            │
              │  • ≥50 picks        │
              │  • ≥24h since last  │
              │                     │
              │  Algorithm:         │
              │  1. Factor → hit    │
              │     correlation     │
              │  2. Softmax optimal │
              │  3. 70/30 smoothing │
              │  4. Bounds 2%-35%   │
              │  5. Normalize       │
              │  6. Save if >0.5%   │
              │     change          │
              └────────┬───────────┘
                       │
              ┌────────▼───────────┐
              │  TuningLog table    │
              │  + AppSettings      │
              │  → /probability     │
              │    dashboard        │
              └────────────────────┘
```
