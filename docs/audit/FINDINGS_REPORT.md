# PuntyAI Full Solution Audit — Findings Report

**Date:** 2026-02-13
**Auditor:** Claude Code (Opus 4.6)
**Scope:** Complete dry-run audit of codebase, data pipeline, AI integration, and infrastructure

---

## Executive Summary

PuntyAI is a well-engineered async Python application that combines multi-source racing data scraping, a 10-factor probability engine, LLM-powered content generation, and multi-platform delivery into a cohesive automated pipeline. The system serves dual purposes: an admin dashboard (app.punty.ai) and a public tips website (punty.ai) from a single FastAPI instance.

**Overall Grade: B+** — Strong foundation with excellent architecture, comprehensive data coverage, and sound probability maths. Key areas for improvement: cost visibility, settlement edge cases, public API security, and database infrastructure.

| Dimension | Grade | Notes |
|-----------|-------|-------|
| Architecture & Design | A- | Clean separation, async-first, well-structured modules |
| Data Layer & Scrapers | A- | Comprehensive multi-source redundancy, robust error handling |
| Probability Engine | A- | Sound maths, 10-factor model, self-tuning, exotic Harville model |
| AI / LLM Integration | B+ | Exhaustive prompts, strong RAG pipeline, but cost visibility missing |
| Settlement Logic | B+ | Correct maths verified, but dead heats and scratched exotics unhandled |
| App & Website | B | Functional and responsive, but missing rate limiting, indexes, caching |
| Prompt Quality | A | 388-line early mail prompt with performance accountability |
| Testing | A- | 759 unit tests + 103 e2e tests, CI/CD pipeline |
| Security | B- | OAuth + CSRF good, but no rate limiting, no session timeout |
| Deployment & Ops | C+ | Manual SSH deploy, no staging, no backups, no zero-downtime |

---

## Phase 2 Findings

### 2.1 — Data Layer

#### Strengths

- **Multi-Source Redundancy**: PuntingForm API (primary) → Racing.com GraphQL (fallback) → TAB (supplement). If one source fails, others fill the gap.
- **Comprehensive Data Coverage**: 70+ fields per runner from 12 scraper sources covering form, odds (5 bookmakers), speed maps, weather, stewards, sectionals, pedigree, gear, and strike rates.
- **Intelligent Conflict Resolution**: `_is_more_specific()` prevents TAB overwriting "Good 4" with bare "Good". Orchestrator merges data sources intelligently.
- **Speed Map Quality Gate**: Auto-unselects meetings with <30% speed map coverage.
- **Dual-Key Runner Matching**: Primary match by `race_id + saddlecloth`, fallback to `horse_name`. Stable across re-scrapes.

#### Issues Found

| ID | Severity | Issue | Location |
|----|----------|-------|----------|
| D1 | Low | GraphQL JSON parse errors not logged | `racing_com.py:107` |
| D2 | Low | Form history lazy loading may miss horses (best-effort) | `racing_com.py:279-318` |
| D3 | Low | `dam_sire` field rarely populated | Runner model |
| D4 | Medium | No PF sectional benchmarks (requires Modeller+ tier) | PuntingForm API |

#### Data Gaps

- Dam sire data rarely populated by racing.com scraper
- Sectional benchmarks require PF Modeller+ subscription
- No stable confidence indicators or vet reports
- Jockey/trainer strike rates national only (no state-level breakdown)

---

### 2.2 — Probability Engine

#### Strengths

- **10-Factor Model**: Market (22%), Form (15%), Deep Learning (10%), Pace (11%), Barrier (9%), Jockey/Trainer (11%), Movement (7%), Class (5%), Weight (5%), Horse Profile (5%). All configurable from UI.
- **Correct Exotic Maths**: Harville conditional probabilities for exacta/trifecta/first4/quinella verified. Box and standout combos calculated correctly. Flexi P&L formula verified.
- **Dynamic Market Weight Boost**: When >40% of non-market factors are near-neutral, market weight increases up to ~50%. Prevents maiden/low-data races from compressing favourites.
- **Market Floor**: After normalization, if model gives <50% of market-implied probability, blends 50/50 with market. Re-normalizes after.
- **Self-Tuning System**: Point-biserial correlation between factor scores and win outcomes. 70/30 smoothing. Bounds: 2%-35%. 24h cooldown. Minimum 50 picks.
- **Quarter-Kelly Staking**: Conservative 0.25 Kelly fraction, $1 minimum, $20 cap, rounds to 50c.

#### Settlement Verification

All bet maths verified correct:
- Win/Place/Saver Win: `pnl = odds × stake - stake` ✅
- Each Way: Half win + half place, correctly split ✅
- Trifecta/Exacta/Quinella Box: Flexi = `dividend × (stake / combos) - stake` ✅
- Standout Trifecta: `combos = (n-1) × (n-2)` ✅
- Quaddie/Big6: Leg-by-leg winner validation, flexi calculation ✅
- Big3 Multi: All 3 races must complete before settlement ✅

#### Issues Found

| ID | Severity | Issue | Location |
|----|----------|-------|----------|
| P1 | Medium | Dead heats not handled (no dividend adjustment) | `picks.py` |
| P2 | Medium | Scratched horses in exotic combos not handled | `picks.py` |
| P3 | Low | Standout quinella code path (impossible bet type) | `picks.py:413-417` |
| P4 | Low | Market floor may be too aggressive (50%) | `probability.py:765` |
| P5 | Low | Weight factor missing distance interaction | `probability.py:617` |
| P6 | Low | Form rating no sample size protection (2 starts minimum) | `probability.py:306` |
| P7 | Low | Pace factor no fallback when PF map_factor is None | `probability.py:401` |
| P8 | Low | Exotic type normalization inconsistency (standout → box) | `parser.py:149` |

---

### 2.3 — AI / OpenAI Integration

#### Full Prompt Audit

8 prompt templates totaling ~970 lines:

| Prompt | Lines | Purpose | Key Features |
|--------|-------|---------|--------------|
| `early_mail.md` | 388 | Pre-race tips (main product) | Probability matching, exotic hierarchy, sequence construction, performance accountability |
| `personality.md` | 163 | Punty's voice & style | 86-entry racing slang glossary, numbers discipline, age/sex domain knowledge |
| `wrap_up.md` | 175 | Post-race review | 11-section structure, calculation rules, tone guidance by outcome |
| `weekly_blog.md` | 99 | Friday recap column | 7 required sections, awards, patterns, ledger |
| `initialise.md` | 76 | Pre-generation setup | Lean list, race weighting, focus races |
| `results.md` | 72 | Post-race commentary | Running P&L, sectional analysis, tone by outcome |
| `speed_map_update.md` | 40 | Pace change alerts | Impact assessment, revised recommendations |
| `race_preview.md` | 30 | Single race focus | 200-300 word max, 6-section structure |

#### What the AI Receives (per Early Mail)

| Component | Tokens | Source |
|-----------|--------|--------|
| System prompt (personality + analysis weights) | ~8K | Fixed |
| Meeting context (formatted markdown) | 40-80K | Scales with races/runners |
| RAG strategy context (performance + directives) | 5-10K | Scales with settled picks |
| RAG assessment context (similar race learnings) | 2-5K | 2-5 assessments retrieved |
| Runner-level memories | 0-2K | Only if settled_memories >= 10 |
| **Total Input** | **56-105K** | |
| **Output** | **10-20K** | Full early mail content |

#### Cost Analysis

**Estimated monthly cost: ~$213-$289/month** (at GPT-4o rates; GPT-5.2 may differ)

| Activity | Count/Day | Avg Tokens | Daily Cost |
|----------|----------|------------|-----------|
| Early Mail generation | 8 | 92K | $2.80 |
| Meeting Wrapup | 8 | 56K | $2.24 |
| Post-Race Assessments | 40 | 22K | $3.20 |
| Content Fixes/Regen | 5 | 7K | $0.20 |
| Results Commentary | 10 | 27K | $1.00 |
| Embedding calls | 200 | 0.5K | $0.20 |

#### RAG Pipeline Assessment

**Is it functional?** Yes — multi-source RAG with 4 distinct knowledge layers:
1. **Strategy context**: Real P&L per bet type + automated directives
2. **Race assessments**: LLM-generated post-race learnings, retrieved by hybrid similarity (SQL filter + embedding + attribute bonuses)
3. **Deep learning patterns**: 844+ patterns from 280K+ historical runners
4. **Domain knowledge**: Track specs, handicapping rules, bias patterns for 60+ tracks

**Is it measurable?** Partially — strategy directives are data-driven from actual settlement results, but there's no A/B testing to measure the impact of RAG context on prediction quality.

**Scalability concern**: `find_similar_situations` loads ALL embeddings into memory for cosine similarity (O(n) per query). Won't scale past ~10K memories.

#### Value vs Cost: What Requires LLM?

| Task | LLM Value | Recommendation |
|------|-----------|----------------|
| Content generation (early mail, wrapup, blog) | **ESSENTIAL** | Keep — no deterministic alternative for natural language |
| Post-race assessments | HIGH | Batch to 1/meeting (reduce 80%) |
| Content fixes/tone adjustment | MEDIUM | Keep, improve initial prompt quality |
| Exotic selection | MEDIUM | Pre-select highest-value combo, LLM confirms |
| Sequence lane construction | LOW-MEDIUM | Pre-build Skinny/Balanced/Wide algorithmically |
| Stake allocation | LOW | Pre-calculate via Kelly, present as recommended |
| Bet type selection | LOW-MEDIUM | Apply decision tree, present as recommended |
| Punty's Pick | MEDIUM | Pre-calculate EV, present as suggested |

**Already deterministic (correctly designed)**: Probability model, exotic probabilities, sequence leg confidence, market movement, stewards parsing, time ratings, weight analysis, pace scenario.

#### Issues Found

| ID | Severity | Issue | Location |
|----|----------|-------|----------|
| A1 | High | No token usage logging — cannot track costs | `client.py:85-113` |
| A2 | High | Excessive retry multiplication (3×3=9 attempts possible) | `generator.py:182-208` |
| A3 | Medium | Personality cache never expires (needs restart) | `generator.py:42-55` |
| A4 | Medium | Context snapshot race condition (concurrent version collision) | `versioning.py:60-67` |
| A5 | Medium | Assessment embeddings fail silently (invisible to RAG) | `assessment.py:695-721` |
| A6 | Low | Odds movement formatted as integers not decimals | `generator.py:849-857` |
| A7 | Medium | No post-generation probability validation | — |
| A8 | High | Single AI provider dependency (no Claude fallback despite CLAUDE.md) | `client.py` |
| A9 | Medium | Strategy directives on small samples (10+ bets) | `strategy.py:693-812` |
| A10 | Medium | 10-min timeout risk with 8 concurrent meetings | `generator.py` |

---

### 2.4 — App & Website

#### Strengths

- **Async-First Stack**: FastAPI + SQLAlchemy 2.0 async + aiosqlite + httpx. No blocking I/O.
- **Hostname-Based Routing**: Single FastAPI instance serves both `punty.ai` (public) and `app.punty.ai` (admin) via middleware URL rewriting.
- **Comprehensive Content Lifecycle**: `draft → pending_review → approved → sent` with supersede, reject, regenerate paths. Old content superseded when new approved.
- **SSE Progress Streaming**: Real-time progress for scrape, speed maps, and generation operations.
- **Public Site SEO**: Dynamic sitemaps, robots.txt, llms.txt, meta tags, proper lastmod dates.
- **Cyberpunk Design**: Consistent Orbitron/Rajdhani/Source Sans Pro fonts, magenta/cyan/orange gradients.
- **759 Unit Tests + 103 E2E Tests**: CI/CD via GitHub Actions with separate unit and e2e jobs.

#### Issues Found

| ID | Severity | Issue | Location |
|----|----------|-------|----------|
| W1 | High | No rate limiting on public API (DoS risk) | `/api/public/*` |
| W2 | High | No database backup strategy | Production |
| W3 | High | Missing database indexes on frequently queried columns | Multiple |
| W4 | Medium | SSE stream session leak (db session may close before stream completes) | `meets.py:305-353` |
| W5 | Medium | No session timeout (sessions persist indefinitely) | `auth.py` |
| W6 | Medium | SQLite single-writer bottleneck under concurrent load | Multiple |
| W7 | Medium | Per-request stats computation on homepage (no caching) | `public/routes.py:143-400` |
| W8 | Medium | N+1 query pattern in tips page | `public/routes.py:976-988` |
| W9 | Medium | No zero-downtime deployment | Production |
| W10 | Low | Dead endpoints: `POST /api/meets/scrape`, `POST /api/scheduler/run/{job_type}` | `meets.py:490`, `scheduler.py:89` |
| W11 | Low | Legacy `daily_morning_prep()` function (230 lines, deprecated) | `jobs.py:411-641` |
| W12 | Low | Review detail prev/next uses current date only | `routes.py:383-399` |
| W13 | Medium | Facebook App Secret stored in plain text in DB | `delivery.py:172` |
| W14 | Medium | Telegram bash tool allows arbitrary code execution | `tools.py:42` |

---

## Phase 3 — Strategic Analysis

### Architecture Questions

#### 1. Could the prediction pipeline run locally without OpenAI?

**Yes, partially.** The probability model (10 factors), exotic calculations (Harville), sequence leg confidence, and all context enrichment (stewards, time ratings, weight analysis) are already deterministic. These produce the numbers that drive betting decisions.

What OpenAI provides that can't be replicated locally:
- **Natural language generation**: Punty's personality, humor, narrative flow
- **Subjective analysis**: Post-race assessments, decisive factor identification
- **Format compliance**: Consistent output structure across 388 lines of rules

**Recommendation**: Keep LLM for content generation, move remaining semi-deterministic decisions (exotic selection, sequence lanes, stake allocation, bet type, Punty's Pick) to pre-calculation. This reduces LLM cognitive load and improves consistency on publicly-tracked metrics.

#### 2. Would an Agents-based architecture be better?

**Not at current scale.** The single-model approach with structured prompts works well because:
- Each generation is a single, well-defined task (not multi-step reasoning)
- Context is assembled deterministically before the LLM call
- The prompt is exhaustive (388 lines) — minimal ambiguity

An agents architecture would add value if:
- Multiple LLM calls per meeting were needed (e.g., per-race analysis then synthesis)
- Dynamic tool use was required (e.g., LLM queries DB mid-generation)
- Different models were optimal for different tasks (e.g., fast model for exotics, reasoning model for narrative)

**Future consideration**: If deterministic pre-selections are implemented, the LLM's job shrinks to pure narrative — a smaller, faster model could handle it.

#### 3. Should manual weights be removed entirely?

**No — hybrid approach is better.** The self-tuning system already adjusts weights based on settled results, but with conservative smoothing (70/30) and bounds (2%-35%). Pure data-driven weights risk:
- **Small sample instability**: With only 50+ picks needed for tuning, weights could oscillate
- **Regime change blindness**: If racing conditions shift (new track, rule change), historical data may be misleading
- **Factor correlation**: Some factors are correlated (market + movement), pure data-driven could double-count

**Recommendation**: Keep self-tuning with current constraints. The 70/30 smoothing and 24h cooldown are appropriate. Consider increasing minimum sample to 100+ for more stable tuning.

#### 4. Could race-type-specific weight profiles work?

**Yes, and the architecture supports it.** The deep learning patterns already analyze by venue, distance, condition, and pace style. To implement race-type-specific weights:

1. Store separate weight profiles in `AppSettings` (e.g., `probability_weights_sprint`, `probability_weights_staying`)
2. In `calculate_race_probabilities`, select profile based on race characteristics
3. Tuning system would need separate tuning loops per profile (requires more settled data per profile)

**Risk**: Smaller sample sizes per profile means noisier tuning. Sprint races (1000-1200m) are most common, staying races (2000m+) much rarer. A sprint-specific model could be reliable quickly; a staying-specific model would take months.

**Recommendation**: Start with 2 profiles — Sprint (≤1200m) and Standard (>1200m). Only if sample sizes reach 200+ per profile within 2 months.

### Performance Questions

#### 5. Strongest aspects of the current system

1. **Data breadth**: 70+ fields per runner from 12 sources — more comprehensive than any single tipster
2. **Probability model**: 10-factor engine with Harville exotics is mathematically sound
3. **RAG feedback loop**: Strategy directives from actual betting results create genuine self-improvement
4. **Automation pipeline**: Calendar → scrape → generate → approve → post → settle runs hands-free
5. **Content quality**: 388-line prompt with performance accountability produces consistent, entertaining output
6. **Value-betting philosophy**: The system explicitly looks for overs, not just favourites

#### 6. Weakest points and biggest risks

1. **No cost visibility**: Cannot track token spend or identify expensive generations
2. **Settlement edge cases**: Dead heats and scratched exotic runners not handled
3. **Database fragility**: Single SQLite file, no backups, no migration system, write contention
4. **Public API exposure**: Unauthenticated, unthrottled, complex queries
5. **Single AI provider**: OpenAI outage = no content generation
6. **LLM decision inconsistency**: Exotics, stakes, bet types still decided by LLM (variable quality)

#### 7. Highest-impact improvements for prediction accuracy

1. **Pre-calculate deterministic decisions** (exotics, stakes, bet types, Punty's Pick): Removes LLM variance from measurable metrics
2. **Add distance interaction to weight factor**: Weight impacts sprints more than staying races
3. **Form rating Bayesian shrinkage**: Prevent 1-from-1 (100% SR) from inflating scores
4. **Pace factor fallback**: Use settle position when PF map_factor unavailable
5. **Relax market floor from 50% to 33%**: Allow model to disagree more with market
6. **Increase strategy directive sample thresholds**: 30+ bets for directional advice, 50+ for "DROP"

#### 8. Missing industry-standard data/approaches

- **Sectional benchmarks**: Available via PF Modeller+ subscription — measures raw ability (time-adjusted)
- **Class ratings**: Timeform/Ratings-based class assessment (partially covered by handicap_rating)
- **Wet track sire statistics**: Pedigree-based wet track aptitude
- **Pace maps with predicted sectionals**: More granular than leader/on_pace/midfield/backmarker
- **Stable confidence indicators**: Following money patterns

None of these would compromise PuntyAI's value-betting identity — they'd strengthen the probability model's inputs while Punty still finds the overs.

### Growth Questions

#### 9. What would it take to attract and retain users?

The public site already has strong foundations (tips pages, stats, blog, SEO). To attract users:

1. **Consistent Punty's Pick strike rate >30%**: The publicly tracked stat is the credibility metric
2. **Real-time race-day engagement**: Live updates, celebrations, pace analysis (already built)
3. **Subscription tier with "Talk to Punty"**: Conversational AI over the data (planned)
4. **Social proof**: Display win streaks, big exotic hits, and ROI prominently
5. **Email/SMS alerts**: Push tips before race time (email already implemented)
6. **Free tier**: Daily "Punty's Best Bet" (1 pick) to hook users; paid tier for full early mail

#### 10. Quick wins vs longer-term investments

**Quick wins (1-2 days each)**:
- Add database indexes (1 hour, immediate query improvement)
- Add token usage logging (2 hours, cost visibility)
- Consolidate retry logic (1 hour, prevent cost multiplication)
- Add personality cache TTL (30 minutes, Settings UI works properly)
- Public API rate limiting (3 hours, security improvement)
- Database backups (2 hours, disaster recovery)

**Medium-term (1-2 weeks)**:
- Deterministic pre-selections (3-4 days, consistency improvement)
- Batch assessment generation (2 days, 80% cost reduction on assessments)
- Post-generation probability validation (3 days, catch errors pre-delivery)
- Staging environment setup (see CI/CD section below)

**Longer-term (1+ months)**:
- Claude fallback provider (3-4 days, eliminate SPOF)
- Alembic migration system (4 hours initial, ongoing discipline)
- Vector DB migration for RAG (1-2 weeks)
- Race-type-specific weight profiles (2 weeks + data collection)
- Paid subscription + "Talk to Punty" (multi-week project)

---

## CI/CD & Staging Environment (No Docker)

Since Docker Desktop requires Windows 10 Pro/Enterprise (Hyper-V), here are alternatives for Windows 10 Home:

### Option A: Branch-Based Staging on Same Server (Recommended)

```
Production: /opt/puntyai/          → app.punty.ai (master branch)
Staging:    /opt/puntyai-staging/  → staging.punty.ai (staging branch)
```

- **Caddy** reverse proxy routes `staging.punty.ai` to port 8001
- Separate systemd service `puntyai-staging.service`
- Separate SQLite database at `/opt/puntyai-staging/data/punty.db`
- Deploy workflow: `push to staging branch → SSH deploy to staging → test → merge to master → SSH deploy to production`

**GitHub Actions workflow addition**:
```yaml
deploy-staging:
  if: github.ref == 'refs/heads/staging'
  steps:
    - run: ssh root@app.punty.ai "cd /opt/puntyai-staging && git fetch && git checkout -f origin/staging && systemctl restart puntyai-staging"

deploy-production:
  if: github.ref == 'refs/heads/master'
  needs: [unit-test, e2e-test]
  steps:
    - run: ssh root@app.punty.ai "cd /opt/puntyai && git fetch && git checkout -f origin/master && chmod 666 prompts/*.md && systemctl restart puntyai"
```

**Pros**: Simple, uses existing server, no extra cost, real production-like environment
**Cons**: Shares server resources, staging weather/scraping could interfere

### Option B: WSL2 Local Staging (Dev Machine)

Windows 10 Home supports WSL2 (Windows Subsystem for Linux):
- Install Ubuntu via `wsl --install`
- Clone repo in WSL2, run with `uvicorn` on localhost:8001
- Use a separate SQLite database for staging
- Test locally before pushing to production

**Pros**: Free, fast iteration, no server impact
**Cons**: Not production-equivalent (different OS, no Caddy, no systemd)

### Option C: Cheap VPS Staging ($5/month)

A small VPS (DigitalOcean $4/mo, Hetzner $3.29/mo, or Vultr $3.50/mo):
- Mirror production setup exactly
- Separate staging.punty.ai subdomain
- Automated deploy from `staging` branch

**Pros**: True production parity, isolated from production
**Cons**: Small monthly cost, another server to manage

### Recommendation

**Option A** (branch-based on same server) for immediate improvement, with **Option C** as a future upgrade when revenue justifies it. Option A gives you 90% of the benefit at zero extra cost.

---

## Security Summary

### Current Protections ✅
- Google OAuth with email whitelist (no self-registration)
- CSRF protection via HMAC-SHA256 tokens
- SQLAlchemy parameterized queries (no SQL injection)
- Path traversal protection in Telegram tools
- API keys stored in DB (not in source code)
- Sensitive keys redacted in audit log

### Gaps ❌
- No rate limiting on public API endpoints
- No session timeout (sessions persist until logout)
- API keys stored unencrypted in `app_settings` table
- Telegram bash tool allows arbitrary code execution
- Facebook App Secret in plain text in DB
- No database backup strategy
- SSH deploy as root user

---

## Test Coverage Summary

| Suite | Tests | Status | Coverage |
|-------|-------|--------|----------|
| Unit Tests | 759 | All passing | Parser, picks, settlement, probability (156), stewards (43), time ratings (31), weight analysis (14), PF API (31), WillyWeather (40), Facebook (23), Telegram (33), strategy (31), tuning (40) |
| E2E Tests | 103 | All passing | Public pages/API, admin pages/API, auth/CSRF, security, mobile/tablet/desktop viewports, performance |
| CI/CD | GitHub Actions | Passing | Separate unit-test and e2e-test jobs |

---

*Full Feature Registry, Data Flow Diagram, Prompt Documentation, and Recommendations Roadmap available as separate documents in this directory.*
