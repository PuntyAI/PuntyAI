# PuntyAI Recommendations Roadmap

**Date:** 2026-02-13 | **Priority-ranked improvements with estimated impact**

---

## Priority Matrix

### CRITICAL — This Week

| # | Recommendation | Effort | Impact | Files |
|---|---------------|--------|--------|-------|
| 1 | **Add database backup automation** | 2h | Prevent catastrophic data loss | New cron job + cloud storage |
| 2 | **Add token usage logging** | 2h | Cost visibility and budget enforcement | `ai/client.py` |
| 3 | **Consolidate retry logic** (remove stream-level retries) | 1h | Prevent 9x cost multiplication | `ai/generator.py:182-208` |
| 4 | **Add public API rate limiting** | 3h | Prevent DoS on complex stat queries | `main.py`, new middleware |
| 5 | **Add database indexes** | 1h | Immediate query performance improvement | `models/database.py` |

**Database indexes to add:**
```sql
CREATE INDEX ix_picks_settled_hit ON picks(settled, hit);
CREATE INDEX ix_picks_settled_at ON picks(settled_at);
CREATE INDEX ix_content_meeting_type_status ON content(meeting_id, content_type, status);
CREATE INDEX ix_content_blog_slug ON content(blog_slug);
CREATE INDEX ix_live_updates_meeting ON live_updates(meeting_id);
```

---

### HIGH — Next 2 Weeks

| # | Recommendation | Effort | Impact | Files |
|---|---------------|--------|--------|-------|
| 6 | **Deterministic pre-selections** (exotic, bet type, stake, Punty's Pick) | 3-4d | 15-20% decision consistency improvement | `context/builder.py` |
| 7 | **Batch assessment generation** (1/meeting not 1/race) | 2d | $2.56/day savings ($77/month) | `memory/assessment.py` |
| 8 | **Post-generation probability validation** | 3d | Catch probability/stake errors before delivery | New `validation/` module |
| 9 | **Fix SSE session management** (own sessions in generators) | 30m | Prevent DB session leaks during streaming | `api/meets.py:305-353` |
| 10 | **Add session timeout** (7-day max, 24h idle) | 1h | Security improvement | `auth.py` |
| 11 | **Add personality cache TTL** (10-min expiry) | 30m | Settings UI personality edits work without restart | `ai/generator.py:42-55` |
| 12 | **Pre-generate sequence lanes** (Skinny/Balanced/Wide) | 2d | Eliminate maths errors in sequences | `context/builder.py` |

---

### MEDIUM — Next Month

| # | Recommendation | Effort | Impact | Files |
|---|---------------|--------|--------|-------|
| 13 | **Implement database migration system** (Alembic) | 4h | Repeatable schema changes | New `alembic/` directory |
| 14 | **Cache public stats queries** (TTL=60s) | 2h | Homepage and stats page performance | `public/routes.py` |
| 15 | **Add context token limit** (truncate at 80K) | 2d | Prevent generation failures on large meetings | `ai/generator.py` |
| 16 | **Implement Anthropic Claude fallback** | 3-4d | Eliminate single AI provider SPOF | `ai/client.py` |
| 17 | **Staging environment** (branch-based on same server) | 4h | Test changes before production | Caddy config, systemd, GitHub Actions |
| 18 | **Increase strategy directive sample thresholds** | 1h | More reliable strategy advice | `memory/strategy.py` |
| 19 | **Handle dead heats in settlement** | 2h | Correct P&L when dead heats occur | `results/picks.py` |
| 20 | **Handle scratched horses in exotics** | 2h | Correct exotic settlement edge cases | `results/picks.py` |
| 21 | **Fix context snapshot race condition** | 1h | Data integrity under concurrent generation | `context/versioning.py` |
| 22 | **Remove dead code** (legacy morning prep, dead endpoints) | 30m | Codebase hygiene | `scheduler/jobs.py`, `api/meets.py`, `api/scheduler.py` |

---

### LOW — Next Quarter

| # | Recommendation | Effort | Impact | Files |
|---|---------------|--------|--------|-------|
| 23 | **Migrate RAG to vector DB** (pgvector or dedicated) | 1-2w | RAG scalability past 10K memories | `memory/store.py` |
| 24 | **Modularize early mail prompt** | 1w | Easier prompt maintenance, enable A/B testing | `prompts/early_mail.md` |
| 25 | **Add prompt versioning** | 1d | Track which prompt produced which content | `models/content.py`, `ai/generator.py` |
| 26 | **Zero-downtime deployment** | 4h | No service interruption during deploys | systemd, gunicorn |
| 27 | **Consolidate tips page queries** (reduce 8+ to 3-4) | 3h | Tips page load performance | `public/routes.py` |
| 28 | **Add generation concurrency limits** (semaphore=3) | 2h | Prevent resource exhaustion during peak | `ai/generator.py` |
| 29 | **Experiment with JSON context format** | 1w | Potential 20-30% token reduction | `context/builder.py` |
| 30 | **Race-type-specific weight profiles** | 2w | Sprint vs staying optimization | `probability.py` |
| 31 | **Add health check integration** (Caddy + systemd) | 1h | Automated restart on failure | Caddy config, systemd |
| 32 | **Deploy as non-root user** | 1h | Reduce blast radius of compromised deploy key | Server config |

---

## Impact on Strike Rate & ROI

### Immediate Impact (Weeks 1-2)

| Change | Expected Impact |
|--------|----------------|
| Deterministic pre-selections (#6) | +2-5% Punty's Pick consistency (removes LLM variance) |
| Pre-generate sequence lanes (#12) | Eliminate combo maths errors (currently ~10% of sequences have maths issues) |
| Post-generation validation (#8) | Catch probability mismatches before delivery |

### Medium-Term Impact (Month 1-2)

| Change | Expected Impact |
|--------|----------------|
| Increase directive sample thresholds (#18) | More reliable "LEAN INTO" / "DROP" advice |
| Dead heat handling (#19) | Correct P&L for ~2% of settled races |
| Race-type weight profiles (#30) | +1-3% accuracy for sprint-specific predictions |

### Long-Term Impact (Quarter 1)

| Change | Expected Impact |
|--------|----------------|
| Self-tuning with more data | Gradual weight optimization as sample grows |
| Sectional benchmarks (if PF Modeller+ acquired) | Better raw ability measurement |
| Claude fallback (#16) | Zero downtime during OpenAI outages |

---

## Cost Optimization Summary

| Optimization | Savings/Month |
|-------------|--------------|
| Batch assessments (#7) | $77 |
| Consolidate retries (#3) | Prevents up to $50+ in worst-case retry storms |
| Token usage logging (#2) | Enables identifying expensive generations |
| Pre-generate exotics/sequences (#6, #12) | ~$15 from reduced prompt complexity |
| Truncate form history to 3 starts | ~$18 |
| **Total Potential Savings** | **~$125/month (43% reduction)** |
| **Optimized Monthly Cost** | **~$165/month** |
