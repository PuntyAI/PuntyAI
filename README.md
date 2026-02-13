# PuntyAI

AI-powered Australian horse racing content generator and betting tracker. Punty is the cheeky Aussie tipster who crunches form, reads the track, and finds the bloody winners.

## What It Does

PuntyAI automates the full racing analysis pipeline:

1. **Scrapes** meeting data from 12+ sources (PuntingForm API, Racing.com GraphQL, TAB, WillyWeather)
2. **Calculates** probabilities via a 10-factor model with Harville exotic pricing
3. **Generates** humorous early mail tips via GPT-5.2 with Punty's personality
4. **Tracks** every bet from tip to settlement with full P&L
5. **Delivers** across Twitter/X, Facebook, Email with platform-specific formatting
6. **Learns** from results via RAG (strategy context, race assessments, deep learning patterns)

## Features

- **Admin Dashboard** (app.punty.ai) — Manage meetings, review content, track performance
- **Public Tips Site** (punty.ai) — Tips pages, stats, blog, form guide, live updates
- **10-Factor Probability Engine** — Market, form, pace, barrier, jockey/trainer, movement, class, weight, horse profile, deep learning
- **Self-Tuning Weights** — Auto-adjusts factor weights from settled results
- **Calibration Dashboard** — Visualize model accuracy, factor performance, value tracking
- **Multi-Source Scraping** — PuntingForm API (primary), Racing.com GraphQL (supplement), TAB, WillyWeather
- **AI Content Generation** — 8 prompt templates, SSE streaming, personality consistency
- **RAG Learning System** — Strategy directives from actual P&L, race assessments, 844+ deep learning patterns
- **Automated Pipeline** — Calendar → scrape → generate → approve → post → settle → learn
- **Live Race Monitoring** — Results, celebrations, pace analysis, weather alerts
- **Multi-Platform Delivery** — Twitter (25K), Facebook (63K), Email (HTML + plain text)
- **Telegram Bot** — Remote server management via Claude API agent loop
- **759 Unit Tests + 103 E2E Tests** — CI/CD via GitHub Actions

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11+, FastAPI, SQLAlchemy 2.0 (async), SQLite |
| Frontend | Jinja2, Tailwind CSS, HTMX, Alpine.js, Chart.js |
| AI | OpenAI GPT-5.2 (Responses API, Reasoning) |
| Scraping | httpx (async), Playwright (browser automation) |
| Scheduling | APScheduler |
| Delivery | Tweepy (Twitter), httpx (Facebook/Email) |
| Testing | pytest, pytest-asyncio, Playwright (e2e) |

## Quick Start

```bash
# Install dependencies
pip install -e .

# Install Playwright browsers (for scraping)
playwright install chromium

# Configure environment
cp .env.example .env
# Edit .env: set PUNTY_SECRET_KEY, PUNTY_GOOGLE_CLIENT_ID/SECRET, PUNTY_ALLOWED_EMAILS

# Run
uvicorn punty.main:app --reload
```

Open http://localhost:8000 — configure API keys (OpenAI, PuntingForm, WillyWeather) in Settings.

## Configuration

### Environment Variables (PUNTY_ prefix)

| Variable | Required | Description |
|----------|----------|-------------|
| `PUNTY_SECRET_KEY` | Yes | Session/CSRF signing key |
| `PUNTY_DATABASE_URL` | No | SQLite path (default: `sqlite+aiosqlite:///data/punty.db`) |
| `PUNTY_DEBUG` | No | Debug mode |
| `PUNTY_GOOGLE_CLIENT_ID` | Yes | Google OAuth client ID |
| `PUNTY_GOOGLE_CLIENT_SECRET` | Yes | Google OAuth secret |
| `PUNTY_ALLOWED_EMAILS` | Yes | Comma-separated authorized emails |

### API Keys (stored in Settings page)

| Key | Required | Description |
|-----|----------|-------------|
| OpenAI | Yes | Content generation (GPT-5.2) |
| PuntingForm | Yes | Primary racing data source |
| WillyWeather | Recommended | Live weather for racecourses |
| Twitter | Optional | Auto-post to Twitter/X |
| Facebook | Optional | Auto-post to Facebook Page |
| Telegram | Optional | Remote bot management |

## Project Structure

```
punty/
├── ai/              # AI content generation (client, generator, reviewer)
├── api/             # FastAPI route handlers (7 routers, 73 endpoints)
├── context/         # Context building + enrichment (stewards, time, weight)
├── deep_learning/   # Historical pattern analysis (18 analyses, 844+ patterns)
├── delivery/        # Platform delivery (Twitter, Facebook, Email, WhatsApp)
├── formatters/      # Platform-specific formatting
├── memory/          # RAG system (strategy, assessments, knowledge, embeddings)
├── models/          # SQLAlchemy models (11 tables)
├── patterns/        # Live pattern engine + awards
├── public/          # Public website (routes + templates)
├── results/         # Settlement, parser, monitor, celebrations
├── scheduler/       # APScheduler jobs + automation
├── scrapers/        # Data scraping (12+ sources)
├── telegram/        # Telegram bot (agent + tools)
└── web/             # Admin dashboard (routes + templates)
prompts/             # 8 AI prompt templates
tests/
├── unit/            # 759 unit tests
└── e2e/             # 103 end-to-end tests
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Unit tests only (faster)
pytest tests/unit -v

# E2E tests (requires Playwright)
pytest tests/e2e -v

# With coverage
pytest tests/ --cov=punty --cov-report=term-missing
```

## Deployment

Production runs on an Ubuntu VM with uvicorn behind Caddy:

```bash
# Deploy to production
ssh -i ~/.ssh/id_ed25519 root@app.punty.ai \
  "cd /opt/puntyai && git fetch origin master && git checkout -f origin/master && chmod 666 prompts/*.md && systemctl restart puntyai"
```

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — System design, data flow, tech stack
- [CLAUDE.md](CLAUDE.md) — Developer guide, models, config, operations
- [docs/audit/](docs/audit/) — Full solution audit (Feb 2026)
  - [Findings Report](docs/audit/FINDINGS_REPORT.md) — Strengths, bugs, risks, strategic analysis
  - [Recommendations](docs/audit/RECOMMENDATIONS.md) — Prioritized improvement roadmap
  - [Feature Registry](docs/audit/FEATURE_REGISTRY.md) — Complete component inventory
  - [Data Flow](docs/audit/DATA_FLOW.md) — End-to-end data journey diagrams
  - [Prompt Documentation](docs/audit/PROMPT_DOCUMENTATION.md) — Full AI prompt audit

## Gamble Responsibly

1800 858 858 | [gamblinghelponline.org.au](https://www.gamblinghelponline.org.au)
