# PuntyAI — Feature & Tool Registry

**Complete cross-referenced inventory of every component, feature, script, and tool.**

---

## 1. Core Application Features

### 1.1 Data Scraping Pipeline

| Feature | Files | Description |
|---------|-------|-------------|
| Calendar Discovery | `scrapers/calendar.py` | Playwright scrapes racing.com/calendar daily at 00:05 |
| PuntingForm API | `scrapers/punting_form.py` | Primary data source: meetings, fields, form (10 runs), speed maps, ratings, conditions, scratchings |
| Racing.com Scraper | `scrapers/racing_com.py` | GraphQL + Playwright: odds (5 bookmakers), comments, stewards, sectionals, results |
| TAB Scraper | `scrapers/tab.py` | API + HTML: live odds, results, dividends |
| WillyWeather | `scrapers/willyweather.py` | Weather API: wind, temp, humidity, rain, radar, cameras |
| Scraper Orchestrator | `scrapers/orchestrator.py` | Coordinates scrapers, merges data, manages conflicts |
| Track Conditions | `scrapers/track_conditions.py` | Playwright: racingaustralia.horse + sportsbetform (legacy) |
| Future Races | `scrapers/future_races.py` | PF API: upcoming Group races + nominations |
| News Scraper | `scrapers/news.py` | Google News RSS + 7NEWS HTML for weekly blog |
| Trainer Rankings | `scrapers/racing_australia.py` | BeautifulSoup: RA premierships + TRC global rankings |
| TabTouch | `scrapers/tabtouch.py` | HTML: exotic dividends (WA source) |
| Punters.com.au | `scrapers/punters.py` | Legacy (not actively used) |
| Base Scraper | `scrapers/base.py` | httpx client, ID generation utilities |
| Playwright Base | `scrapers/playwright_base.py` | Browser singleton for JS-heavy scraping |

### 1.2 Probability Engine

| Feature | Files | Description |
|---------|-------|-------------|
| 10-Factor Model | `probability.py` | Market, form, DL, pace, barrier, J/T, movement, class, weight, profile |
| Harville Exotics | `probability.py` | Conditional probability for exacta/trifecta/first4/quinella |
| Quarter-Kelly Staking | `probability.py` | Conservative stake recommendation from $20 pool |
| Dynamic Market Boost | `probability.py` | Increases market weight when other factors are neutral |
| Market Floor | `probability.py` | Prevents model from contradicting market by >50% |
| Sequence Leg Confidence | `probability.py` | HIGH/MED/LOW rating per race with suggested width |
| Self-Tuning | `probability_tuning.py` | Point-biserial correlation, 70/30 smoothing, bounds/cooldown |
| Calibration Dashboard | `web/templates/probability.html` | Read-only visualization of tuning metrics |

### 1.3 AI Content Generation

| Feature | Files | Description |
|---------|-------|-------------|
| AI Client | `ai/client.py` | OpenAI Responses API wrapper (GPT-5.2 Reasoning) |
| Content Generator | `ai/generator.py` | Orchestrates: context → snapshot → prompts → generate → save |
| Content Reviewer | `ai/reviewer.py` | AI-assisted content fixes (6 issue types) |
| SSE Streaming | `ai/generator.py` | Real-time progress events for scrape/generate operations |

### 1.4 Context & Enrichment

| Feature | Files | Description |
|---------|-------|-------------|
| Context Builder | `context/builder.py` | Assembles 60+ data sources into prompt-ready markdown |
| Blog Context Builder | `context/blog_builder.py` | Weekly patterns, awards, ledger, news for blog |
| Context Versioning | `context/versioning.py` | SHA256 snapshots for change detection |
| Change Detection | `context/diff.py` | Detect scratchings, odds shifts, speed map changes |
| Stewards Parsing | `context/stewards.py` | 8-category excuse extraction from form history |
| Time Ratings | `context/time_ratings.py` | FAST/STANDARD/SLOW vs venue standard times |
| Weight Analysis | `context/weight_analysis.py` | Performance at different weight bands |

### 1.5 RAG Learning System

| Feature | Files | Description |
|---------|-------|-------------|
| Strategy Context | `memory/strategy.py` | Bet-type P&L aggregation, directives, deep patterns |
| Race Assessments | `memory/assessment.py` | LLM-generated post-race learnings with embeddings |
| Memory Store | `memory/store.py` | Runner-level situation memories (embedding similarity) |
| Embedding Service | `memory/embeddings.py` | OpenAI text-embedding-3-small |
| Racing Knowledge | `memory/racing_knowledge.py` | 60+ tracks, conditions, handicapping rules, bias patterns |
| Memory Models | `memory/models.py` | RaceMemory, PatternInsight, RaceAssessment tables |

### 1.6 Deep Learning Patterns

| Feature | Files | Description |
|---------|-------|-------------|
| Pattern Runner | `deep_learning/runner.py` | Orchestrates 18 pattern analyses |
| 18 Pattern Types | `deep_learning/patterns.py` | Track/dist/cond, barrier, pace, form cycles, J/T, class, season, market, sectional, track bias, pace collapse, coming-into-form, class movers, speed quality, standard times, weight impact, bounce-back |
| Historical Importer | `deep_learning/importer.py` | Import historical data for analysis |
| DL Models | `deep_learning/models.py` | HistoricalMeeting/Race/Runner tables |

### 1.7 Results & Settlement

| Feature | Files | Description |
|---------|-------|-------------|
| Results Monitor | `results/monitor.py` | Background polling: odds, scratchings, track, results, settlement |
| Parser | `results/parser.py` | Regex extraction: Big3, selections, exotics, sequences from AI text |
| Picks & Settlement | `results/picks.py` | P&L calculation for all bet types, flexi exotics, multi settlement |
| Change Detection | `results/change_detection.py` | Track changes to selections, exotics, sequences, odds, speed maps, weather |
| Celebrations | `results/celebrations.py` | Victory phrase generator for social media |

### 1.8 Delivery Channels

| Feature | Files | Description |
|---------|-------|-------------|
| Twitter/X | `delivery/twitter.py` | Long-form posting (25K), threading, standalone tweets |
| Facebook | `delivery/facebook.py` | Page posts (63K), token exchange, Graph API v21.0 |
| Email | `delivery/email.py` | Resend API + SMTP fallback |
| WhatsApp | `delivery/whatsapp.py` | WhatsApp Business API (legacy) |

### 1.9 Formatters

| Feature | Files | Description |
|---------|-------|-------------|
| Twitter Formatter | `formatters/twitter.py` | Markdown → Unicode bold, profanity substitution, hashtags |
| Facebook Formatter | `formatters/facebook.py` | Markdown → plain text, profanity rotation |
| HTML Formatter | `formatters/html.py` | Markdown → HTML for public site, profanity rotation (20 alt greetings) |
| Email Formatter | `formatters/email.py` | Markdown → HTML email body |

### 1.10 Scheduler & Automation

| Feature | Files | Description |
|---------|-------|-------------|
| Job Definitions | `scheduler/jobs.py` | Daily calendar, morning scrape, weekly patterns, blog |
| Automation Pipeline | `scheduler/automation.py` | Auto-approve validation, auto-post to social media |
| Meeting Automation | `scheduler/jobs.py` | Per-meeting pre-race and post-race jobs |

### 1.11 Telegram Bot

| Feature | Files | Description |
|---------|-------|-------------|
| Bot Lifecycle | `telegram/bot.py` | Polling, auth check, message chunking (4096 char) |
| Claude Agent | `telegram/agent.py` | Agentic loop (max 25 turns), conversation memory |
| Server Tools | `telegram/tools.py` | bash, read_file, write_file, edit_file, list_files, query_db |

---

## 2. API Endpoints (73 total)

### Meetings (`/api/meets`) — 14 endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | List all meetings |
| GET | `/today` | Today's meetings |
| GET | `/selected` | Selected meetings for today |
| GET | `/incomplete-data-check` | Check speed map completeness |
| PUT | `/bulk/select-all` | Bulk select/deselect |
| GET | `/bulk/scrape-stream` | SSE bulk scrape |
| GET | `/bulk/speed-maps-stream` | SSE bulk speed maps |
| GET | `/bulk/scrape-and-speedmaps-stream` | SSE combined |
| GET | `/bulk/generate-early-mail-stream` | SSE bulk generate |
| GET | `/{id}` | Get meeting with races/runners |
| POST | `/{id}/scrape-full` | Full scrape |
| GET | `/{id}/scrape-stream` | SSE single scrape |
| POST | `/{id}/refresh-odds` | Quick odds refresh |
| GET | `/{id}/races/{num}/probabilities` | Runner probabilities |

### Content (`/api/content`) — 14 endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | List content with filters |
| GET | `/review-queue` | Pending review items |
| POST | `/reject-all-pending` | Bulk reject |
| GET | `/review-count` | Badge count (plain text) |
| GET | `/generate-stream` | SSE content generation |
| GET | `/blog/generate-stream` | SSE blog generation |
| GET | `/blog/latest` | Latest published blog |
| GET | `/{id}` | Get content item |
| POST | `/generate` | Non-streaming generate |
| POST | `/{id}/review` | Approve/reject/regenerate/fix |
| PUT | `/{id}` | Manual text update |
| POST | `/{id}/format` | Re-format for Twitter |
| GET | `/{id}/picks` | Get picks for content |
| POST | `/{id}/reparse-picks` | Re-extract picks from text |

### Results (`/api/results`) — 11 endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/monitor-status` | Monitor status |
| POST | `/monitor/start` | Start monitor |
| POST | `/monitor/stop` | Stop monitor |
| POST | `/{id}/check` | Manual results check |
| GET | `/performance` | Today's P&L |
| GET | `/performance/history` | Date range P&L |
| GET | `/{id}/summary` | Meeting P&L |
| POST | `/{id}/race/{num}/sectionals` | Scrape sectionals |
| GET | `/{id}/race/{num}/sectionals` | Get sectionals |
| POST | `/settle-past` | Settle all unsettled |
| GET | `/wins/recent` | Recent wins |

### Delivery (`/api/delivery`) — 10 endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/twitter/status` | Twitter config |
| POST | `/twitter/post/{id}` | Post to Twitter |
| POST | `/twitter/tweet` | Standalone tweet |
| DELETE | `/twitter/tweet/{id}` | Delete tweet |
| GET | `/facebook/status` | Facebook config |
| POST | `/facebook/post/{id}` | Post to Facebook |
| DELETE | `/facebook/post/{id}` | Delete Facebook post |
| POST | `/facebook/exchange-token` | Exchange token |
| POST | `/send` | Multi-platform send |
| GET | `/preview/{id}` | Preview formatted |

### Settings (`/api/settings`) — 14 endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | All settings |
| GET | `/personality` | Personality prompt |
| PUT | `/personality` | Save personality |
| GET | `/audit-log` | Settings audit log |
| GET | `/probability-weights` | Weights + metadata |
| PUT | `/probability-weights` | Update weights |
| POST | `/probability-weights/reset` | Reset to defaults |
| GET | `/{key}` | Get setting |
| PUT | `/{key}` | Update setting |
| PUT | `/api-keys/{provider}` | Update API keys |
| POST | `/initialize` | Initialize defaults |
| POST | `/test-email` | Test email config |
| GET | `/learnings` | Race assessments |
| POST | `/cleanup` | Delete stale data |

### Scheduler (`/api/scheduler`) — 9 endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/jobs` | List jobs |
| POST | `/jobs` | Create job |
| PUT | `/jobs/{id}/toggle` | Enable/disable |
| DELETE | `/jobs/{id}` | Delete job |
| GET | `/status` | Scheduler status |
| POST | `/morning-prep/run` | Manual morning scrape |
| POST | `/morning-prep/toggle` | Pause/resume |
| GET | `/activity-log` | Recent activity |
| GET | `/full-status` | Complete status |

### Weather (`/api/weather`) — 4 endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/status` | API config check |
| GET | `/{id}` | Full weather data |
| POST | `/{id}/refresh` | Force refresh |
| GET | `/{id}/radar` | Radar overlays |

### Public (`/api/public`) — 6 endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/stats` | Winner stats |
| GET | `/wins` | Recent wins ticker |
| GET | `/next-race` | Next race countdown |
| GET | `/bet-type-stats` | Filterable performance |
| GET | `/filter-options` | Autocomplete values |
| GET | `/venues` | Distinct venues |

---

## 3. Web Pages

### Admin (app.punty.ai) — 9 pages
| Path | Template | Description |
|------|----------|-------------|
| `/` | `dashboard.html` | Hero stats, ticker, monitor, P&L chart, activity log |
| `/meets` | `meets.html` | Today + paginated archive |
| `/meets/{id}` | `meet_detail.html` | Races, runners, content, picks |
| `/content` | `content.html` | Content management |
| `/review` | `review.html` | Pending review queue |
| `/review/{id}` | `review_detail.html` | Review with prev/next |
| `/probability` | `probability.html` | Calibration dashboard |
| `/settings` | `settings.html` | API keys, weights, prompts |
| `/settings/learnings` | — | Race assessments |

### Public (punty.ai) — 14 pages
| Path | Template | Description |
|------|----------|-------------|
| `/` | `index.html` | Hero, best bet, stats, strike rates, recent tips |
| `/about` | `about.html` | About Punty |
| `/how-it-works` | `how-it-works.html` | System explanation |
| `/contact` | `contact.html` | Contact info |
| `/glossary` | `glossary.html` | Racing terms |
| `/calculator` | `calculator.html` | Betting calculator |
| `/tips` | `tips.html` | Tips calendar |
| `/tips/{id}` | `tips_detail.html` | Meeting tips with form guide |
| `/stats` | `stats.html` | Filterable performance stats |
| `/blog` | `blog.html` | Blog listing |
| `/blog/{slug}` | `blog_detail.html` | Individual blog post |
| `/terms` | `terms.html` | Terms of service |
| `/privacy` | `privacy.html` | Privacy policy |
| `/data-deletion` | `data-deletion.html` | Data deletion |

### SEO
| Path | Description |
|------|-------------|
| `/robots.txt` | Robots file |
| `/llms.txt` | AI crawler discovery |
| `/sitemap.xml` | Sitemap index |
| `/sitemap-static.xml` | Static pages |
| `/sitemap-blog.xml` | Blog posts |
| `/sitemap-tips.xml` | Tips pages (limit 500) |

---

## 4. Scheduler Jobs

| Job | Schedule | Function |
|-----|----------|----------|
| `daily-calendar-scrape` | 00:05 AEDT | Find + select meetings, initial scrape |
| `daily-morning-scrape` | 05:00 AEDT | Full re-scrape + speed maps |
| `weekly-pattern-refresh` | Thu 22:00 | Deep patterns, awards, ledger, news |
| `weekly-blog` | Fri 08:00 | Generate + approve + post blog |
| `{meeting}-pre-race` | R1 - 2h | Odds → weather → speed maps → generate → approve → post |
| `{meeting}-post-race` | R(last) + 30m | Settlement → wrapup → post → tune weights |
| `ResultsMonitor` | 30-90s loop | Live race-day monitoring |

---

## 5. Database Tables (11)

| Table | PK | Records |
|-------|-----|---------|
| `meetings` | `{venue}-{date}` | Meetings |
| `races` | `{meeting}-r{num}` | Races |
| `runners` | `{race}-{sc}-{horse}` | 199K+ runners |
| `content` | UUID | Generated content |
| `picks` | UUID(16) | Betting picks |
| `app_settings` | key string | Configuration |
| `live_updates` | autoincrement | Race-day posts |
| `context_snapshots` | UUID | Context versions |
| `race_assessments` | autoincrement | Post-race learnings |
| `pattern_insights` | autoincrement | 844+ betting patterns |
| `probability_tuning_log` | autoincrement | Weight change history |

---

## 6. Prompt Templates (8)

| File | Lines | Purpose |
|------|-------|---------|
| `early_mail.md` | 388 | Main tips product |
| `personality.md` | 163 | Voice & style |
| `wrap_up.md` | 175 | Post-race review |
| `weekly_blog.md` | 99 | Friday column |
| `initialise.md` | 76 | Meeting setup |
| `results.md` | 72 | Race commentary |
| `speed_map_update.md` | 40 | Pace alerts |
| `race_preview.md` | 30 | Single race |

---

## 7. Test Suites

| File | Tests | Coverage |
|------|-------|---------|
| `test_parser.py` | ~50 | Pick extraction regex |
| `test_picks.py` | ~60 | Settlement P&L |
| `test_probability.py` | 156 | 10-factor model |
| `test_probability_tuning.py` | 40 | Self-tuning system |
| `test_stewards.py` | 43 | Excuse parsing |
| `test_time_ratings.py` | 31 | Speed ratings |
| `test_weight_analysis.py` | 14 | Weight form |
| `test_punting_form.py` | 31 | PF API scraper |
| `test_willyweather.py` | 40 | Weather API |
| `test_facebook.py` | 23 | Facebook delivery |
| `test_telegram.py` | 33 | Telegram bot |
| `test_strategy.py` | 31 | RAG strategy |
| `test_generator.py` | ~30 | Content generation |
| E2E suite | 103 | Full stack tests |

---

## 8. Configuration

### Environment Variables (PUNTY_ prefix)
| Variable | Description |
|----------|-------------|
| `PUNTY_SECRET_KEY` | Session/CSRF signing key |
| `PUNTY_DATABASE_URL` | SQLite path |
| `PUNTY_DEBUG` | Debug mode |
| `PUNTY_GOOGLE_CLIENT_ID` | OAuth client ID |
| `PUNTY_GOOGLE_CLIENT_SECRET` | OAuth secret |
| `PUNTY_ALLOWED_EMAILS` | Authorized email whitelist |

### AppSettings Keys (DB)
| Key | Description |
|-----|-------------|
| `openai_api_key` | OpenAI API key |
| `anthropic_api_key` | Claude API key |
| `twitter_*` | Twitter OAuth credentials |
| `facebook_*` | Facebook page token + app credentials |
| `telegram_*` | Bot token + owner ID |
| `punting_form_api_key` | PF API key |
| `willyweather_api_key` | Weather API key |
| `probability_weights` | JSON factor weights |
| `unit_value` | Dollar value per unit |
| `smtp_*` | Email SMTP settings |
| `resend_api_key` | Resend email API |
