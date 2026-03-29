"""Load all audit plan items into Vibe Kanban cloud."""
import json
import subprocess
import time


def get_token():
    result = subprocess.run(
        ["curl", "-s", "http://127.0.0.1:60021/api/auth/token"],
        capture_output=True, text=True
    )
    return json.loads(result.stdout)["data"]["access_token"]


PROJECT = "232a48ab-ad79-4f38-acc8-37b0cc5735aa"
TODO = "855bdaa0-e5c0-4074-8789-1fb50f3aeaf2"
DONE = "2029d484-a3df-41d1-94ac-70e9ec064393"
BACKLOG = "a3e8ea62-71f2-4171-9413-f85bd201685b"

issues = [
    # Done items (1-7)
    (DONE, "#1 Database Backup Automation", "COMPLETED. Created scripts/backup_db.sh - SQLite backup with daily/weekly rotation, gzip compressed.\nEffort: 2h | Audit: #1 Critical"),
    (DONE, "#2 Database Indexes", "COMPLETED. Added 3 composite indexes: ix_picks_settled_hit, ix_picks_settled_at, ix_content_meeting_type_status.\nEffort: 1h | Audit: #5 Critical"),
    (DONE, "#3 Token Usage Logging", "COMPLETED. TokenUsage dataclass, token_usage table, logging on all 7 generation call sites, AI Cost widget on dashboard.\nEffort: 2h | Audit: #2 Critical"),
    (DONE, "#4 Consolidate Retry Logic", "COMPLETED. Removed stream-level rate limit retries (was 3x3=9, now client-only max 4).\nEffort: 1h | Audit: #3 Critical"),
    (DONE, "#5 Personality Cache TTL", "COMPLETED. Added 10-min expiry with timestamp to personality prompt cache.\nEffort: 30min | Audit: #11 High"),
    (DONE, "#6 Fix SSE Session Management", "COMPLETED. Verified already fixed - all SSE endpoints create own sessions.\nEffort: 30min | Audit: #9 High"),
    (DONE, "#7 Remove Dead Code", "COMPLETED. Deleted daily_morning_prep() (231 lines), stub endpoints, UI references.\nEffort: 30min | Audit: #22 Medium"),

    # Security (Todo)
    (TODO, "#8 Public API Rate Limiting", "Add rate limiting middleware (slowapi or custom). 60 req/min per IP for public API, 10 req/min for expensive stat endpoints.\n\nFiles: punty/main.py, new rate limit module\nEffort: 3h | Audit: #4 Critical | Category: Security"),
    (TODO, "#9 Session Timeout", "Add 7-day max session lifetime + 24h idle timeout. Check timestamps in auth middleware, clear expired sessions.\n\nFiles: punty/auth.py\nEffort: 1h | Audit: #10 High | Category: Security"),

    # Settlement
    (TODO, "#10 Dead Heat Handling", "Verify TAB dividend already reflects dead heat. Handle case where our tip = dead heat winner (position =1 or DH).\n\nFiles: punty/results/picks.py\nEffort: 2h | Audit: #19 Medium | Category: Settlement"),
    (TODO, "#11 Scratched Horses in Exotics", "Detect scratched runners in exotic legs. Apply TAB rules (scratched = refund if before set time, otherwise counts).\n\nFiles: punty/results/picks.py\nEffort: 2h | Audit: #20 Medium | Category: Settlement"),

    # Prediction Consistency
    (TODO, "#12 Deterministic Pre-Selections", "Pre-calculate best exotic combo (Harville), bet type per runner, stakes via Kelly, Puntys Pick (highest EV). Inject as RECOMMENDED in context.\n\nFiles: punty/context/builder.py, new punty/context/pre_selections.py\nEffort: 3-4 days | Audit: #6 High | Category: Prediction"),
    (TODO, "#13 Pre-Generate Sequence Lanes", "Build Skinny/Balanced/Wide algorithmically from probability rankings. Calculate exact combos and unit price targeting $20 total.\n\nFiles: punty/context/builder.py\nEffort: 2 days | Audit: #12 High | Category: Prediction"),
    (TODO, "#14 Post-Generation Probability Validation", "Validate AI output: Puntys Pick prob >= 15%, win bets >= 10%, stake totals = $20, exotic runners exist, sequence legs valid.\n\nFiles: new punty/validation/content_validator.py\nEffort: 3 days | Audit: #8 High | Category: Prediction"),

    # Cost & Correctness
    (TODO, "#15 Batch Assessment Generation", "Combine all race results into single assessment prompt per meeting (currently 1 per race = ~40/day). Saves ~$77/month.\n\nFiles: punty/memory/assessment.py\nEffort: 2 days | Audit: #7 High | Category: Cost"),
    (TODO, "#16 Increase Strategy Directive Thresholds", "Increase to 30+ bets for LEAN INTO/REDUCE, 50+ for DROP directives.\n\nFiles: punty/memory/strategy.py:693-812\nEffort: 1h | Audit: #18 Medium | Category: Correctness"),
    (TODO, "#17 Context Token Limit", "Count tokens in formatted context, truncate low-ranked runner details when approaching 80K. Preserve top-4 runners per race.\n\nFiles: punty/ai/generator.py, punty/context/builder.py\nEffort: 2 days | Audit: #15 Medium | Category: Cost"),
    (TODO, "#18 Fix Context Snapshot Race Condition", "Concurrent generation for same meeting can collide on context snapshot versioning. Add meeting-level lock or INSERT OR REPLACE with version check.\n\nFiles: punty/context/versioning.py:60-67\nEffort: 1h | Audit: #21 Medium | Category: Correctness"),

    # Infrastructure
    (TODO, "#19 Cache Public Stats Queries", "Cache stats results for 60 seconds in-memory. Invalidate on new settlement.\n\nFiles: punty/public/routes.py:143-400\nEffort: 2h | Audit: #14 Medium | Category: Performance"),
    (TODO, "#20 Alembic Migration System", "Initialize Alembic with current schema as baseline. Migrate existing _add_missing_columns() to Alembic revisions.\n\nFiles: new alembic/ directory, alembic.ini\nEffort: 4h | Audit: #13 Medium | Category: Infrastructure"),
    (TODO, "#21 Staging Environment", "Branch-based staging: /opt/puntyai-staging/ with separate DB and systemd service. staging.punty.ai via Caddy. GitHub Actions deploy from staging branch.\n\nEffort: 4h | Audit: #17 Medium | Category: Infrastructure"),
    (TODO, "#22 Claude Fallback Provider", "Abstract AI client interface. Add Anthropic Claude implementation. Auto-switch on OpenAI failure.\n\nFiles: punty/ai/client.py (refactor), new punty/ai/anthropic_client.py\nEffort: 3-4 days | Audit: #16 Medium | Category: Infrastructure"),

    # Performance & Future
    (TODO, "#23 Consolidate Tips Page Queries", "Tips page makes 8+ DB queries. Consolidate to 3-4 with JOINs.\n\nFiles: punty/public/routes.py:976-988\nEffort: 3h | Audit: #27 Low | Category: Performance"),
    (TODO, "#24 Generation Concurrency Limits", "Peak racing days could have 8 concurrent generations. Add asyncio.Semaphore(3).\n\nFiles: punty/ai/generator.py\nEffort: 2h | Audit: #28 Low | Category: Performance"),
    (TODO, "#25 Zero-Downtime Deployment", "Use gunicorn with --graceful-timeout or start new process, health check, swap traffic, stop old.\n\nFiles: server systemd config\nEffort: 4h | Audit: #26 Low | Category: Infrastructure"),
    (TODO, "#26 Health Check + Non-Root Deploy", "Add /health endpoint (DB ping, uptime, version). Create puntyai user, stop deploying as root. Configure systemd restart on failure.\n\nFiles: punty/main.py, server config\nEffort: 2h | Audit: #31-32 Low | Category: Security"),
    (TODO, "#27 Prompt Versioning", "Hash prompt content, store on Content model. Enables A/B testing later.\n\nFiles: punty/models/content.py, punty/ai/generator.py\nEffort: 1 day | Audit: #25 Low | Category: Infrastructure"),
    (BACKLOG, "#28 Vector DB Migration", "find_similar_situations loads ALL embeddings into memory. Migrate to pgvector, ChromaDB, or FAISS.\n\nFiles: punty/memory/store.py, punty/memory/assessment.py\nEffort: 1-2 weeks | Audit: #23 Low | Category: Infrastructure"),
    (BACKLOG, "#29 Modularize Early Mail Prompt", "388-line monolithic prompt. Split into composable sections for easier maintenance and A/B testing.\n\nFiles: prompts/early_mail.md -> multiple files\nEffort: 1 week | Audit: #24 Low | Category: Maintenance"),
    (BACKLOG, "#30 Race-Type Weight Profiles", "Sprint (<=1200m) and Standard (>1200m) profiles for probability model. Only viable when 200+ picks per profile.\n\nFiles: punty/probability.py, punty/probability_tuning.py\nEffort: 2 weeks | Audit: #30 Low | Category: Prediction"),
]

token = get_token()

for i, (status_id, title, desc) in enumerate(issues):
    payload = json.dumps({
        "project_id": PROJECT,
        "title": title,
        "description": desc,
        "status_id": status_id,
        "sort_order": i,
        "extension_metadata": {}
    })

    result = subprocess.run(
        ["curl", "-s", "-X", "POST", "https://api.vibekanban.com/v1/issues",
         "-H", f"Authorization: Bearer {token}",
         "-H", "Content-Type: application/json",
         "-d", payload],
        capture_output=True, text=True
    )

    try:
        resp = json.loads(result.stdout)
        sid = resp["data"]["simple_id"]
        print(f"  {sid:8s} {title}")
    except Exception as e:
        # Token might have expired
        if "expired" in result.stdout.lower() or "unauthorized" in result.stdout.lower() or "token" in result.stdout.lower():
            token = get_token()
            result = subprocess.run(
                ["curl", "-s", "-X", "POST", "https://api.vibekanban.com/v1/issues",
                 "-H", f"Authorization: Bearer {token}",
                 "-H", "Content-Type: application/json",
                 "-d", payload],
                capture_output=True, text=True
            )
            try:
                resp = json.loads(result.stdout)
                sid = resp["data"]["simple_id"]
                print(f"  {sid:8s} {title} (retried)")
            except:
                print(f"  ERROR: {title} -> {result.stdout[:200]}")
        else:
            print(f"  ERROR: {title} -> {result.stdout[:200]}")

    time.sleep(0.15)

print("\n=== ALL 30 ISSUES CREATED ===")
