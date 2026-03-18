#!/bin/bash
# Weekly LGBM retrain — run via cron or manually
# Pulls the latest live DB from the server, trains with combined
# Proform + live data, and deploys if the new model improves metrics.
#
# Usage:
#   bash scripts/retrain_weekly.sh
#
# Cron example (every Sunday at 3 AM):
#   0 3 * * 0 cd /path/to/PuntyAI && bash scripts/retrain_weekly.sh >> logs/retrain.log 2>&1

set -euo pipefail

cd "$(dirname "$0")/.."

echo "============================================================"
echo "  LGBM Weekly Retrain — $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

python scripts/train_lgbm_v2.py --auto --db-path data/punty_live.db

echo ""
echo "  Retrain complete — $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
