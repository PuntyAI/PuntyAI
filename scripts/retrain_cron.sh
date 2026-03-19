#!/bin/bash
# Weekly LGBM retrain cron job
# Runs every Monday at 4:00 AM AEST (before race day generation)
# Crontab entry: 0 4 * * 1 /opt/puntyai/scripts/retrain_cron.sh >> /var/log/puntyai-retrain.log 2>&1

set -e

PUNTY_DIR="/opt/puntyai"
VENV="$PUNTY_DIR/venv/bin/activate"
LOG_PREFIX="[$(date '+%Y-%m-%d %H:%M:%S')]"

echo "$LOG_PREFIX Starting weekly LGBM retrain..."

cd "$PUNTY_DIR"
source "$VENV"

# Pull latest code
git fetch origin master
git checkout -f origin/master
chmod 666 prompts/*.md

# Run retrain with auto mode (compares with existing model, only deploys if improved)
python3 scripts/train_lgbm_v2.py \
    --data-dir /opt/puntyai/proform_data \
    --db-path /opt/puntyai/data/punty.db \
    --auto

RETRAIN_EXIT=$?

if [ $RETRAIN_EXIT -eq 0 ]; then
    echo "$LOG_PREFIX Retrain completed successfully"
    # Restart service to pick up new model
    systemctl restart puntyai
    echo "$LOG_PREFIX Service restarted"
else
    echo "$LOG_PREFIX Retrain exited with code $RETRAIN_EXIT (model not improved or error)"
fi

# Retrain Betfair meta-model (bet selector)
echo "$LOG_PREFIX Starting Betfair meta-model retrain..."
python3 scripts/train_bf_meta.py \
    --data-dir /opt/puntyai/proform_data \
    2>&1

META_EXIT=$?

if [ $META_EXIT -eq 0 ]; then
    echo "$LOG_PREFIX Meta-model retrain completed successfully"
    # Restart to pick up new meta-model
    systemctl restart puntyai
    echo "$LOG_PREFIX Service restarted with new meta-model"
else
    echo "$LOG_PREFIX Meta-model retrain exited with code $META_EXIT"
fi

echo "$LOG_PREFIX Done"
