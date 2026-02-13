#!/bin/bash
# PuntyAI Database Backup Script
# Run via cron: 0 2 * * * /opt/puntyai/scripts/backup_db.sh
#
# Creates daily SQLite backups with rotation:
#   - 7 daily backups (last 7 days)
#   - 4 weekly backups (Sunday snapshots, last 4 weeks)

set -euo pipefail

# Configuration
DB_PATH="/opt/puntyai/data/punty.db"
BACKUP_DIR="/opt/puntyai/backups"
DAILY_DIR="${BACKUP_DIR}/daily"
WEEKLY_DIR="${BACKUP_DIR}/weekly"
DAILY_RETAIN=7
WEEKLY_RETAIN=4
LOG_FILE="/opt/puntyai/logs/backup.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Create directories
mkdir -p "$DAILY_DIR" "$WEEKLY_DIR" "$(dirname "$LOG_FILE")"

# Check database exists
if [ ! -f "$DB_PATH" ]; then
    log "ERROR: Database not found at $DB_PATH"
    exit 1
fi

# Generate filename with timestamp
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
DAILY_BACKUP="${DAILY_DIR}/punty_${TIMESTAMP}.db"

# Use SQLite .backup command for consistency (handles WAL mode correctly)
log "Starting backup..."
if sqlite3 "$DB_PATH" ".backup '${DAILY_BACKUP}'"; then
    BACKUP_SIZE=$(du -h "$DAILY_BACKUP" | cut -f1)
    log "Daily backup created: ${DAILY_BACKUP} (${BACKUP_SIZE})"
else
    log "ERROR: SQLite backup failed"
    exit 1
fi

# Compress the backup
if gzip "$DAILY_BACKUP"; then
    COMPRESSED_SIZE=$(du -h "${DAILY_BACKUP}.gz" | cut -f1)
    log "Compressed: ${DAILY_BACKUP}.gz (${COMPRESSED_SIZE})"
else
    log "WARNING: Compression failed, keeping uncompressed backup"
fi

# Weekly backup on Sundays
DOW=$(date '+%u')  # 7 = Sunday
if [ "$DOW" -eq 7 ]; then
    WEEKLY_BACKUP="${WEEKLY_DIR}/punty_weekly_${TIMESTAMP}.db.gz"
    if [ -f "${DAILY_BACKUP}.gz" ]; then
        cp "${DAILY_BACKUP}.gz" "$WEEKLY_BACKUP"
        log "Weekly backup created: ${WEEKLY_BACKUP}"
    fi
fi

# Rotate daily backups (keep last N)
DAILY_COUNT=$(find "$DAILY_DIR" -name "punty_*.db*" -type f | wc -l)
if [ "$DAILY_COUNT" -gt "$DAILY_RETAIN" ]; then
    REMOVE_COUNT=$((DAILY_COUNT - DAILY_RETAIN))
    find "$DAILY_DIR" -name "punty_*.db*" -type f | sort | head -n "$REMOVE_COUNT" | while read -r old_backup; do
        rm "$old_backup"
        log "Rotated daily: $(basename "$old_backup")"
    done
fi

# Rotate weekly backups (keep last N)
WEEKLY_COUNT=$(find "$WEEKLY_DIR" -name "punty_weekly_*.db*" -type f | wc -l)
if [ "$WEEKLY_COUNT" -gt "$WEEKLY_RETAIN" ]; then
    REMOVE_COUNT=$((WEEKLY_COUNT - WEEKLY_RETAIN))
    find "$WEEKLY_DIR" -name "punty_weekly_*.db*" -type f | sort | head -n "$REMOVE_COUNT" | while read -r old_backup; do
        rm "$old_backup"
        log "Rotated weekly: $(basename "$old_backup")"
    done
fi

log "Backup complete. Daily: ${DAILY_COUNT} -> $(find "$DAILY_DIR" -name "punty_*.db*" -type f | wc -l), Weekly: $(find "$WEEKLY_DIR" -name "punty_weekly_*.db*" -type f | wc -l)"
