-- Add PF AI prediction columns to runners table
-- Run on production: sqlite3 /opt/puntyai/data/punty.db < scripts/migrate_pf_ai_columns.sql

ALTER TABLE runners ADD COLUMN pf_ai_score REAL DEFAULT NULL;
ALTER TABLE runners ADD COLUMN pf_ai_price REAL DEFAULT NULL;
ALTER TABLE runners ADD COLUMN pf_ai_rank INTEGER DEFAULT NULL;
ALTER TABLE runners ADD COLUMN pf_assessed_price REAL DEFAULT NULL;
