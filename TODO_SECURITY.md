# Security Fixes TODO

**Created:** 2026-02-04
**Status:** Pending

## Issues to Address

### 1. API Keys Stored Unencrypted (HIGH)
- **Location:** `app_settings` table in database
- **Issue:** Anthropic, OpenAI, WhatsApp, Twitter API keys stored as plaintext
- **Fix:** Encrypt at rest using Fernet or similar, decrypt on read
- **Note:** Need migration strategy for existing keys

### 2. CSRF Token in Query Params (MEDIUM)
- **Location:** Glory competition forms
- **Issue:** CSRF token passed in URL can leak via referrer headers/logs
- **Fix:** Move to POST body or header-based CSRF

### 3. Error Message Leakage (LOW)
- **Location:** Various API endpoints
- **Issue:** Stack traces may expose internal paths
- **Fix:** Sanitize error responses in production mode

---

*Delete this file once addressed.*
