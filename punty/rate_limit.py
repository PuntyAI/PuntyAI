"""In-memory rate limiting middleware for public API endpoints."""

import logging
import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Rate limit configuration: (max_requests, window_seconds)
RATE_LIMITS = {
    # Public API endpoints - 60 req/min per IP
    "/api/public/": (60, 60),
    # Expensive stat endpoints - 10 req/min per IP
    "/api/public/stats": (10, 60),
    "/api/public/bet-type-stats": (10, 60),
    "/api/public/filter-options": (10, 60),
}

# Endpoints exempt from rate limiting (already behind auth)
EXEMPT_PREFIXES = ("/api/meets", "/api/content", "/api/scheduler",
                   "/api/delivery", "/api/settings", "/api/results",
                   "/api/weather", "/api/csrf-token")


class _TokenBucket:
    """Simple token bucket for rate limiting."""

    __slots__ = ("max_tokens", "refill_rate", "tokens", "last_refill")

    def __init__(self, max_tokens: int, refill_rate: float):
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate  # tokens per second
        self.tokens = float(max_tokens)
        self.last_refill = time.monotonic()

    def consume(self) -> bool:
        """Try to consume a token. Returns True if allowed."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limit public API endpoints by client IP."""

    def __init__(self, app):
        super().__init__(app)
        # {(ip, rule_prefix): TokenBucket}
        self._buckets: dict[tuple[str, str], _TokenBucket] = {}
        self._last_cleanup = time.monotonic()

    def _get_client_ip(self, request) -> str:
        """Extract client IP, respecting X-Forwarded-For from Caddy."""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"

    def _cleanup_stale_buckets(self):
        """Remove buckets that haven't been used in 5 minutes."""
        now = time.monotonic()
        if now - self._last_cleanup < 300:
            return
        self._last_cleanup = now
        stale = [k for k, b in self._buckets.items() if now - b.last_refill > 300]
        for k in stale:
            del self._buckets[k]

    async def dispatch(self, request, call_next):
        path = request.url.path

        # Only rate limit public-facing paths
        if not path.startswith("/api/public/") and not path.startswith("/public"):
            return await call_next(request)

        # Skip auth-protected endpoints
        if any(path.startswith(p) for p in EXEMPT_PREFIXES):
            return await call_next(request)

        # Find the most specific matching rule
        matched_rule = None
        for prefix, limit in RATE_LIMITS.items():
            if path.startswith(prefix) or path == prefix.rstrip("/"):
                if matched_rule is None or len(prefix) > len(matched_rule[0]):
                    matched_rule = (prefix, limit)

        if not matched_rule:
            # Default: 60 req/min for any /public path
            matched_rule = ("/public", (60, 60))

        rule_prefix, (max_req, window) = matched_rule
        client_ip = self._get_client_ip(request)
        bucket_key = (client_ip, rule_prefix)

        # Get or create bucket
        if bucket_key not in self._buckets:
            refill_rate = max_req / window
            self._buckets[bucket_key] = _TokenBucket(max_req, refill_rate)

        bucket = self._buckets[bucket_key]
        if not bucket.consume():
            logger.warning(f"Rate limit hit: {client_ip} on {path} (rule: {rule_prefix})")
            return JSONResponse(
                {"detail": "Rate limit exceeded. Please slow down."},
                status_code=429,
                headers={"Retry-After": str(window)},
            )

        # Periodic cleanup
        self._cleanup_stale_buckets()

        return await call_next(request)
