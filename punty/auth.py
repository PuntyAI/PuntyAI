"""Google OAuth authentication for PuntyAI."""

import hashlib
import hmac
import logging
import secrets

from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from punty.config import settings

logger = logging.getLogger(__name__)

# --- OAuth client -----------------------------------------------------------

oauth = OAuth()

if settings.google_client_id:
    oauth.register(
        name="google",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )

# --- Auth middleware ----------------------------------------------------------

# Paths that never require login
PUBLIC_PATHS = {"/login", "/login/google", "/auth/callback", "/health"}
PUBLIC_PREFIXES = ("/static/", "/api/webhook/", "/api/public/", "/public")

# Public site paths (served on punty.ai, not app.punty.ai)
PUBLIC_SITE_PATHS = {"/", "/about", "/how-it-works", "/contact", "/terms", "/privacy", "/tips", "/sitemap.xml", "/robots.txt"}
# Also allow /tips/* paths for individual meeting pages
PUBLIC_SITE_PREFIXES_EXTRA = ("/tips/",)
PUBLIC_SITE_HOSTS = {"punty.ai", "www.punty.ai", "localhost:8000", "127.0.0.1:8000"}


class AuthMiddleware(BaseHTTPMiddleware):
    """Redirect unauthenticated users to login page."""

    async def dispatch(self, request, call_next):
        path = request.url.path
        host = request.headers.get("host", "").lower()

        # Allow public paths
        if path in PUBLIC_PATHS or any(path.startswith(p) for p in PUBLIC_PREFIXES):
            return await call_next(request)

        # Allow public site paths on public domains (punty.ai, not app.punty.ai)
        # This lets the public website work without authentication
        is_public_host = any(host.startswith(h) or host == h for h in PUBLIC_SITE_HOSTS)
        if is_public_host:
            if path in PUBLIC_SITE_PATHS:
                return await call_next(request)
            # Also allow /tips/* paths for meeting detail pages
            if any(path.startswith(p) for p in PUBLIC_SITE_PREFIXES_EXTRA):
                return await call_next(request)

        # Check session
        user = request.session.get("user")
        if user:
            return await call_next(request)

        # Not authenticated
        if path.startswith("/api/"):
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)
        return RedirectResponse(url="/login")


# --- CSRF middleware ----------------------------------------------------------

CSRF_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
CSRF_EXEMPT_PREFIXES = ("/api/webhook/",)


def _get_session_csrf_secret(request) -> str:
    """Get or create a per-session CSRF secret."""
    csrf_secret = request.session.get("_csrf_secret")
    if not csrf_secret:
        csrf_secret = secrets.token_hex(32)
        request.session["_csrf_secret"] = csrf_secret
    return csrf_secret


def _generate_csrf_token(session_secret: str) -> str:
    """Generate a CSRF token tied to the session."""
    nonce = secrets.token_hex(16)
    sig = hmac.new(
        session_secret.encode(), nonce.encode(), hashlib.sha256
    ).hexdigest()[:32]
    return f"{nonce}.{sig}"


def _verify_csrf_token(session_secret: str, token: str) -> bool:
    """Verify a CSRF token."""
    if not token or "." not in token:
        return False
    nonce, sig = token.split(".", 1)
    expected = hmac.new(
        session_secret.encode(), nonce.encode(), hashlib.sha256
    ).hexdigest()[:32]
    return hmac.compare_digest(sig, expected)


class CSRFMiddleware(BaseHTTPMiddleware):
    """Verify CSRF token on state-changing requests."""

    async def dispatch(self, request, call_next):
        # Safe methods don't need CSRF
        if request.method in CSRF_SAFE_METHODS:
            return await call_next(request)

        path = request.url.path

        # Exempt webhook endpoints (external callbacks)
        if any(path.startswith(p) for p in CSRF_EXEMPT_PREFIXES):
            return await call_next(request)

        # Exempt auth callback (OAuth redirect)
        if path in {"/auth/callback"}:
            return await call_next(request)

        # Check token from header (htmx/fetch) only - query params can leak in logs/referer
        token = request.headers.get("X-CSRF-Token")

        csrf_secret = request.session.get("_csrf_secret", "")
        if not csrf_secret or not _verify_csrf_token(csrf_secret, token or ""):
            if path.startswith("/api/"):
                return JSONResponse({"detail": "CSRF validation failed"}, status_code=403)
            return RedirectResponse(url="/login?error=Session+expired.+Please+try+again.")

        return await call_next(request)


# --- Auth routes --------------------------------------------------------------

router = APIRouter()

from pathlib import Path

_templates_dir = Path(__file__).parent / "web" / "templates"
_templates = Jinja2Templates(directory=_templates_dir)


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    error = request.query_params.get("error")
    return _templates.TemplateResponse("login.html", {"request": request, "error": error})


@router.get("/privacy", response_class=HTMLResponse)
async def privacy_page(request: Request):
    return _templates.TemplateResponse("privacy.html", {"request": request})


@router.get("/login/google")
async def login_google(request: Request):
    if not settings.google_client_id:
        return RedirectResponse(url="/login?error=Google+OAuth+not+configured")
    redirect_uri = request.url_for("auth_callback")
    return await oauth.google.authorize_redirect(request, str(redirect_uri))


@router.get("/auth/callback")
async def auth_callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception as e:
        logger.error(f"OAuth error: {e}")
        return RedirectResponse(url="/login?error=Authentication+failed")

    user_info = token.get("userinfo", {})
    email = user_info.get("email", "")

    allowed = {e.strip().lower() for e in settings.allowed_emails.split(",")}
    if email.lower() not in allowed:
        logger.warning(f"Rejected login from {email}")
        return RedirectResponse(url="/login?error=Access+denied.+Not+authorised.")

    request.session["user"] = {
        "email": email,
        "name": user_info.get("name", ""),
        "picture": user_info.get("picture", ""),
    }
    logger.info(f"User logged in: {email}")
    return RedirectResponse(url="/", status_code=302)


@router.get("/api/csrf-token")
async def csrf_token(request: Request):
    """Return a fresh CSRF token for JS to use."""
    csrf_secret = _get_session_csrf_secret(request)
    token = _generate_csrf_token(csrf_secret)
    return {"csrf_token": token}


@router.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login")
