"""Email/password authentication for Group One Glory."""

import hashlib
import hmac
import logging
import secrets
import uuid
from datetime import datetime, timedelta

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from punty.models.glory import G1User

# Try to import bcrypt, fall back to passlib if not available
try:
    import bcrypt
    _USE_BCRYPT = True
except ImportError:
    _USE_BCRYPT = False

logger = logging.getLogger(__name__)

# Session key prefix for glory users (separate from main Punty auth)
GLORY_USER_SESSION_KEY = "glory_user"
GLORY_CSRF_SECRET_KEY = "glory_csrf_secret"

# Session duration
SESSION_DURATION_DAYS = 30


def hash_password(password: str) -> str:
    """Hash a password using bcrypt or fallback to PBKDF2."""
    if _USE_BCRYPT:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    else:
        # Fallback to PBKDF2 with SHA256
        salt = secrets.token_hex(16)
        key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
        return f"pbkdf2:{salt}:{key.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    if _USE_BCRYPT and not password_hash.startswith("pbkdf2:"):
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    elif password_hash.startswith("pbkdf2:"):
        # PBKDF2 fallback
        _, salt, stored_key = password_hash.split(":")
        key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
        return hmac.compare_digest(key.hex(), stored_key)
    return False


def generate_user_id() -> str:
    """Generate a unique user ID."""
    return f"g1u_{uuid.uuid4().hex[:16]}"


async def get_user_by_email(db: AsyncSession, email: str) -> G1User | None:
    """Get a user by their email address."""
    result = await db.execute(
        select(G1User).where(G1User.email == email.lower())
    )
    return result.scalar_one_or_none()


async def get_user_by_id(db: AsyncSession, user_id: str) -> G1User | None:
    """Get a user by their ID."""
    result = await db.execute(
        select(G1User).where(G1User.id == user_id)
    )
    return result.scalar_one_or_none()


async def create_user(
    db: AsyncSession,
    email: str,
    password: str,
    display_name: str,
    is_admin: bool = False
) -> G1User:
    """Create a new user."""
    user = G1User(
        id=generate_user_id(),
        email=email.lower(),
        password_hash=hash_password(password),
        display_name=display_name,
        is_admin=is_admin,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    logger.info(f"Created new user: {email}")
    return user


async def authenticate_user(db: AsyncSession, email: str, password: str) -> G1User | None:
    """Authenticate a user by email and password."""
    user = await get_user_by_email(db, email)
    if user and verify_password(password, user.password_hash):
        logger.info(f"User authenticated: {email}")
        return user
    logger.warning(f"Failed authentication attempt for: {email}")
    return None


def login_user(request: Request, user: G1User) -> None:
    """Log a user in by setting session data."""
    request.session[GLORY_USER_SESSION_KEY] = {
        "id": user.id,
        "email": user.email,
        "display_name": user.display_name,
        "is_admin": user.is_admin,
    }


def logout_user(request: Request) -> None:
    """Log a user out by clearing glory session data."""
    if GLORY_USER_SESSION_KEY in request.session:
        del request.session[GLORY_USER_SESSION_KEY]
    if GLORY_CSRF_SECRET_KEY in request.session:
        del request.session[GLORY_CSRF_SECRET_KEY]


def get_current_user(request: Request) -> dict | None:
    """Get the current logged-in user from session."""
    return request.session.get(GLORY_USER_SESSION_KEY)


def require_user(request: Request) -> dict:
    """Get the current user or raise 401."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def require_admin(request: Request) -> dict:
    """Get the current user and verify admin status or raise 403."""
    user = require_user(request)
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# --- CSRF for Glory routes ----------------------------------------------------

def _get_glory_csrf_secret(request: Request) -> str:
    """Get or create a per-session CSRF secret for glory routes."""
    csrf_secret = request.session.get(GLORY_CSRF_SECRET_KEY)
    if not csrf_secret:
        csrf_secret = secrets.token_hex(32)
        request.session[GLORY_CSRF_SECRET_KEY] = csrf_secret
    return csrf_secret


def generate_csrf_token(request: Request) -> str:
    """Generate a CSRF token tied to the session."""
    session_secret = _get_glory_csrf_secret(request)
    nonce = secrets.token_hex(16)
    sig = hmac.new(
        session_secret.encode(), nonce.encode(), hashlib.sha256
    ).hexdigest()[:32]
    return f"{nonce}.{sig}"


def verify_csrf_token(request: Request, token: str) -> bool:
    """Verify a CSRF token."""
    session_secret = request.session.get(GLORY_CSRF_SECRET_KEY, "")
    if not session_secret or not token or "." not in token:
        return False
    nonce, sig = token.split(".", 1)
    expected = hmac.new(
        session_secret.encode(), nonce.encode(), hashlib.sha256
    ).hexdigest()[:32]
    return hmac.compare_digest(sig, expected)


# --- Auth Middleware for Glory routes -----------------------------------------

# Paths within /group1glory/ that don't require login
GLORY_PUBLIC_PATHS = {
    "/group1glory/",
    "/group1glory/login",
    "/group1glory/register",
    "/group1glory/about",
}

# Prefixes that are public
GLORY_PUBLIC_PREFIXES = (
    "/group1glory/api/auth/",
    "/group1glory/static/",
)

# Safe HTTP methods that don't need CSRF
CSRF_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


class GloryAuthMiddleware(BaseHTTPMiddleware):
    """Auth middleware for Group One Glory routes."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Only handle glory routes
        if not path.startswith("/group1glory"):
            return await call_next(request)

        # Allow public paths
        if path in GLORY_PUBLIC_PATHS or any(path.startswith(p) for p in GLORY_PUBLIC_PREFIXES):
            return await call_next(request)

        # Check session
        user = get_current_user(request)
        if user:
            return await call_next(request)

        # Not authenticated
        if "/api/" in path:
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)
        return RedirectResponse(url="/group1glory/login")


class GloryCSRFMiddleware(BaseHTTPMiddleware):
    """CSRF middleware for Group One Glory routes."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Only handle glory routes
        if not path.startswith("/group1glory"):
            return await call_next(request)

        # Safe methods don't need CSRF
        if request.method in CSRF_SAFE_METHODS:
            return await call_next(request)

        # Auth endpoints are exempt (login/register POST)
        if path.startswith("/group1glory/api/auth/"):
            return await call_next(request)

        # Check token from header or query param
        token = (
            request.headers.get("X-CSRF-Token")
            or request.query_params.get("_csrf")
        )

        if not verify_csrf_token(request, token or ""):
            if "/api/" in path:
                return JSONResponse({"detail": "CSRF validation failed"}, status_code=403)
            return RedirectResponse(
                url="/group1glory/login?error=Session+expired.+Please+try+again."
            )

        return await call_next(request)
