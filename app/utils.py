"""Small shared helpers: IST clock, market hours, and credential redaction."""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta, timezone

# India has no DST, so a fixed offset is exact and avoids depending on a
# platform IANA tz database (which Windows does not ship).
IST = timezone(timedelta(hours=5, minutes=30), name="IST")

MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)


def now_ist() -> datetime:
    return datetime.now(IST)


def to_ist(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=IST)
    return dt.astimezone(IST)


def is_trading_day(d: date) -> bool:
    """Weekday check only — NSE trading holidays are not modelled.

    Documented as a known limitation: a holiday looks like an open market with
    no ticks, which the UI surfaces as stale data rather than a wrong price.
    """
    return d.weekday() < 5


def market_status(at: datetime | None = None) -> str:
    """Return ``open``, ``pre_open``, or ``closed`` for the given IST moment."""
    at = at or now_ist()
    if not is_trading_day(at.date()):
        return "closed"
    t = at.time()
    if time(9, 0) <= t < MARKET_OPEN:
        return "pre_open"
    if MARKET_OPEN <= t <= MARKET_CLOSE:
        return "open"
    return "closed"


def session_start(at: datetime | None = None) -> datetime:
    """Start of the current (or most recent) trading session, in IST."""
    at = at or now_ist()
    d = at.date()
    if at.time() < MARKET_OPEN:
        d -= timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return datetime.combine(d, MARKET_OPEN, tzinfo=IST)


def floor_minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)


# --- credential hygiene -----------------------------------------------------

_SECRET_KEYS = (
    "api_key",
    "apikey",
    "mpin",
    "password",
    "totp",
    "jwtToken",
    "refreshToken",
    "feedToken",
    "clientcode",
    "client_code",
)


def redact(message: object, secrets: tuple[str, ...] = ()) -> str:
    """Scrub secrets out of any string before it reaches a log or an API body.

    Two passes: exact removal of known secret values, then a regex sweep over
    ``key=value`` / ``"key": "value"`` shapes so an unexpected SDK error string
    cannot leak a token we never explicitly handed it.
    """
    text = str(message)
    for secret in secrets:
        if secret and len(secret) >= 4:
            text = text.replace(secret, "***REDACTED***")
    for key in _SECRET_KEYS:
        text = re.sub(
            rf'("?{key}"?\s*[:=]\s*"?)([^\s,"}}]+)',
            r"\1***REDACTED***",
            text,
            flags=re.IGNORECASE,
        )
    return text


# --- formatting -------------------------------------------------------------


def indian_number(value: float | int | None) -> str:
    """Format a quantity with lakh/crore suffixes, e.g. 1_240_000 -> '12.4L'."""
    if value is None:
        return "—"
    v = float(value)
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1e7:
        return f"{sign}{v / 1e7:.2f}Cr"
    if v >= 1e5:
        return f"{sign}{v / 1e5:.1f}L"
    if v >= 1e3:
        return f"{sign}{v / 1e3:.1f}K"
    return f"{sign}{v:.0f}"
