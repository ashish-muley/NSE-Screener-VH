"""Configuration loading.

Two sources, deliberately separated:

* ``config.yaml`` — tunable, non-secret runtime settings. Committed.
* ``.env``        — credentials and the mock/live switch. Never committed.

Nothing in this module ever logs or returns a credential value.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
STATIC_DIR = ROOT / "static"

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Credentials:
    """Angel One SmartAPI credentials pulled from the environment.

    ``__repr__`` is overridden so a credential can never leak into a log line,
    a traceback, or an API response through accidental interpolation.
    """

    api_key: str = ""
    client_code: str = ""
    mpin: str = ""
    totp_secret: str = ""

    @property
    def complete(self) -> bool:
        return all([self.api_key, self.client_code, self.mpin, self.totp_secret])

    @property
    def missing(self) -> list[str]:
        names = {
            "ANGEL_API_KEY": self.api_key,
            "ANGEL_CLIENT_CODE": self.client_code,
            "ANGEL_MPIN": self.mpin,
            "ANGEL_TOTP_SECRET": self.totp_secret,
        }
        return [k for k, v in names.items() if not v]

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"Credentials(complete={self.complete}, missing={self.missing})"

    __str__ = __repr__


@dataclass
class Settings:
    """Everything the app needs to run, resolved once at import time."""

    mode: str = "mock"
    log_level: str = "INFO"
    host: str = "127.0.0.1"
    port: int = 8000
    credentials: Credentials = field(default_factory=Credentials)
    raw: dict[str, Any] = field(default_factory=dict)

    # -- section accessors ---------------------------------------------------
    @property
    def screener(self) -> dict[str, Any]:
        return self.raw.get("screener", {})

    @property
    def poll(self) -> dict[str, Any]:
        return self.raw.get("poll", {})

    @property
    def universe(self) -> dict[str, Any]:
        return self.raw.get("universe", {})

    @property
    def indicators(self) -> dict[str, Any]:
        return self.raw.get("indicators", {})

    @property
    def ml(self) -> dict[str, Any]:
        return self.raw.get("ml", {})

    @property
    def mock(self) -> dict[str, Any]:
        return self.raw.get("mock", {})

    @property
    def is_mock(self) -> bool:
        return self.mode == "mock"

    @property
    def poll_interval(self) -> float:
        key = "interval_seconds_mock" if self.is_mock else "interval_seconds_live"
        return float(self.poll.get(key, 2 if self.is_mock else 5))

    def model_path(self) -> Path:
        return ROOT / str(self.ml.get("model_path", "data/model.pkl"))


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        log.warning("config.yaml not found at %s — falling back to defaults", path)
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_settings() -> Settings:
    """Read ``.env`` + ``config.yaml`` and return the resolved settings."""
    load_dotenv(ROOT / ".env")

    raw = _load_yaml(CONFIG_PATH)
    server = raw.get("server", {})

    mode = (os.getenv("MODE") or "mock").strip().lower()
    if mode not in ("mock", "live"):
        log.warning("Unknown MODE=%r — defaulting to 'mock'", mode)
        mode = "mock"

    creds = Credentials(
        api_key=(os.getenv("ANGEL_API_KEY") or "").strip(),
        client_code=(os.getenv("ANGEL_CLIENT_CODE") or "").strip(),
        mpin=(os.getenv("ANGEL_MPIN") or "").strip(),
        totp_secret=(os.getenv("ANGEL_TOTP_SECRET") or "").strip(),
    )

    # A placeholder left over from .env.example is not a credential.
    if creds.complete and "your_" in creds.api_key:
        log.warning("ANGEL_API_KEY still looks like the .env.example placeholder")

    settings = Settings(
        mode=mode,
        log_level=(os.getenv("LOG_LEVEL") or "INFO").strip().upper(),
        host=os.getenv("HOST") or str(server.get("host", "127.0.0.1")),
        port=int(os.getenv("PORT") or server.get("port", 8000)),
        credentials=creds,
        raw=raw,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return settings


def configure_logging(level: str = "INFO") -> None:
    # Windows consoles default to cp1252 and mangle (or raise on) the ₹ / — we
    # use in messages. Force UTF-8 on the streams before attaching handlers.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover - non-tty streams
            pass

    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s  %(levelname)-7s %(name)-18s %(message)s",
        datefmt="%H:%M:%S",
    )
    # These are chatty and tell us nothing useful.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("smartapi").setLevel(logging.WARNING)


settings = load_settings()
