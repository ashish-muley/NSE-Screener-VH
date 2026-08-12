"""Broker adapters.

Two implementations behind one interface:

* :class:`MockBroker`  — synthetic but realistic data, zero credentials.
* :class:`AngelBroker` — Angel One SmartAPI, **read endpoints only**.

READ-ONLY GUARANTEE
-------------------
This module imports and calls only quote/candle/instrument endpoints. No order
placement, modification, or cancellation method is imported, wrapped, or
referenced anywhere in this codebase.

Both brokers emit the *same* normalised quote shape, which mirrors Angel's
``getMarketData`` FULL response so the screener never branches on mode::

    {
      "token", "symbol", "ltp", "open", "high", "low", "close",
      "netChange", "percentChange", "avgPrice", "tradeVolume",
      "totBuyQuan", "totSellQuan", "exchFeedTime",
      "depth": {"buy": [{"price","quantity","orders"} x5], "sell": [...]}
    }
"""

from __future__ import annotations

import json
import logging
import random
import time as _time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import requests

from .config import DATA_DIR, Settings
from .indicators import smma_last
from .utils import IST, MARKET_CLOSE, MARKET_OPEN, floor_minute, is_trading_day, now_ist, redact, session_start

log = logging.getLogger(__name__)

SCRIP_MASTER_URL = (
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
)
UNIVERSE_PATH = DATA_DIR / "universe.json"

# Angel's getMarketData FULL mode accepts at most 50 tokens per request.
MAX_TOKENS_PER_REQUEST = 50

# --- mock tuning ---
# One-minute bars of synthetic history generated per symbol at startup.
MOCK_HISTORY_BARS = 260
# Crossover candidates start this far below their SMMA120, in percent, so a
# positive drift walks them through the line within a minute or two.
CROSS_SETUP_GAP_PCT = -0.05
# Bars at the end of the path that the shaping ramp acts on.
TILT_BARS = 45


class BrokerError(RuntimeError):
    """Raised for any broker-side failure. Messages are always redacted."""


@dataclass(frozen=True)
class Instrument:
    token: str
    symbol: str
    name: str = ""

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class Candle:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


def trading_minutes_back(count: int, end: datetime | None = None) -> list[datetime]:
    """The last ``count`` NSE trading minutes at or before ``end``, ascending.

    Walks backwards through 09:15–15:30 windows, skipping weekends, so
    synthetic candle timestamps look like a real session rather than a
    continuous 24h stream.
    """
    end = floor_minute(end or now_ist())
    cursor = end
    # Rewind to the most recent valid trading minute.
    guard = 0
    while guard < 30:
        guard += 1
        if not is_trading_day(cursor.date()):
            cursor = datetime.combine(cursor.date() - timedelta(days=1), MARKET_CLOSE, tzinfo=IST)
            continue
        if cursor.time() < MARKET_OPEN:
            cursor = datetime.combine(cursor.date() - timedelta(days=1), MARKET_CLOSE, tzinfo=IST)
            continue
        if cursor.time() > MARKET_CLOSE:
            cursor = datetime.combine(cursor.date(), MARKET_CLOSE, tzinfo=IST)
        break

    out: list[datetime] = []
    while len(out) < count:
        out.append(cursor)
        cursor -= timedelta(minutes=1)
        if cursor.time() < MARKET_OPEN:
            day = cursor.date() - timedelta(days=1)
            while not is_trading_day(day):
                day -= timedelta(days=1)
            cursor = datetime.combine(day, MARKET_CLOSE, tzinfo=IST)
    return list(reversed(out))


class BaseBroker(ABC):
    """The only surface the rest of the app is allowed to touch."""

    mode: str = "base"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.connected = False
        self.last_error: str | None = None

    @abstractmethod
    def connect(self) -> bool:
        """Establish a session. Returns True on success, never raises."""

    @abstractmethod
    def get_universe(self, refresh: bool = False) -> list[Instrument]:
        """Symbol universe to screen."""

    @abstractmethod
    def get_quotes(self, instruments: list[Instrument]) -> dict[str, dict[str, Any]]:
        """Normalised FULL quotes keyed by token. Missing tokens are omitted."""

    @abstractmethod
    def get_candles(self, instrument: Instrument, bars: int) -> list[Candle]:
        """Most recent ``bars`` one-minute candles, oldest first."""

    def intraday_anchor(self) -> datetime:
        """Moment at which cumulative traded volume resets to zero."""
        return session_start()


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------

# Real NSE tickers in the ₹30–500 band. Every price, quantity and candle
# attached to them below is SIMULATED — the dashboard always renders a
# "DEMO DATA" badge in mock mode so this can never be mistaken for real data.
MOCK_SYMBOLS: list[tuple[str, str, float]] = [
    ("NATIONALUM-EQ", "National Aluminium", 192.0),
    ("YESBANK-EQ", "Yes Bank", 34.0),
    ("SUZLON-EQ", "Suzlon Energy", 62.0),
    ("IRFC-EQ", "Indian Railway Finance", 148.0),
    ("PNB-EQ", "Punjab National Bank", 104.0),
    ("SAIL-EQ", "Steel Authority of India", 122.0),
    ("IOC-EQ", "Indian Oil Corporation", 138.0),
    ("NHPC-EQ", "NHPC Limited", 84.0),
    ("BHEL-EQ", "Bharat Heavy Electricals", 232.0),
    ("TATASTEEL-EQ", "Tata Steel", 146.0),
    ("ONGC-EQ", "Oil & Natural Gas Corp", 244.0),
    ("GAIL-EQ", "GAIL India", 188.0),
    ("BANKBARODA-EQ", "Bank of Baroda", 242.0),
    ("CANBK-EQ", "Canara Bank", 108.0),
    ("NMDC-EQ", "NMDC Limited", 68.0),
    ("ASHOKLEY-EQ", "Ashok Leyland", 232.0),
    ("FEDERALBNK-EQ", "Federal Bank", 196.0),
    ("RVNL-EQ", "Rail Vikas Nigam", 348.0),
    ("IRCON-EQ", "Ircon International", 178.0),
    ("HFCL-EQ", "HFCL Limited", 92.0),
    ("ZOMATO-EQ", "Zomato Limited", 268.0),
    ("IDFCFIRSTB-EQ", "IDFC First Bank", 72.0),
    ("UNIONBANK-EQ", "Union Bank of India", 122.0),
    ("PFC-EQ", "Power Finance Corp", 438.0),
    ("RECLTD-EQ", "REC Limited", 486.0),
    ("MOTHERSON-EQ", "Samvardhana Motherson", 152.0),
    ("JPPOWER-EQ", "Jaiprakash Power", 42.0),
    ("TATAPOWER-EQ", "Tata Power", 398.0),
    ("EXIDEIND-EQ", "Exide Industries", 386.0),
    ("MANAPPURAM-EQ", "Manappuram Finance", 178.0),
    ("TRIDENT-EQ", "Trident Limited", 36.0),
    ("SJVN-EQ", "SJVN Limited", 108.0),
]


class _MockSymbolState:
    """Random-walk state for one simulated instrument."""

    __slots__ = (
        "instrument",
        "prev_close",
        "ltp",
        "day_open",
        "day_high",
        "day_low",
        "cum_volume",
        "drift",
        "sigma",
        "liquidity",
        "tot_buy",
        "tot_sell",
        "will_cross",
    )

    def __init__(self, instrument: Instrument, base: float, rng: random.Random, will_cross: bool):
        self.instrument = instrument
        self.prev_close = round(base * rng.uniform(0.97, 1.03), 2)
        self.ltp = round(self.prev_close * rng.uniform(0.985, 1.015), 2)
        self.day_open = self.ltp
        self.day_high = self.ltp
        self.day_low = self.ltp
        self.cum_volume = rng.randint(2_000_000, 40_000_000)
        # Candidates get a decisive upward drift so their SMMA20 climbs through
        # SMMA120 within the first minutes of the demo.
        self.drift = rng.uniform(0.00035, 0.00060) if will_cross else rng.uniform(-0.00012, 0.00012)
        self.sigma = rng.uniform(0.0006, 0.0016)
        self.liquidity = rng.uniform(0.4, 3.0)
        self.tot_buy = rng.randint(400_000, 3_000_000)
        self.tot_sell = rng.randint(400_000, 3_000_000)
        self.will_cross = will_cross


class MockBroker(BaseBroker):
    """Synthetic market data. Same interface, no network, no credentials."""

    mode = "mock"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        cfg = settings.mock
        self._rng = random.Random(int(cfg.get("seed", 7)))
        self._count = min(int(cfg.get("symbol_count", 30)), len(MOCK_SYMBOLS))
        self._states: dict[str, _MockSymbolState] = {}
        self._history: dict[str, list[Candle]] = {}
        self._build_states()

    # -- lifecycle -----------------------------------------------------------
    def connect(self) -> bool:
        self.connected = True
        log.info("MockBroker ready — %d simulated symbols (DEMO DATA)", len(self._states))
        return True

    def _build_states(self) -> None:
        """Build history first, then anchor the live walk to the end of it.

        Generating candles up front (rather than lazily on the first
        ``get_candles`` call) means the live LTP, the day's open/high/low and
        the cumulative volume are all consistent with the history the SMMAs are
        seeded from — no discontinuity on the first render.
        """
        chosen = MOCK_SYMBOLS[: self._count]
        # 3 symbols are engineered to produce a crossover early in the demo.
        cross_idx = set(self._rng.sample(range(len(chosen)), k=min(3, len(chosen))))

        for i, (symbol, name, base) in enumerate(chosen):
            # Keep every simulated price inside the ₹30–500 screening band so
            # the demo has something to show; the liquidity filter still bites.
            base = min(max(base, 32.0), 480.0)
            inst = Instrument(token=str(90000 + i), symbol=symbol, name=name)
            st = _MockSymbolState(inst, base, self._rng, i in cross_idx)

            candles = self._generate_history(st, MOCK_HISTORY_BARS)
            self._history[inst.token] = candles
            self._anchor_to_history(st, candles)
            self._states[inst.token] = st

    def _anchor_to_history(self, st: _MockSymbolState, candles: list[Candle]) -> None:
        if not candles:
            return
        last_day = candles[-1].ts.date()
        today = [c for c in candles if c.ts.date() == last_day]
        st.ltp = candles[-1].close
        st.day_open = today[0].open
        st.day_high = max(c.high for c in today)
        st.day_low = min(c.low for c in today)
        # Previous close sits a small gap away from the open, as it would after
        # an overnight move — so "% change" is not just change-from-open.
        st.prev_close = round(today[0].open * self._rng.uniform(0.990, 1.010), 2)
        # Cumulative volume must equal the sum of the day's bar volumes, or the
        # tracker's backwards reconstruction of the volume windows won't line up.
        st.cum_volume = sum(c.volume for c in today)

    # -- data ----------------------------------------------------------------
    def get_universe(self, refresh: bool = False) -> list[Instrument]:
        return [s.instrument for s in self._states.values()]

    def get_quotes(self, instruments: list[Instrument]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        ts = now_ist()
        for inst in instruments:
            st = self._states.get(inst.token)
            if st is None:
                continue
            self._step(st)
            out[inst.token] = self._quote(st, ts)
        return out

    def _step(self, st: _MockSymbolState) -> None:
        shock = self._rng.gauss(0.0, st.sigma)
        st.ltp = round(max(1.0, st.ltp * (1.0 + st.drift + shock)), 2)
        # Mean-revert hard if the walk drifts out of a plausible daily range.
        band = st.prev_close * 0.08
        st.ltp = round(min(max(st.ltp, st.prev_close - band), st.prev_close + band), 2)
        st.day_high = max(st.day_high, st.ltp)
        st.day_low = min(st.day_low, st.ltp)

        st.cum_volume += int(abs(self._rng.gauss(0, 1)) * 9000 * st.liquidity) + 250

        # Depth totals wander so filter membership changes over a session.
        for attr in ("tot_buy", "tot_sell"):
            cur = getattr(st, attr)
            cur = int(cur * self._rng.uniform(0.97, 1.035))
            setattr(st, attr, min(max(cur, 200_000), 3_200_000))

    def _quote(self, st: _MockSymbolState, ts: datetime) -> dict[str, Any]:
        tick = 0.05
        half_spread = max(tick, round(st.ltp * 0.0004 / tick) * tick)
        bid0 = round(st.ltp - half_spread, 2)
        ask0 = round(st.ltp + half_spread, 2)

        def depth(side: str) -> list[dict[str, Any]]:
            levels = []
            for lvl in range(5):
                price = (bid0 - lvl * tick) if side == "buy" else (ask0 + lvl * tick)
                qty = int(self._rng.uniform(400, 9000) * st.liquidity * (1.0 + lvl * 0.55))
                levels.append(
                    {"price": round(price, 2), "quantity": qty, "orders": self._rng.randint(1, 45)}
                )
            return levels

        change = round(st.ltp - st.prev_close, 2)
        return {
            "token": st.instrument.token,
            "symbol": st.instrument.symbol,
            "name": st.instrument.name,
            "ltp": st.ltp,
            "open": st.day_open,
            "high": st.day_high,
            "low": st.day_low,
            "close": st.prev_close,
            "netChange": change,
            "percentChange": round(change / st.prev_close * 100.0, 2) if st.prev_close else 0.0,
            "avgPrice": round((st.day_high + st.day_low + st.ltp) / 3.0, 2),
            "tradeVolume": st.cum_volume,
            "totBuyQuan": st.tot_buy,
            "totSellQuan": st.tot_sell,
            "exchFeedTime": ts.isoformat(),
            "depth": {"buy": depth("buy"), "sell": depth("sell")},
            "simulated": True,
        }

    def get_candles(self, instrument: Instrument, bars: int) -> list[Candle]:
        return self._history.get(instrument.token, [])[-bars:]

    # -- history generation --------------------------------------------------
    def _generate_history(self, st: _MockSymbolState, bars: int) -> list[Candle]:
        rng = random.Random(f"{st.instrument.token}:history")
        stamps = trading_minutes_back(bars)

        closes = [st.ltp]
        for _ in range(bars - 1):
            closes.append(max(1.0, closes[-1] * (1.0 + rng.gauss(0.0, 0.0013))))

        if st.will_cross:
            closes = self._tilt_to_gap(closes, CROSS_SETUP_GAP_PCT)

        out: list[Candle] = []
        prev = closes[0]
        for ts, close in zip(stamps, closes):
            hi = max(prev, close) * (1.0 + abs(rng.gauss(0, 0.0006)))
            lo = min(prev, close) * (1.0 - abs(rng.gauss(0, 0.0006)))
            volume = int(abs(rng.gauss(0, 1)) * 30_000 * st.liquidity) + 800
            out.append(Candle(ts, round(prev, 2), round(hi, 2), round(lo, 2), round(close, 2), volume))
            prev = close
        return out

    @staticmethod
    def _tilt_to_gap(closes: list[float], target_gap_pct: float) -> list[float]:
        """Bend the tail of a price path until SMMA20 sits ``target_gap_pct``
        below SMMA120.

        Applies a linear ramp over the last :data:`TILT_BARS` closes and
        bisects on its amplitude. The SMMA gap rises monotonically with the
        ramp, so bisection converges — which makes "this symbol will cross
        soon" a guarantee rather than a hand-tuned hope.
        """
        n = len(closes)
        span = min(TILT_BARS, n)
        weights = [0.0] * (n - span) + [i / span for i in range(span)]

        def gap_after(amplitude: float) -> float:
            tilted = [c * (1.0 + amplitude * w) for c, w in zip(closes, weights)]
            fast = smma_last(tilted, 20)
            slow = smma_last(tilted, 120)
            if not fast or not slow:
                return 0.0
            return (fast - slow) / slow * 100.0

        lo, hi = -0.30, 0.30
        if not (gap_after(lo) <= target_gap_pct <= gap_after(hi)):
            return closes  # path is too short to shape; leave it alone
        for _ in range(48):
            mid = (lo + hi) / 2.0
            if gap_after(mid) < target_gap_pct:
                lo = mid
            else:
                hi = mid
        amplitude = (lo + hi) / 2.0
        return [max(1.0, c * (1.0 + amplitude * w)) for c, w in zip(closes, weights)]


# ---------------------------------------------------------------------------
# Angel One SmartAPI
# ---------------------------------------------------------------------------


class AngelBroker(BaseBroker):
    """Live NSE data over Angel One SmartAPI. Read endpoints only."""

    mode = "live"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._api: Any = None
        self._last_call = 0.0
        self._min_interval = float(settings.poll.get("batch_sleep_seconds", 1.1))

    # -- auth ----------------------------------------------------------------
    def connect(self) -> bool:
        creds = self.settings.credentials
        if not creds.complete:
            self.last_error = f"Missing credentials: {', '.join(creds.missing)}"
            log.error("Cannot connect to Angel One — %s", self.last_error)
            return False
        try:
            from SmartApi import SmartConnect  # imported lazily: mock mode needs no SDK
            import pyotp

            api = SmartConnect(api_key=creds.api_key)
            totp = pyotp.TOTP(creds.totp_secret).now()
            resp = api.generateSession(creds.client_code, creds.mpin, totp)
            if not isinstance(resp, dict) or not resp.get("status"):
                msg = (resp or {}).get("message", "unknown error") if isinstance(resp, dict) else "bad response"
                raise BrokerError(f"generateSession failed: {msg}")
            self._api = api
            self.connected = True
            self.last_error = None
            log.info("Connected to Angel One SmartAPI as %s***", creds.client_code[:2])
            return True
        except ImportError as exc:
            self.last_error = f"smartapi-python not installed ({exc})"
            log.error(self.last_error)
        except Exception as exc:  # noqa: BLE001 - never leak a credential
            self.last_error = self._safe(exc)
            log.error("Angel One login failed: %s", self.last_error)
        self.connected = False
        return False

    def _safe(self, exc: object) -> str:
        c = self.settings.credentials
        return redact(exc, (c.api_key, c.client_code, c.mpin, c.totp_secret))

    # -- throttling / retries ------------------------------------------------
    def _throttle(self) -> None:
        gap = _time.monotonic() - self._last_call
        if gap < self._min_interval:
            _time.sleep(self._min_interval - gap)
        self._last_call = _time.monotonic()

    def _call(self, label: str, fn, *args, **kwargs) -> Any:
        """Invoke a read endpoint with throttling and exponential backoff."""
        attempts = int(self.settings.poll.get("retry_attempts", 3))
        base = float(self.settings.poll.get("retry_base_seconds", 1.0))
        cap = float(self.settings.poll.get("retry_max_seconds", 20.0))
        last: str = "no attempt made"
        for attempt in range(attempts):
            try:
                self._throttle()
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last = self._safe(exc)
                wait = min(base * (2**attempt), cap)
                log.warning("%s failed (attempt %d/%d): %s — retrying in %.1fs",
                            label, attempt + 1, attempts, last, wait)
                _time.sleep(wait)
        raise BrokerError(f"{label} failed after {attempts} attempts: {last}")

    # -- universe ------------------------------------------------------------
    def get_universe(self, refresh: bool = False) -> list[Instrument]:
        cached = self._load_universe_cache()
        if cached and not refresh:
            return cached
        try:
            instruments = self._download_universe()
        except Exception as exc:  # noqa: BLE001
            log.error("Scrip master download failed: %s", self._safe(exc))
            if cached:
                log.warning("Falling back to the stale cached universe")
                return cached
            raise BrokerError("Could not build the symbol universe") from None
        self._save_universe_cache(instruments)
        return instruments

    def _load_universe_cache(self) -> list[Instrument]:
        if not UNIVERSE_PATH.exists():
            return []
        max_age = timedelta(hours=float(self.settings.universe.get("refresh_hours", 24)))
        age = datetime.now().timestamp() - UNIVERSE_PATH.stat().st_mtime
        if age > max_age.total_seconds():
            log.info("Cached universe is %.1fh old — refreshing", age / 3600)
            return []
        try:
            payload = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
            return [Instrument(**row) for row in payload.get("instruments", [])]
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not read universe cache: %s", exc)
            return []

    def _save_universe_cache(self, instruments: list[Instrument]) -> None:
        UNIVERSE_PATH.write_text(
            json.dumps(
                {
                    "generated_at": now_ist().isoformat(),
                    "count": len(instruments),
                    "instruments": [i.as_dict() for i in instruments],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        log.info("Saved %d instruments to %s", len(instruments), UNIVERSE_PATH.name)

    def _download_universe(self) -> list[Instrument]:
        log.info("Downloading Angel scrip master (~10MB, first run only)...")
        resp = requests.get(SCRIP_MASTER_URL, timeout=90)
        resp.raise_for_status()
        rows = resp.json()

        eq = [
            Instrument(token=str(r["token"]), symbol=r["symbol"], name=r.get("name", ""))
            for r in rows
            if r.get("exch_seg") == "NSE" and str(r.get("symbol", "")).endswith("-EQ")
        ]
        log.info("Scrip master: %d NSE -EQ instruments", len(eq))
        return self._trim(eq)

    def _trim(self, instruments: list[Instrument]) -> list[Instrument]:
        """Cap the universe. ``max_symbols: 0`` means screen all of NSE."""
        cap = int(self.settings.universe.get("max_symbols", 0))
        if cap <= 0:
            log.info("Universe: all %d NSE -EQ instruments (no cap)", len(instruments))
            self._warn_cycle_time(len(instruments))
            return instruments

        priority = list(self.settings.universe.get("priority_symbols", []))
        by_symbol = {i.symbol: i for i in instruments}

        picked: list[Instrument] = []
        seen: set[str] = set()
        for sym in priority:
            inst = by_symbol.get(sym)
            if inst and inst.token not in seen:
                picked.append(inst)
                seen.add(inst.token)
        for inst in instruments:
            if len(picked) >= cap:
                break
            if inst.token not in seen:
                picked.append(inst)
                seen.add(inst.token)
        log.info("Universe trimmed to %d symbols (cap=%d)", len(picked), cap)
        self._warn_cycle_time(len(picked))
        return picked

    def _warn_cycle_time(self, count: int) -> None:
        """Tell the operator up front how long one full sweep will take.

        Silently taking 45s per cycle would look like a hang; the number is a
        consequence of Angel's published quote rate limit, so it is better
        stated than discovered.
        """
        batches = (count + MAX_TOKENS_PER_REQUEST - 1) // MAX_TOKENS_PER_REQUEST
        seconds = batches * self._min_interval
        level = log.warning if seconds > 15 else log.info
        level(
            "%d symbols = %d quote batches ~= %.0fs per full poll cycle "
            "(Angel allows ~1 quote request/sec)",
            count, batches, seconds,
        )

    # -- quotes --------------------------------------------------------------
    def get_quotes(self, instruments: list[Instrument]) -> dict[str, dict[str, Any]]:
        if not self._api:
            raise BrokerError("Not connected")

        by_token = {i.token: i for i in instruments}
        out: dict[str, dict[str, Any]] = {}
        for batch in _chunks([i.token for i in instruments], MAX_TOKENS_PER_REQUEST):
            try:
                resp = self._call(
                    "getMarketData", self._api.getMarketData, "FULL", {"NSE": list(batch)}
                )
            except BrokerError as exc:
                # Degrade this batch to stale rather than killing the poll loop.
                log.error("Quote batch dropped: %s", exc)
                continue
            data = (resp or {}).get("data") or {}
            for row in data.get("fetched", []) or []:
                token = str(row.get("symbolToken", ""))
                inst = by_token.get(token)
                if inst is None:
                    continue
                out[token] = self._normalise(row, inst)
            for row in data.get("unfetched", []) or []:
                log.debug("Unfetched token %s: %s", row.get("symbolToken"), row.get("message"))
        return out

    @staticmethod
    def _normalise(row: dict[str, Any], inst: Instrument) -> dict[str, Any]:
        depth = row.get("depth") or {}

        def levels(side: str) -> list[dict[str, Any]]:
            return [
                {
                    "price": float(lv.get("price") or 0.0),
                    "quantity": int(lv.get("quantity") or 0),
                    "orders": int(lv.get("orders") or 0),
                }
                for lv in (depth.get(side) or [])
            ]

        return {
            "token": inst.token,
            "symbol": inst.symbol,
            "name": inst.name,
            "ltp": float(row.get("ltp") or 0.0),
            "open": float(row.get("open") or 0.0),
            "high": float(row.get("high") or 0.0),
            "low": float(row.get("low") or 0.0),
            "close": float(row.get("close") or 0.0),
            "netChange": float(row.get("netChange") or 0.0),
            "percentChange": float(row.get("percentChange") or 0.0),
            "avgPrice": float(row.get("avgPrice") or 0.0),
            "tradeVolume": int(row.get("tradeVolume") or 0),
            "totBuyQuan": int(row.get("totBuyQuan") or 0),
            "totSellQuan": int(row.get("totSellQuan") or 0),
            "exchFeedTime": row.get("exchFeedTime") or now_ist().isoformat(),
            "depth": {"buy": levels("buy"), "sell": levels("sell")},
            "simulated": False,
        }

    # -- candles -------------------------------------------------------------
    def get_candles(self, instrument: Instrument, bars: int) -> list[Candle]:
        if not self._api:
            raise BrokerError("Not connected")
        days = int(self.settings.indicators.get("warmup_days", 3))
        to_dt = now_ist()
        from_dt = to_dt - timedelta(days=max(days, 1) + 2)
        params = {
            "exchange": "NSE",
            "symboltoken": instrument.token,
            "interval": "ONE_MINUTE",
            "fromdate": from_dt.strftime("%Y-%m-%d %H:%M"),
            "todate": to_dt.strftime("%Y-%m-%d %H:%M"),
        }
        try:
            resp = self._call("getCandleData", self._api.getCandleData, params)
        except BrokerError as exc:
            log.warning("No history for %s: %s", instrument.symbol, exc)
            return []

        rows = (resp or {}).get("data") or []
        out: list[Candle] = []
        for r in rows:
            try:
                out.append(
                    Candle(
                        ts=datetime.fromisoformat(r[0]).astimezone(IST),
                        open=float(r[1]),
                        high=float(r[2]),
                        low=float(r[3]),
                        close=float(r[4]),
                        volume=int(r[5]),
                    )
                )
            except (ValueError, IndexError, TypeError):
                continue
        return out[-bars:]


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def make_broker(settings: Settings) -> BaseBroker:
    """Pick the broker implementation for the configured mode."""
    if settings.is_mock:
        return MockBroker(settings)
    return AngelBroker(settings)
