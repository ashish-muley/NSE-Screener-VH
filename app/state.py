"""Live state.

Two objects live here:

* :class:`SymbolTracker` — everything remembered about one instrument: the tick
  sample ring, the rolling 1-minute bar series, both SMMAs, and crossover
  history.
* :class:`AppState` — the process-wide store. The polling thread writes; the
  HTTP handlers only ever read an already-built snapshot dict, so a slow broker
  can never block a request.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Deque

from .broker import Candle
from .indicators import SMMA, adx, atr_pct, crossovers_in, pct_slope, smma_series, vwap
from .utils import floor_minute, now_ist

log = logging.getLogger(__name__)

# Registering a flip needs the SMMAs to actually separate, and the same symbol
# cannot fire twice inside the cooldown. Without these, a pair of lines sitting
# on top of each other emits dozens of meaningless events per minute.
CROSS_EPSILON_PCT = 0.004
CROSS_COOLDOWN_SECONDS = 45

MAX_BARS = 260


class Bar:
    __slots__ = ("ts", "open", "high", "low", "close", "volume")

    def __init__(self, ts: datetime, o: float, h: float, l: float, c: float, v: int) -> None:
        self.ts, self.open, self.high, self.low, self.close, self.volume = ts, o, h, l, c, v


class SymbolTracker:
    """Rolling per-symbol state, owned exclusively by the polling thread."""

    def __init__(self, token: str, symbol: str, name: str, cfg: dict[str, Any]) -> None:
        self.token = token
        self.symbol = symbol
        self.name = name

        self.fast_n = int(cfg.get("smma_fast", 20))
        self.slow_n = int(cfg.get("smma_slow", 120))
        self.history_minutes = int(cfg.get("sample_history_minutes", 90))
        self.volume_windows = list(cfg.get("volume_windows", [5, 20, 60]))
        self.avg_windows = list(cfg.get("avg_price_windows", [20, 60]))

        # (timestamp, cumulative_volume, ltp) — cumulative, never per-tick sums.
        self.samples: Deque[tuple[datetime, int, float]] = deque(maxlen=6000)
        self.bars: Deque[Bar] = deque(maxlen=MAX_BARS)
        self.cross_times: Deque[datetime] = deque(maxlen=200)

        self.smma_fast = SMMA(self.fast_n)
        self.smma_slow = SMMA(self.slow_n)
        self.smma_ready = False

        self.quote: dict[str, Any] = {}
        self.last_sign = 0
        self.last_cross_at: datetime | None = None
        self.last_signal: str = "NEUTRAL"
        self.ml: dict[str, Any] = {}
        self.stale = False
        self.last_update: datetime | None = None
        self.volume_anchor: datetime | None = None

        self._bar_ts: datetime | None = None
        self._bar_open = 0.0
        self._bar_high = 0.0
        self._bar_low = 0.0
        self._bar_close = 0.0
        self._bar_vol_start = 0

    # -- warm-up -------------------------------------------------------------
    def warm_up(self, candles: list[Candle], quote: dict[str, Any], anchor: datetime) -> bool:
        """Seed SMMAs, the bar series and the sample ring from historical candles.

        Cumulative volume at each historical bar is reconstructed by walking the
        day's per-bar volumes backwards from the quote's cumulative
        ``tradeVolume``. That keeps the sample ring in exactly the same
        (timestamp, cumulative_volume, ltp) form the live path appends to, so
        the traded-quantity windows are populated from the first render instead
        of showing "—" for an hour.
        """
        if not candles:
            self.smma_ready = False
            return False

        self.bars.clear()
        for c in candles[-MAX_BARS:]:
            self.bars.append(Bar(c.ts, c.open, c.high, c.low, c.close, c.volume))

        closes = [b.close for b in self.bars]
        self.smma_fast.seed(closes)
        self.smma_slow.seed(closes)
        self.smma_ready = self.smma_fast.ready and self.smma_slow.ready

        if self.smma_ready:
            fast_series = smma_series(closes, self.fast_n)
            slow_series = smma_series(closes, self.slow_n)
            bars = list(self.bars)
            for idx in crossovers_in(fast_series, slow_series):
                self.cross_times.append(bars[idx].ts)
            # Establish the baseline sign without emitting a synthetic event.
            f, s = self.smma_fast.value, self.smma_slow.value
            if f is not None and s is not None:
                self.last_sign = 1 if f > s else (-1 if f < s else 0)

        self._seed_samples(quote, anchor)
        return self.smma_ready

    def _seed_samples(self, quote: dict[str, Any], anchor: datetime) -> None:
        cum_now = int(quote.get("tradeVolume") or 0)
        if cum_now <= 0:
            return
        cutoff = now_ist() - timedelta(minutes=self.history_minutes)
        today_bars = [b for b in self.bars if b.ts >= anchor]
        if not today_bars:
            return
        self.volume_anchor = today_bars[0].ts

        # Walk backwards: cumulative volume at the close of bar i is the current
        # cumulative total minus everything traded after it.
        running = cum_now
        seeded: list[tuple[datetime, int, float]] = []
        for bar in reversed(today_bars):
            seeded.append((bar.ts + timedelta(minutes=1), max(running, 0), bar.close))
            running -= bar.volume
        seeded.reverse()
        self.samples.extend([s for s in seeded if s[0] >= cutoff])

    # -- live updates --------------------------------------------------------
    def on_quote(self, quote: dict[str, Any], ts: datetime | None = None) -> str | None:
        """Fold one quote into the rolling state. Returns 'BUY'/'SELL' on a flip."""
        ts = ts or now_ist()
        self.quote = quote
        self.stale = False
        self.last_update = ts

        ltp = float(quote.get("ltp") or 0.0)
        cum_volume = int(quote.get("tradeVolume") or 0)
        if ltp <= 0:
            return None

        self.samples.append((ts, cum_volume, ltp))
        self._trim_samples(ts)
        self._roll_bar(ts, ltp, cum_volume)
        return self._detect_cross(ts, ltp)

    def _trim_samples(self, ts: datetime) -> None:
        cutoff = ts - timedelta(minutes=self.history_minutes)
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def _roll_bar(self, ts: datetime, ltp: float, cum_volume: int) -> None:
        minute = floor_minute(ts)
        if self._bar_ts is None:
            self._start_bar(minute, ltp, cum_volume)
            return
        if minute > self._bar_ts:
            volume = max(0, cum_volume - self._bar_vol_start)
            self.bars.append(
                Bar(self._bar_ts, self._bar_open, self._bar_high, self._bar_low, self._bar_close, volume)
            )
            # A closed bar is what the confirmed SMMA advances on.
            self.smma_fast.close_bar(self._bar_close)
            self.smma_slow.close_bar(self._bar_close)
            self.smma_ready = self.smma_fast.ready and self.smma_slow.ready
            self._start_bar(minute, ltp, cum_volume)
            return
        self._bar_high = max(self._bar_high, ltp)
        self._bar_low = min(self._bar_low, ltp)
        self._bar_close = ltp

    def _start_bar(self, minute: datetime, ltp: float, cum_volume: int) -> None:
        self._bar_ts = minute
        self._bar_open = self._bar_high = self._bar_low = self._bar_close = ltp
        self._bar_vol_start = cum_volume

    def _detect_cross(self, ts: datetime, ltp: float) -> str | None:
        fast, slow = self.smma_projected(ltp)
        if fast is None or slow is None or slow == 0:
            return None
        gap_pct = (fast - slow) / slow * 100.0
        if abs(gap_pct) < CROSS_EPSILON_PCT:
            return None  # too tight to call a side

        sign = 1 if gap_pct > 0 else -1
        prev = self.last_sign
        self.last_sign = sign
        if prev == 0 or sign == prev:
            return None
        if self.last_cross_at and (ts - self.last_cross_at).total_seconds() < CROSS_COOLDOWN_SECONDS:
            return None

        self.last_cross_at = ts
        self.cross_times.append(ts)
        self.last_signal = "BUY" if sign > 0 else "SELL"
        return self.last_signal

    # -- derived values ------------------------------------------------------
    def smma_projected(
        self, ltp: float | None = None, extra_bars: int = 0
    ) -> tuple[float | None, float | None]:
        """SMMA values including the in-progress bar.

        ``extra_bars`` rolls the recursion forward that many *additional* bars
        at the same price. The training script measures ``smma_gap_pct`` one bar
        after the cross — at the crossing bar itself the gap is ~0 by
        definition, which would be a useless feature — so signal scoring passes
        ``extra_bars=1`` to reproduce that measurement live.
        """
        if not self.smma_ready:
            return None, None
        price = ltp if ltp is not None else float(self.quote.get("ltp") or 0.0)
        if price <= 0:
            return self.smma_fast.value, self.smma_slow.value

        fast, slow = self.smma_fast.value, self.smma_slow.value
        for _ in range(extra_bars + 1):
            fast = (fast * (self.fast_n - 1) + price) / self.fast_n
            slow = (slow * (self.slow_n - 1) + price) / self.slow_n
        return fast, slow

    def traded_quantity(self, minutes: int) -> int | None:
        """Cumulative volume now minus cumulative volume ``minutes`` ago.

        Returns ``None`` — never a guess — when the sample ring does not yet
        span the window, or when the window reaches back past the point where
        the exchange's cumulative counter reset.
        """
        if len(self.samples) < 2:
            return None
        now_ts, cum_now, _ = self.samples[-1]
        cutoff = now_ts - timedelta(minutes=minutes)
        if self.volume_anchor and cutoff < self.volume_anchor:
            return None
        if self.samples[0][0] > cutoff:
            return None
        past = None
        for sample in self.samples:
            if sample[0] <= cutoff:
                past = sample
            else:
                break
        if past is None:
            return None
        return max(0, cum_now - past[1])

    def average_ltp(self, minutes: int) -> float | None:
        """Simple mean of the LTPs sampled inside the window."""
        if not self.samples:
            return None
        now_ts = self.samples[-1][0]
        cutoff = now_ts - timedelta(minutes=minutes)
        if self.samples[0][0] > cutoff:
            return None  # window not fully covered yet
        window = [s[2] for s in self.samples if s[0] >= cutoff]
        if len(window) < 2:
            return None
        return sum(window) / len(window)

    def crossovers_last(self, minutes: int = 60) -> int:
        cutoff = now_ist() - timedelta(minutes=minutes)
        return sum(1 for t in self.cross_times if t >= cutoff)

    def feature_context(self, extra_bars: int = 0) -> dict[str, Any]:
        """Raw inputs the ML layer turns into a feature vector.

        Pass ``extra_bars=1`` when scoring a crossover the instant it fires, to
        match the training script's "measured 1 bar after cross" convention.
        """
        bars = list(self.bars)
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]
        ltp = float(self.quote.get("ltp") or 0.0)
        fast, slow = self.smma_projected(ltp, extra_bars)

        fast_series = smma_series(closes, self.fast_n) if closes else []
        slow_series = smma_series(closes, self.slow_n) if closes else []

        vol_surge = None
        if len(volumes) >= 35:
            recent = sum(volumes[-5:]) / 5.0
            prior = sum(volumes[-35:-5]) / 30.0
            vol_surge = (recent / prior) if prior > 0 else None

        session_vwap = vwap(highs, lows, closes, volumes) if bars else None
        dist_vwap = None
        if session_vwap and ltp > 0:
            dist_vwap = (ltp - session_vwap) / session_vwap * 100.0

        last_bar = bars[-1] if bars else None
        body_ratio = None
        if last_bar and (last_bar.high - last_bar.low) > 0:
            body_ratio = abs(last_bar.close - last_bar.open) / (last_bar.high - last_bar.low)

        now = now_ist()
        minutes_since_open = max(0.0, (now - now.replace(hour=9, minute=15, second=0, microsecond=0)).total_seconds() / 60.0)

        return {
            "direction": "BUY" if (fast or 0) >= (slow or 0) else "SELL",
            "smma_gap_pct": ((fast - slow) / slow * 100.0) if (fast and slow) else None,
            "smma20_slope": pct_slope(fast_series, 5),
            "smma120_slope": pct_slope(slow_series, 5),
            "adx_14": adx(highs, lows, closes, 14),
            "atr_pct": atr_pct(highs, lows, closes, 14),
            "volume_surge": vol_surge,
            "crossovers_last_60_bars": self.crossovers_last(60),
            "dist_from_vwap_pct": dist_vwap,
            "minutes_since_open": min(minutes_since_open, 375.0),
            "body_ratio": body_ratio,
        }

    # -- serialisation -------------------------------------------------------
    def to_row(self) -> dict[str, Any]:
        q = self.quote
        depth = q.get("depth") or {"buy": [], "sell": []}
        buy = depth.get("buy") or []
        sell = depth.get("sell") or []
        ltp = float(q.get("ltp") or 0.0)
        fast, slow = self.smma_projected(ltp)

        return {
            "token": self.token,
            "symbol": self.symbol.replace("-EQ", ""),
            "name": self.name,
            "ltp": ltp,
            "prev_close": float(q.get("close") or 0.0),
            "change": float(q.get("netChange") or 0.0),
            "change_pct": float(q.get("percentChange") or 0.0),
            "bid": (buy[0]["price"] if buy else None),
            "bid_qty": (buy[0]["quantity"] if buy else None),
            "ask": (sell[0]["price"] if sell else None),
            "ask_qty": (sell[0]["quantity"] if sell else None),
            "total_buy_qty": int(q.get("totBuyQuan") or 0),
            "total_sell_qty": int(q.get("totSellQuan") or 0),
            "vol_5m": self.traded_quantity(5),
            "vol_20m": self.traded_quantity(20),
            "vol_60m": self.traded_quantity(60),
            "avg_20m": self.average_ltp(20),
            "avg_60m": self.average_ltp(60),
            "smma20": fast,
            "smma120": slow,
            "smma_ready": self.smma_ready,
            "smma_gap_pct": ((fast - slow) / slow * 100.0) if (fast and slow) else None,
            "signal": self.last_signal,
            "signal_at": self.last_cross_at.isoformat() if self.last_cross_at else None,
            "crossovers_60m": self.crossovers_last(60),
            "ml": self.ml,
            "depth": {"buy": buy[:5], "sell": sell[:5]},
            "stale": self.stale,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "day_open": float(q.get("open") or 0.0),
            "day_high": float(q.get("high") or 0.0),
            "day_low": float(q.get("low") or 0.0),
            "avg_price": float(q.get("avgPrice") or 0.0),
        }


class AppState:
    """Process-wide state. All mutation happens on the polling thread."""

    def __init__(self, max_signals: int = 50) -> None:
        self._lock = threading.Lock()
        self.trackers: dict[str, SymbolTracker] = {}
        self.signals: Deque[dict[str, Any]] = deque(maxlen=max_signals)
        self._snapshot: dict[str, Any] = {
            "mode": "mock",
            "status": "starting",
            "market_status": "unknown",
            "last_updated": None,
            "universe_size": 0,
            "screened_count": 0,
            "stocks": [],
            "signals": [],
            "health": {},
        }
        self.health: dict[str, Any] = {
            "mode": "mock",
            "connected": False,
            "model_loaded": False,
            "scorer": "unknown",
            "last_poll_ts": None,
            "poll_duration_ms": 0,
            "errors_last_hour": 0,
            "poll_count": 0,
            "warmed_up": 0,
        }
        self._errors: Deque[datetime] = deque(maxlen=500)

    # -- writes (polling thread) --------------------------------------------
    def record_error(self, message: str) -> None:
        self._errors.append(now_ist())
        log.debug("recorded error: %s", message)

    def errors_last_hour(self) -> int:
        cutoff = now_ist() - timedelta(hours=1)
        return sum(1 for t in self._errors if t >= cutoff)

    def add_signal(self, event: dict[str, Any]) -> None:
        with self._lock:
            self.signals.appendleft(event)

    def publish(self, snapshot: dict[str, Any]) -> None:
        with self._lock:
            self._snapshot = snapshot

    def recent_signals(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.signals)[:limit]

    # -- reads (HTTP thread) -------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot

    def health_payload(self) -> dict[str, Any]:
        with self._lock:
            payload = dict(self.health)
        payload["errors_last_hour"] = self.errors_last_hour()
        return payload


state = AppState()
