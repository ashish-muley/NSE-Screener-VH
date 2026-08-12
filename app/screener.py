"""The screening pipeline and the background polling thread.

One cycle:

1. quote the whole universe (batched, throttled inside the broker)
2. filter on price band and both-sided depth
3. warm up any new survivor from 1-minute history (budgeted per cycle)
4. fold the quote into that symbol's tracker; emit a crossover if the sign flips
5. score every screened row with the ML layer
6. publish an immutable snapshot dict for the HTTP handlers to serve

Nothing here raises out of the loop: a failure marks symbols stale, records an
error, and the thread keeps polling.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from .broker import BaseBroker, Instrument
from .config import Settings
from .ml import SignalScorer
from .state import AppState, SymbolTracker
from .utils import market_status, now_ist

log = logging.getLogger(__name__)


class Screener:
    def __init__(
        self,
        settings: Settings,
        broker: BaseBroker,
        state: AppState,
        scorer: SignalScorer,
    ) -> None:
        self.settings = settings
        self.broker = broker
        self.state = state
        self.scorer = scorer

        sc = settings.screener
        self.min_price = float(sc.get("min_price", 30.0))
        self.max_price = float(sc.get("max_price", 500.0))
        self.min_buy_qty = int(sc.get("min_total_buy_qty", 1_000_000))
        self.min_sell_qty = int(sc.get("min_total_sell_qty", 1_000_000))

        ind = settings.indicators
        self.slow_n = int(ind.get("smma_slow", 120))
        self.warmup_bars = max(self.slow_n * 2, 240)

        key = "warmup_per_cycle_mock" if settings.is_mock else "warmup_per_cycle_live"
        self.warmup_budget = int(settings.poll.get(key, 40 if settings.is_mock else 5))

        self.universe: list[Instrument] = []
        self._warm_attempts: dict[str, int] = {}

    # -- setup ---------------------------------------------------------------
    def bootstrap(self) -> bool:
        if not self.broker.connect():
            self.state.record_error(f"connect failed: {self.broker.last_error}")
            return False
        try:
            self.universe = self.broker.get_universe()
        except Exception as exc:  # noqa: BLE001
            log.error("Universe load failed: %s", exc)
            self.state.record_error(str(exc))
            return False
        log.info("Universe loaded: %d symbols", len(self.universe))
        return True

    # -- filtering -----------------------------------------------------------
    def passes(self, quote: dict[str, Any]) -> bool:
        ltp = float(quote.get("ltp") or 0.0)
        if not (self.min_price <= ltp <= self.max_price):
            return False
        if int(quote.get("totBuyQuan") or 0) <= self.min_buy_qty:
            return False
        if int(quote.get("totSellQuan") or 0) <= self.min_sell_qty:
            return False
        return True

    # -- one cycle -----------------------------------------------------------
    def poll_once(self) -> None:
        started = time.perf_counter()
        by_token = {i.token: i for i in self.universe}

        try:
            quotes = self.broker.get_quotes(self.universe)
        except Exception as exc:  # noqa: BLE001 - the loop must survive anything
            log.error("Quote fetch failed: %s", exc)
            self.state.record_error(str(exc))
            self._mark_all_stale()
            self._publish([], time.perf_counter() - started)
            return

        survivors = {t: q for t, q in quotes.items() if self.passes(q)}
        ts = now_ist()
        warmed_this_cycle = 0

        for token, quote in survivors.items():
            inst = by_token.get(token)
            if inst is None:
                continue
            tracker = self.state.trackers.get(token)
            if tracker is None:
                tracker = SymbolTracker(token, inst.symbol, inst.name, self.settings.indicators)
                self.state.trackers[token] = tracker

            if not tracker.smma_ready and warmed_this_cycle < self.warmup_budget:
                if self._warm_up(tracker, inst, quote):
                    warmed_this_cycle += 1

            direction = tracker.on_quote(quote, ts)
            if direction:
                self._emit_signal(tracker, direction, ts)

        # Anything that dropped out of the filter keeps its history but is no
        # longer rendered; anything that vanished from the feed goes stale.
        for token, tracker in self.state.trackers.items():
            if token not in quotes:
                tracker.stale = True

        self._score_all(list(survivors.keys()))
        self._publish(list(survivors.keys()), time.perf_counter() - started)

    def _warm_up(self, tracker: SymbolTracker, inst: Instrument, quote: dict[str, Any]) -> bool:
        attempts = self._warm_attempts.get(inst.token, 0)
        if attempts >= 3:
            return False
        self._warm_attempts[inst.token] = attempts + 1
        try:
            candles = self.broker.get_candles(inst, self.warmup_bars)
        except Exception as exc:  # noqa: BLE001
            log.warning("Warm-up failed for %s: %s", inst.symbol, exc)
            self.state.record_error(f"warmup {inst.symbol}: {exc}")
            return False
        if not candles:
            log.info("No history for %s — rendering as warming up", inst.symbol)
            return False
        ready = tracker.warm_up(candles, quote, self.broker.intraday_anchor())
        log.debug("Warmed %s: %d bars, smma_ready=%s", inst.symbol, len(candles), ready)
        return True

    def _emit_signal(self, tracker: SymbolTracker, direction: str, ts) -> None:
        # extra_bars=1 reproduces the training convention: the SMMA gap is
        # measured one bar after the cross, not at the cross itself.
        ctx = tracker.feature_context(extra_bars=1)
        ctx["direction"] = direction
        score = self.scorer.score(ctx)
        tracker.ml = score.as_dict()
        event = {
            "ts": ts.isoformat(),
            "symbol": tracker.symbol.replace("-EQ", ""),
            "direction": direction,
            "price": float(tracker.quote.get("ltp") or 0.0),
            "probability": round(score.probability, 4),
            "verdict": score.verdict,
            "reasons": score.reasons,
            "top_reason": score.reasons[0] if score.reasons else "",
            "scorer": score.scorer,
        }
        self.state.add_signal(event)
        log.info(
            "%s %s @ %.2f  p=%.2f  %s  (%s)",
            direction,
            event["symbol"],
            event["price"],
            score.probability,
            score.verdict,
            event["top_reason"],
        )

    def _score_all(self, tokens: list[str]) -> None:
        """Score the current SMMA configuration for every screened row.

        For a row that has not crossed over yet this is a forward-looking read
        — "if this crossover fired right now, how would the model rate it" —
        which is what fills the Confidence column. The row's Signal pill stays
        NEUTRAL until an actual flip happens.
        """
        for token in tokens:
            tracker = self.state.trackers.get(token)
            if tracker is None or not tracker.smma_ready:
                continue
            try:
                ctx = tracker.feature_context()
                tracker.ml = self.scorer.score(ctx).as_dict()
            except Exception as exc:  # noqa: BLE001
                log.debug("Scoring %s failed: %s", tracker.symbol, exc)

    def _mark_all_stale(self) -> None:
        for tracker in self.state.trackers.values():
            tracker.stale = True

    def _publish(self, tokens: list[str], elapsed_seconds: float) -> None:
        rows = []
        for token in tokens:
            tracker = self.state.trackers.get(token)
            if tracker is not None:
                rows.append(tracker.to_row())
        rows.sort(key=lambda r: r["symbol"])

        warmed = sum(1 for t in self.state.trackers.values() if t.smma_ready)
        health = {
            "mode": self.settings.mode,
            "connected": bool(self.broker.connected),
            "model_loaded": self.scorer.model is not None,
            "scorer": self.scorer.scorer_name,
            "last_poll_ts": now_ist().isoformat(),
            "poll_duration_ms": round(elapsed_seconds * 1000, 1),
            "errors_last_hour": self.state.errors_last_hour(),
            "poll_count": int(self.state.health.get("poll_count", 0)) + 1,
            "warmed_up": warmed,
            "universe_size": len(self.universe),
            "screened_count": len(rows),
        }
        self.state.health = health

        signals = self.state.recent_signals(50)
        self.state.publish(
            {
                "mode": self.settings.mode,
                "simulated": self.settings.is_mock,
                "status": "ok" if self.broker.connected else "disconnected",
                "market_status": market_status(),
                "last_updated": health["last_poll_ts"],
                "universe_size": len(self.universe),
                "screened_count": len(rows),
                "filters": {
                    "min_price": self.min_price,
                    "max_price": self.max_price,
                    "min_total_buy_qty": self.min_buy_qty,
                    "min_total_sell_qty": self.min_sell_qty,
                },
                "thresholds": {
                    "take": self.scorer.take_threshold,
                    "caution": self.scorer.caution_threshold,
                },
                "stocks": rows,
                "signals": signals,
                "health": health,
            }
        )


class PollerThread(threading.Thread):
    """Daemon thread that drives :meth:`Screener.poll_once` on an interval."""

    def __init__(self, screener: Screener, interval: float) -> None:
        super().__init__(name="poller", daemon=True)
        self.screener = screener
        self.interval = max(0.5, interval)
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:  # pragma: no cover - exercised by running the app
        if not self.screener.bootstrap():
            log.error("Bootstrap failed — poller is idle. Fix credentials or use MODE=mock.")
            self.screener._publish([], 0.0)
            return
        log.info("Polling every %.1fs", self.interval)
        while not self._stop.is_set():
            cycle_started = time.perf_counter()
            try:
                self.screener.poll_once()
            except Exception as exc:  # noqa: BLE001 - last line of defence
                log.exception("Unhandled error in poll cycle: %s", exc)
                self.screener.state.record_error(str(exc))
            sleep_for = self.interval - (time.perf_counter() - cycle_started)
            if sleep_for > 0:
                self._stop.wait(sleep_for)
