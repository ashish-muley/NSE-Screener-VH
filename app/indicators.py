"""Pure indicator math — no I/O, no state beyond what is passed in.

Deliberately dependency-free (plain Python lists) so that the offline training
script and the live runtime compute features with *exactly* the same code. A
feature that is calculated differently at train time and at inference time is
the single most common way an ML trading model quietly stops working.

SMMA (Wilder's smoothed moving average) — NOT an EMA::

    SMMA[0] = SMA(prices[0:n])                     # seed
    SMMA[i] = (SMMA[i-1] * (n - 1) + price[i]) / n # recursive

An EMA would use a 2/(n+1) multiplier; Wilder's uses 1/n, which smooths roughly
half as fast. Using the wrong one shifts every crossover by several bars.
"""

from __future__ import annotations

from typing import Sequence


def sma(prices: Sequence[float], n: int) -> float | None:
    if len(prices) < n or n <= 0:
        return None
    return sum(prices[-n:]) / n


def smma_series(prices: Sequence[float], n: int) -> list[float | None]:
    """SMMA aligned to ``prices``; ``None`` until the seed window is complete."""
    out: list[float | None] = [None] * len(prices)
    if n <= 0 or len(prices) < n:
        return out
    value = sum(prices[:n]) / n
    out[n - 1] = value
    for i in range(n, len(prices)):
        value = (value * (n - 1) + prices[i]) / n
        out[i] = value
    return out


def smma_last(prices: Sequence[float], n: int) -> float | None:
    series = smma_series(prices, n)
    return series[-1] if series else None


class SMMA:
    """Incremental SMMA holding the value as of the last *closed* bar.

    ``project(price)`` returns what the SMMA would be if the in-progress bar
    closed at ``price`` right now. The dashboard shows the projected value so
    the line moves with the market instead of jumping once a minute; the
    confirmed value is what gets carried forward when the bar actually closes.
    """

    __slots__ = ("period", "value", "_seed_buffer")

    def __init__(self, period: int) -> None:
        self.period = period
        self.value: float | None = None
        self._seed_buffer: list[float] = []

    @property
    def ready(self) -> bool:
        return self.value is not None

    def seed(self, prices: Sequence[float]) -> bool:
        """Seed from historical closes. Returns True if a value was produced."""
        self.value = smma_last(prices, self.period)
        self._seed_buffer = [] if self.ready else list(prices)[-self.period :]
        return self.ready

    def close_bar(self, price: float) -> float | None:
        """Roll a completed bar into the confirmed value."""
        if self.value is None:
            self._seed_buffer.append(price)
            if len(self._seed_buffer) >= self.period:
                self.value = sum(self._seed_buffer[-self.period :]) / self.period
                self._seed_buffer = []
            return self.value
        self.value = (self.value * (self.period - 1) + price) / self.period
        return self.value

    def project(self, price: float) -> float | None:
        if self.value is None:
            return None
        return (self.value * (self.period - 1) + price) / self.period


def true_ranges(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]) -> list[float]:
    tr: list[float] = []
    for i in range(1, len(closes)):
        tr.append(
            max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        )
    return tr


def _wilder_smooth(values: Sequence[float], n: int) -> list[float]:
    """Wilder's running sum smoothing: first = sum(n), then prev - prev/n + x."""
    if len(values) < n:
        return []
    out = [sum(values[:n])]
    for x in values[n:]:
        out.append(out[-1] - out[-1] / n + x)
    return out


def atr(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], n: int = 14) -> float | None:
    tr = true_ranges(highs, lows, closes)
    smoothed = _wilder_smooth(tr, n)
    if not smoothed:
        return None
    return smoothed[-1] / n


def atr_pct(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], n: int = 14) -> float | None:
    value = atr(highs, lows, closes, n)
    if value is None or not closes or closes[-1] == 0:
        return None
    return value / closes[-1] * 100.0


def adx(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], n: int = 14) -> float | None:
    """Wilder's ADX(14). Needs roughly ``2n`` bars before it returns a value."""
    if len(closes) < 2 * n + 1:
        return None

    plus_dm: list[float] = []
    minus_dm: list[float] = []
    for i in range(1, len(closes)):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm.append(up if (up > down and up > 0) else 0.0)
        minus_dm.append(down if (down > up and down > 0) else 0.0)

    tr = true_ranges(highs, lows, closes)
    tr_s = _wilder_smooth(tr, n)
    plus_s = _wilder_smooth(plus_dm, n)
    minus_s = _wilder_smooth(minus_dm, n)
    if not tr_s or not plus_s or not minus_s:
        return None

    dx: list[float] = []
    for t, p, m in zip(tr_s, plus_s, minus_s):
        if t == 0:
            dx.append(0.0)
            continue
        pdi = 100.0 * p / t
        mdi = 100.0 * m / t
        denom = pdi + mdi
        dx.append(0.0 if denom == 0 else 100.0 * abs(pdi - mdi) / denom)

    if len(dx) < n:
        return None
    # ADX is itself a Wilder average of DX.
    value = sum(dx[:n]) / n
    for x in dx[n:]:
        value = (value * (n - 1) + x) / n
    return value


def vwap(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], volumes: Sequence[float]) -> float | None:
    """Session VWAP from typical price. Falls back to a simple mean if no volume."""
    total_pv = 0.0
    total_v = 0.0
    for h, l, c, v in zip(highs, lows, closes, volumes):
        typical = (h + l + c) / 3.0
        total_pv += typical * v
        total_v += v
    if total_v <= 0:
        return sum(closes) / len(closes) if closes else None
    return total_pv / total_v


def pct_slope(values: Sequence[float | None], lookback: int = 5) -> float | None:
    """Percentage change of a series over the prior ``lookback`` points."""
    clean = [v for v in values if v is not None]
    if len(clean) <= lookback:
        return None
    past = clean[-(lookback + 1)]
    now = clean[-1]
    if past == 0:
        return None
    return (now - past) / abs(past) * 100.0


def crossovers_in(fast: Sequence[float | None], slow: Sequence[float | None]) -> list[int]:
    """Indices where sign(fast - slow) flips. Used for the whipsaw counter."""
    flips: list[int] = []
    prev_sign = 0
    for i, (f, s) in enumerate(zip(fast, slow)):
        if f is None or s is None:
            continue
        sign = 1 if f > s else (-1 if f < s else 0)
        if sign != 0 and prev_sign != 0 and sign != prev_sign:
            flips.append(i)
        if sign != 0:
            prev_sign = sign
    return flips
