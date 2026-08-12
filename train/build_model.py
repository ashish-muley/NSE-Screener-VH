"""Offline training for the crossover-quality model. Run once; commit the output.

    python train/build_model.py                # uses MODE from .env
    python train/build_model.py --months 3     # more history
    python train/build_model.py --force        # ignore the parquet cache

What it does
------------
1. Pulls ~3 months of 1-minute candles for ~40 liquid NSE stocks, caching each
   symbol to ``data/cache/*.parquet`` so a re-run costs no API calls.
2. Computes SMMA(20)/SMMA(120) and finds every historical crossover.
3. Labels each one with a triple-barrier rule: over the next 30 bars, did price
   reach +1.0% before −0.7% (inverted for SELL)?
4. Trains LightGBM, wraps it in ``CalibratedClassifierCV(sigmoid)`` so the
   output probability means something.
5. Writes ``data/model.pkl`` and ``data/model_report.txt``.

Feature construction goes through ``app.ml.build_features`` — the same function
the live app calls — so there is exactly one definition of the feature vector.

DATA SOURCE: with ``MODE=live`` and valid credentials this trains on real NSE
history. Without credentials it trains on a simulated regime-switching price
process, and both the report and the model bundle are stamped accordingly.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.broker import Candle, make_broker  # noqa: E402
from app.config import CACHE_DIR, DATA_DIR, configure_logging, settings  # noqa: E402
from app.indicators import adx, atr_pct, pct_slope, smma_series, vwap  # noqa: E402
from app.ml import FEATURE_NAMES, build_features  # noqa: E402
from app.utils import IST, MARKET_OPEN, is_trading_day  # noqa: E402

log = logging.getLogger("train")

# ~40 liquid NSE names. Deliberately hardcoded: the training universe should be
# reproducible and not depend on whatever the live scrip master returns today.
TRAIN_SYMBOLS: list[tuple[str, str]] = [
    ("SBIN-EQ", "3045"), ("TATASTEEL-EQ", "3499"), ("ONGC-EQ", "2475"),
    ("ITC-EQ", "1660"), ("COALINDIA-EQ", "20374"), ("NTPC-EQ", "11630"),
    ("POWERGRID-EQ", "14977"), ("BPCL-EQ", "526"), ("IOC-EQ", "1624"),
    ("GAIL-EQ", "4717"), ("HINDALCO-EQ", "1363"), ("JSWSTEEL-EQ", "11723"),
    ("VEDL-EQ", "3063"), ("SAIL-EQ", "2963"), ("NMDC-EQ", "15332"),
    ("BANKBARODA-EQ", "4668"), ("PNB-EQ", "10666"), ("CANBK-EQ", "10794"),
    ("UNIONBANK-EQ", "10753"), ("IDFCFIRSTB-EQ", "11184"), ("FEDERALBNK-EQ", "1023"),
    ("YESBANK-EQ", "11915"), ("TATAPOWER-EQ", "3426"), ("ASHOKLEY-EQ", "212"),
    ("TATAMOTORS-EQ", "3456"), ("MOTHERSON-EQ", "4204"), ("EXIDEIND-EQ", "676"),
    ("MANAPPURAM-EQ", "19061"), ("RECLTD-EQ", "15355"), ("PFC-EQ", "14299"),
    ("IRFC-EQ", "2029"), ("NHPC-EQ", "17400"), ("SJVN-EQ", "18883"),
    ("SUZLON-EQ", "12018"), ("HFCL-EQ", "21951"), ("TRIDENT-EQ", "13404"),
    ("NATIONALUM-EQ", "6364"), ("BHEL-EQ", "438"), ("RVNL-EQ", "9552"),
    ("JPPOWER-EQ", "13014"),
]

FAST_N, SLOW_N = 20, 120
HORIZON_BARS = 30
TAKE_PROFIT_PCT = 1.0
STOP_LOSS_PCT = 0.7
MIN_HISTORY_BARS = SLOW_N + 40
MODEL_THRESHOLD = 0.55


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------


def cache_path(symbol: str, months: int) -> Path:
    return CACHE_DIR / f"{symbol.replace('-', '_')}_{months}m.parquet"


def fetch_live_candles(broker: Any, symbol: str, token: str, months: int) -> pd.DataFrame:
    """Pull 1-minute candles in 25-day chunks (Angel caps ONE_MINUTE at ~30 days)."""
    from app.broker import Instrument

    inst = Instrument(token=token, symbol=symbol)
    end = datetime.now(IST)
    start = end - timedelta(days=months * 31)
    frames: list[pd.DataFrame] = []

    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=25), end)
        params = {
            "exchange": "NSE",
            "symboltoken": token,
            "interval": "ONE_MINUTE",
            "fromdate": cursor.strftime("%Y-%m-%d %H:%M"),
            "todate": chunk_end.strftime("%Y-%m-%d %H:%M"),
        }
        try:
            resp = broker._call("getCandleData", broker._api.getCandleData, params)
            rows = (resp or {}).get("data") or []
        except Exception as exc:  # noqa: BLE001
            log.warning("  chunk %s..%s failed: %s", cursor.date(), chunk_end.date(), exc)
            rows = []
        if rows:
            frames.append(
                pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
            )
        cursor = chunk_end

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], format="mixed", utc=True).dt.tz_convert(IST)
    return df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)


def simulate_candles(symbol: str, months: int, seed: int) -> pd.DataFrame:
    """Regime-switching synthetic 1-minute series used when no broker is available.

    Days alternate between trending and choppy regimes with persistence. That
    matters: in a pure random walk every crossover is equally worthless and
    there is nothing for a model to learn. Regime structure is a real property
    of markets, so the model has something genuine — if synthetic — to find.
    """
    rng = random.Random(f"{symbol}:{seed}")
    price = rng.uniform(45.0, 460.0)
    rows: list[dict[str, Any]] = []

    day = datetime.now(IST).date() - timedelta(days=int(months * 31))
    sessions = 0
    target_sessions = months * 21
    regime = "chop"

    while sessions < target_sessions:
        if not is_trading_day(day):
            day += timedelta(days=1)
            continue

        # Regimes persist: a trending day is likely to be followed by another.
        if rng.random() < 0.35:
            regime = rng.choices(["up", "down", "chop"], weights=[0.3, 0.28, 0.42])[0]
        # Volatility is calibrated to real NSE mid-cap intraday behaviour:
        # ~0.15-0.30% per 1-minute bar. An earlier, tamer setting made the
        # +1.0% / -0.7% barriers almost unreachable inside 30 bars, which
        # collapsed the label to ~12% positives and produced a degenerate model.
        if regime == "chop":
            drift, vol = 0.0, rng.uniform(0.0020, 0.0032)
        else:
            direction = 1 if regime == "up" else -1
            drift = direction * rng.uniform(0.00012, 0.00030)
            vol = rng.uniform(0.0014, 0.0023)

        ts = datetime.combine(day, MARKET_OPEN, tzinfo=IST)
        price *= 1.0 + rng.gauss(0, 0.006)  # overnight gap
        for bar in range(375):
            ret = rng.gauss(drift, vol)
            # Chop mean-reverts; trends do not. This is what ADX can pick up.
            if regime == "chop" and rows:
                ret -= 0.22 * (price / rows[-1]["close"] - 1.0)
            open_ = price
            price = max(1.0, price * (1.0 + ret))
            high = max(open_, price) * (1 + abs(rng.gauss(0, 0.0005)))
            low = min(open_, price) * (1 - abs(rng.gauss(0, 0.0005)))
            volume = int(abs(ret) * 3.5e6 + abs(rng.gauss(0, 1)) * 9000 + 500)
            # Volume clusters at the open and the close.
            if bar < 20 or bar > 355:
                volume = int(volume * 2.1)
            rows.append({"ts": ts, "open": round(open_, 2), "high": round(high, 2),
                         "low": round(low, 2), "close": round(price, 2), "volume": volume})
            ts += timedelta(minutes=1)

        sessions += 1
        day += timedelta(days=1)

    return pd.DataFrame(rows)


def load_symbol(broker: Any, symbol: str, token: str, months: int,
                simulated: bool, force: bool, seed: int) -> pd.DataFrame:
    path = cache_path(symbol, months)
    if path.exists() and not force:
        try:
            return pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001
            log.warning("  cache unreadable for %s (%s) — refetching", symbol, exc)

    df = simulate_candles(symbol, months, seed) if simulated \
        else fetch_live_candles(broker, symbol, token, months)
    if df.empty:
        return df
    try:
        df.to_parquet(path, index=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("  could not cache %s: %s", symbol, exc)
    return df


# ---------------------------------------------------------------------------
# Labelling and feature extraction
# ---------------------------------------------------------------------------


def triple_barrier(highs: list[float], lows: list[float], entry_idx: int,
                   entry: float, direction: str, session_end: int) -> int | None:
    """1 if the profit barrier is touched before the stop, else 0.

    Bars where both barriers fall inside the same candle are scored as losses —
    we cannot know the intrabar path, and the pessimistic read is the honest one.
    Returns ``None`` when the horizon runs past the end of the session.
    """
    last = entry_idx + HORIZON_BARS
    if last >= session_end:
        return None

    if direction == "BUY":
        tp, sl = entry * (1 + TAKE_PROFIT_PCT / 100), entry * (1 - STOP_LOSS_PCT / 100)
        for j in range(entry_idx + 1, last + 1):
            if lows[j] <= sl:
                return 0
            if highs[j] >= tp:
                return 1
    else:
        tp, sl = entry * (1 - TAKE_PROFIT_PCT / 100), entry * (1 + STOP_LOSS_PCT / 100)
        for j in range(entry_idx + 1, last + 1):
            if highs[j] >= sl:
                return 0
            if lows[j] <= tp:
                return 1
    return 0


def extract_rows(df: pd.DataFrame, symbol: str) -> list[dict[str, Any]]:
    """Every labelled crossover in one symbol's history."""
    if len(df) < MIN_HISTORY_BARS + HORIZON_BARS:
        return []

    ts = pd.to_datetime(df["ts"])
    opens = df["open"].astype(float).tolist()
    highs = df["high"].astype(float).tolist()
    lows = df["low"].astype(float).tolist()
    closes = df["close"].astype(float).tolist()
    volumes = df["volume"].astype(float).tolist()

    fast = smma_series(closes, FAST_N)
    slow = smma_series(closes, SLOW_N)

    dates = ts.dt.date.to_numpy()
    # Index one past the last bar of each session, for the horizon check.
    session_end_at: dict[Any, int] = {}
    for i, d in enumerate(dates):
        session_end_at[d] = i + 1

    # Rolling crossover history, for the whipsaw feature.
    flip_indices: list[int] = []
    prev_sign = 0
    out: list[dict[str, Any]] = []

    for i in range(SLOW_N, len(closes) - 1):
        f, s = fast[i], slow[i]
        if f is None or s is None or s == 0:
            continue
        sign = 1 if f > s else -1
        if prev_sign == 0:
            prev_sign = sign
            continue
        if sign == prev_sign:
            continue
        prev_sign = sign
        flip_indices.append(i)

        if i < MIN_HISTORY_BARS:
            continue

        direction = "BUY" if sign > 0 else "SELL"
        label = triple_barrier(highs, lows, i, closes[i], direction, session_end_at[dates[i]])
        if label is None:
            continue

        # Session-to-date slices, so VWAP and minutes-since-open are per-day.
        day_start = i
        while day_start > 0 and dates[day_start - 1] == dates[i]:
            day_start -= 1
        lo = max(0, i - 200)

        gap_at_next = None
        if fast[i + 1] is not None and slow[i + 1]:
            gap_at_next = (fast[i + 1] - slow[i + 1]) / slow[i + 1] * 100.0

        session_vwap = vwap(highs[day_start:i + 1], lows[day_start:i + 1],
                            closes[day_start:i + 1], volumes[day_start:i + 1])
        dist_vwap = ((closes[i] - session_vwap) / session_vwap * 100.0) if session_vwap else None

        vol_surge = None
        if i >= 35:
            prior = sum(volumes[i - 34:i - 4]) / 30.0
            vol_surge = (sum(volumes[i - 4:i + 1]) / 5.0 / prior) if prior > 0 else None

        rng_bar = highs[i] - lows[i]
        ctx = {
            "direction": direction,
            "smma_gap_pct": gap_at_next,
            "smma20_slope": pct_slope([v for v in fast[lo:i + 1] if v is not None], 5),
            "smma120_slope": pct_slope([v for v in slow[lo:i + 1] if v is not None], 5),
            "adx_14": adx(highs[lo:i + 1], lows[lo:i + 1], closes[lo:i + 1], 14),
            "atr_pct": atr_pct(highs[lo:i + 1], lows[lo:i + 1], closes[lo:i + 1], 14),
            "volume_surge": vol_surge,
            "crossovers_last_60_bars": sum(1 for k in flip_indices[:-1] if k >= i - 60),
            "dist_from_vwap_pct": dist_vwap,
            "minutes_since_open": float(i - day_start),
            "body_ratio": (abs(closes[i] - opens[i]) / rng_bar) if rng_bar > 0 else None,
        }

        row = build_features(ctx)
        row.update({"label": label, "symbol": symbol, "direction": direction,
                    "ts": ts.iloc[i], "price": closes[i]})
        out.append(row)

    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(dataset: pd.DataFrame, simulated: bool, months: int) -> tuple[Any, Any, str]:
    from lightgbm import LGBMClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (brier_score_loss, classification_report,
                                 roc_auc_score)

    dataset = dataset.sort_values("ts").reset_index(drop=True)
    X = dataset[FEATURE_NAMES].astype(float)
    y = dataset["label"].astype(int)

    # Chronological split. A random split would leak the future into the past.
    cut = int(len(dataset) * 0.8)
    X_train, X_test = X.iloc[:cut], X.iloc[cut:]
    y_train, y_test = y.iloc[:cut], y.iloc[cut:]

    params = dict(
        n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=6,
        min_child_samples=40, subsample=0.85, subsample_freq=1,
        colsample_bytree=0.85, reg_lambda=1.0, random_state=42, verbose=-1,
    )

    calibrated = CalibratedClassifierCV(LGBMClassifier(**params), method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train)

    # A separate un-calibrated fit gives SHAP a stable single booster to explain.
    explainer_model = LGBMClassifier(**params).fit(X_train, y_train)

    probs = calibrated.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    baseline = float(y_test.mean())
    taken = probs >= MODEL_THRESHOLD
    model_win = float(y_test[taken].mean()) if taken.sum() else float("nan")
    coverage = float(taken.mean())

    try:
        auc = roc_auc_score(y_test, probs)
    except ValueError:
        auc = float("nan")
    brier = brier_score_loss(y_test, probs)

    importances = sorted(
        zip(FEATURE_NAMES, explainer_model.feature_importances_),
        key=lambda kv: kv[1], reverse=True,
    )

    # A single fixed threshold can be vacuous: if the base rate is well under
    # 50%, a *correctly* calibrated model will rarely emit 0.55+, so coverage
    # collapses to zero and the headline comparison says nothing. The sweep
    # shows where the model actually earns its keep.
    sweep_lines = []
    for thr in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        sel = probs >= thr
        n = int(sel.sum())
        if n == 0:
            sweep_lines.append(f"{thr:>5.2f}{0:>10}{0.0:>11.1f}%{'—':>13}{'—':>14}")
            continue
        win = float(y_test[sel].mean())
        sweep_lines.append(
            f"{thr:>5.2f}{n:>10}{sel.mean() * 100:>10.1f}%{win * 100:>12.2f}%"
            f"{(win - baseline) * 100:>+13.2f}"
        )

    def pct_or_dash(value: float) -> str:
        return "n/a (no signals)" if value != value else f"{value * 100:.2f}%"

    lift_text = "n/a" if model_win != model_win else f"{(model_win - baseline) * 100:+.2f} pp"

    # Cut-points implied by this model's own output distribution, so the
    # dashboard's TAKE/CAUTION/AVOID bands can be re-centred on the base rate
    # instead of on the generic 0.60 / 0.45 defaults.
    rec_take = round(float(np.quantile(probs, 0.90)), 2)
    rec_caution = round(float(np.quantile(probs, 0.55)), 2)
    rec_take_win = float(y_test[probs >= rec_take].mean()) if (probs >= rec_take).sum() else float("nan")

    banner = (
        "!! TRAINED ON SIMULATED DATA — no broker credentials were available.\n"
        "!! The pipeline is real; these numbers describe a synthetic\n"
        "!! regime-switching price process, NOT the NSE. Re-run with\n"
        "!! MODE=live and valid credentials to train on real history.\n"
        if simulated else
        "Trained on real Angel One 1-minute NSE history.\n"
    )

    report = f"""{'=' * 74}
SMMA CROSSOVER QUALITY MODEL — TRAINING REPORT
{'=' * 74}
Generated       : {datetime.now(IST):%Y-%m-%d %H:%M:%S %Z}
Data source     : {'SIMULATED' if simulated else 'Angel One SmartAPI (live)'}
History         : ~{months} months of 1-minute candles
Symbols         : {dataset['symbol'].nunique()}
Crossovers      : {len(dataset)}  (BUY {int((dataset['direction'] == 'BUY').sum())} /
                  SELL {int((dataset['direction'] == 'SELL').sum())})
Train / test    : {len(X_train)} / {len(X_test)}  (chronological 80/20 split)
Label           : +{TAKE_PROFIT_PCT}% before -{STOP_LOSS_PCT}% within {HORIZON_BARS} bars
Positive rate   : {y.mean():.3f} overall, {baseline:.3f} in the test window

{banner}
{'-' * 74}
HEADLINE COMPARISON  (test window)
{'-' * 74}
Baseline — take EVERY crossover      : {baseline * 100:.2f}% win rate  (n={len(y_test)})
Model    — take when p >= {MODEL_THRESHOLD}       : {pct_or_dash(model_win)}  (n={int(taken.sum())})
Improvement                          : {lift_text}
Signal coverage (share taken)        : {coverage * 100:.2f}%

ROC AUC                              : {auc:.4f}      (0.50 = no skill)
Brier score (lower is better)        : {brier:.4f}
Predicted probability range          : {probs.min():.3f} – {probs.max():.3f}
                                       (median {float(np.median(probs)):.3f})

{'-' * 74}
THRESHOLD SWEEP  (the headline number above is one row of this table)
{'-' * 74}
  thr         n  coverage    win rate    vs baseline
""" + "\n".join(sweep_lines) + f"""

Cut-points implied by this model's own distribution:
    ml.take_threshold    : {rec_take:.2f}   (top decile, {pct_or_dash(rec_take_win)} win rate)
    ml.caution_threshold : {rec_caution:.2f}
The shipped config.yaml keeps the generic 0.60 / 0.45 defaults; swap in the
values above if you want the dashboard's verdict bands centred on this model.

{'-' * 74}
CLASSIFICATION REPORT  (threshold 0.50)
{'-' * 74}
{classification_report(y_test, preds, target_names=['fail', 'win'], zero_division=0)}
{'-' * 74}
FEATURE IMPORTANCE  (LightGBM split gain)
{'-' * 74}
""" + "\n".join(f"{name:<28}{int(score):>8}" for name, score in importances) + f"""

{'-' * 74}
HOW TO READ THIS
{'-' * 74}
The model does not predict price. It predicts whether a crossover that has
already fired will reach +{TAKE_PROFIT_PCT}% before -{STOP_LOSS_PCT}%. Its value is in declining
signals, not in finding winners.

Probabilities are calibrated with CalibratedClassifierCV(sigmoid), so "0.42"
should mean roughly a 42% hit rate over many signals — the Brier score above
is the check on that claim. A consequence worth stating plainly: because the
base rate here is {baseline * 100:.0f}%, a correctly calibrated model SHOULD rarely emit
0.60+. Seeing mostly CAUTION and AVOID verdicts in the dashboard is the
calibration working, not a bug. The runtime cut-points live in config.yaml
under `ml.take_threshold` / `ml.caution_threshold` if you want to re-centre
them on this base rate.
"""
    return calibrated, explainer_model, report


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the crossover-quality model.")
    parser.add_argument("--months", type=int, default=3, help="months of history (default 3)")
    parser.add_argument("--symbols", type=int, default=len(TRAIN_SYMBOLS), help="cap the symbol list")
    parser.add_argument("--force", action="store_true", help="ignore the parquet cache")
    parser.add_argument("--seed", type=int, default=11, help="seed for simulated data")
    args = parser.parse_args()

    configure_logging("INFO")

    broker = make_broker(settings)
    simulated = settings.is_mock
    if not simulated:
        if not broker.connect():
            log.error("Live mode requested but login failed: %s", broker.last_error)
            log.error("Set MODE=mock to train on simulated data instead.")
            return 1
    else:
        log.warning("MODE=mock — training on SIMULATED history. The report will say so.")

    symbols = TRAIN_SYMBOLS[: args.symbols]
    frames: list[dict[str, Any]] = []

    for n, (symbol, token) in enumerate(symbols, 1):
        log.info("[%2d/%d] %s", n, len(symbols), symbol)
        df = load_symbol(broker, symbol, token, args.months, simulated, args.force, args.seed)
        if df.empty:
            log.warning("  no candles — skipped")
            continue
        rows = extract_rows(df, symbol)
        log.info("  %d bars -> %d labelled crossovers", len(df), len(rows))
        frames.extend(rows)

    if len(frames) < 200:
        log.error("Only %d crossovers found — too few to train. Try --months 6.", len(frames))
        return 1

    dataset = pd.DataFrame(frames)
    log.info("Dataset: %d crossovers, positive rate %.3f", len(dataset), dataset["label"].mean())

    model, explainer_model, report = train(dataset, simulated, args.months)

    report_path = DATA_DIR / "model_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print("\n" + report)

    import joblib

    bundle = {
        "model": model,
        "base_estimator": explainer_model,
        "features": FEATURE_NAMES,
        "trained_at": datetime.now(IST).isoformat(),
        "simulated": simulated,
        "months": args.months,
        "n_samples": len(dataset),
        "metrics": {"report_path": str(report_path)},
    }
    out = settings.model_path()
    joblib.dump(bundle, out)
    log.info("Saved model -> %s", out)
    log.info("Saved report -> %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
