# NSE Real-Time Stock Screener with ML Signal Analysis

A read-only NSE equity screener. A background thread polls market data, filters on
price and depth, computes Wilder-smoothed moving averages, detects SMMA(20)/SMMA(120)
crossovers, scores each one with a calibrated LightGBM model, and serves the whole
picture as JSON. A single hand-written HTML page renders it as a live trading terminal.

It runs with **zero credentials** in mock mode.

> **Read-only.** This application consumes quote, candle, and instrument endpoints
> only. No order placement, modification, or cancellation method is imported,
> wrapped, or referenced anywhere in this codebase.

---

## Screenshot

![Dashboard](docs/screenshot.png)

<!-- Replace docs/screenshot.png with a capture of the running dashboard.
     Run `python -m app.server`, open http://127.0.0.1:8000, and screenshot it. -->

---

## Features

| | |
|---|---|
| **Live screening** | Price band + both-sided depth filter, re-evaluated every poll cycle |
| **Market depth** | Best bid/ask with quantities in the grid; full 5-level ladder on row expand |
| **Traded quantity** | Exchange-traded quantity over 5 / 20 / 60 minutes, from cumulative-volume deltas |
| **Average LTP** | Mean sampled LTP over 20 / 60 minutes |
| **SMMA** | Wilder's SMMA(20) and SMMA(120) on 1-minute candles, seeded from history |
| **Crossover detection** | Sign flips emit BUY/SELL events; last 50 kept in memory |
| **ML scoring** | Calibrated LightGBM + SHAP, with a rule-based fallback if no model is present |
| **Plain-English reasons** | The top 3 factors pushing a prediction down, in words |
| **Price-band control** | Narrow the displayed band live from the header; defaults to the configured ₹30–500 |
| **Sortable grid** | Click any column header; click a row for depth, indicators, and every reason |
| **Mock mode** | 30 simulated symbols, engineered crossovers, badged `DEMO DATA` |
| **Honest gaps** | Insufficient history renders `—` or `warming up`, never a fabricated number |

---

## Architecture

```
                          ┌──────────────────────────────┐
                          │  Angel One SmartAPI (READ)   │
                          │  getMarketData · getCandle   │
                          │  Data · scrip master JSON    │
                          └──────────────┬───────────────┘
                                         │  MODE=live
      ┌──────────────────────────────────┴───────────────┐
      │                broker.py  (one interface)        │
      │   AngelBroker            │          MockBroker   │
      │   throttled, retried     │   synthetic, seeded   │
      └──────────────────────────┬───────────────────────┘
                                 │  normalised quotes + candles
                                 ▼
            ┌────────────────────────────────────────┐
            │            screener.py                 │
            │  1. quote universe (batches of 50)     │
            │  2. filter price + depth               │
            │  3. warm up from 1-min history         │
            │  4. fold quote → tracker, detect flip  │
            │  5. score with ml.py                   │
            │  6. publish an immutable snapshot      │
            └───────┬────────────────────┬───────────┘
                    │                    │
                    ▼                    ▼
        ┌───────────────────┐   ┌──────────────────────┐
        │    state.py       │   │       ml.py          │
        │ SymbolTracker ×N  │   │ LightGBM + SHAP      │
        │  · sample ring    │   │   ↓ fallback         │
        │  · 1-min bars     │   │ rule-based scorer    │
        │  · SMMA 20 / 120  │   └──────────────────────┘
        │ AppState (locked) │
        └─────────┬─────────┘
                  │  snapshot dict, pre-built
                  ▼
        ┌───────────────────┐        ┌──────────────────────┐
        │    server.py      │◄───────│  static/index.html   │
        │  GET /            │  2s    │  vanilla JS + CSS    │
        │  GET /api/snapshot│  poll  │  no build step       │
        │  GET /api/health  │        └──────────────────────┘
        └───────────────────┘
```

The polling thread is the only writer. HTTP handlers return an already-assembled
dict, so a slow or failing broker can never block a request.

---

## Quick Start

### Mock mode — 3 commands, no credentials

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
python -m app.server
```

Open <http://127.0.0.1:8000>. The header shows an amber **DEMO DATA** badge.

On Linux/macOS the activate line is `source .venv/bin/activate`.

### Live mode — 5 steps

```bash
cp .env.example .env          # 1. create your env file
#                               2. edit .env: set MODE=live and fill in the four
#                                  ANGEL_* values (see below)
pip install -r requirements.txt
python -m app.server --refresh-universe
```

`.env` needs all four of these, or the app logs which are missing and stays idle:

```ini
MODE=live
ANGEL_API_KEY=...          # from smartapi.angelbroking.com
ANGEL_CLIENT_CODE=...      # e.g. A123456
ANGEL_MPIN=...             # login MPIN, not your web password
ANGEL_TOTP_SECRET=...      # base32 string behind the 2FA QR code
```

Also required on Angel's side: your SmartAPI app must have the machine's public
IP whitelisted, or `generateSession` fails regardless of a correct key.

`.env` is git-ignored. The first live run downloads the ~10 MB scrip master and
caches it to `data/universe.json`; it refreshes automatically after 24 hours, or
immediately with `--refresh-universe`.

### Train the model

```bash
python train/build_model.py --months 3
```

Writes `data/model.pkl` and `data/model_report.txt`. Raw candles are cached to
`data/cache/*.parquet`, so re-runs cost no API calls. Add `--force` to refetch.

---

## Configuration reference

All non-secret settings live in `config.yaml`.

| Key | Default | Meaning |
|---|---|---|
| `screener.min_price` / `max_price` | `30` / `500` | LTP band, in rupees |
| `screener.min_total_buy_qty` | `1000000` | `totBuyQuan` must exceed this |
| `screener.min_total_sell_qty` | `1000000` | `totSellQuan` must exceed this |
| `poll.interval_seconds_mock` | `2` | Poll cadence in mock mode |
| `poll.interval_seconds_live` | `5` | Poll cadence in live mode |
| `poll.batch_size` | `50` | Tokens per `getMarketData` call (Angel's FULL-mode cap) |
| `poll.batch_sleep_seconds` | `1.1` | Minimum gap between API calls |
| `poll.retry_attempts` | `3` | Attempts before a batch degrades to stale |
| `poll.warmup_per_cycle_live` | `5` | Candle warm-ups budgeted per cycle |
| `universe.max_symbols` | `0` | Cap on the live universe; `0` = every NSE `-EQ` symbol |
| `universe.refresh_hours` | `24` | Scrip-master cache lifetime |
| `universe.priority_symbols` | list | Always kept when trimming to the cap |
| `indicators.smma_fast` / `smma_slow` | `20` / `120` | SMMA periods |
| `indicators.sample_history_minutes` | `90` | Tick sample ring depth |
| `indicators.warmup_days` | `3` | Calendar days of history requested at warm-up |
| `ml.take_threshold` | `0.60` | `p ≥` this → **TAKE** |
| `ml.caution_threshold` | `0.45` | `p ≥` this → **CAUTION**, below → **AVOID** |
| `ml.max_signals` | `50` | Crossover events retained in memory |
| `mock.symbol_count` / `seed` | `30` / `7` | Simulated universe size and RNG seed |

---

## The price filter

Filtering happens in two places, and the distinction matters:

**Backend screen — authoritative.** Every poll cycle, `screener.passes()` drops any
symbol whose LTP falls outside `screener.min_price … max_price` (default ₹30–500) or
whose `totBuyQuan` / `totSellQuan` fail the depth test. Rejected symbols are never
sent to the browser. This runs across the *whole* universe — with
`universe.max_symbols: 0`, that is every NSE `-EQ` instrument, so the ₹30–500 band is
applied to all of NSE, not to a preselected watchlist.

**UI band — display only.** The `Price ₹ [min] – [max]` control in the table header
narrows what is rendered *within* the rows the backend already returned. It shows an
amber `N hidden` count, updates the stat cards, and `Reset` restores the configured
band. It cannot widen past the backend screen — to do that, change `config.yaml` and
restart:

```yaml
screener:
  min_price: 30.0
  max_price: 500.0
```

The panel header always states the backend band (`Backend screen: ₹30–₹500 · …`) so
the two can never be confused.

---

## How SMMA is computed

SMMA is **Wilder's smoothed moving average**, not an EMA:

```
SMMA[0] = SMA(prices[0:n])                        # seed: simple mean of the first n
SMMA[i] = (SMMA[i-1] * (n - 1) + price[i]) / n    # recursive
```

An EMA uses a `2/(n+1)` multiplier; Wilder's uses `1/n`, smoothing roughly half as
fast. Using the wrong one shifts every crossover by several bars, so it is
implemented directly in `app/indicators.py` rather than borrowed from a library.

**Warm-up.** At startup each surviving symbol fetches ~240 one-minute candles
(`indicators.warmup_days` back) and seeds both SMMAs from that history. If no
history is available the symbol renders as `warming up` — never with a wrong number.

**Live updates.** Incoming LTPs roll into the current minute's candle. When the
minute closes, that close advances the *confirmed* SMMA. Between closes the
dashboard shows a **projected** value — what the SMMA would be if the in-progress
bar closed at the current price:

```
SMMA_projected = (SMMA_confirmed * (n - 1) + ltp) / n
```

This is why the lines move continuously instead of stepping once a minute.

**Crossover detection** tracks `sign(SMMA20 − SMMA120)`. A flip to positive emits
**BUY**, to negative emits **SELL**. Two guards prevent meaningless spam when the
lines sit on top of each other: the gap must exceed `0.004%`, and the same symbol
cannot fire twice within 45 seconds.

---

## How the ML model works

The model does **not** predict price. It answers one narrow question: *this
crossover has already fired — will it reach +1.0% before −0.7% within 30 bars?*

### Labels

Triple-barrier. From the crossover bar, look forward 30 one-minute bars. Label `1`
if the profit barrier is touched before the stop (inverted for SELL), else `0`.
When both barriers fall inside the same candle the intrabar path is unknowable, so
it is scored as a loss — the pessimistic reading is the honest one. Crossovers whose
30-bar horizon would run past the session close are dropped rather than truncated.

### Features

| Feature | Definition |
|---|---|
| `smma_gap_pct` | `(SMMA20 − SMMA120) / SMMA120 × 100`, measured **1 bar after** the cross |
| `smma20_slope` | % change in SMMA20 over the prior 5 bars |
| `smma120_slope` | % change in SMMA120 over the prior 5 bars |
| `adx_14` | Wilder's ADX(14) — trend strength |
| `atr_pct` | `ATR(14) / close × 100` |
| `volume_surge` | mean volume of last 5 bars ÷ mean of the prior 30 |
| `crossovers_last_60_bars` | whipsaw counter |
| `dist_from_vwap_pct` | distance from session VWAP |
| `minutes_since_open` | bars since 09:15 |
| `body_ratio` | `\|close − open\| / (high − low)` of the crossover candle |

Features whose sign is meaningful (`smma_gap_pct`, both slopes, `dist_from_vwap_pct`)
are flipped for SELL signals, so one symmetric model serves both directions.

The feature vector is built by `app.ml.build_features` — **the training script imports
the same function the live app calls**, so there is exactly one definition. Measuring
`smma_gap_pct` one bar after the cross matters: at the crossing bar the gap is ~0 by
construction, which would make it a useless feature. The live scorer reproduces this
by projecting the SMMA recursion one bar forward (`feature_context(extra_bars=1)`).

### Model

LightGBM binary classifier wrapped in `CalibratedClassifierCV(method="sigmoid")`, so
the output probability is meaningful rather than an arbitrary score. Train/test is a
**chronological** 80/20 split — a random split would leak the future into the past.

### Verdicts and reasons

`TAKE` at `p ≥ 0.60`, `CAUTION` at `0.45 ≤ p < 0.60`, `AVOID` below `0.45`.

`shap.TreeExplainer` returns per-prediction contributions. The three features pushing
the prediction *down* hardest are mapped to plain English, e.g.
*"Choppy: 4 crossovers in the last hour"*, *"Weak trend (ADX 14.2) — crossover likely
to fail"*. Each message reads the actual value, so a high ADX reports
*"trend already extended — late entry risk"* rather than mislabelling it weak.

SHAP explains the underlying booster rather than the calibrated wrapper, which is not
a tree model. Calibration is a monotonic sigmoid, so it rescales probabilities without
changing the *ranking* of contributions — and ranking is all the reasons depend on.

### Measured performance

From `data/model_report.txt`, on the held-out chronological test window:

| | |
|---|---|
| Crossovers in dataset | 3,472 (40 symbols, 3 months) |
| Test window | 695 crossovers |
| **Baseline — take every crossover** | **37.99%** win rate |
| **Model — take when `p ≥ 0.40`** | **45.21%** win rate (+7.22 pp, 10.5% coverage) |
| ROC AUC | 0.6064 |
| Brier score | 0.2322 |

⚠️ **These numbers come from simulated data.** No broker credentials were available at
training time, so `build_model.py` fell back to a synthetic regime-switching price
process. The pipeline is real and the numbers are honestly measured, but they describe
that simulation, **not the NSE**. Re-run with `MODE=live` and valid credentials to
train on real history; the report stamps its own data source at the top.

Note the interaction between calibration and the verdict bands: with a 38% base rate,
a *correctly* calibrated model should rarely emit 0.60+. The shipped model's
probabilities span 0.32–0.43, so every live verdict is currently `AVOID`. That is the
calibration working, not a bug — 42% odds on a +1.0%/−0.7% payoff genuinely is not a
good trade. `model_report.txt` prints the cut-points implied by the model's own
distribution (`take 0.40 / caution 0.37`) if you want to re-centre the bands.

---

## Design Decisions & Assumptions

Every shortcut taken, and why.

1. **Average LTP is a simple mean of sampled LTPs**, not a time- or volume-weighted
   average. Polls are near-evenly spaced, so the difference is small, and a plain mean
   is the one definition nobody has to guess at.

2. **Traded quantity is a cumulative-volume delta**, `cum_volume_now − cum_volume_at_T−N`,
   read from a per-symbol ring of `(timestamp, cumulative_volume, ltp)` samples capped
   at 90 minutes. Summing per-tick volumes would double-count.

3. **The sample ring is seeded from historical candles at warm-up.** Cumulative volume
   at each past bar is reconstructed by walking the day's bar volumes backwards from
   the quote's `tradeVolume`. Without this the 20m and 60m columns would show `—` for
   the first hour of every run. Windows reaching back past the day's first bar — where
   the exchange's cumulative counter resets — still return `null`.

4. **`null` is rendered, never faked.** Insufficient history shows `—`; an unseeded
   SMMA shows `warming up`. No interpolation, no partial-window averages.

5. **The live universe defaults to the whole NSE `-EQ` list** (`universe.max_symbols: 0`).
   Know the arithmetic before running it: ~2,000 instruments ÷ 50 tokens per request
   × ~1 request/second ≈ **40–45 seconds per full poll cycle**. That is Angel's rate
   limit, not a code limit, and the app logs the projected cycle time at startup rather
   than letting it look like a hang. Set `max_symbols` to a number (e.g. `300`) for a
   faster refresh on a smaller watchlist; `priority_symbols` are always quoted in the
   first batches either way.

6. **Mock mode uses real NSE ticker symbols with entirely simulated prices.** Invented
   tickers look obviously fake and undercut the demo. The amber `DEMO DATA` badge, the
   `MODE MOCK` chip, and `"simulated": true` on every payload make the distinction
   unmissable. Symbols whose real price sits outside the ₹30–500 band were swapped out
   rather than clamped.

7. **Three mock symbols are engineered to cross within a minute or two.** Their price
   history is bent by a linear ramp whose amplitude is found by bisection until SMMA20
   sits 0.05% below SMMA120. Bisection converges because the gap rises monotonically
   with the ramp, so "this will cross soon" is a guarantee, not a hope.

8. **Crossover hysteresis**: a 0.004% minimum gap and a 45-second per-symbol cooldown.
   Without them, two SMMAs resting on each other emit dozens of meaningless events a
   minute.

9. **Every screened row is scored, not just ones that have crossed.** For a row that
   has not flipped yet the confidence answers *"if this crossover fired right now, how
   would the model rate it?"*. The Signal pill stays `NEUTRAL` until a real flip occurs.

10. **Warm-up is budgeted per cycle** (5 symbols in live mode). Fetching 200 candle
    histories serially at 1 req/sec would stall the first poll for three minutes.

11. **Market holidays are not modelled** — only weekends. An NSE holiday looks like an
    open market with no ticks, which surfaces as stale data rather than a wrong price.

12. **IST is a fixed +05:30 offset** rather than a `zoneinfo` lookup. India has no DST,
    so the offset is exact, and it avoids depending on a platform IANA database that
    Windows does not ship.

13. **The trained model ships even though it was trained on simulated data**, because
    the alternative — no `model.pkl` — would leave the SHAP path unexercised. Both the
    report and the model bundle carry a `simulated` flag.

14. **Config thresholds keep the specified 0.60 / 0.45 defaults** rather than being
    silently re-centred on the shipped model's distribution. The recommendation is
    printed in the report instead; changing behaviour quietly to make a demo look
    better is the wrong trade.

15. **The UI price band narrows, it never widens.** The header control filters the
    rows the backend already screened; it cannot request symbols the server rejected.
    Letting the browser widen the band would mean either shipping every unscreened
    symbol over the wire or making the client's view disagree with what "screened"
    means. The panel header prints the backend band alongside the control so the two
    are never mistaken for each other.

---

## Known Limitations

- **Full-universe polling is slow by construction.** Screening all ~2,000 NSE `-EQ`
  symbols costs ~40–45s per cycle at Angel's quote rate limit, so "real-time" means
  once every ~45 seconds, not every 5. Cap `universe.max_symbols` to trade breadth for
  freshness. See decision 5.
- **Warm-up across a full-NSE run is gradual.** Only symbols that pass the screen get
  a candle history, and those are fetched `poll.warmup_per_cycle_live` (8) at a time to
  keep the cycle moving. With a few hundred survivors, expect SMMA columns to fill in
  over the first several cycles; until then those rows honestly read `warming up`.
- **REST polling, not WebSocket.** Angel offers a streaming feed; this polls on an
  interval. Polling is simpler, degrades more gracefully, and is well inside the rate
  limits — but it is not tick-by-tick.
- **The model is trained on limited, simulated history.** Three months, 40 symbols,
  synthetic. Treat the reported edge as a demonstration of the pipeline, not evidence
  of a tradeable strategy.
- **Feature drift between mock and live.** The mock feed's statistics differ from the
  training simulation's, so live mock probabilities cluster more tightly than the
  test-set distribution.
- **In-memory state only.** Restarting loses tick history and the signal feed; both
  rebuild from candles on the next warm-up.
- **Single process.** No horizontal scaling; the poll thread and the API share one
  interpreter.
- **No backtest harness.** The model report measures classification quality, not
  strategy P&L after costs, slippage, or impact.
- **`getCandleData` gaps.** Illiquid symbols return sparse or empty 1-minute history;
  those rows stay in `warming up` rather than being seeded from bad data.

---

## Read-only guarantee

This application **cannot place, modify, or cancel orders.**

- `app/broker.py` imports `SmartConnect` and calls exactly three methods:
  `generateSession`, `getMarketData`, and `getCandleData`, plus one plain HTTPS GET
  for the public scrip master.
- No order-placement, order-modification, or order-cancellation method is imported,
  wrapped, aliased, or referenced anywhere in the codebase.
- There is no endpoint, form, or code path that submits anything to the broker.

Verify it yourself:

```bash
grep -rniE "placeorder|modifyorder|cancelorder|squareoff|placeOrderFullResponse" app/ train/ static/
```

## Credential handling

- `.gitignore` was the first file created in this repository; it excludes `.env`.
- Only `.env.example`, containing placeholders, is committed.
- `Credentials.__repr__` is overridden to print completeness, never values, so a
  credential cannot leak through accidental interpolation into a log or traceback.
- Every Angel exception passes through `utils.redact()` before being logged: it
  removes the known secret values, then regex-sweeps `key=value` shapes so an
  unexpected SDK error string cannot leak a token the code never handled directly.
- No credential appears in any API response.

## Rate limits

Angel's published quote limit is respected by deliberate throttling, not by hoping:

- `getMarketData` FULL mode is called with at most **50 tokens per request** — Angel's
  documented cap.
- A monotonic-clock throttle enforces a minimum `poll.batch_sleep_seconds` (default
  1.1s) between *every* API call, batches included.
- Failures retry with exponential backoff (`1s → 2s → 4s`, capped at 20s).
- A batch that exhausts its retries degrades the affected symbols to **stale** and the
  polling thread continues. One bad batch never kills the loop.

---

## Project layout

```
app/
  broker.py      MockBroker + AngelBroker behind one interface (read endpoints only)
  config.py      .env + config.yaml loading; credential redaction
  indicators.py  SMMA, ADX, ATR, VWAP, slopes — pure functions, no I/O
  ml.py          feature building, LightGBM + SHAP, rule-based fallback
  screener.py    the poll cycle and the background thread
  server.py      FastAPI: /, /api/snapshot, /api/health
  state.py       SymbolTracker (per symbol) + AppState (process-wide, locked)
  utils.py       IST clock, market hours, redaction, Indian number formatting
static/
  index.html     the entire dashboard — vanilla JS + CSS, no build step
train/
  build_model.py offline training; caches candles to data/cache/*.parquet
scripts/
  smoke_broker.py    Phase-1 check: prints quotes and candles
  smoke_pipeline.py  Phase-2 check: runs poll cycles in-process, no HTTP
data/
  model.pkl          committed
  model_report.txt   committed
  cache/             git-ignored
  universe.json      git-ignored
```

### API

`GET /api/snapshot`

```jsonc
{
  "mode": "mock", "simulated": true, "status": "ok",
  "market_status": "open", "last_updated": "2026-08-12T13:48:25+05:30",
  "universe_size": 30, "screened_count": 20,
  "filters":    { "min_price": 30.0, "max_price": 500.0, "...": 1000000 },
  "thresholds": { "take": 0.60, "caution": 0.45 },
  "stocks":  [ { "symbol": "PNB", "ltp": 106.87, "vol_5m": 660000, "smma20": 102.97,
                 "smma_ready": true, "signal": "BUY",
                 "ml": { "probability": 0.41, "verdict": "AVOID", "reasons": ["..."] },
                 "depth": { "buy": [ /* 5 */ ], "sell": [ /* 5 */ ] } } ],
  "signals": [ { "ts": "...", "symbol": "PNB", "direction": "BUY",
                 "probability": 0.41, "verdict": "AVOID", "top_reason": "..." } ],
  "health":  { "connected": true, "poll_duration_ms": 25.1, "scorer": "lightgbm" }
}
```

`GET /api/health` → `{ mode, connected, last_poll_ts, poll_duration_ms, errors_last_hour, scorer, model_loaded, poll_count, warmed_up }`

---

## Disclaimer

This is an **educational and research tool**. It is **not investment advice**.

ML predictions are probabilistic estimates derived from historical patterns and are
**not guarantees of future performance**. The shipped model was trained on simulated
data. Markets are adversarial and non-stationary; a pattern that held in the training
window may not hold tomorrow. Nothing here has been validated for live trading, and no
result shown accounts for brokerage, taxes, slippage, or market impact.

Do your own research. Consult a SEBI-registered adviser before risking capital.
