"""Phase-1 smoke test: prove the broker interface works end to end.

    python scripts/smoke_broker.py

Runs against whatever MODE is set in .env (default: mock).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.broker import make_broker  # noqa: E402
from app.config import configure_logging, settings  # noqa: E402
from app.utils import indian_number  # noqa: E402


def main() -> int:
    configure_logging(settings.log_level)
    print(f"\nMODE = {settings.mode}\n")

    broker = make_broker(settings)
    if not broker.connect():
        print(f"connect() failed: {broker.last_error}")
        return 1

    universe = broker.get_universe()
    print(f"universe: {len(universe)} instruments\n")

    quotes = broker.get_quotes(universe[:8])
    header = f"{'SYMBOL':<14}{'LTP':>9}{'CHG%':>8}{'BID':>9}{'ASK':>9}{'TOTBUY':>10}{'TOTSELL':>10}"
    print(header)
    print("-" * len(header))
    for q in quotes.values():
        bid = q["depth"]["buy"][0]["price"]
        ask = q["depth"]["sell"][0]["price"]
        print(
            f"{q['symbol']:<14}{q['ltp']:>9.2f}{q['percentChange']:>8.2f}"
            f"{bid:>9.2f}{ask:>9.2f}"
            f"{indian_number(q['totBuyQuan']):>10}{indian_number(q['totSellQuan']):>10}"
        )

    inst = universe[0]
    candles = broker.get_candles(inst, bars=200)
    print(f"\ncandles for {inst.symbol}: {len(candles)} bars")
    if candles:
        first, last = candles[0], candles[-1]
        print(f"  first {first.ts:%Y-%m-%d %H:%M}  close={first.close:.2f}  vol={first.volume}")
        print(f"  last  {last.ts:%Y-%m-%d %H:%M}  close={last.close:.2f}  vol={last.volume}")

    # Prove prices actually move between polls.
    again = broker.get_quotes([inst])
    print(f"\n{inst.symbol}: {quotes[inst.token]['ltp']:.2f} -> {again[inst.token]['ltp']:.2f}")
    print("\nOK\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
