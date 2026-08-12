"""Phase-2 smoke test: run the screening pipeline in-process, no HTTP.

    python scripts/smoke_pipeline.py [cycles]

Prints what /api/snapshot would return after N poll cycles.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.broker import make_broker  # noqa: E402
from app.config import configure_logging, settings  # noqa: E402
from app.ml import make_scorer  # noqa: E402
from app.screener import Screener  # noqa: E402
from app.state import state  # noqa: E402
from app.utils import indian_number  # noqa: E402


def fmt(value, spec: str = ".2f") -> str:
    return "—" if value is None else format(value, spec)


def main() -> int:
    cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    configure_logging(settings.log_level)

    screener = Screener(settings, make_broker(settings), state, make_scorer(settings))
    if not screener.bootstrap():
        print("bootstrap failed")
        return 1

    for i in range(cycles):
        screener.poll_once()
        snap = state.snapshot()
        print(
            f"cycle {i + 1}: screened={snap['screened_count']}/{snap['universe_size']} "
            f"warmed={snap['health']['warmed_up']} "
            f"poll={snap['health']['poll_duration_ms']}ms "
            f"signals={len(snap['signals'])}"
        )
        time.sleep(0.4)

    snap = state.snapshot()
    cols = f"{'SYM':<13}{'LTP':>9}{'CHG%':>7}{'V5m':>8}{'V20m':>9}{'AVG20':>9}{'SMMA20':>9}{'SMMA120':>9}{'SIG':>8}{'P':>7} {'VERDICT':<8} REASON"
    print("\n" + cols)
    print("-" * 130)
    for row in snap["stocks"][:12]:
        ml = row.get("ml") or {}
        reason = (ml.get("reasons") or [""])[0]
        print(
            f"{row['symbol']:<13}{row['ltp']:>9.2f}{row['change_pct']:>7.2f}"
            f"{indian_number(row['vol_5m']):>8}{indian_number(row['vol_20m']):>9}"
            f"{fmt(row['avg_20m']):>9}{fmt(row['smma20']):>9}{fmt(row['smma120']):>9}"
            f"{row['signal']:>8}{fmt(ml.get('probability'), '.2f'):>7} "
            f"{str(ml.get('verdict', '-')):<8} {reason[:44]}"
        )

    print(f"\nsignals emitted: {len(snap['signals'])}")
    for s in snap["signals"][:5]:
        print(f"  {s['ts'][11:19]}  {s['symbol']:<12} {s['direction']:<5} "
              f"p={s['probability']:.2f} {s['verdict']:<8} {s['top_reason']}")

    keys = sorted(snap.keys())
    print(f"\nsnapshot keys: {keys}")
    print(f"snapshot bytes: {len(json.dumps(snap))}")
    print("\nOK\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
