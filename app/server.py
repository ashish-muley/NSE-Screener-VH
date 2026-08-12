"""FastAPI application.

The HTTP layer is deliberately thin: every endpoint returns state that the
polling thread has already assembled, so a slow or failing broker can never
stall a request.

Run it with::

    python -m app.server
    # or
    uvicorn app.server:app --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from .broker import make_broker
from .config import STATIC_DIR, configure_logging, settings
from .ml import make_scorer
from .screener import PollerThread, Screener
from .state import state

log = logging.getLogger(__name__)

INDEX = STATIC_DIR / "index.html"
_poller: PollerThread | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _poller
    configure_logging(settings.log_level)
    log.info("Starting in %s mode", settings.mode.upper())
    if settings.is_mock:
        log.warning("MOCK MODE — all market data is simulated and badged DEMO in the UI")

    broker = make_broker(settings)
    scorer = make_scorer(settings)
    screener = Screener(settings, broker, state, scorer)

    state.health.update(
        {
            "mode": settings.mode,
            "connected": False,
            "model_loaded": scorer.model is not None,
            "scorer": scorer.scorer_name,
        }
    )

    _poller = PollerThread(screener, settings.poll_interval)
    _poller.start()
    try:
        yield
    finally:
        if _poller:
            _poller.stop()
        log.info("Shutting down")


app = FastAPI(
    title="NSE Real-Time Stock Screener",
    description="Read-only NSE screener with SMMA crossover detection and ML signal scoring.",
    version="1.0.0",
    lifespan=lifespan,
)

_NO_STORE = {"Cache-Control": "no-store, max-age=0"}


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(INDEX, headers=_NO_STORE)


@app.get("/api/snapshot")
async def snapshot() -> JSONResponse:
    """Everything the dashboard renders, assembled by the last poll cycle."""
    return JSONResponse(state.snapshot(), headers=_NO_STORE)


@app.get("/api/health")
async def health() -> JSONResponse:
    payload = state.health_payload()
    return JSONResponse(
        {
            "mode": payload.get("mode", settings.mode),
            "connected": payload.get("connected", False),
            "last_poll_ts": payload.get("last_poll_ts"),
            "poll_duration_ms": payload.get("poll_duration_ms", 0),
            "errors_last_hour": payload.get("errors_last_hour", 0),
            "scorer": payload.get("scorer", "unknown"),
            "model_loaded": payload.get("model_loaded", False),
            "poll_count": payload.get("poll_count", 0),
            "warmed_up": payload.get("warmed_up", 0),
        },
        headers=_NO_STORE,
    )


def main() -> None:  # pragma: no cover
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="NSE screener (read-only).")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument(
        "--refresh-universe",
        action="store_true",
        help="re-download the Angel scrip master before starting (live mode only)",
    )
    args = parser.parse_args()

    configure_logging(settings.log_level)

    if args.refresh_universe:
        if settings.is_mock:
            log.warning("--refresh-universe has no effect in mock mode")
        else:
            broker = make_broker(settings)
            if broker.connect():
                broker.get_universe(refresh=True)
            else:
                log.error("Cannot refresh the universe: %s", broker.last_error)

    uvicorn.run(
        "app.server:app",
        host=args.host,
        port=args.port,
        log_level=settings.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
