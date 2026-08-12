"""Signal scoring.

Primary path: the calibrated LightGBM classifier produced by
``train/build_model.py``, explained per-prediction with ``shap.TreeExplainer``.

Fallback path: a transparent rule-based scorer used when ``data/model.pkl`` is
absent or unloadable. It is a *fallback*, not the method — the README says so
too, and the API/UI always report which scorer produced a number.

``build_features`` is imported by the training script as well, so the feature
vector at train time and at inference time is produced by one piece of code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

# Canonical order. The saved model carries its own copy; a mismatch is fatal
# to accuracy, so it is checked at load time.
FEATURE_NAMES: list[str] = [
    "smma_gap_pct",
    "smma20_slope",
    "smma120_slope",
    "adx_14",
    "atr_pct",
    "volume_surge",
    "crossovers_last_60_bars",
    "dist_from_vwap_pct",
    "minutes_since_open",
    "body_ratio",
]

# Neutral stand-ins for a feature that cannot be computed yet. Chosen to be
# uninformative rather than flattering.
FEATURE_DEFAULTS: dict[str, float] = {
    "smma_gap_pct": 0.0,
    "smma20_slope": 0.0,
    "smma120_slope": 0.0,
    "adx_14": 20.0,
    "atr_pct": 0.50,
    "volume_surge": 1.0,
    "crossovers_last_60_bars": 0.0,
    "dist_from_vwap_pct": 0.0,
    "minutes_since_open": 90.0,
    "body_ratio": 0.50,
}

# Features whose sign is meaningful relative to the trade direction. Flipping
# them for SELL lets one model serve both sides symmetrically.
DIRECTIONAL = {"smma_gap_pct", "smma20_slope", "smma120_slope", "dist_from_vwap_pct"}


def build_features(ctx: dict[str, Any]) -> dict[str, float]:
    """Turn a raw context dict into the model's feature vector.

    ``ctx["direction"]`` must be ``"BUY"`` or ``"SELL"``.
    """
    sign = -1.0 if str(ctx.get("direction", "BUY")).upper() == "SELL" else 1.0
    out: dict[str, float] = {}
    for name in FEATURE_NAMES:
        raw = ctx.get(name)
        value = FEATURE_DEFAULTS[name] if raw is None else float(raw)
        if name in DIRECTIONAL:
            value *= sign
        out[name] = value
    return out


# --- plain-English explanations --------------------------------------------

# Every entry must read the actual value and describe *that*. An earlier
# version assumed each feature only ever hurt in one direction and emitted
# "Weak trend (ADX 43.3)" — which is a strong trend. A reason string that
# contradicts the number beside it destroys trust in the whole panel.
#
# Sign convention: directional features are already flipped for SELL, so a
# negative value always means "against the proposed trade".
REASON_TEXT: dict[str, Callable[[float], str]] = {
    "crossovers_last_60_bars": lambda v: (
        f"Choppy: {int(v)} crossovers in the last hour"
        if v >= 2
        else "One recent crossover already — mild whipsaw risk"
    ),
    "adx_14": lambda v: (
        f"Weak trend (ADX {v:.1f}) — crossover likely to fail" if v < 20
        else f"Trend strength only moderate (ADX {v:.1f})" if v < 35
        else f"Trend already extended (ADX {v:.1f}) — late entry risk"
    ),
    "smma_gap_pct": lambda v: (
        f"SMMAs still inverted ({v:.2f}%) — crossover not confirmed" if v < 0
        else f"SMMA separation only {v:.2f}% — too tight to be decisive" if v < 0.15
        else f"SMMAs already {v:.2f}% apart — much of the move may be done"
    ),
    "smma20_slope": lambda v: (
        f"SMMA20 turning against the signal ({v:+.2f}% over 5 bars)" if v < 0
        else f"SMMA20 barely turning ({v:+.2f}% over 5 bars)" if v < 0.10
        else f"SMMA20 already ran {v:+.2f}% in 5 bars — entry may be late"
    ),
    "smma120_slope": lambda v: (
        f"Long-term trend opposes the signal ({v:+.2f}% over 5 bars)" if v < 0
        else f"Long-term trend flat ({v:+.2f}% over 5 bars)" if v < 0.05
        else f"Long-term trend already steep ({v:+.2f}% over 5 bars)"
    ),
    "volume_surge": lambda v: (
        f"No volume confirmation ({v:.1f}x average)" if v < 1.3
        else f"Volume spike ({v:.1f}x average) — risk of an exhaustion move"
    ),
    "atr_pct": lambda v: (
        f"Volatility compressed (ATR {v:.2f}% of price) — little room to the target" if v < 0.25
        else f"Volatility elevated (ATR {v:.2f}% of price) — wide stops needed"
    ),
    "minutes_since_open": lambda v: (
        "First 15 minutes — unreliable signal window" if v < 15
        else f"Late in the session ({int(v)} min in) — limited follow-through time" if v > 330
        else f"Mid-session entry ({int(v)} min in)"
    ),
    "dist_from_vwap_pct": lambda v: (
        f"Price {abs(v):.2f}% on the wrong side of VWAP" if v < 0
        else f"Already {v:.2f}% extended from VWAP — chasing"
    ),
    "body_ratio": lambda v: (
        f"Indecisive candle (body only {v * 100:.0f}% of range)" if v < 0.5
        else f"Full-bodied candle ({v * 100:.0f}% of range) — entry far from the low"
    ),
}


def explain(feature: str, value: float) -> str:
    fn = REASON_TEXT.get(feature)
    return fn(value) if fn else f"{feature} = {value:.2f}"


@dataclass
class Score:
    probability: float
    verdict: str
    reasons: list[str] = field(default_factory=list)
    scorer: str = "rules"

    def as_dict(self) -> dict[str, Any]:
        return {
            "probability": round(self.probability, 4),
            "verdict": self.verdict,
            "reasons": self.reasons,
            "scorer": self.scorer,
        }


class SignalScorer:
    """Loads the model if it exists; degrades to rules if it does not."""

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.take_threshold = float(settings.ml.get("take_threshold", 0.60))
        self.caution_threshold = float(settings.ml.get("caution_threshold", 0.45))
        self.model: Any = None
        self.features: list[str] = list(FEATURE_NAMES)
        self.metrics: dict[str, Any] = {}
        self._explainer: Any = None
        self.scorer_name = "rules"

    # -- loading -------------------------------------------------------------
    def load(self) -> bool:
        path: Path = self.settings.model_path()
        if not path.exists():
            log.warning(
                "No ML model at %s — using the documented rule-based fallback scorer. "
                "Run `python train/build_model.py` to train one.",
                path,
            )
            return False
        try:
            import joblib

            bundle = joblib.load(path)
            self.model = bundle["model"]
            self.features = list(bundle.get("features") or FEATURE_NAMES)
            self.metrics = dict(bundle.get("metrics") or {})
            if self.features != FEATURE_NAMES:
                log.warning("Model feature order differs from the runtime list — using the model's")
            self._build_explainer(bundle.get("base_estimator"))
            self.scorer_name = "lightgbm"
            log.info("Loaded ML model from %s (%d features)", path.name, len(self.features))
            return True
        except Exception as exc:  # noqa: BLE001 - never fatal
            log.error("Could not load %s (%s) — falling back to rules", path.name, exc)
            self.model = None
            self.scorer_name = "rules"
            return False

    def _build_explainer(self, base_estimator: Any) -> None:
        """SHAP explains the raw tree margin; calibration only rescales it.

        The calibrated wrapper is not itself a tree model, so TreeExplainer is
        pointed at the underlying LightGBM booster. Contribution *ranking* is
        what drives the reason strings, and that ranking is unaffected by the
        monotonic sigmoid calibration applied on top.
        """
        if base_estimator is None:
            return
        try:
            import warnings

            import shap

            # SHAP warns on *every* call that a LightGBM binary classifier now
            # returns a list of ndarray. That is informational, it is the shape
            # `_shap_reasons` already handles, and at one warning per scored row
            # per poll it buries every real log line. Silence this one message
            # only — no blanket filter.
            warnings.filterwarnings(
                "ignore",
                message=".*binary classifier with TreeExplainer.*",
                category=UserWarning,
            )
            self._explainer = shap.TreeExplainer(base_estimator)
            log.info("SHAP explainer ready")
        except Exception as exc:  # noqa: BLE001
            log.warning("SHAP unavailable (%s) — reasons will use the rule heuristics", exc)
            self._explainer = None

    # -- scoring -------------------------------------------------------------
    def verdict_for(self, probability: float) -> str:
        if probability >= self.take_threshold:
            return "TAKE"
        if probability >= self.caution_threshold:
            return "CAUTION"
        return "AVOID"

    def score(self, ctx: dict[str, Any]) -> Score:
        features = build_features(ctx)
        if self.model is not None:
            try:
                return self._score_model(features)
            except Exception as exc:  # noqa: BLE001 - a bad predict must not kill the poll
                log.error("Model scoring failed (%s) — using rules for this cycle", exc)
        return self._score_rules(features)

    def _score_model(self, features: dict[str, float]) -> Score:
        import numpy as np

        row = np.array([[features[name] for name in self.features]], dtype=float)
        probability = float(self.model.predict_proba(row)[0][1])
        reasons = self._shap_reasons(row, features) or self._rule_reasons(features)
        return Score(probability, self.verdict_for(probability), reasons[:3], "lightgbm")

    def _shap_reasons(self, row: Any, features: dict[str, float]) -> list[str]:
        """Top 3 features pushing the prediction DOWN, in plain English."""
        if self._explainer is None:
            return []
        try:
            values = self._explainer.shap_values(row)
            # Older SHAP returns a list per class; newer returns one array.
            if isinstance(values, list):
                values = values[1] if len(values) > 1 else values[0]
            contributions = list(values[0])
        except Exception as exc:  # noqa: BLE001
            log.debug("SHAP failed: %s", exc)
            return []

        negatives = sorted(
            ((name, float(c)) for name, c in zip(self.features, contributions) if c < 0),
            key=lambda kv: kv[1],
        )
        reasons = [explain(name, features[name]) for name, _ in negatives[:3]]
        return reasons or ["Model found no material negatives in this setup"]

    # -- rule-based fallback -------------------------------------------------
    def _rule_deltas(self, f: dict[str, float]) -> list[tuple[str, float]]:
        """Fixed, hand-written weights. Transparent by design."""
        deltas: list[tuple[str, float]] = []

        adx_v = f["adx_14"]
        deltas.append(("adx_14", 0.12 if adx_v >= 30 else 0.07 if adx_v >= 25 else
                       0.02 if adx_v >= 20 else -0.05 if adx_v >= 15 else -0.12))

        gap = abs(f["smma_gap_pct"])
        deltas.append(("smma_gap_pct", 0.08 if gap >= 0.35 else 0.04 if gap >= 0.15 else
                       -0.02 if gap >= 0.06 else -0.09))

        surge = f["volume_surge"]
        deltas.append(("volume_surge", 0.10 if surge >= 2.0 else 0.05 if surge >= 1.3 else
                       0.0 if surge >= 0.8 else -0.07))

        whip = f["crossovers_last_60_bars"]
        deltas.append(("crossovers_last_60_bars", 0.05 if whip <= 0 else 0.0 if whip == 1 else
                       -0.06 if whip == 2 else -0.11 if whip == 3 else -0.16))

        deltas.append(("minutes_since_open", -0.08 if f["minutes_since_open"] < 15 else 0.0))
        deltas.append(("atr_pct", -0.06 if f["atr_pct"] > 1.5 else 0.0))
        deltas.append(("smma20_slope", 0.04 if f["smma20_slope"] > 0 else -0.04))
        deltas.append(("body_ratio", 0.02 if f["body_ratio"] >= 0.5 else -0.03))
        return deltas

    def _rule_reasons(self, f: dict[str, float]) -> list[str]:
        negatives = sorted(
            ((name, d) for name, d in self._rule_deltas(f) if d < 0), key=lambda kv: kv[1]
        )
        reasons = [explain(name, f[name]) for name, _ in negatives[:3]]
        return reasons or ["Clean setup — no material negatives detected"]

    def _score_rules(self, f: dict[str, float]) -> Score:
        probability = 0.50 + sum(d for _, d in self._rule_deltas(f))
        probability = max(0.05, min(0.95, probability))
        return Score(probability, self.verdict_for(probability), self._rule_reasons(f), "rules")


def make_scorer(settings: Any) -> SignalScorer:
    scorer = SignalScorer(settings)
    scorer.load()
    return scorer
