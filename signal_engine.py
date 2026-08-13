"""Deterministic trend-following signal engine.

Pure, importable module: no CLI, no printing, no network at import time and
no network anywhere in this file. Callers fetch OHLCV themselves (e.g. via
``signal_check.fetch_ohlcv``) and pass plain DataFrames in, so every function
here is testable offline.

Strategy (see docs/superpowers/specs/2026-08-12-deterministic-signal-engine-design.md):

1. Trend filter (must pass): EMA9 > EMA21 > EMA50 for LONG (mirrored for
   SHORT) plus ADX > 25 with DI+/DI- agreeing with the direction.
2. Entry timing: last close within 0.75xATR of EMA21 (pullback zone) OR an
   EMA9/EMA21 crossover within the last 3 closed bars, plus an RSI guard
   (LONG 45-65, SHORT 35-55).
3. Confluence score -> confidence: base 50, +8 per confirming indicator
   (MACD histogram, VWAP side, volume spike, OBV slope, 1d trend), cap 95.
4. ATR-based trade plan: SL 2.0xATR, TP1 4.0xATR, TP2 7.0xATR.

(Default parameters are the result of a grid sweep + walk-forward validation
over ~13 months of 4h data on BTC/ETH/SOL/BNB — see the tuning notes in
docs/superpowers/specs/2026-08-12-deterministic-signal-engine-design.md.
Wide stops were decisively more robust across regimes than tight ones.)

Indicator columns follow the naming of ``signal_check.compute_indicators``
(``rsi_14``, ``atr_14``, ``adx``, ``di_plus``, ``di_minus``, ``macd_hist``,
``ema_50``, ``obv``, ``vwap_14``, ``vol_sma_20``) and are computed with the
same ``ta`` library calls. ``signal_check.compute_indicators`` returns only
the last bar as a dict of scalars, while this engine needs per-bar series
(crossovers, histogram slope, OBV slope), so it computes its own frame
(``add_indicators``) and adds ``ema_9``/``ema_21`` in the same style.

In-progress candle policy: ``evaluate_symbol`` ALWAYS drops the last row of
``df`` (and of ``df_1d``) before doing anything, treating it as a possibly
still-forming candle. Callers that already sliced to closed bars must pass
their data as-is and accept that one more trailing bar is ignored; feeding
one extra (live) candle is the intended usage.

``symbol`` / ``timeframe``: ``evaluate_symbol`` does not know the symbol; it
returns ``symbol=None`` and ``scan_symbols`` fills it in. ``timeframe`` is an
optional keyword-only pass-through stored on the evaluation dict so that
``build_trade_plan`` can emit it in the plan.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
import ta

# ----------------------------------------------------------------- tunables

ATR_SL_MULT = 2.0
ATR_TP1_MULT = 4.0
ATR_TP2_MULT = 7.0
ADX_MIN = 25
RSI_LONG = (45, 65)
RSI_SHORT = (35, 55)
CONFLUENCE_BASE = 50
CONFLUENCE_STEP = 8
CONFIDENCE_CAP = 95
PULLBACK_ATR_FRAC = 0.75
CROSS_LOOKBACK_BARS = 3
VOLUME_MULT = 1.2
REQUIRE_1D_ALIGNMENT = False  # hard filter: trade only with the daily trend


@dataclass(frozen=True)
class SignalParams:
    """Tunable knobs for the engine; defaults are the module constants above.

    Construct overrides from a dict with ``SignalParams(**{...})``; use
    ``params_to_dict`` for a plain-dict echo (tuples become lists so the
    result is JSON-serialisable).
    """

    ATR_SL_MULT: float = ATR_SL_MULT
    ATR_TP1_MULT: float = ATR_TP1_MULT
    ATR_TP2_MULT: float = ATR_TP2_MULT
    ADX_MIN: float = ADX_MIN
    RSI_LONG: tuple = RSI_LONG
    RSI_SHORT: tuple = RSI_SHORT
    CONFLUENCE_BASE: float = CONFLUENCE_BASE
    CONFLUENCE_STEP: float = CONFLUENCE_STEP
    CONFIDENCE_CAP: float = CONFIDENCE_CAP
    PULLBACK_ATR_FRAC: float = PULLBACK_ATR_FRAC
    CROSS_LOOKBACK_BARS: int = CROSS_LOOKBACK_BARS
    VOLUME_MULT: float = VOLUME_MULT
    REQUIRE_1D_ALIGNMENT: bool = REQUIRE_1D_ALIGNMENT


DEFAULT_PARAMS = SignalParams()


def params_to_dict(params: SignalParams) -> dict:
    """Plain-dict view of a SignalParams (tuples -> lists for JSON safety)."""
    return {k: list(v) if isinstance(v, tuple) else v
            for k, v in asdict(params).items()}

MIN_BARS = 60      # need EMA50 + ADX(14) warmup on the trading frame
MIN_BARS_1D = 55   # need EMA50 warmup on the daily frame

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
INDICATOR_COLUMNS = [
    "ema_9", "ema_21", "ema_50",
    "rsi_14", "atr_14",
    "adx", "di_plus", "di_minus",
    "macd_hist", "obv", "vwap_14", "vol_sma_20",
]


# ------------------------------------------------------------ indicatori

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with the indicator columns the engine relies on.

    Same ``ta`` calls (and column names) as ``signal_check.compute_indicators``,
    but kept as per-bar series instead of last-bar scalars.
    """
    df = df.copy()
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    for n in (9, 21, 50):
        df[f"ema_{n}"] = ta.trend.EMAIndicator(c, n).ema_indicator()
    df["rsi_14"] = ta.momentum.RSIIndicator(c, 14).rsi()
    df["atr_14"] = ta.volatility.AverageTrueRange(h, l, c, 14).average_true_range()
    adx = ta.trend.ADXIndicator(h, l, c, 14)
    df["adx"], df["di_plus"], df["di_minus"] = adx.adx(), adx.adx_pos(), adx.adx_neg()
    df["macd_hist"] = ta.trend.MACD(c).macd_diff()
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    df["vwap_14"] = ta.volume.VolumeWeightedAveragePrice(h, l, c, v, 14).volume_weighted_average_price()
    df["vol_sma_20"] = v.rolling(20).mean()
    return df


def _daily_emas(df_1d: pd.DataFrame) -> tuple[float, float] | None:
    """EMA20/EMA50 on the daily close, or None if data is insufficient."""
    if df_1d is None or len(df_1d) < MIN_BARS_1D:
        return None
    c = df_1d["close"]
    ema20 = ta.trend.EMAIndicator(c, 20).ema_indicator().iloc[-1]
    ema50 = ta.trend.EMAIndicator(c, 50).ema_indicator().iloc[-1]
    if pd.isna(ema20) or pd.isna(ema50):
        return None
    return float(ema20), float(ema50)


# ------------------------------------------------------------- evaluation

def _cross_within(df: pd.DataFrame, direction: str, lookback: int) -> bool:
    """True if EMA9 crossed EMA21 (in ``direction``) within the last bars."""
    n = len(df)
    for i in range(n - lookback, n):
        prev_diff = df["ema_9"].iloc[i - 1] - df["ema_21"].iloc[i - 1]
        diff = df["ema_9"].iloc[i] - df["ema_21"].iloc[i]
        if direction == "LONG" and prev_diff <= 0 < diff:
            return True
        if direction == "SHORT" and prev_diff >= 0 > diff:
            return True
    return False


def evaluate_symbol(
    df: pd.DataFrame,
    df_1d: pd.DataFrame | None = None,
    *,
    min_confidence: float = 70.0,
    timeframe: str | None = None,
    params: SignalParams | None = None,
) -> dict:
    """Evaluate one symbol on pre-fetched OHLCV data.

    ``df`` is the trading-timeframe frame (columns open/high/low/close/volume,
    plus optionally the pre-computed indicator columns — if present they are
    used as-is, which lets unit tests and the backtester skip recomputation).
    The LAST ROW IS ALWAYS DROPPED as a possibly in-progress candle (see
    module docstring). Returns a dict; never raises for ordinary "no signal"
    outcomes — check ``passed_filter`` / ``direction`` / ``reasons``.

    ``params`` overrides the module-level tunables; None uses the defaults,
    which is byte-for-byte the historical behaviour.
    """
    p = params if params is not None else DEFAULT_PARAMS
    df = df.iloc[:-1].copy()
    if not all(col in df.columns for col in INDICATOR_COLUMNS):
        df = add_indicators(df)
    if df_1d is not None:
        df_1d = df_1d.iloc[:-1]

    result = {
        "symbol": None,  # filled in by scan_symbols / the caller
        "timeframe": timeframe,
        "min_confidence": min_confidence,
        "passed_filter": False,
        "direction": "NONE",
        "confidence": 0.0,
        "reasons": [],
        "entry_type": None,
        "market_regime": None,
        "confluence_indicators": [],
        "divergent_indicators": [],
        "indicators": {},
    }

    if len(df) < MIN_BARS:
        result["reasons"].append(f"insufficient_bars ({len(df)} < {MIN_BARS})")
        return result

    last = df.iloc[-1]
    close = float(last["close"])
    atr = float(last["atr_14"])
    rsi = float(last["rsi_14"])
    adx = float(last["adx"])
    ema9 = float(last["ema_9"])
    ema21 = float(last["ema_21"])
    ema50 = float(last["ema_50"])
    di_plus = float(last["di_plus"])
    di_minus = float(last["di_minus"])
    vwap = float(last["vwap_14"])
    vol_sma = float(last["vol_sma_20"])
    volume = float(last["volume"])
    volume_ratio = volume / vol_sma if vol_sma else 0.0
    macd_hist = float(last["macd_hist"])
    macd_hist_prev = float(df["macd_hist"].iloc[-2])
    obv_slope_5 = float(df["obv"].iloc[-1] - df["obv"].iloc[-6])

    result["indicators"] = {
        "close": close, "atr": atr, "rsi": rsi, "adx": adx,
        "ema9": ema9, "ema21": ema21, "ema50": ema50,
        "vwap": vwap, "volume_ratio": volume_ratio,
        "di_plus": di_plus, "di_minus": di_minus,
        "macd_hist": macd_hist, "macd_hist_prev": macd_hist_prev,
        "obv_slope_5": obv_slope_5,
    }

    reasons = result["reasons"]

    # --- 1. trend filter -------------------------------------------------
    ema_long = ema9 > ema21 > ema50
    ema_short = ema9 < ema21 < ema50
    if not (ema_long or ema_short):
        reasons.append(
            f"no_ema_alignment (ema9={ema9:.6g}, ema21={ema21:.6g}, ema50={ema50:.6g})"
        )
    if adx <= p.ADX_MIN:
        reasons.append(f"adx_too_weak ({adx:.1f} <= {p.ADX_MIN})")
    if ema_long and di_plus <= di_minus:
        reasons.append(f"di_disagrees_with_long (+DI {di_plus:.1f} <= -DI {di_minus:.1f})")
    if ema_short and di_minus <= di_plus:
        reasons.append(f"di_disagrees_with_short (-DI {di_minus:.1f} <= +DI {di_plus:.1f})")

    direction = "NONE"
    if ema_long and adx > p.ADX_MIN and di_plus > di_minus:
        direction = "LONG"
    elif ema_short and adx > p.ADX_MIN and di_minus > di_plus:
        direction = "SHORT"
    if direction == "NONE":
        return result

    # --- 1b. hard daily-trend alignment filter ----------------------------
    daily = _daily_emas(df_1d)
    if p.REQUIRE_1D_ALIGNMENT and daily is not None:
        ema20_1d, ema50_1d = daily
        result["indicators"]["ema20_1d"] = ema20_1d
        result["indicators"]["ema50_1d"] = ema50_1d
        aligned = ema20_1d > ema50_1d if direction == "LONG" else ema20_1d < ema50_1d
        if not aligned:
            reasons.append(
                f"daily_trend_misaligned ({direction} vs 1d EMA20 "
                f"{ema20_1d:.6g} {'>' if direction == 'LONG' else '<'} EMA50 {ema50_1d:.6g})"
            )
            return result

    # --- 2. entry timing -------------------------------------------------
    pullback = abs(close - ema21) <= p.PULLBACK_ATR_FRAC * atr
    cross = _cross_within(df, direction, p.CROSS_LOOKBACK_BARS)
    if not (pullback or cross):
        reasons.append(
            f"no_entry_timing (|close-ema21|={abs(close - ema21):.6g} > "
            f"{p.PULLBACK_ATR_FRAC}xATR={p.PULLBACK_ATR_FRAC * atr:.6g}, no EMA9/21 cross "
            f"in last {p.CROSS_LOOKBACK_BARS} bars)"
        )
    rsi_lo, rsi_hi = p.RSI_LONG if direction == "LONG" else p.RSI_SHORT
    if not (rsi_lo <= rsi <= rsi_hi):
        reasons.append(f"rsi_outside_guard (rsi={rsi:.1f} not in [{rsi_lo}, {rsi_hi}])")
    if reasons:
        return result

    result["passed_filter"] = True
    result["direction"] = direction
    result["entry_type"] = "pullback_to_ema21" if pullback else "ema9_21_cross"
    result["market_regime"] = "trending_long" if direction == "LONG" else "trending_short"

    # --- 3. confluence score ---------------------------------------------
    checks: list[tuple[str, str, bool]] = []  # (hit name, miss name, passed)
    macd_ok = (
        (macd_hist > 0 and macd_hist > macd_hist_prev)
        if direction == "LONG"
        else (macd_hist < 0 and macd_hist < macd_hist_prev)
    )
    checks.append(("macd_histogram_confirms", "macd_histogram_divergent", macd_ok))
    vwap_ok = close > vwap if direction == "LONG" else close < vwap
    checks.append(("price_above_vwap" if direction == "LONG" else "price_below_vwap",
                   "price_wrong_side_of_vwap", vwap_ok))
    checks.append(("volume_above_average", "volume_below_average",
                   volume > p.VOLUME_MULT * vol_sma))
    obv_ok = obv_slope_5 > 0 if direction == "LONG" else obv_slope_5 < 0
    checks.append(("obv_slope_confirms", "obv_slope_divergent", obv_ok))

    daily = _daily_emas(df_1d)
    if daily is not None:
        ema20_1d, ema50_1d = daily
        result["indicators"]["ema20_1d"] = ema20_1d
        result["indicators"]["ema50_1d"] = ema50_1d
        daily_ok = ema20_1d > ema50_1d if direction == "LONG" else ema20_1d < ema50_1d
        checks.append(("daily_trend_aligned", "daily_trend_divergent", daily_ok))
    # df_1d is None (or too short): daily check is skipped — no points, not a failure.

    hits = 0
    for hit_name, miss_name, ok in checks:
        if ok:
            hits += 1
            result["confluence_indicators"].append(hit_name)
        else:
            result["divergent_indicators"].append(miss_name)
    result["confidence"] = float(
        min(p.CONFIDENCE_CAP, p.CONFLUENCE_BASE + p.CONFLUENCE_STEP * hits)
    )
    return result


# ------------------------------------------------------------- trade plan

def _build_reasoning(ev: dict, plan: dict, p: SignalParams) -> str:
    ind = ev["indicators"]
    direction = ev["direction"]
    return (
        f"{direction} setup on {plan['symbol']} ({plan['timeframe']}): "
        f"EMA stack aligned (EMA9 {ind['ema9']:.6g} "
        f"{'>' if direction == 'LONG' else '<'} EMA21 {ind['ema21']:.6g} "
        f"{'>' if direction == 'LONG' else '<'} EMA50 {ind['ema50']:.6g}), "
        f"ADX {ind['adx']:.1f} > {p.ADX_MIN} with "
        f"{'+DI' if direction == 'LONG' else '-DI'} leading "
        f"(+DI {ind['di_plus']:.1f} / -DI {ind['di_minus']:.1f}). "
        f"Entry via {ev['entry_type']} at close {ind['close']:.6g}, "
        f"RSI {ind['rsi']:.1f} inside the guard band. "
        f"Confidence {plan['confidence']:.0f} from confluence: "
        f"{', '.join(ev['confluence_indicators']) or 'none'}"
        + (
            f"; divergent: {', '.join(ev['divergent_indicators'])}."
            if ev["divergent_indicators"]
            else "."
        )
        + f" ATR {ind['atr']:.6g} sizes the risk: SL {plan['stop_loss']:.6g} "
        f"({p.ATR_SL_MULT}xATR), TP1 {plan['take_profit_1']:.6g} ({p.ATR_TP1_MULT}xATR, "
        f"RR {plan['risk_reward_tp1']:.2f}), TP2 {plan['take_profit_2']:.6g} "
        f"({p.ATR_TP2_MULT}xATR, RR {plan['risk_reward_tp2']:.2f})."
    )


def build_trade_plan(
    evaluation: dict,
    *,
    params: SignalParams | None = None,
) -> dict | None:
    """Turn a passing ``evaluate_symbol`` result into a strategist-shaped plan.

    Returns None when the evaluation did not pass the filter or its
    confidence is below ``min_confidence`` (read back from the evaluation
    dict, defaulting to 70.0). ``params`` overrides the module-level
    SL/TP tunables; None uses the defaults.
    """
    p = params if params is not None else DEFAULT_PARAMS
    if not evaluation.get("passed_filter"):
        return None
    confidence = float(evaluation["confidence"])
    if confidence < float(evaluation.get("min_confidence", 70.0)):
        return None

    direction = evaluation["direction"]
    ind = evaluation["indicators"]
    entry = ind["close"]
    atr = ind["atr"]
    sign = 1.0 if direction == "LONG" else -1.0
    stop_loss = entry - sign * p.ATR_SL_MULT * atr
    take_profit_1 = entry + sign * p.ATR_TP1_MULT * atr
    take_profit_2 = entry + sign * p.ATR_TP2_MULT * atr
    risk = abs(entry - stop_loss)
    rr_tp1 = abs(take_profit_1 - entry) / risk
    rr_tp2 = abs(take_profit_2 - entry) / risk

    plan = {
        "valid": True,
        "invalid_reason": None,
        "symbol": evaluation.get("symbol"),
        "timeframe": evaluation.get("timeframe"),
        "signal": "BUY" if direction == "LONG" else "SELL",
        "confidence": confidence,
        "entry_price": entry,
        "stop_loss": stop_loss,
        "take_profit_1": take_profit_1,
        "take_profit_2": take_profit_2,
        "risk_reward_tp1": rr_tp1,
        "risk_reward_tp2": rr_tp2,
        "market_regime": evaluation["market_regime"],
        "trend_direction": direction,
        "confluence_indicators": list(evaluation["confluence_indicators"]),
        "divergent_indicators": list(evaluation["divergent_indicators"]),
        "reasoning": None,  # filled below
        "context_factors": {
            "entry_type": evaluation["entry_type"],
            "adx": ind["adx"],
            "di_plus": ind["di_plus"],
            "di_minus": ind["di_minus"],
            "macd_hist": ind["macd_hist"],
            "macd_hist_prev": ind["macd_hist_prev"],
            "obv_slope_5": ind["obv_slope_5"],
            "ema20_1d": ind.get("ema20_1d"),
            "ema50_1d": ind.get("ema50_1d"),
        },
        "indicators": {
            "close": ind["close"],
            "atr": ind["atr"],
            "rsi": ind["rsi"],
            "adx": ind["adx"],
            "ema9": ind["ema9"],
            "ema21": ind["ema21"],
            "ema50": ind["ema50"],
            "vwap": ind["vwap"],
            "volume_ratio": ind["volume_ratio"],
        },
    }
    plan["reasoning"] = _build_reasoning(evaluation, plan, p)
    return plan


# ------------------------------------------------------------------ scan

def scan_symbols(
    data: dict[str, pd.DataFrame],
    data_1d: dict[str, pd.DataFrame] | None = None,
    *,
    min_confidence: float = 70.0,
    cooldown_symbols: set[str] | None = None,
    timeframe: str | None = None,
    params: SignalParams | None = None,
) -> dict:
    """Evaluate every symbol in ``data`` and pick the best trade plan.

    Symbols in ``cooldown_symbols`` are skipped entirely. Per-symbol failures
    are collected in ``errors`` and never abort the scan. Candidates are
    ranked by confidence (ties -> higher risk_reward_tp1). Returns
    ``{"best": plan | None, "scores": [...], "errors": [...]}``.
    ``params`` overrides the module-level tunables; None uses the defaults.
    """
    p = params if params is not None else DEFAULT_PARAMS
    cooldown = cooldown_symbols or set()
    scores: list[dict] = []
    errors: list[dict] = []
    candidates: list[dict] = []

    for symbol, df in data.items():
        if symbol in cooldown:
            continue
        try:
            ev = evaluate_symbol(
                df,
                (data_1d or {}).get(symbol),
                min_confidence=min_confidence,
                timeframe=timeframe,
                params=p,
            )
        except Exception as exc:  # per-symbol failure must not abort the scan
            errors.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
            continue
        ev["symbol"] = symbol
        scores.append({
            "symbol": symbol,
            "passed_filter": ev["passed_filter"],
            "confidence": ev["confidence"],
            "direction": ev["direction"],
            "reasons": ev["reasons"],
        })
        if ev["passed_filter"] and ev["confidence"] >= min_confidence:
            candidates.append(ev)

    def _rank_key(ev: dict) -> tuple[float, float]:
        atr = ev["indicators"]["atr"]
        rr_tp1 = (p.ATR_TP1_MULT * atr) / (p.ATR_SL_MULT * atr) if atr else 0.0
        return (ev["confidence"], rr_tp1)

    best = None
    if candidates:
        best = build_trade_plan(max(candidates, key=_rank_key), params=p)

    return {"best": best, "scores": scores, "errors": errors}
