"""Deterministic multi-timeframe volatility breakout engine.

Pure, importable module: no CLI, no printing, no network at import time and
no network anywhere in this file. Callers fetch OHLCV themselves and pass
plain DataFrames in (columns ``ts``/``open``/``high``/``low``/``close``/
``volume``, same shape as ``backtest_core.load_candles`` produces), so every
function here is testable offline.

Strategy (see docs/superpowers/specs/2026-08-13-volatility-breakout-engine-design.md):

1. 4h regime (must pass): close vs SMA(100) with the SMA sloping in the same
   direction over SLOPE_BARS bars, plus a Kaufman Efficiency Ratio over
   ER_BARS >= ER_MIN as the chop filter (replaces ADX).
2. 1h setup: Bollinger(20, 2) bandwidth squeeze (minimum bandwidth inside the
   last BB_LOOKBACK bars sitting in the lowest BB_SQUEEZE_PCTL quantile of
   that window), then expansion: bandwidth rising for 2 consecutive bars, the
   last closed 1h bar ranging > EXPANSION_RANGE_MULT x ATR(14) with volume >
   EXPANSION_VOL_MULT x SMA20, and the expansion bar's direction agreeing with
   the 4h regime.
3. 15m trigger: breakout-and-retest against the nearest level in the regime
   direction — anchored VWAP (from the compression-window start), prior-day
   high/low, or the 1h consolidation high/low. LONG: a 15m close above the
   level with volume > TRIGGER_VOL_MULT x SMA20, then within
   RETEST_WINDOW_BARS a bar whose low comes within RETEST_ATR_FRAC x ATR(14)
   of the level and closes back above it. Entry = the retest bar's close.
   The retest must land on the last closed bar for the signal to be
   actionable; setups that broke out and never retested within the window are
   reported as expired, never chased.
4. Structural trade plan: SL on the far side of the retest bar / broken level
   plus SL_BUFFER_ATR x ATR(15m), TP1 = TP1_RISK_MULT x risk, TP2 =
   TP2_RISK_MULT x risk, rejected when RR to TP1 < MIN_RR_TP1 (a parameter-
   consistency floor: with TP1_RISK_MULT >= MIN_RR_TP1 it passes by
   construction and only bites on misconfigured params or degenerate risk).
5. Funding filter: reject LONG when the current 8h funding rate exceeds
   +FUNDING_EXTREME, SHORT when below -FUNDING_EXTREME. funding_rate=None
   means unknown: the filter is skipped and the evaluation/plan is flagged
   ``funding_unknown`` (never silently trusted).

Confidence carries a deterministic structural-quality score so vetting works
uniformly across engines: base CONFIDENCE_BASE, + ER_CONFIDENCE_WEIGHT x
min(1, ER), + FAST_RETEST_BONUS if the retest held within FAST_RETEST_BARS of
the breakout, + VOLUME_SPIKE_BONUS if the breakout bar's volume exceeded
VOLUME_SPIKE_MULT x SMA20; capped at CONFIDENCE_CAP.

In-progress candle policy: ``evaluate_breakout`` ALWAYS drops the last row of
every input frame before doing anything, treating it as a possibly still-
forming candle (same convention as ``signal_engine.evaluate_symbol``).

``symbol`` / ``timeframe``: ``evaluate_breakout`` does not know the symbol; it
returns ``symbol=None`` and ``scan_breakout`` fills it in. ``timeframe`` is a
keyword-only pass-through stored on the evaluation dict so that
``build_trade_plan`` can emit it in the plan.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import ta

# ----------------------------------------------------------------- tunables

SMA_SLOW = 100               # 4h regime SMA
SLOPE_BARS = 10              # SMA slope lookback (4h)
ER_BARS = 20                 # Kaufman ER window (4h)
ER_MIN = 0.30                # chop filter
BB_LEN = 20                  # 1h Bollinger
BB_MULT = 2.0
BB_SQUEEZE_PCTL = 0.20       # bandwidth in lowest quintile of lookback
BB_LOOKBACK = 100
EXPANSION_RANGE_MULT = 1.5   # 1h range > 1.5 x ATR(14)1h
EXPANSION_VOL_MULT = 1.5     # 1h volume > 1.5 x SMA20
TRIGGER_VOL_MULT = 1.5       # 15m breakout bar volume
RETEST_ATR_FRAC = 0.25       # retest touch tolerance (15m ATR)
RETEST_WINDOW_BARS = 6       # setup expiry
SL_BUFFER_ATR = 0.5          # SL beyond broken level (15m ATR)
TP1_RISK_MULT = 1.5
TP2_RISK_MULT = 2.5
MIN_RR_TP1 = 1.5             # in-engine rejection floor
FUNDING_EXTREME = 0.0005     # 0.05% per 8h

# Supporting knobs (named here rather than hardcoded; see module docstring).
SQUEEZE_MAX_AGE_BARS = 20    # argmin of bandwidth must be this recent (1h bars)
ATR_LEN = 14                 # ATR length on both 1h and 15m
VOL_SMA_LEN = 20             # volume SMA length on both 1h and 15m
CONFIDENCE_BASE = 55
CONFIDENCE_CAP = 95
ER_CONFIDENCE_WEIGHT = 25
FAST_RETEST_BARS = 3
FAST_RETEST_BONUS = 10
VOLUME_SPIKE_MULT = 2.0
VOLUME_SPIKE_BONUS = 10


@dataclass(frozen=True)
class BreakoutParams:
    """Tunable knobs for the engine; defaults are the module constants above.

    Construct overrides from a dict with ``BreakoutParams(**{...})``; use
    ``params_to_dict`` for a plain-dict echo (JSON-serialisable).
    """

    SMA_SLOW: int = SMA_SLOW
    SLOPE_BARS: int = SLOPE_BARS
    ER_BARS: int = ER_BARS
    ER_MIN: float = ER_MIN
    BB_LEN: int = BB_LEN
    BB_MULT: float = BB_MULT
    BB_SQUEEZE_PCTL: float = BB_SQUEEZE_PCTL
    BB_LOOKBACK: int = BB_LOOKBACK
    EXPANSION_RANGE_MULT: float = EXPANSION_RANGE_MULT
    EXPANSION_VOL_MULT: float = EXPANSION_VOL_MULT
    TRIGGER_VOL_MULT: float = TRIGGER_VOL_MULT
    RETEST_ATR_FRAC: float = RETEST_ATR_FRAC
    RETEST_WINDOW_BARS: int = RETEST_WINDOW_BARS
    SL_BUFFER_ATR: float = SL_BUFFER_ATR
    TP1_RISK_MULT: float = TP1_RISK_MULT
    TP2_RISK_MULT: float = TP2_RISK_MULT
    MIN_RR_TP1: float = MIN_RR_TP1
    FUNDING_EXTREME: float = FUNDING_EXTREME
    SQUEEZE_MAX_AGE_BARS: int = SQUEEZE_MAX_AGE_BARS
    ATR_LEN: int = ATR_LEN
    VOL_SMA_LEN: int = VOL_SMA_LEN
    CONFIDENCE_BASE: float = CONFIDENCE_BASE
    CONFIDENCE_CAP: float = CONFIDENCE_CAP
    ER_CONFIDENCE_WEIGHT: float = ER_CONFIDENCE_WEIGHT
    FAST_RETEST_BARS: int = FAST_RETEST_BARS
    FAST_RETEST_BONUS: float = FAST_RETEST_BONUS
    VOLUME_SPIKE_MULT: float = VOLUME_SPIKE_MULT
    VOLUME_SPIKE_BONUS: float = VOLUME_SPIKE_BONUS


DEFAULT_PARAMS = BreakoutParams()


def params_to_dict(params: BreakoutParams) -> dict:
    """Plain-dict view of a BreakoutParams (JSON-safe)."""
    return asdict(params)

# Minimum closed bars per frame (after the in-progress candle is dropped).
MIN_BARS_4H = SMA_SLOW + SLOPE_BARS + 1
MIN_BARS_1H = BB_LEN + BB_LOOKBACK + 2
MIN_BARS_15M = VOL_SMA_LEN + RETEST_WINDOW_BARS + ATR_LEN

# Tolerance when comparing RR against the MIN_RR_TP1 floor: TP1 is derived
# from risk, so RR == TP1_RISK_MULT up to floating-point noise.
RR_EPS = 1e-9


# --------------------------------------------------------------- indicators

def _kaufman_er(closes: pd.Series, bars: int) -> float:
    """Efficiency Ratio over the last ``bars`` bars: |net move| / sum|moves|."""
    window = closes.iloc[-(bars + 1):]
    net = abs(float(window.iloc[-1]) - float(window.iloc[0]))
    denom = float(window.diff().abs().iloc[1:].sum())
    if denom == 0.0:
        return 0.0
    return net / denom


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    return ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], length
    ).average_true_range()


def _anchored_vwap(df_15m: pd.DataFrame, anchor_ts) -> float | None:
    """VWAP of typical price over all 15m bars at/after ``anchor_ts``."""
    if "ts" not in df_15m.columns:
        return None
    seg = df_15m[df_15m["ts"] >= anchor_ts]
    if seg.empty:
        return None
    vol = float(seg["volume"].sum())
    if vol <= 0.0:
        return None
    typical = (seg["high"] + seg["low"] + seg["close"]) / 3.0
    return float((typical * seg["volume"]).sum() / vol)


def _prior_day_high_low(df_15m: pd.DataFrame) -> tuple[float | None, float | None]:
    """High/low of the previous UTC day, from 15m bars."""
    if "ts" not in df_15m.columns or df_15m.empty:
        return None, None
    days = df_15m["ts"].dt.normalize()
    prev_day = days.iloc[-1] - pd.Timedelta(days=1)
    prev = df_15m[days == prev_day]
    if prev.empty:
        return None, None
    return float(prev["high"].max()), float(prev["low"].min())


# ------------------------------------------------------------- evaluation

def evaluate_breakout(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    funding_rate: float | None,
    *,
    params: BreakoutParams | None = None,
    timeframe: str = "15m",
) -> dict:
    """Evaluate one symbol on pre-fetched 15m/1h/4h OHLCV data.

    The LAST ROW OF EVERY FRAME IS ALWAYS DROPPED as a possibly in-progress
    candle (see module docstring). Returns a dict mirroring
    ``signal_engine.evaluate_symbol``; never raises for ordinary "no signal"
    outcomes — check ``passed_filter`` / ``direction`` / ``reasons``.
    ``min_confidence`` is echoed for shape compatibility but NOT enforced:
    confidence here is a structural score, not a filter.

    On a pass the evaluation also carries the structural plan fields
    (``entry_price`` / ``stop_loss`` / ``take_profit_*`` / ``risk_reward_*``)
    which ``build_trade_plan`` reshapes into a strategist-shaped plan.
    """
    p = params if params is not None else DEFAULT_PARAMS
    df_15m = df_15m.iloc[:-1].copy()
    df_1h = df_1h.iloc[:-1].copy()
    df_4h = df_4h.iloc[:-1].copy()

    result = {
        "symbol": None,  # filled in by scan_breakout / the caller
        "timeframe": timeframe,
        "min_confidence": None,  # structural score only; no confidence floor
        "passed_filter": False,
        "direction": "NONE",
        "confidence": 0.0,
        "reasons": [],
        "entry_type": None,
        "market_regime": None,
        "funding_unknown": funding_rate is None,
        "confluence_indicators": [],
        "divergent_indicators": [],
        "indicators": {},
    }
    reasons = result["reasons"]
    ind = result["indicators"]

    if len(df_4h) < MIN_BARS_4H:
        reasons.append(f"insufficient_bars_4h ({len(df_4h)} < {MIN_BARS_4H})")
        return result
    if len(df_1h) < MIN_BARS_1H:
        reasons.append(f"insufficient_bars_1h ({len(df_1h)} < {MIN_BARS_1H})")
        return result
    if len(df_15m) < MIN_BARS_15M:
        reasons.append(f"insufficient_bars_15m ({len(df_15m)} < {MIN_BARS_15M})")
        return result

    # --- 1. 4h regime: direction (price/SMA + slope) and strength (ER) -----
    close4 = df_4h["close"]
    sma = close4.rolling(p.SMA_SLOW).mean()
    sma_last = float(sma.iloc[-1])
    sma_prev = float(sma.iloc[-1 - p.SLOPE_BARS])
    px4 = float(close4.iloc[-1])
    er = _kaufman_er(close4, p.ER_BARS)
    ind["er"] = er
    ind["sma_slow"] = sma_last
    ind["funding_rate"] = funding_rate

    if er < p.ER_MIN:
        reasons.append(f"chop_regime (ER {er:.2f} < {p.ER_MIN})")
        return result
    if px4 > sma_last and sma_last > sma_prev:
        direction = "LONG"
    elif px4 < sma_last and sma_last < sma_prev:
        direction = "SHORT"
    else:
        reasons.append(
            f"no_regime_direction (close {px4:.6g} vs SMA{p.SMA_SLOW} {sma_last:.6g}, "
            f"SMA slope {'up' if sma_last > sma_prev else 'down'} over {p.SLOPE_BARS} bars)"
        )
        return result

    # --- funding veto (None = unknown: skip, flagged funding_unknown) ------
    if funding_rate is not None:
        if direction == "LONG" and funding_rate > p.FUNDING_EXTREME:
            reasons.append(
                f"funding_extreme_long (funding {funding_rate:.5f} > {p.FUNDING_EXTREME})"
            )
            return result
        if direction == "SHORT" and funding_rate < -p.FUNDING_EXTREME:
            reasons.append(
                f"funding_extreme_short (funding {funding_rate:.5f} < {-p.FUNDING_EXTREME})"
            )
            return result

    # --- 2. 1h setup: bandwidth squeeze then expansion ---------------------
    o1, h1, l1, c1, v1 = (df_1h[k] for k in ("open", "high", "low", "close", "volume"))
    n1 = len(df_1h)
    mid = c1.rolling(p.BB_LEN).mean()
    std = c1.rolling(p.BB_LEN).std(ddof=0)
    upper = mid + p.BB_MULT * std
    lower = mid - p.BB_MULT * std
    bw = ((upper - lower) / mid).to_numpy()
    ind["bb_bandwidth"] = float(bw[-1])

    window = bw[n1 - p.BB_LOOKBACK:]
    squeeze_q = float(np.quantile(window, p.BB_SQUEEZE_PCTL))
    j = n1 - p.BB_LOOKBACK + int(np.argmin(window))  # compression start (VWAP anchor)
    bw_min = float(bw[j])
    ind["bb_bandwidth_min"] = bw_min

    # bw_min <= squeeze_q holds by construction (the min is always in the
    # lowest quantile of its own window); the selective conditions are the
    # recency of the squeeze and the expansion that follows it.
    if not bw_min <= squeeze_q:  # pragma: no cover - defensive
        reasons.append(
            f"no_squeeze (min bandwidth {bw_min:.4g} > "
            f"{p.BB_SQUEEZE_PCTL:.0%} quantile {squeeze_q:.4g})"
        )
        return result
    if j < n1 - 2 - p.SQUEEZE_MAX_AGE_BARS:
        reasons.append(
            f"squeeze_stale (bandwidth low {n1 - 1 - j} bars ago "
            f"> {p.SQUEEZE_MAX_AGE_BARS})"
        )
        return result
    if not (bw[-2] > bw[-3] and bw[-1] > bw[-2]):
        reasons.append("bandwidth_not_expanding (need 2 consecutive rises after the squeeze)")
        return result

    atr1h = _atr(df_1h, p.ATR_LEN)
    atr1h_last = float(atr1h.iloc[-1])
    bar_range = float(h1.iloc[-1]) - float(l1.iloc[-1])
    if not bar_range > p.EXPANSION_RANGE_MULT * atr1h_last:
        reasons.append(
            f"expansion_range_too_small ({bar_range:.6g} <= "
            f"{p.EXPANSION_RANGE_MULT}xATR1h {p.EXPANSION_RANGE_MULT * atr1h_last:.6g})"
        )
        return result
    vol_sma1h = v1.rolling(p.VOL_SMA_LEN).mean()
    vol_sma1h_last = float(vol_sma1h.iloc[-1])
    if not float(v1.iloc[-1]) > p.EXPANSION_VOL_MULT * vol_sma1h_last:
        reasons.append(
            f"expansion_volume_too_small ({float(v1.iloc[-1]):.6g} <= "
            f"{p.EXPANSION_VOL_MULT}xSMA20 {p.EXPANSION_VOL_MULT * vol_sma1h_last:.6g})"
        )
        return result
    if direction == "LONG" and not float(c1.iloc[-1]) > float(o1.iloc[-1]):
        reasons.append("expansion_direction_disagrees (bearish 1h expansion bar vs LONG regime)")
        return result
    if direction == "SHORT" and not float(c1.iloc[-1]) < float(o1.iloc[-1]):
        reasons.append("expansion_direction_disagrees (bullish 1h expansion bar vs SHORT regime)")
        return result

    # Consolidation = the compression window up to (excluding) the expansion bar.
    cons = df_1h.iloc[j:n1 - 1]
    cons_high = float(cons["high"].max())
    cons_low = float(cons["low"].min())
    anchor_ts = df_1h["ts"].iloc[j] if "ts" in df_1h.columns else None

    # --- 3. 15m trigger: breakout-and-retest against the nearest level -----
    c15, h15, l15, v15 = (df_15m[k] for k in ("close", "high", "low", "volume"))
    n15 = len(df_15m)
    r = n15 - 1  # last closed bar; the retest must land here to be actionable
    atr15 = _atr(df_15m, p.ATR_LEN)
    atr15_last = float(atr15.iloc[-1])
    vol_sma15 = v15.rolling(p.VOL_SMA_LEN).mean()
    close_now = float(c15.iloc[-1])
    ind["close"] = close_now
    ind["atr_15m"] = atr15_last

    avwap = _anchored_vwap(df_15m, anchor_ts) if anchor_ts is not None else None
    pdh, pdl = _prior_day_high_low(df_15m)
    levels: dict[str, float] = {}
    if avwap is not None:
        levels["avwap"] = avwap
    if direction == "LONG":
        if pdh is not None:
            levels["pdh"] = pdh
        levels["consolidation"] = cons_high
    else:
        if pdl is not None:
            levels["pdl"] = pdl
        levels["consolidation"] = cons_low

    if direction == "LONG":
        cands = {k: v for k, v in levels.items() if v < close_now}
        if not cands:
            reasons.append(
                f"no_level_below_price (no avwap/pdh/consolidation level below {close_now:.6g})"
            )
            return result
        level_type = max(cands, key=cands.get)  # nearest below = highest
    else:
        cands = {k: v for k, v in levels.items() if v > close_now}
        if not cands:
            reasons.append(
                f"no_level_above_price (no avwap/pdl/consolidation level above {close_now:.6g})"
            )
            return result
        level_type = min(cands, key=cands.get)  # nearest above = lowest
    level = cands[level_type]
    ind["level"] = level
    ind["level_type"] = level_type

    def _crossed(i: int) -> bool:
        if direction == "LONG":
            return float(c15.iloc[i - 1]) <= level < float(c15.iloc[i])
        return float(c15.iloc[i - 1]) >= level > float(c15.iloc[i])

    # Most recent breakout bar inside the retest window (with volume).
    b = None
    for i in range(r - 1, max(r - 1 - p.RETEST_WINDOW_BARS, 0), -1):
        vsma = float(vol_sma15.iloc[i])
        if np.isnan(vsma) or vsma <= 0.0:
            continue
        if _crossed(i) and float(v15.iloc[i]) > p.TRIGGER_VOL_MULT * vsma:
            b = i
            break
    if b is None:
        last_cross = next((i for i in range(r - 1, 0, -1) if _crossed(i)), None)
        if last_cross is not None and r - last_cross > p.RETEST_WINDOW_BARS:
            reasons.append(
                f"setup_expired (breakout {r - last_cross} bars ago, outside the "
                f"{p.RETEST_WINDOW_BARS}-bar retest window)"
            )
        else:
            reasons.append(
                f"no_breakout (no 15m close "
                f"{'above' if direction == 'LONG' else 'below'} {level_type} "
                f"{level:.6g} with volume > {p.TRIGGER_VOL_MULT}xSMA20 "
                f"in the last {p.RETEST_WINDOW_BARS} bars)"
            )
        return result

    # Retest-and-hold on the last closed bar.
    tol = p.RETEST_ATR_FRAC * atr15_last
    if direction == "LONG":
        held = float(l15.iloc[r]) <= level + tol and float(c15.iloc[r]) > level
    else:
        held = float(h15.iloc[r]) >= level - tol and float(c15.iloc[r]) < level
    if not held:
        reasons.append(
            f"awaiting_retest (breakout {r - b} bars ago; last bar did not "
            f"retest-and-hold {level:.6g} within {p.RETEST_ATR_FRAC}xATR15)"
        )
        return result

    # --- 4. structural trade plan ------------------------------------------
    sign = 1.0 if direction == "LONG" else -1.0
    entry = close_now
    if direction == "LONG":
        stop = min(float(l15.iloc[r]), level) - p.SL_BUFFER_ATR * atr15_last
    else:
        stop = max(float(h15.iloc[r]), level) + p.SL_BUFFER_ATR * atr15_last
    risk = abs(entry - stop)
    if risk <= 0.0:  # pragma: no cover - defensive; SL is structurally beyond entry
        reasons.append("degenerate_risk (entry == stop)")
        return result
    tp1 = entry + sign * p.TP1_RISK_MULT * risk
    tp2 = entry + sign * p.TP2_RISK_MULT * risk
    rr_tp1 = abs(tp1 - entry) / risk
    rr_tp2 = abs(tp2 - entry) / risk
    if rr_tp1 < p.MIN_RR_TP1 - RR_EPS:
        reasons.append(f"rr_below_floor (RR to TP1 {rr_tp1:.2f} < {p.MIN_RR_TP1})")
        return result

    # --- 5. confidence: deterministic structural-quality score -------------
    retest_bars = r - b
    breakout_vol = float(v15.iloc[b])
    breakout_vsma = float(vol_sma15.iloc[b])
    fast_retest = retest_bars <= p.FAST_RETEST_BARS
    volume_spike = breakout_vol > p.VOLUME_SPIKE_MULT * breakout_vsma
    confidence = p.CONFIDENCE_BASE + p.ER_CONFIDENCE_WEIGHT * min(1.0, er)
    if fast_retest:
        confidence += p.FAST_RETEST_BONUS
    if volume_spike:
        confidence += p.VOLUME_SPIKE_BONUS
    confidence = float(min(p.CONFIDENCE_CAP, confidence))

    confluence = [
        "regime_trend_aligned",
        "efficiency_ratio_above_min",
        "bandwidth_squeeze",
        "expansion_range_confirmed",
        "expansion_volume_confirmed",
    ]
    divergent = []
    if fast_retest:
        confluence.append("fast_retest_hold")
    else:
        divergent.append("slow_retest")
    if volume_spike:
        confluence.append("breakout_volume_spike")
    else:
        divergent.append("breakout_volume_moderate")
    if funding_rate is None:
        divergent.append("funding_unknown")

    result.update({
        "passed_filter": True,
        "direction": direction,
        "confidence": confidence,
        "entry_type": f"{level_type}_retest",
        "market_regime": "breakout_long" if direction == "LONG" else "breakout_short",
        "confluence_indicators": confluence,
        "divergent_indicators": divergent,
        # structural plan fields consumed by build_trade_plan
        "entry_price": entry,
        "stop_loss": stop,
        "take_profit_1": tp1,
        "take_profit_2": tp2,
        "risk_reward_tp1": rr_tp1,
        "risk_reward_tp2": rr_tp2,
        "retest_bars": retest_bars,
        "breakout_volume_ratio": breakout_vol / breakout_vsma if breakout_vsma else 0.0,
    })
    return result


# ------------------------------------------------------------- trade plan

def _build_reasoning(ev: dict, plan: dict, p: BreakoutParams) -> str:
    ind = ev["indicators"]
    direction = ev["direction"]
    side = "above" if direction == "LONG" else "below"
    return (
        f"{direction} volatility breakout on {plan['symbol']} ({plan['timeframe']}): "
        f"4h regime aligned — close {side} SMA{p.SMA_SLOW} ({ind['sma_slow']:.6g}) "
        f"with the SMA sloping {side.replace('above', 'up').replace('below', 'down')} "
        f"and ER {ind['er']:.2f} >= {p.ER_MIN}. "
        f"1h Bollinger bandwidth squeezed to {ind['bb_bandwidth_min']:.4g} "
        f"(lowest {p.BB_SQUEEZE_PCTL:.0%} of the last {p.BB_LOOKBACK} bars), then "
        f"expanded with range and volume confirmation in the regime direction. "
        f"15m broke the {ind['level_type']} level {ind['level']:.6g} and retested it "
        f"in {ev['retest_bars']} bar(s); entry at the retest close "
        f"{plan['entry_price']:.6g}. "
        f"Confidence {plan['confidence']:.0f} from structural quality: "
        f"{', '.join(ev['confluence_indicators']) or 'none'}"
        + (
            f"; divergent: {', '.join(ev['divergent_indicators'])}."
            if ev["divergent_indicators"]
            else "."
        )
        + f" SL {plan['stop_loss']:.6g} ({p.SL_BUFFER_ATR}xATR15 beyond the broken "
        f"level), TP1 {plan['take_profit_1']:.6g} ({p.TP1_RISK_MULT}x risk, "
        f"RR {plan['risk_reward_tp1']:.2f}), TP2 {plan['take_profit_2']:.6g} "
        f"({p.TP2_RISK_MULT}x risk, RR {plan['risk_reward_tp2']:.2f})."
    )


def build_trade_plan(
    evaluation: dict,
    *,
    params: BreakoutParams | None = None,
) -> dict | None:
    """Turn a passing ``evaluate_breakout`` result into a strategist-shaped plan.

    Same key contract as ``signal_engine.build_trade_plan``; ``context_factors``
    additionally carries ``entry_type``, ``level``, ``level_type``,
    ``funding_rate``, ``er`` and ``funding_unknown``. Returns None when the
    evaluation did not pass the filter. ``min_confidence`` is NOT enforced —
    confidence is a structural score emitted for the vet, not a gate.
    """
    p = params if params is not None else DEFAULT_PARAMS
    if not evaluation.get("passed_filter"):
        return None

    ind = evaluation["indicators"]
    plan = {
        "valid": True,
        "invalid_reason": None,
        "symbol": evaluation.get("symbol"),
        "timeframe": evaluation.get("timeframe"),
        "signal": "BUY" if evaluation["direction"] == "LONG" else "SELL",
        "confidence": float(evaluation["confidence"]),
        "entry_price": evaluation["entry_price"],
        "stop_loss": evaluation["stop_loss"],
        "take_profit_1": evaluation["take_profit_1"],
        "take_profit_2": evaluation["take_profit_2"],
        "risk_reward_tp1": evaluation["risk_reward_tp1"],
        "risk_reward_tp2": evaluation["risk_reward_tp2"],
        "market_regime": evaluation["market_regime"],
        "trend_direction": evaluation["direction"],
        "confluence_indicators": list(evaluation["confluence_indicators"]),
        "divergent_indicators": list(evaluation["divergent_indicators"]),
        "reasoning": None,  # filled below
        "context_factors": {
            "entry_type": evaluation["entry_type"],
            "level": ind["level"],
            "level_type": ind["level_type"],
            "funding_rate": ind["funding_rate"],
            "er": ind["er"],
            "funding_unknown": evaluation["funding_unknown"],
        },
        "indicators": {
            "close": ind["close"],
            "er": ind["er"],
            "sma_slow": ind["sma_slow"],
            "bb_bandwidth": ind["bb_bandwidth"],
            "atr_15m": ind["atr_15m"],
            "level": ind["level"],
            "level_type": ind["level_type"],
            "funding_rate": ind["funding_rate"],
        },
    }
    plan["reasoning"] = _build_reasoning(evaluation, plan, p)
    return plan


# ------------------------------------------------------------------ scan

def scan_breakout(
    data: dict[str, dict],
    *,
    cooldown_symbols: set[str] | None = None,
    params: BreakoutParams | None = None,
) -> dict:
    """Evaluate every symbol in ``data`` and pick the best trade plan.

    ``data`` maps symbol -> {"15m": df, "1h": df, "4h": df, "funding":
    float | None}. Symbols in ``cooldown_symbols`` are skipped entirely.
    Per-symbol failures are collected in ``errors`` and never abort the scan.
    Candidates are ranked by confidence (ties -> higher risk_reward_tp1).
    Returns ``{"best": plan | None, "scores": [...], "errors": [...]}``.
    """
    p = params if params is not None else DEFAULT_PARAMS
    cooldown = cooldown_symbols or set()
    scores: list[dict] = []
    errors: list[dict] = []
    candidates: list[dict] = []

    for symbol, frames in data.items():
        if symbol in cooldown:
            continue
        try:
            ev = evaluate_breakout(
                frames["15m"], frames["1h"], frames["4h"],
                frames.get("funding"),
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
        if ev["passed_filter"]:
            candidates.append(ev)

    def _rank_key(ev: dict) -> tuple[float, float]:
        return (ev["confidence"], ev["risk_reward_tp1"])

    best = None
    if candidates:
        best = build_trade_plan(max(candidates, key=_rank_key), params=p)

    return {"best": best, "scores": scores, "errors": errors}
