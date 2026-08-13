"""Importable backtest core for the deterministic signal engine.

No argparse, no printing, no subprocesses — safe to import from the sweep
harness or the CLI runner (``ai_runners/backtest.py`` is a thin wrapper over
this module). Network access happens ONLY inside ``load_candles``; the
replay itself (``run_backtest``) is pure in-memory.

Simulation rules (unchanged from the original ai_runners/backtest.py):
  - One open trade per symbol at a time; a new signal may re-enter after a close.
  - Entry at the plan's entry_price (last closed bar's close).
  - From the next bar on, high/low decide whether SL or TP1 is hit first; if a
    single bar touches both, the SL is assumed hit first (conservative).
  - Risk is a fixed ``risk_pct`` of current equity per trade (compounding).
  - A win realizes RR_tp1 x risk; a loss realizes -risk — MINUS fees on both.

Fee model
---------
``fee_bps`` is the taker fee in basis points of NOTIONAL per side (default 10
= 0.1%, i.e. 0.2% round trip — conservative for AsterDEX-style perps).

The simulation risks ``risk`` (a currency amount) per trade and the stop is
``sl_frac = |entry - SL| / entry`` away from entry, so the position notional
implied by that risk is::

    notional = risk / sl_frac          # = margin x effective leverage
    fee_cost = 2 * (fee_bps / 1e4) * notional        # entry + exit, taker
             = 2 * (fee_bps / 1e4) * risk / sl_frac

That cost is deducted from EVERY trade's PnL, winners and losers alike:
a win nets ``rr * risk - fee_cost``, a loss nets ``-risk - fee_cost``.
In R-multiple terms (risk units) the fee is ``2 * fee_bps / 1e4 / sl_frac``
per trade, so tight stops (small sl_frac) pay proportionally more.

Candle caching
--------------
``load_candles`` pickles fetched OHLCV frames under ``cache_dir`` (default
``/tmp/astertrade_candles``), keyed by exchange/symbol/timeframe/days. A cache
file written on the SAME CALENDAR DAY (UTC) is reused as-is; anything older is
refetched. Callers that need warmup bars must include them in ``days``.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

WARMUP_BARS = 200        # extra history before the window so indicators are mature
WARMUP_BARS_1D = 80      # daily warmup (engine needs ~55 daily bars)
MIN_BARS = 60            # mirrors signal_engine.MIN_BARS

DEFAULT_CACHE_DIR = "/tmp/astertrade_candles"

# evaluate_symbol (with pre-computed indicator columns) never reads further
# back than 6 bars; slice this many trailing rows per evaluation instead of
# the whole history so the replay stays O(n) instead of O(n^2).
_EVAL_TAIL_BARS = 72


def to_ccxt_symbol(symbol: str, quote: str = "USDT") -> str:
    if "/" in symbol:
        return symbol
    for suffix in (quote, "USDT", "USDC", "USD"):
        if symbol.endswith(suffix):
            return f"{symbol[: -len(suffix)]}/{suffix}"
    return symbol


def fetch_candles(exchange, symbol: str, timeframe: str, since_ms: int) -> pd.DataFrame:
    """Paginated OHLCV fetch from ``since_ms`` to now; signal_check.fetch_ohlcv
    caps at 400 bars, which is too short for long windows (4h x 90d = 540)."""
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    rows = []
    since = since_ms
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not batch:
            break
        if rows and batch[-1][0] <= rows[-1][0]:
            break  # no forward progress — defensive
        rows.extend(batch)
        since = batch[-1][0] + tf_ms
        if len(batch) < 1000:
            break  # reached the present

    dedup = {row[0]: row for row in rows}
    ordered = [dedup[ts] for ts in sorted(dedup)]
    df = pd.DataFrame(ordered, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.reset_index(drop=True)


def load_candles(
    symbol: str,
    timeframe: str,
    days: int,
    exchange: str = "binance",
    quote: str = "USDT",
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Fetch ``days`` of OHLCV ending now, with a same-calendar-day disk cache.

    ``exchange`` may be a ccxt exchange id (str) or an already-constructed
    ccxt exchange instance. Cache files are pickled DataFrames named
    ``<exchange>_<symbol>_<timeframe>_<days>d.pkl`` under ``cache_dir``; a file
    whose mtime falls on today (UTC) is reused, older files are refetched.
    """
    import ccxt

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    safe_symbol = symbol.replace("/", "-").upper()
    exchange_id = exchange if isinstance(exchange, str) else exchange.id
    key = f"{exchange_id}_{safe_symbol}_{timeframe}_{int(days)}d"
    path = cache_path / f"{key}.pkl"

    today = datetime.now(timezone.utc).date()
    if path.exists():
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).date()
        if mtime == today:
            return pd.read_pickle(path)

    ex = getattr(ccxt, exchange)({"enableRateLimit": True}) if isinstance(exchange, str) else exchange
    since_ms = ex.milliseconds() - int(days) * 86_400_000
    df = fetch_candles(ex, to_ccxt_symbol(symbol, quote), timeframe, since_ms)
    df.to_pickle(path)
    return df


def run_backtest(
    df: pd.DataFrame,
    df_1d: pd.DataFrame | None,
    *,
    min_confidence: float = 70.0,
    params=None,
    initial_equity: float = 10000.0,
    risk_pct: float = 2.0,
    fee_bps: float = 10.0,
    timeframe: str | None = None,
    window_start: pd.Timestamp | None = None,
    cooldown_bars: int = 0,
    tp1_frac: float = 0.5,
) -> dict:
    """Replay the signal engine over one symbol; returns per-symbol metrics.

    ``df`` is the trading-timeframe OHLCV frame (with a ``ts`` column),
    INCLUDING warmup bars before the window; ``window_start`` marks the first
    bar whose close may generate a signal (defaults to the frame start).
    ``params`` is a signal_engine.SignalParams (None -> engine defaults).
    ``cooldown_bars`` models the platform's post-loss cooldown: after a losing
    trade, no new position is opened on this symbol for that many bars
    (0 disables — the historical behaviour).
    ``tp1_frac`` models the platform's TP1/TP2 bracket split: this fraction of
    the position exits at TP1, the remainder rides to TP2 or the SL
    (1.0 = single target at TP1, the historical behaviour). A trade's R is
    ``tp1_frac * RR1 + (1 - tp1_frac) * RR2`` when both targets hit,
    ``tp1_frac * RR1 - (1 - tp1_frac)`` when only TP1 hits, ``-1`` otherwise.

    Metric keys are exactly those of the historical runner output, plus the
    internal ``_gross_profit`` / ``_gross_loss`` (R-multiple sums, net of
    fees) which callers strip before serialising.
    """
    import signal_engine

    # Compute indicators once on the full frame: every indicator the engine
    # uses is causal, so values at bar i are identical to recomputing on a
    # slice — and evaluate_symbol uses pre-computed columns as-is.
    df = signal_engine.add_indicators(df)

    risk_fraction = risk_pct / 100.0
    fee_rate_roundtrip = 2.0 * fee_bps / 1e4  # both sides, as fraction of notional

    equity = float(initial_equity)
    peak = equity
    max_drawdown_pct = 0.0
    trades = []  # realized R multiples, net of fees
    position = None
    blocked_until = -1  # post-loss cooldown: no entries before this bar index

    if window_start is not None:
        # the cached frame's ts dtype may be datetime64[ms]; match its unit
        window_start = pd.Timestamp(window_start).as_unit(df["ts"].dt.unit)
        window_idx = int(df["ts"].searchsorted(window_start))
        start = max(window_idx, MIN_BARS)
    else:
        start = MIN_BARS

    i = start
    while i < len(df) - 1:  # -1: the engine always drops one trailing bar
        bar = df.iloc[i]

        if position is not None:
            long = position["direction"] == "LONG"
            hit_sl = bar["low"] <= position["sl"] if long else bar["high"] >= position["sl"]
            hit_tp1 = bar["high"] >= position["tp1"] if long else bar["low"] <= position["tp1"]
            hit_tp2 = bar["high"] >= position["tp2"] if long else bar["low"] <= position["tp2"]
            r = None
            banked = position["rr1"] * position["tp1_frac"] if position["tp1_done"] else 0.0
            rest = 1.0 - position["tp1_frac"] if position["tp1_done"] else 1.0
            if hit_sl:  # SL first within a bar (conservative)
                r = banked - rest
            elif hit_tp1 and not position["tp1_done"]:
                position["tp1_done"] = True
                banked = position["rr1"] * position["tp1_frac"]
                rest = 1.0 - position["tp1_frac"]
                if rest <= 0:
                    r = banked
                elif hit_tp2:
                    r = banked + rest * position["rr2"]
            elif hit_tp2 and position["tp1_done"]:
                r = banked + rest * position["rr2"]
            if r is not None:
                r -= position["fee_r"]
                equity += r * position["risk"]
                trades.append(r)
                if r < 0 and cooldown_bars > 0:
                    blocked_until = i + cooldown_bars
                peak = max(peak, equity)
                if peak > 0:
                    max_drawdown_pct = max(max_drawdown_pct, (peak - equity) / peak * 100)
                position = None
            i += 1
            continue

        # No open position: evaluate as of the close of bar i. The slice ends
        # at bar i+1, which the engine drops as "in progress", so no future
        # data leaks into the decision. Indicators are pre-computed, so a
        # short trailing slice is enough (see _EVAL_TAIL_BARS).
        if i < blocked_until:
            i += 1
            continue
        window = df.iloc[max(0, i + 2 - _EVAL_TAIL_BARS): i + 2]
        day_slice = None
        if df_1d is not None:
            day_slice = df_1d[df_1d["ts"] <= bar["ts"]]
        ev = signal_engine.evaluate_symbol(
            window, day_slice, min_confidence=min_confidence,
            timeframe=timeframe, params=params,
        )
        plan = signal_engine.build_trade_plan(ev, params=params)
        if plan is not None:
            entry = float(plan["entry_price"])
            sl = float(plan["stop_loss"])
            sl_frac = abs(entry - sl) / entry if entry else 0.0
            position = {
                "direction": ev["direction"],
                "sl": sl,
                "tp1": float(plan["take_profit_1"]),
                "tp2": float(plan["take_profit_2"]),
                "rr1": float(plan["risk_reward_tp1"]),
                "rr2": float(plan["risk_reward_tp2"]),
                "tp1_frac": tp1_frac,
                "tp1_done": False,
                "risk": risk_fraction * equity,
                # fee as an R multiple: notional = risk / sl_frac
                "fee_r": fee_rate_roundtrip / sl_frac if sl_frac > 0 else 0.0,
            }
        i += 1
    # A position still open at the end of the window is left uncounted.

    wins = sum(1 for r in trades if r > 0)
    losses = len(trades) - wins
    gross_profit = sum(r for r in trades if r > 0)
    gross_loss = -sum(r for r in trades if r < 0)
    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / len(trades), 4) if trades else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else None,
        "avg_rr": round(sum(trades) / len(trades), 4) if trades else 0.0,
        "max_drawdown_pct": round(max_drawdown_pct, 4),
        "sl_hit_rate": round(losses / len(trades), 4) if trades else 0.0,
        "final_equity": round(equity, 2),
        "_gross_profit": gross_profit,  # internal, stripped before output
        "_gross_loss": gross_loss,
    }


def aggregate(per_symbol: dict, initial_equity: float) -> dict:
    """Pool trades across symbols; equity fields sum the independent sleeves."""
    trades = sum(m["trades"] for m in per_symbol.values())
    wins = sum(m["wins"] for m in per_symbol.values())
    losses = trades - wins
    gross_profit = sum(m["_gross_profit"] for m in per_symbol.values())
    gross_loss = sum(m["_gross_loss"] for m in per_symbol.values())
    rr_sum = sum(m["avg_rr"] * m["trades"] for m in per_symbol.values())
    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / trades, 4) if trades else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else None,
        "avg_rr": round(rr_sum / trades, 4) if trades else 0.0,
        "max_drawdown_pct": max((m["max_drawdown_pct"] for m in per_symbol.values()),
                                default=0.0),
        "sl_hit_rate": round(losses / trades, 4) if trades else 0.0,
        "final_equity": round(sum(m["final_equity"] for m in per_symbol.values()), 2),
        "initial_equity_per_symbol": initial_equity,
    }


def slice_window(df: pd.DataFrame, window_start: pd.Timestamp,
                 window_end: pd.Timestamp, timeframe: str,
                 exchange=None) -> pd.DataFrame:
    """Trim a cached frame to ``[window_start - anything, window_end)``.

    Keeps all warmup bars before ``window_start`` and ONE extra bar at or
    after ``window_end``: the replay loop never evaluates the final bar (the
    engine drops it as in-progress), so the extra bar lets the last in-window
    bar be evaluated without leaking future data into any decision.
    """
    if exchange is not None:
        tf_ms = exchange.parse_timeframe(timeframe) * 1000
    else:
        import ccxt
        tf_ms = ccxt.Exchange().parse_timeframe(timeframe) * 1000
    cutoff = window_end + pd.Timedelta(milliseconds=tf_ms)
    return df[df["ts"] < cutoff].reset_index(drop=True)


def warmup_days(timeframe: str, bars: int = WARMUP_BARS) -> int:
    """Whole days of extra history needed for ``bars`` of ``timeframe``."""
    import ccxt
    tf_seconds = ccxt.Exchange().parse_timeframe(timeframe)
    return math.ceil(bars * tf_seconds / 86_400)
