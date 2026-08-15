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

import hashlib
import json
import math
import os
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import pairs_engine

WARMUP_BARS = 200        # extra history before the window so indicators are mature
WARMUP_BARS_1D = 80      # daily warmup (engine needs ~55 daily bars)
MIN_BARS = 60            # mirrors signal_engine.MIN_BARS

DEFAULT_CACHE_DIR = "/tmp/astertrade_candles"

# evaluate_symbol (with pre-computed indicator columns) never reads further
# back than 6 bars; slice this many trailing rows per evaluation instead of
# the whole history so the replay stays O(n) instead of O(n^2).
_EVAL_TAIL_BARS = 72

# Trailing-slice sizes for the breakout engine (evaluate_breakout recomputes
# its indicators on whatever slice it is given, so the slice must be long
# enough for every lookback the engine can reach):
#   15m: prior-day high/low (96 bars) and the anchored-VWAP window (anchor is
#        the 1h compression start, at most BB_LOOKBACK=100 1h bars ~ 400 15m
#        bars back), plus ATR/vol-SMA warmup.
#   1h:  BB_LEN + BB_LOOKBACK + 2 = 122 closed bars minimum, plus warmup.
#   4h:  SMA_SLOW + SLOPE_BARS + 1 = 111 closed bars minimum, plus warmup.
_BO_TAIL_15M = 512
_BO_TAIL_1H = 224
_BO_TAIL_4H = 160


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


_FUNDING_COLUMNS = ["ts", "funding_rate"]


def _empty_funding() -> pd.DataFrame:
    return pd.DataFrame({
        "ts": pd.Series(dtype="datetime64[ns, UTC]"),
        "funding_rate": pd.Series(dtype="float64"),
    })


def load_funding(
    symbol: str,
    days: int,
    exchange: str = "binance",
    quote: str = "USDT",
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Fetch ``days`` of 8h funding-rate history ending now, disk-cached.

    Same cache convention as ``load_candles`` (same-calendar-day reuse), with
    a ``funding_`` key prefix so it never collides with candle frames.
    Returns a frame with columns ``ts`` (datetime64, funding timestamp) and
    ``funding_rate`` (float). ``exchange`` may be a ccxt id or instance;
    ``"binance"`` is mapped to ``binanceusdm`` because USDT-M perp funding
    lives on the futures API (spot binance returns nothing). On ANY fetch
    failure or an empty result an EMPTY frame with those columns is returned
    — callers treat missing funding as unknown and skip the funding filter.
    """
    import ccxt

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    safe_symbol = symbol.replace("/", "-").replace(":", "-").upper()
    exchange_id = exchange if isinstance(exchange, str) else exchange.id
    key = f"funding_{exchange_id}_{safe_symbol}_{int(days)}d"
    path = cache_path / f"{key}.pkl"

    today = datetime.now(timezone.utc).date()
    if path.exists():
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).date()
        if mtime == today:
            return pd.read_pickle(path)

    try:
        if isinstance(exchange, str):
            # USDT-M perp funding requires the futures API; spot has none.
            ex_id = "binanceusdm" if exchange == "binance" else exchange
            ex = getattr(ccxt, ex_id)({"enableRateLimit": True})
        else:
            ex = exchange
        since_ms = ex.milliseconds() - int(days) * 86_400_000
        ccxt_symbol = to_ccxt_symbol(symbol, quote)
        rows = []
        since = since_ms
        while True:
            batch = ex.fetch_funding_rate_history(ccxt_symbol, since=since, limit=1000)
            if not batch:
                break
            if rows and batch[-1]["timestamp"] <= rows[-1]["timestamp"]:
                break  # no forward progress — defensive
            rows.extend(batch)
            since = batch[-1]["timestamp"] + 1
            if len(batch) < 1000:
                break
        if not rows:
            return _empty_funding()
        dedup = {r["timestamp"]: float(r["fundingRate"]) for r in rows
                 if r.get("fundingRate") is not None}
        df = pd.DataFrame(
            {"ts": sorted(dedup), "funding_rate": [dedup[ts] for ts in sorted(dedup)]}
        )
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.reset_index(drop=True)
    except Exception:  # noqa: BLE001 - missing funding must never kill a run
        return _empty_funding()
    df.to_pickle(path)
    return df


@dataclass(frozen=True)
class PairMarketData:
    pair: str
    alt_symbol: str
    btc_symbol: str
    alt_1h: pd.DataFrame
    btc_1h: pd.DataFrame
    alt_15m: pd.DataFrame
    btc_15m: pd.DataFrame
    alt_funding: pd.DataFrame
    btc_funding: pd.DataFrame


@dataclass(frozen=True)
class PairsBacktestConfig:
    initial_equity: float = 10_000.0
    risk_pct: float = 1.0
    fee_bps: float = 10.0
    slippage_bps: float = 2.0
    one_leg_execution_shock_bps: float = 0.0


@dataclass(frozen=True)
class PairTrade:
    pair: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    alt_side: str
    btc_side: str
    alt_weight: float
    btc_weight: float
    entry_z: float
    exit_z: float | None
    exit_reason: str
    price_return: float
    fee_return: float
    slippage_return: float
    funding_return: float
    execution_shock_return: float
    net_return: float
    realized_btc_beta: float | None
    mfe_z: float
    mae_z: float
    mfe_return: float
    mae_return: float
    funding_events: int
    forced_close: bool
    gross_notional: float
    equity_before: float
    pnl: float


@dataclass(frozen=True)
class PairsBacktestResult:
    pair: str
    trades: list[PairTrade]
    metrics: dict[str, object]
    invalid_reasons: list[str]


@dataclass(frozen=True)
class WalkForwardWindow:
    formation_start: pd.Timestamp
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class WalkForwardSchedule:
    common_start: pd.Timestamp
    common_end: pd.Timestamp
    holdout_start: pd.Timestamp
    development_windows: list[WalkForwardWindow]


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    failed_conditions: list[str]


def _pairs_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "ts" not in frame.columns:
        raise ValueError("pairs market data requires a ts column")
    normalized = frame.copy()
    normalized["ts"] = pd.to_datetime(normalized["ts"], utc=True)
    return normalized.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)


def _pairs_rows(frame: pd.DataFrame) -> dict[pd.Timestamp, pd.Series]:
    return {pd.Timestamp(row["ts"]): row for _, row in _pairs_frame(frame).iterrows()}


def _finite_price(row: pd.Series | None, column: str) -> float | None:
    if row is None or column not in row:
        return None
    try:
        value = float(row[column])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or value <= 0.0:
        return None
    return value


def _pair_prices(
    state: dict[str, object], ts: pd.Timestamp, column: str,
) -> tuple[float, float] | None:
    alt = _finite_price(state["alt_rows"].get(ts), column)
    btc = _finite_price(state["btc_rows"].get(ts), column)
    if alt is None or btc is None:
        return None
    return alt, btc


def _pair_zscore(
    state: dict[str, object], ts: pd.Timestamp,
) -> tuple[pairs_engine.RelationshipSnapshot | None, float | None]:
    observation = state["latest_observation"]
    snapshot = observation.snapshot if observation is not None else None
    prices = _pair_prices(state, ts, "close")
    if snapshot is None or prices is None:
        return snapshot, None
    values = (
        snapshot.alpha, snapshot.beta, snapshot.residual_mean, snapshot.residual_std,
    )
    if not all(np.isfinite(value) for value in values) or snapshot.residual_std <= 0.0:
        return snapshot, None
    alt_close, btc_close = prices
    residual = math.log(alt_close) - (snapshot.alpha + snapshot.beta * math.log(btc_close))
    zscore = (residual - snapshot.residual_mean) / snapshot.residual_std
    return snapshot, float(zscore) if np.isfinite(zscore) else None


def _side_sign(side: str) -> float:
    return 1.0 if side == "LONG" else -1.0


def _fill_price(price: float, side: str, *, entry: bool, slip: float) -> float:
    side_sign = _side_sign(side)
    adjustment = side_sign * slip if entry else -side_sign * slip
    return price * (1.0 + adjustment)


def _weighted_price_return(
    position: dict[str, object], alt_exit: float, btc_exit: float, *, slipped: bool,
) -> float:
    alt_entry = float(position["alt_entry"])
    btc_entry = float(position["btc_entry"])
    if slipped:
        alt_entry_fill = float(position["alt_entry_fill"])
        btc_entry_fill = float(position["btc_entry_fill"])
        alt_exit_fill = _fill_price(
            alt_exit, position["signal"].alt_side, entry=False, slip=float(position["slip"]),
        )
        btc_exit_fill = _fill_price(
            btc_exit, position["signal"].btc_side, entry=False, slip=float(position["slip"]),
        )
    else:
        alt_entry_fill = alt_entry
        btc_entry_fill = btc_entry
        alt_exit_fill = alt_exit
        btc_exit_fill = btc_exit
    signal = position["signal"]
    alt_return = (
        signal.alt_weight * _side_sign(signal.alt_side)
        * (alt_exit_fill - alt_entry_fill) / alt_entry
    )
    btc_return = (
        signal.btc_weight * _side_sign(signal.btc_side)
        * (btc_exit_fill - btc_entry_fill) / btc_entry
    )
    return float(alt_return + btc_return)


def _funding_rows(frame: pd.DataFrame) -> dict[pd.Timestamp, float]:
    if not {"ts", "funding_rate"}.issubset(frame.columns):
        return {}
    rates: dict[pd.Timestamp, float] = {}
    for _, row in _pairs_frame(frame).iterrows():
        try:
            rate = float(row["funding_rate"])
        except (TypeError, ValueError):
            continue
        if np.isfinite(rate):
            rates[pd.Timestamp(row["ts"])] = rate
    return rates


def _expected_funding_times(entry_ts: pd.Timestamp, exit_ts: pd.Timestamp) -> list[pd.Timestamp]:
    return [
        pd.Timestamp(ts)
        for ts in pd.date_range(
            entry_ts.normalize(), exit_ts.normalize() + pd.Timedelta(days=1),
            freq="8h", inclusive="left",
        )
        if entry_ts < ts <= exit_ts
    ]


def _trade_funding(
    position: dict[str, object], exit_ts: pd.Timestamp, invalid_reasons: list[str],
) -> tuple[float, int]:
    market = position["market"]
    signal = position["signal"]
    entry_ts = position["entry_ts"]
    alt_rates = _funding_rows(market.alt_funding)
    btc_rates = _funding_rows(market.btc_funding)
    funding_return = 0.0
    funding_events = 0
    for rates, side, weight in (
        (alt_rates, signal.alt_side, signal.alt_weight),
        (btc_rates, signal.btc_side, signal.btc_weight),
    ):
        for funding_ts, rate in rates.items():
            if entry_ts < funding_ts <= exit_ts:
                funding_return += -_side_sign(side) * weight * rate
                funding_events += 1
    for funding_ts in _expected_funding_times(entry_ts, exit_ts):
        if funding_ts not in alt_rates or funding_ts not in btc_rates:
            reason = f"missing_funding:{market.pair}:{funding_ts.isoformat()}"
            if reason not in invalid_reasons:
                invalid_reasons.append(reason)
    return float(funding_return), funding_events


def _realized_btc_beta(position: dict[str, object]) -> float | None:
    pair_marks = np.asarray(position["pair_marks"], dtype=float)
    btc_logs = np.asarray(position["btc_logs"], dtype=float)
    mark_ts = pd.DatetimeIndex(pd.to_datetime(position["mark_ts"], utc=True))
    if not (len(pair_marks) == len(btc_logs) == len(mark_ts)) or len(pair_marks) < 4:
        return None
    pair_increments = np.diff(pair_marks)
    btc_increments = np.diff(btc_logs)
    consecutive = (mark_ts[1:] - mark_ts[:-1]) == pd.Timedelta(minutes=15)
    finite = np.isfinite(pair_increments) & np.isfinite(btc_increments) & consecutive
    if int(finite.sum()) < 3:
        return None
    x = btc_increments[finite]
    y = pair_increments[finite]
    centered_x = x - x.mean()
    variance = float(centered_x @ centered_x)
    if not np.isfinite(variance) or variance <= 0.0:
        return None
    slope = float(centered_x @ (y - y.mean()) / variance)
    return slope if np.isfinite(slope) else None


def _append_mark(
    position: dict[str, object], mark_ts: pd.Timestamp, zscore: float | None,
    alt_close: float, btc_close: float,
) -> None:
    gross_return = _weighted_price_return(position, alt_close, btc_close, slipped=False)
    position["mfe_return"] = max(position["mfe_return"], gross_return)
    position["mae_return"] = min(position["mae_return"], gross_return)
    position["pair_marks"].append(gross_return)
    position["btc_logs"].append(math.log(btc_close))
    position["mark_ts"].append(mark_ts)
    if zscore is not None and np.isfinite(zscore):
        pair_direction = -1.0 if position["entry_z"] > 0.0 else 1.0
        z_excursion = pair_direction * (zscore - position["entry_z"])
        position["mfe_z"] = max(position["mfe_z"], z_excursion)
        position["mae_z"] = min(position["mae_z"], z_excursion)
        position["last_z"] = zscore


def _four_fill_fee_return(*, alt_weight: float, btc_weight: float, fee_rate: float) -> float:
    return float(-fee_rate * (alt_weight + btc_weight + alt_weight + btc_weight))


def _complete_pair_trade(
    position: dict[str, object], *, exit_ts: pd.Timestamp, exit_reason: str,
    exit_z: float | None, invalid_reasons: list[str],
) -> PairTrade | None:
    prices = _pair_prices(position["state"], exit_ts, "open")
    if prices is None:
        return None
    alt_exit, btc_exit = prices
    signal = position["signal"]
    price_return = _weighted_price_return(position, alt_exit, btc_exit, slipped=False)
    slipped_price_return = _weighted_price_return(position, alt_exit, btc_exit, slipped=True)
    slippage_return = slipped_price_return - price_return
    fee_rate = float(position["fee_rate"])
    fee_return = _four_fill_fee_return(
        alt_weight=signal.alt_weight,
        btc_weight=signal.btc_weight,
        fee_rate=fee_rate,
    )
    funding_return, funding_events = _trade_funding(position, exit_ts, invalid_reasons)
    execution_shock_return = float(position["execution_shock_return"])
    net_return = (
        price_return + fee_return + slippage_return
        + funding_return + execution_shock_return
    )
    equity_before = float(position["equity_before"])
    gross_notional = float(position["gross_notional"])
    pnl = gross_notional * net_return
    return PairTrade(
        pair=position["market"].pair,
        entry_ts=position["entry_ts"],
        exit_ts=exit_ts,
        alt_side=signal.alt_side,
        btc_side=signal.btc_side,
        alt_weight=signal.alt_weight,
        btc_weight=signal.btc_weight,
        entry_z=float(position["entry_z"]),
        exit_z=exit_z,
        exit_reason=exit_reason,
        price_return=price_return,
        fee_return=float(fee_return),
        slippage_return=float(slippage_return),
        funding_return=funding_return,
        execution_shock_return=execution_shock_return,
        net_return=float(net_return),
        realized_btc_beta=_realized_btc_beta(position),
        mfe_z=float(position["mfe_z"]),
        mae_z=float(position["mae_z"]),
        mfe_return=float(position["mfe_return"]),
        mae_return=float(position["mae_return"]),
        funding_events=funding_events,
        forced_close=exit_reason in {"data_gap", "window_boundary"},
        gross_notional=gross_notional,
        equity_before=equity_before,
        pnl=float(pnl),
    )


def _observation_rows(
    market: PairMarketData, params: pairs_engine.PairsParams,
) -> list[pairs_engine.PairObservation]:
    hourly = pairs_engine.align_hourly_prices(market.alt_1h, market.btc_1h)
    frame = pairs_engine.build_hourly_observations(hourly, params, pair=market.pair)
    observations = []
    for _, row in frame.iterrows():
        if row.get("pair") != market.pair:
            raise ValueError(
                f"observation pair {row.get('pair')!r} does not match market pair {market.pair!r}"
            )
        snapshot = row.get("snapshot")
        if not isinstance(snapshot, pairs_engine.RelationshipSnapshot):
            snapshot = None
        zscore = row.get("zscore")
        zscore = float(zscore) if zscore is not None and np.isfinite(zscore) else None
        direction = row.get("direction")
        observations.append(pairs_engine.PairObservation(
            pair=market.pair,
            # CCXT labels OHLCV by candle open; the hourly close is usable one hour later.
            ts=pd.Timestamp(row["ts"]) + pd.Timedelta(hours=1),
            snapshot=snapshot,
            zscore=zscore,
            direction=direction if isinstance(direction, str) else None,
        ))
    return sorted(observations, key=lambda observation: observation.ts)


def _advance_observations(
    state: dict[str, object], available_ts: pd.Timestamp, *, suppress_entries: bool,
) -> None:
    observations = state["observations"]
    while (
        state["observation_index"] < len(observations)
        and observations[state["observation_index"]].ts <= available_ts
    ):
        state["latest_observation"] = observations[state["observation_index"]]
        if suppress_entries:
            state["last_signal_observation_ts"] = state["latest_observation"].ts
            state["pending_confirmation"] = None
        state["observation_index"] += 1


def _pairs_metrics(
    trades: list[PairTrade], *, initial_equity: float, final_equity: float,
    rejected_entries: int,
) -> dict[str, object]:
    weighted_beta_numerator = 0.0
    weighted_beta_denominator = 0.0
    for trade in trades:
        if trade.realized_btc_beta is None or not np.isfinite(trade.realized_btc_beta):
            continue
        duration = max((trade.exit_ts - trade.entry_ts).total_seconds(), 0.0)
        weight = trade.gross_notional * duration
        weighted_beta_numerator += abs(trade.realized_btc_beta) * weight
        weighted_beta_denominator += weight
    aggregate_beta = (
        weighted_beta_numerator / weighted_beta_denominator
        if weighted_beta_denominator > 0.0 else None
    )
    return {
        "completed_trades": len(trades),
        "initial_equity": float(initial_equity),
        "final_equity": float(final_equity),
        "total_pnl": float(sum(trade.pnl for trade in trades)),
        "absolute_realized_btc_beta": aggregate_beta,
        "rejected_entries": rejected_entries,
    }


def run_pairs_backtest(
    data: dict[str, PairMarketData], *, window_start: pd.Timestamp,
    window_end: pd.Timestamp, config: PairsBacktestConfig,
    params: pairs_engine.PairsParams = pairs_engine.DEFAULT_PARAMS,
) -> dict[str, PairsBacktestResult]:
    """Replay the fixed BTC-relative pairs on one synchronized 15m clock."""
    window_start = pd.Timestamp(window_start)
    window_end = pd.Timestamp(window_end)
    window_start = (
        window_start.tz_localize("UTC") if window_start.tzinfo is None
        else window_start.tz_convert("UTC")
    )
    window_end = (
        window_end.tz_localize("UTC") if window_end.tzinfo is None
        else window_end.tz_convert("UTC")
    )
    if window_end <= window_start:
        raise ValueError("window_end must be after window_start")

    ordered_pairs = [pair for pair in pairs_engine.FIXED_PAIRS if pair in data]
    unknown_pairs = set(data) - set(pairs_engine.FIXED_PAIRS)
    if unknown_pairs:
        raise ValueError(f"unsupported pairs: {sorted(unknown_pairs)}")
    if not ordered_pairs:
        return {}

    states: dict[str, dict[str, object]] = {}
    invalid_reasons = {pair: [] for pair in ordered_pairs}
    trades = {pair: [] for pair in ordered_pairs}
    rejected_entries = {pair: 0 for pair in ordered_pairs}
    clock_values: set[pd.Timestamp] = set()
    for pair in ordered_pairs:
        market = data[pair]
        if market.pair != pair:
            raise ValueError(f"pair key {pair} does not match market data {market.pair}")
        alt_rows = _pairs_rows(market.alt_15m)
        btc_rows = _pairs_rows(market.btc_15m)
        clock_values.update(alt_rows)
        clock_values.update(btc_rows)
        states[pair] = {
            "market": market,
            "alt_rows": alt_rows,
            "btc_rows": btc_rows,
            "observations": _observation_rows(market, params),
            "observation_index": 0,
            "latest_observation": None,
            "pending_confirmation": None,
            "last_signal_observation_ts": None,
            "missing_bars": 0,
        }

    clock_values.update(pd.date_range(window_start, window_end, freq="15min"))
    clock = sorted(ts for ts in clock_values if window_start <= ts <= window_end)
    equity = float(config.initial_equity)
    position: dict[str, object] | None = None
    pending_entry: dict[str, object] | None = None
    pending_exit: dict[str, object] | None = None
    slip = float(config.slippage_bps) / 10_000.0
    fee_rate = float(config.fee_bps) / 10_000.0
    execution_shock_return = -float(config.one_leg_execution_shock_bps) / 10_000.0

    for ts in clock:
        entries_suppressed = position is not None or pending_entry is not None
        for state in states.values():
            _advance_observations(state, ts, suppress_entries=entries_suppressed)

        if (
            position is not None
            and pending_exit is not None
            and ts >= pending_exit["decision_ts"]
        ):
            trade = _complete_pair_trade(
                position,
                exit_ts=ts,
                exit_reason=pending_exit["reason"],
                exit_z=pending_exit["zscore"],
                invalid_reasons=invalid_reasons[position["market"].pair],
            )
            if trade is not None:
                trades[trade.pair].append(trade)
                equity += trade.pnl
                position = None
                pending_exit = None
                for state in states.values():
                    state["pending_confirmation"] = None

        if position is None and pending_entry is not None:
            pair = pending_entry["pair"]
            state = states[pair]
            prices = _pair_prices(state, ts, "open")
            latest_observation = state["latest_observation"]
            fill_snapshot = (
                latest_observation.snapshot if latest_observation is not None else None
            )
            if fill_snapshot is None or not fill_snapshot.stable:
                pending_entry = None
            elif prices is not None and pending_entry["decision_ts"] <= ts < window_end:
                signal = pending_entry["signal"]
                estimated_stop_return = (
                    signal.alt_weight * (params.stop_z - abs(pending_entry["entry_z"]))
                    * fill_snapshot.residual_std
                )
                if not np.isfinite(estimated_stop_return) or estimated_stop_return <= 0.0:
                    rejected_entries[pair] += 1
                else:
                    gross_notional = min(
                        equity,
                        equity * float(config.risk_pct) / 100.0 / estimated_stop_return,
                    )
                    alt_entry, btc_entry = prices
                    position = {
                        "market": data[pair],
                        "state": state,
                        "signal": signal,
                        "entry_ts": ts,
                        "entry_z": float(pending_entry["entry_z"]),
                        "last_z": float(pending_entry["entry_z"]),
                        "alt_entry": alt_entry,
                        "btc_entry": btc_entry,
                        "alt_entry_fill": _fill_price(
                            alt_entry, signal.alt_side, entry=True, slip=slip,
                        ),
                        "btc_entry_fill": _fill_price(
                            btc_entry, signal.btc_side, entry=True, slip=slip,
                        ),
                        "slip": slip,
                        "fee_rate": fee_rate,
                        "execution_shock_return": execution_shock_return,
                        "gross_notional": float(gross_notional),
                        "equity_before": equity,
                        "mfe_z": 0.0,
                        "mae_z": 0.0,
                        "mfe_return": 0.0,
                        "mae_return": 0.0,
                        "pair_marks": [0.0],
                        "btc_logs": [math.log(btc_entry)],
                        "mark_ts": [ts],
                    }
                    state["missing_bars"] = 0
                pending_entry = None

        if ts >= window_end:
            continue

        # Fifteen-minute decisions happen at the candle close; the next row opens then.
        close_ts = ts + pd.Timedelta(minutes=15)
        entries_suppressed = position is not None or pending_entry is not None
        for state in states.values():
            _advance_observations(state, close_ts, suppress_entries=entries_suppressed)

        if position is not None:
            state = position["state"]
            snapshot, zscore = _pair_zscore(state, ts)
            close_prices = _pair_prices(state, ts, "close")
            if close_prices is None:
                state["missing_bars"] += 1
            else:
                state["missing_bars"] = 0
                _append_mark(position, close_ts, zscore, close_prices[0], close_prices[1])
            if pending_exit is None:
                exit_signal = pairs_engine.classify_exit(
                    zscore=zscore,
                    stable=snapshot.stable if snapshot is not None else None,
                    entry_ts=position["entry_ts"],
                    decision_ts=close_ts,
                    consecutive_missing_bars=state["missing_bars"],
                    params=params,
                )
                if exit_signal is not None:
                    pending_exit = {
                        "reason": exit_signal.reason,
                        "zscore": zscore,
                        "decision_ts": close_ts,
                    }
            continue

        if pending_entry is not None:
            continue

        candidates = []
        for pair in ordered_pairs:
            state = states[pair]
            observation = state["latest_observation"]
            if (
                observation is not None
                and observation.ts != state["last_signal_observation_ts"]
                and observation.direction is not None
            ):
                state["last_signal_observation_ts"] = observation.ts
                signal = pairs_engine.make_signal(observation, params)
                if signal is not None:
                    elapsed_bars = max(
                        0,
                        int((close_ts - signal.decision_ts) // pd.Timedelta(minutes=15)),
                    )
                    if elapsed_bars <= params.confirmation_bars:
                        state["pending_confirmation"] = pairs_engine.PendingConfirmation(
                            signal=signal,
                            bars_seen=max(0, elapsed_bars - 1),
                        )
            pending = state["pending_confirmation"]
            if pending is None or close_ts <= pending.signal.decision_ts:
                continue
            snapshot, zscore = _pair_zscore(state, ts)
            if snapshot is None or not snapshot.stable:
                state["pending_confirmation"] = None
                continue
            confirmation = pairs_engine.advance_confirmation(pending, zscore, params)
            state["pending_confirmation"] = confirmation.pending
            if (
                confirmation.entered
                and snapshot is not None
                and snapshot.stable
                and zscore is not None
            ):
                candidates.append({
                    "pair": pair,
                    "signal": pending.signal,
                    "snapshot": snapshot,
                    "entry_z": zscore,
                    "decision_ts": close_ts,
                })

        if candidates:
            pair_priority = {pair: index for index, pair in enumerate(pairs_engine.FIXED_PAIRS)}
            candidates.sort(key=lambda candidate: (
                -abs(candidate["entry_z"]), pair_priority[candidate["pair"]],
            ))
            pending_entry = candidates[0]
            for state in states.values():
                state["pending_confirmation"] = None

    if position is not None:
        synchronized = [
            ts for ts in clock
            if position["entry_ts"] <= ts <= window_end
            and _pair_prices(position["state"], ts, "open") is not None
        ]
        if synchronized:
            later_exit_opens = (
                [ts for ts in synchronized if ts >= pending_exit["decision_ts"]]
                if pending_exit is not None else []
            )
            if later_exit_opens:
                exit_ts = later_exit_opens[0]
                reason = pending_exit["reason"]
                exit_z = pending_exit["zscore"]
            else:
                exit_ts = synchronized[-1]
                reason = "window_boundary"
                exit_z = position["last_z"]
            trade = _complete_pair_trade(
                position,
                exit_ts=exit_ts,
                exit_reason=reason,
                exit_z=exit_z,
                invalid_reasons=invalid_reasons[position["market"].pair],
            )
            if trade is not None:
                trades[trade.pair].append(trade)
                equity += trade.pnl

    return {
        pair: PairsBacktestResult(
            pair=pair,
            trades=trades[pair],
            metrics=_pairs_metrics(
                trades[pair], initial_equity=config.initial_equity,
                final_equity=(
                    float(config.initial_equity) + sum(trade.pnl for trade in trades[pair])
                ),
                rejected_entries=rejected_entries[pair],
            ),
            invalid_reasons=invalid_reasons[pair],
        )
        for pair in ordered_pairs
    }


_PAIR_EXIT_REASONS = (
    "convergence",
    "divergence_stop",
    "time_stop",
    "structural",
    "data_gap",
    "window_boundary",
)
_COST_STRESS_GRID = tuple(
    (fee_bps, slippage_bps)
    for fee_bps in (5.0, 10.0, 15.0)
    for slippage_bps in (1.0, 2.0, 5.0)
)
_LEDGER_SCHEMA = {
    "schema_version": 1,
    "holdout": {
        "status": "sealed",
        "opened_at": None,
        "dataset_hash": None,
        "trial_id": None,
    },
    "trials": [],
}
_TRIAL_FIELDS = {
    "trial_id",
    "recorded_at",
    "phase",
    "params",
    "cost_config",
    "dataset_hashes",
    "common_start",
    "common_end",
    "metrics",
    "gate",
    "invalid_reasons",
}


def _utc_timestamp(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC") if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


def build_walk_forward_schedule(
    common_start: pd.Timestamp,
    common_end: pd.Timestamp,
) -> WalkForwardSchedule:
    """Reserve a sealed holdout and build full, non-overlapping development windows."""
    common_start = _utc_timestamp(common_start)
    common_end = _utc_timestamp(common_end)
    history = common_end - common_start
    history_days = int(history // pd.Timedelta(days=1))
    if history < pd.Timedelta(days=360):
        raise ValueError(
            f"insufficient common history: {history_days} days; need at least 360"
        )

    holdout_start = common_end - pd.Timedelta(days=90)
    window_start = common_start + pd.Timedelta(days=60)
    development_windows = []
    while window_start + pd.Timedelta(days=30) <= holdout_start:
        development_windows.append(WalkForwardWindow(
            formation_start=window_start - pd.Timedelta(days=60),
            start=window_start,
            end=window_start + pd.Timedelta(days=30),
        ))
        window_start += pd.Timedelta(days=30)
    return WalkForwardSchedule(
        common_start=common_start,
        common_end=common_end,
        holdout_start=holdout_start,
        development_windows=development_windows,
    )


def _profit_factor(returns: np.ndarray) -> float:
    gross_profit = float(returns[returns > 0.0].sum())
    gross_loss = float(-returns[returns < 0.0].sum())
    if gross_loss > 0.0:
        return gross_profit / gross_loss
    return float("inf") if gross_profit > 0.0 else 0.0


def _basic_pair_summary(trades: list[PairTrade]) -> dict[str, object]:
    returns = np.asarray([trade.net_return for trade in trades], dtype=float)
    pnls = np.asarray([trade.pnl for trade in trades], dtype=float)
    gross_profit = float(sum(max(trade.pnl, 0.0) for trade in trades))
    return {
        "completed_trades": len(trades),
        "profit_factor": _profit_factor(pnls),
        "mean_net_return": float(returns.mean()) if len(returns) else 0.0,
        "median_net_return": float(np.median(returns)) if len(returns) else 0.0,
        "gross_profit": gross_profit,
    }


def _bootstrap_mean_ci(returns: np.ndarray) -> list[float | None]:
    if not len(returns):
        return [None, None]
    block_length = max(1, round(math.sqrt(len(returns))))
    block_count = math.ceil(len(returns) / block_length)
    max_start = len(returns) - block_length
    rng = np.random.default_rng(20260814)
    means = np.empty(2_000, dtype=float)
    for sample_index in range(2_000):
        starts = rng.integers(0, max_start + 1, size=block_count)
        sample = np.concatenate([
            returns[start:start + block_length] for start in starts
        ])[:len(returns)]
        means[sample_index] = float(sample.mean())
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def _deflated_sharpe_probability(returns: np.ndarray, trial_count: int) -> float:
    if len(returns) < 2:
        return 0.0
    standard_deviation = float(np.std(returns, ddof=1))
    if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
        return 1.0 if float(np.mean(returns)) > 0.0 else 0.0

    from scipy.stats import norm

    sharpe = float(np.mean(returns) / standard_deviation)
    trials = max(int(trial_count), 1)
    benchmark = 0.0
    if trials > 1:
        euler_gamma = 0.5772156649015329
        benchmark = float(
            (1.0 - euler_gamma) * norm.ppf(1.0 - 1.0 / trials)
            + euler_gamma * norm.ppf(1.0 - 1.0 / (trials * math.e))
        ) / math.sqrt(max(len(returns) - 1, 1))

    centered = returns - float(np.mean(returns))
    population_std = float(np.std(returns, ddof=0))
    if population_std <= 0.0:
        return 0.0
    skewness = float(np.mean((centered / population_std) ** 3))
    kurtosis = float(np.mean((centered / population_std) ** 4))
    denominator_squared = (
        1.0 - skewness * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe ** 2
    )
    if not np.isfinite(denominator_squared) or denominator_squared <= 0.0:
        return 0.0
    statistic = (
        (sharpe - benchmark) * math.sqrt(len(returns) - 1)
        / math.sqrt(denominator_squared)
    )
    return float(norm.cdf(statistic))


def summarize_pair_trades(
    trades: list[PairTrade], *, trial_count: int,
) -> dict[str, object]:
    """Return deterministic, JSON-safe diagnostics for one causal trade ledger."""
    ordered = sorted(trades, key=lambda trade: (trade.exit_ts, trade.entry_ts, trade.pair))
    returns = np.asarray([trade.net_return for trade in ordered], dtype=float)
    pnls = np.asarray([trade.pnl for trade in ordered], dtype=float)
    wins = int((returns > 0.0).sum())
    sample_std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    sharpe = (
        float(np.mean(returns) / sample_std * math.sqrt(len(returns)))
        if sample_std > 0.0 else 0.0
    )

    portfolio_returns = np.asarray([
        trade.pnl / trade.equity_before if trade.equity_before > 0.0 else 0.0
        for trade in ordered
    ], dtype=float)
    wealth = (
        np.cumprod(1.0 + portfolio_returns)
        if len(portfolio_returns) else np.asarray([], dtype=float)
    )
    if len(wealth):
        wealth_with_origin = np.concatenate(([1.0], wealth))
        peaks = np.maximum.accumulate(wealth_with_origin)
        max_drawdown = float(np.max((peaks - wealth_with_origin) / peaks))
    else:
        max_drawdown = 0.0

    durations = np.asarray([
        max((trade.exit_ts - trade.entry_ts).total_seconds(), 0.0) / 3_600.0
        for trade in ordered
    ], dtype=float)
    elapsed_hours = (
        max((max(trade.exit_ts for trade in ordered) - min(
            trade.entry_ts for trade in ordered
        )).total_seconds(), 0.0) / 3_600.0
        if ordered else 0.0
    )
    total_holding_hours = float(durations.sum())

    beta_numerator = 0.0
    beta_denominator = 0.0
    for trade, duration in zip(ordered, durations):
        if trade.realized_btc_beta is None or not np.isfinite(trade.realized_btc_beta):
            continue
        weight = trade.gross_notional * duration
        beta_numerator += abs(trade.realized_btc_beta) * weight
        beta_denominator += weight
    absolute_realized_btc_beta = (
        float(beta_numerator / beta_denominator) if beta_denominator > 0.0 else None
    )

    per_pair = {
        pair: _basic_pair_summary([trade for trade in ordered if trade.pair == pair])
        for pair in pairs_engine.FIXED_PAIRS
    }
    total_gross_profit = float(sum(summary["gross_profit"] for summary in per_pair.values()))
    for summary in per_pair.values():
        summary["gross_profit_contribution"] = (
            float(summary["gross_profit"] / total_gross_profit)
            if total_gross_profit > 0.0 else 0.0
        )

    def values(attribute: str) -> np.ndarray:
        return np.asarray([getattr(trade, attribute) for trade in ordered], dtype=float)

    components = {
        component: {
            "sum": float(values(component).sum()) if ordered else 0.0,
            "mean": float(values(component).mean()) if ordered else 0.0,
        }
        for component in (
            "price_return",
            "fee_return",
            "slippage_return",
            "funding_return",
            "execution_shock_return",
        )
    }
    long_alt = sum(trade.alt_side == "LONG" for trade in ordered)
    short_alt = sum(trade.alt_side == "SHORT" for trade in ordered)
    return {
        "completed_trades": len(ordered),
        "wins": wins,
        "win_rate": float(wins / len(ordered)) if ordered else 0.0,
        "profit_factor": _profit_factor(pnls),
        "mean_net_return": float(returns.mean()) if ordered else 0.0,
        "median_net_return": float(np.median(returns)) if ordered else 0.0,
        "total_net_return": float(returns.sum()) if ordered else 0.0,
        "total_pnl": float(pnls.sum()) if ordered else 0.0,
        "gross_profit": float(pnls[pnls > 0.0].sum()) if ordered else 0.0,
        "gross_loss": float(-pnls[pnls < 0.0].sum()) if ordered else 0.0,
        "sharpe": sharpe,
        "deflated_sharpe_probability": _deflated_sharpe_probability(
            returns, trial_count,
        ),
        "trial_count": max(int(trial_count), 1),
        "max_drawdown": max_drawdown,
        "total_holding_hours": total_holding_hours,
        "mean_holding_hours": float(durations.mean()) if ordered else 0.0,
        "median_holding_hours": float(np.median(durations)) if ordered else 0.0,
        "exposure": min(total_holding_hours / elapsed_hours, 1.0) if elapsed_hours else 0.0,
        "funding_events": sum(trade.funding_events for trade in ordered),
        "forced_closes": sum(trade.forced_close for trade in ordered),
        "mean_mfe_z": float(values("mfe_z").mean()) if ordered else 0.0,
        "median_mfe_z": float(np.median(values("mfe_z"))) if ordered else 0.0,
        "mean_mae_z": float(values("mae_z").mean()) if ordered else 0.0,
        "median_mae_z": float(np.median(values("mae_z"))) if ordered else 0.0,
        "mean_mfe_return": float(values("mfe_return").mean()) if ordered else 0.0,
        "median_mfe_return": float(np.median(values("mfe_return"))) if ordered else 0.0,
        "mean_mae_return": float(values("mae_return").mean()) if ordered else 0.0,
        "median_mae_return": float(np.median(values("mae_return"))) if ordered else 0.0,
        "return_components": components,
        "absolute_realized_btc_beta": absolute_realized_btc_beta,
        "long_alt_trades": long_alt,
        "short_alt_trades": short_alt,
        "long_alt_fraction": float(long_alt / len(ordered)) if ordered else 0.0,
        "short_alt_fraction": float(short_alt / len(ordered)) if ordered else 0.0,
        "exit_counts": {
            reason: sum(trade.exit_reason == reason for trade in ordered)
            for reason in dict.fromkeys(
                (*_PAIR_EXIT_REASONS, *(trade.exit_reason for trade in ordered))
            )
        },
        "per_pair_trades": {
            pair: int(summary["completed_trades"]) for pair, summary in per_pair.items()
        },
        "per_pair": per_pair,
        "leave_one_pair_out": {
            pair: _basic_pair_summary([trade for trade in ordered if trade.pair != pair])
            for pair in pairs_engine.FIXED_PAIRS
        },
        "bootstrap_block_length": max(1, round(math.sqrt(len(ordered)))) if ordered else 1,
        "bootstrap_resamples": 2_000,
        "bootstrap_mean_net_return_ci_95": _bootstrap_mean_ci(returns),
    }


def _numeric_at_least(metrics: dict[str, object], key: str, threshold: float) -> bool:
    value = metrics.get(key)
    return (
        isinstance(value, (int, float))
        and not np.isnan(value)
        and value >= threshold
    )


def _numeric_at_most(metrics: dict[str, object], key: str, threshold: float) -> bool:
    value = metrics.get(key)
    return isinstance(value, (int, float)) and np.isfinite(value) and value <= threshold


def development_gate(metrics: dict[str, object]) -> GateDecision:
    failed = []
    per_pair_trades = metrics.get("per_pair_trades", {})
    per_pair = metrics.get("per_pair", {})
    conditions = (
        ("completed_trades", _numeric_at_least(metrics, "completed_trades", 60)),
        ("per_pair_trades", all(
            isinstance(per_pair_trades, dict) and per_pair_trades.get(pair, 0) >= 20
            for pair in pairs_engine.FIXED_PAIRS
        )),
        ("profit_factor", _numeric_at_least(metrics, "profit_factor", 1.15)),
        ("win_rate", _numeric_at_least(metrics, "win_rate", 0.50)),
        ("per_pair_mean_net_return", all(
            isinstance(per_pair, dict)
            and isinstance(per_pair.get(pair), dict)
            and per_pair[pair].get("mean_net_return", 0.0) > 0.0
            for pair in pairs_engine.FIXED_PAIRS
        )),
        ("max_drawdown", _numeric_at_most(metrics, "max_drawdown", 0.15)),
        ("absolute_realized_btc_beta", _numeric_at_most(
            metrics, "absolute_realized_btc_beta", 0.15,
        )),
        ("invalid_reasons", not metrics.get("invalid_reasons")),
    )
    failed.extend(name for name, passed in conditions if not passed)
    return GateDecision(passed=not failed, failed_conditions=failed)


def hard_pass_gate(metrics: dict[str, object]) -> GateDecision:
    failed = []
    per_pair_trades = metrics.get("per_pair_trades", {})
    per_pair = metrics.get("per_pair", {})
    stress = metrics.get("cost_stress", {})
    severe_stress = (
        stress.get("15bps_fee_5bps_slippage", {}) if isinstance(stress, dict) else {}
    )
    bootstrap_ci = metrics.get("bootstrap_mean_net_return_ci_95", [None, None])
    bootstrap_excludes_zero = (
        isinstance(bootstrap_ci, (list, tuple))
        and len(bootstrap_ci) == 2
        and all(isinstance(value, (int, float)) for value in bootstrap_ci)
        and (bootstrap_ci[0] > 0.0 or bootstrap_ci[1] < 0.0)
    )
    pair_quality = all(
        isinstance(per_pair, dict)
        and isinstance(per_pair.get(pair), dict)
        and per_pair[pair].get("profit_factor", 0.0) >= 1.05
        and per_pair[pair].get("mean_net_return", 0.0) > 0.0
        for pair in pairs_engine.FIXED_PAIRS
    )
    concentration_ok = all(
        isinstance(per_pair, dict)
        and isinstance(per_pair.get(pair), dict)
        and per_pair[pair].get("gross_profit_contribution", 1.0) <= 0.65
        for pair in pairs_engine.FIXED_PAIRS
    )
    conditions = (
        ("completed_trades", _numeric_at_least(metrics, "completed_trades", 100)),
        ("per_pair_trades", all(
            isinstance(per_pair_trades, dict) and per_pair_trades.get(pair, 0) >= 35
            for pair in pairs_engine.FIXED_PAIRS
        )),
        ("profit_factor", _numeric_at_least(metrics, "profit_factor", 1.25)),
        ("win_rate", _numeric_at_least(metrics, "win_rate", 0.52)),
        ("mean_net_return", _numeric_at_least(metrics, "mean_net_return", 0.0)
         and metrics.get("mean_net_return", 0.0) > 0.0),
        ("median_net_return", _numeric_at_least(metrics, "median_net_return", 0.0)
         and metrics.get("median_net_return", 0.0) > 0.0),
        ("per_pair_quality", pair_quality),
        ("max_drawdown", _numeric_at_most(metrics, "max_drawdown", 0.15)),
        ("absolute_realized_btc_beta", _numeric_at_most(
            metrics, "absolute_realized_btc_beta", 0.15,
        )),
        ("gross_profit_concentration", concentration_ok),
        ("15bps_fee_5bps_slippage", isinstance(severe_stress, dict)
         and severe_stress.get("mean_net_return", -math.inf) >= 0.0),
        ("deflated_sharpe_probability", _numeric_at_least(
            metrics, "deflated_sharpe_probability", 0.95,
        )),
        ("bootstrap_mean_net_return_ci_95", bootstrap_excludes_zero),
        ("invalid_reasons", not metrics.get("invalid_reasons")),
    )
    failed.extend(name for name, passed in conditions if not passed)
    return GateDecision(passed=not failed, failed_conditions=failed)


def dataset_hash(frames: dict[str, pd.DataFrame]) -> str:
    """Hash named frame contents deterministically, independent of mapping order."""
    digest = hashlib.sha256()
    for name in sorted(frames):
        frame = frames[name].copy()
        digest.update(name.encode("utf-8"))
        digest.update(json.dumps(list(frame.columns), separators=(",", ":")).encode("utf-8"))
        digest.update(json.dumps(
            [str(dtype) for dtype in frame.dtypes], separators=(",", ":"),
        ).encode("utf-8"))
        digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return _utc_timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _new_ledger() -> dict[str, object]:
    return json.loads(json.dumps(_LEDGER_SCHEMA))


def _read_trial_ledger(path: Path) -> dict[str, object]:
    if not path.exists():
        return _new_ledger()
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if ledger.get("schema_version") != 1:
        raise ValueError("unsupported trial ledger schema")
    if not isinstance(ledger.get("trials"), list) or not isinstance(ledger.get("holdout"), dict):
        raise ValueError("invalid trial ledger")
    return ledger


def _write_ledger_document(path: Path, ledger: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(ledger), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_trial_ledger(path: Path, trial: dict[str, object]) -> None:
    """Append every trial atomically and irreversibly consume a holdout opening."""
    missing = _TRIAL_FIELDS - set(trial)
    if missing:
        raise ValueError(f"trial missing required fields: {sorted(missing)}")
    ledger = _read_trial_ledger(Path(path))
    if trial.get("holdout_open"):
        if ledger["holdout"].get("status") != "sealed":
            raise ValueError("holdout already opened")
        ledger["holdout"] = {
            "status": "opened",
            "opened_at": trial["recorded_at"],
            "dataset_hash": trial.get("dataset_hash")
            or trial.get("dataset_hashes", {}).get("all"),
            "trial_id": trial["trial_id"],
        }
    ledger["trials"].append(_json_safe(trial))
    _write_ledger_document(Path(path), ledger)


def _reserve_holdout(
    path: Path, *, dataset_digest: str, trial_id: str, opened_at: str,
) -> None:
    ledger = _read_trial_ledger(path)
    if ledger["holdout"].get("status") != "sealed":
        raise ValueError("holdout already opened")
    ledger["holdout"] = {
        "status": "opened",
        "opened_at": opened_at,
        "dataset_hash": dataset_digest,
        "trial_id": trial_id,
    }
    _write_ledger_document(path, ledger)


def _reprice_pair_trades(
    trades: list[PairTrade], *, base_config: PairsBacktestConfig,
    fee_bps: float, slippage_bps: float,
) -> list[PairTrade]:
    """Reprice costs without changing any causal trade identity or gross return."""
    repriced = []
    for trade in trades:
        fee_return = _four_fill_fee_return(
            alt_weight=trade.alt_weight,
            btc_weight=trade.btc_weight,
            fee_rate=float(fee_bps) / 10_000.0,
        )
        if base_config.slippage_bps:
            slippage_return = (
                trade.slippage_return * float(slippage_bps) / float(base_config.slippage_bps)
            )
        else:
            slippage_return = (
                -2.0 * float(slippage_bps) / 10_000.0
                * (trade.alt_weight + trade.btc_weight)
            )
        net_return = (
            trade.price_return + fee_return + slippage_return
            + trade.funding_return + trade.execution_shock_return
        )
        repriced.append(replace(
            trade,
            fee_return=float(fee_return),
            slippage_return=float(slippage_return),
            net_return=float(net_return),
            pnl=float(trade.gross_notional * net_return),
        ))
    return repriced


def _cost_stress_metrics(
    trades: list[PairTrade], *, base_config: PairsBacktestConfig, trial_count: int,
) -> dict[str, dict[str, object]]:
    stress = {}
    for fee_bps, slippage_bps in _COST_STRESS_GRID:
        key = f"{fee_bps:g}bps_fee_{slippage_bps:g}bps_slippage"
        stress[key] = summarize_pair_trades(
            _reprice_pair_trades(
                trades,
                base_config=base_config,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
            ),
            trial_count=trial_count,
        )
    return stress


def _research_frames(
    data: dict[str, PairMarketData],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    if set(data) != set(pairs_engine.FIXED_PAIRS):
        raise ValueError(f"pairs experiment requires exactly {pairs_engine.FIXED_PAIRS}")
    hash_frames = {}
    interval_frames = {}
    for pair in pairs_engine.FIXED_PAIRS:
        market = data[pair]
        alt_symbol, btc_symbol = pair.split("/")
        if (
            market.pair != pair
            or market.alt_symbol != alt_symbol
            or market.btc_symbol != btc_symbol
        ):
            raise ValueError(f"noncanonical market identity for {pair}")
        for leg, timeframe, frame in (
            ("alt", "1h", market.alt_1h),
            ("btc", "1h", market.btc_1h),
            ("alt", "15m", market.alt_15m),
            ("btc", "15m", market.btc_15m),
        ):
            name = f"{pair}:{leg}:{timeframe}"
            normalized = _pairs_frame(frame)
            hash_frames[name] = normalized
            interval_frames[name] = normalized
        hash_frames[f"{pair}:alt:funding"] = _pairs_frame(market.alt_funding)
        hash_frames[f"{pair}:btc:funding"] = _pairs_frame(market.btc_funding)
    return hash_frames, interval_frames


def _common_history_bounds(
    interval_frames: dict[str, pd.DataFrame],
) -> tuple[pd.Timestamp, pd.Timestamp]:
    starts = []
    ends = []
    for name, frame in interval_frames.items():
        if frame.empty:
            raise ValueError(f"empty market history: {name}")
        interval = pd.Timedelta(hours=1) if name.endswith(":1h") else pd.Timedelta(minutes=15)
        starts.append(_utc_timestamp(frame["ts"].min()))
        ends.append(_utc_timestamp(frame["ts"].max()) + interval)
    common_start = max(starts).ceil("1h")
    common_end = min(ends).floor("1h")
    if common_end <= common_start:
        raise ValueError("insufficient common history: 0 days; need at least 360")
    return common_start, common_end


def _run_pairs_windows(
    data: dict[str, PairMarketData], *, windows: list[WalkForwardWindow],
    config: PairsBacktestConfig, params: pairs_engine.PairsParams,
) -> tuple[list[PairTrade], list[str]]:
    trades = []
    invalid_reasons = []
    for window in windows:
        results = run_pairs_backtest(
            data,
            window_start=window.start,
            window_end=window.end,
            config=config,
            params=params,
        )
        for pair in pairs_engine.FIXED_PAIRS:
            result = results[pair]
            trades.extend(result.trades)
            invalid_reasons.extend(result.invalid_reasons)
    return trades, list(dict.fromkeys(invalid_reasons))


def _trial_record(
    *, phase: str, trial_id: str, recorded_at: str,
    params: pairs_engine.PairsParams, config: PairsBacktestConfig,
    dataset_hashes: dict[str, str], schedule: WalkForwardSchedule,
    metrics: dict[str, object], gate: GateDecision, invalid_reasons: list[str],
    cost_stress: dict[str, dict[str, object]],
) -> dict[str, object]:
    return {
        "trial_id": trial_id,
        "recorded_at": recorded_at,
        "phase": phase,
        "params": asdict(params),
        "cost_config": asdict(config),
        "dataset_hashes": dataset_hashes,
        "dataset_hash": dataset_hashes["all"],
        "common_start": schedule.common_start.isoformat(),
        "common_end": schedule.common_end.isoformat(),
        "metrics": metrics,
        "cost_stress": cost_stress,
        "gate": asdict(gate),
        "invalid_reasons": invalid_reasons,
    }


def _schedule_report(schedule: WalkForwardSchedule) -> dict[str, object]:
    return {
        "common_start": schedule.common_start.isoformat(),
        "common_end": schedule.common_end.isoformat(),
        "holdout_start": schedule.holdout_start.isoformat(),
        "development_windows": [
            {
                "formation_start": window.formation_start.isoformat(),
                "start": window.start.isoformat(),
                "end": window.end.isoformat(),
            }
            for window in schedule.development_windows
        ],
    }


def run_pairs_experiment(
    data: dict[str, PairMarketData], *, config: PairsBacktestConfig,
    params: pairs_engine.PairsParams, open_holdout: bool,
    ledger_path: Path,
) -> dict[str, object]:
    """Run development validation and, at most once, the sealed holdout."""
    ledger_path = Path(ledger_path)
    initial_ledger = _read_trial_ledger(ledger_path)
    if open_holdout and initial_ledger["holdout"].get("status") != "sealed":
        raise ValueError("holdout already opened")
    historical_trial_count = len(initial_ledger["trials"])

    hash_frames, interval_frames = _research_frames(data)
    common_start, common_end = _common_history_bounds(interval_frames)
    schedule = build_walk_forward_schedule(common_start, common_end)
    dataset_hashes = {
        name: dataset_hash({name: frame}) for name, frame in sorted(hash_frames.items())
    }
    dataset_hashes["all"] = dataset_hash(hash_frames)

    nominal_config = replace(config, fee_bps=10.0, slippage_bps=2.0)
    development_trades, development_invalid = _run_pairs_windows(
        data,
        windows=schedule.development_windows,
        config=nominal_config,
        params=params,
    )
    development_trial_count = historical_trial_count + 1
    development_metrics = summarize_pair_trades(
        development_trades, trial_count=development_trial_count,
    )
    development_cost_stress = _cost_stress_metrics(
        development_trades,
        base_config=nominal_config,
        trial_count=development_trial_count,
    )
    development_metrics["cost_stress"] = development_cost_stress
    development_metrics["invalid_reasons"] = development_invalid
    development_decision = development_gate(development_metrics)
    development_trial_id = uuid.uuid4().hex
    development_recorded_at = datetime.now(timezone.utc).isoformat()
    development_trial = _trial_record(
        phase="development",
        trial_id=development_trial_id,
        recorded_at=development_recorded_at,
        params=params,
        config=nominal_config,
        dataset_hashes=dataset_hashes,
        schedule=schedule,
        metrics=development_metrics,
        gate=development_decision,
        invalid_reasons=development_invalid,
        cost_stress=development_cost_stress,
    )
    write_trial_ledger(ledger_path, development_trial)

    report = {
        "schedule": _schedule_report(schedule),
        "dataset_hashes": dataset_hashes,
        "development": {
            "metrics": development_metrics,
            "cost_stress": development_cost_stress,
            "gate": asdict(development_decision),
            "invalid_reasons": development_invalid,
            "trial_id": development_trial_id,
        },
        "holdout": {
            "requested": bool(open_holdout),
            "opened": False,
            "status": "sealed",
        },
    }
    if not open_holdout or not development_decision.passed:
        return _json_safe(report)

    holdout_trial_id = uuid.uuid4().hex
    holdout_recorded_at = datetime.now(timezone.utc).isoformat()
    _reserve_holdout(
        ledger_path,
        dataset_digest=dataset_hashes["all"],
        trial_id=holdout_trial_id,
        opened_at=holdout_recorded_at,
    )
    holdout_window = WalkForwardWindow(
        formation_start=schedule.holdout_start - pd.Timedelta(days=60),
        start=schedule.holdout_start,
        end=schedule.common_end,
    )
    holdout_trades, holdout_invalid = _run_pairs_windows(
        data,
        windows=[holdout_window],
        config=nominal_config,
        params=params,
    )
    combined_trades = development_trades + holdout_trades
    combined_invalid = list(dict.fromkeys(development_invalid + holdout_invalid))
    hard_trial_count = historical_trial_count + 2
    hard_metrics = summarize_pair_trades(combined_trades, trial_count=hard_trial_count)
    hard_cost_stress = _cost_stress_metrics(
        combined_trades,
        base_config=nominal_config,
        trial_count=hard_trial_count,
    )
    hard_metrics["cost_stress"] = hard_cost_stress
    hard_metrics["invalid_reasons"] = combined_invalid
    hard_decision = hard_pass_gate(hard_metrics)
    holdout_trial = _trial_record(
        phase="holdout",
        trial_id=holdout_trial_id,
        recorded_at=holdout_recorded_at,
        params=params,
        config=nominal_config,
        dataset_hashes=dataset_hashes,
        schedule=schedule,
        metrics=hard_metrics,
        gate=hard_decision,
        invalid_reasons=combined_invalid,
        cost_stress=hard_cost_stress,
    )
    write_trial_ledger(ledger_path, holdout_trial)
    report["holdout"] = {
        "requested": True,
        "opened": True,
        "status": "opened",
        "trial_id": holdout_trial_id,
        "metrics": hard_metrics,
        "cost_stress": hard_cost_stress,
        "gate": asdict(hard_decision),
        "invalid_reasons": combined_invalid,
    }
    return _json_safe(report)


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


def _funding_as_of(funding_ts, funding_rates, bar_ts):
    """Latest funding rate with ts <= ``bar_ts``; None when unknown/NaN."""
    if funding_ts is None or len(funding_ts) == 0:
        return None
    idx = int(funding_ts.searchsorted(bar_ts, side="right")) - 1
    if idx < 0:
        return None
    rate = float(funding_rates[idx])
    return None if math.isnan(rate) else rate


def run_breakout_backtest(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    funding_df: pd.DataFrame | None,
    *,
    params=None,
    initial_equity: float = 10000.0,
    risk_pct: float = 2.0,
    fee_bps: float = 10.0,
    window_start: pd.Timestamp | None = None,
    cooldown_bars: int = 0,
    tp1_frac: float = 0.5,
) -> dict:
    """Replay the breakout engine over one symbol; same metrics as run_backtest.

    The 15m bars drive the clock; the first decision bar is
    ``max(window_start, MIN_BARS_15M)``. No-lookahead slicing, mirroring the
    trend replay's ``_EVAL_TAIL_BARS`` technique: the decision at bar i uses
    ``df_15m.iloc[:i+2]`` (the engine drops bar i+1 as in-progress, leaving
    data <= bar i); the 1h/4h slices are as-of joins on ts — all bars with
    ``ts <= bar_ts`` plus nothing more, so the engine's drop-last-row policy
    removes the still-forming HTF bar and every indicator sees only fully
    closed bars. The funding rate is the latest entry with ``ts <= bar_ts``
    (empty/NaN -> None, i.e. "unknown": the engine skips the funding filter).

    Fidelity note: the engine only fires on a COMPLETED retest-hold, and the
    entry is that retest bar's close, so entries are already "maker at the
    retest" realistic — no extra fill logic here. Position management is
    identical to ``run_backtest``: SL-first within a bar, TP1 partial
    (``tp1_frac``) + TP2 remainder, the same fee_r formula, compounding
    ``risk_pct``, ``cooldown_bars`` after losses, and a position still open
    at window end is uncounted.
    """
    import breakout_engine

    risk_fraction = risk_pct / 100.0
    fee_rate_roundtrip = 2.0 * fee_bps / 1e4  # both sides, as fraction of notional

    equity = float(initial_equity)
    peak = equity
    max_drawdown_pct = 0.0
    trades = []  # realized R multiples, net of fees
    position = None
    blocked_until = -1  # post-loss cooldown: no entries before this bar index

    # Pre-compute the funding as-of arrays in the 15m frame's datetime unit so
    # searchsorted comparisons are unit-safe (candle frames may be ms or ns).
    funding_ts = funding_rates = None
    if funding_df is not None and not funding_df.empty:
        funding_ts = (
            funding_df["ts"].dt.as_unit(df_15m["ts"].dt.unit).to_numpy()
        )
        funding_rates = funding_df["funding_rate"].to_numpy(dtype=float)

    # As-of cut positions for the HTF frames, in the same unit as df_15m.
    ts_1h = df_1h["ts"].dt.as_unit(df_15m["ts"].dt.unit).to_numpy()
    ts_4h = df_4h["ts"].dt.as_unit(df_15m["ts"].dt.unit).to_numpy()

    if window_start is not None:
        # the cached frame's ts dtype may be datetime64[ms]; match its unit
        window_start = pd.Timestamp(window_start).as_unit(df_15m["ts"].dt.unit)
        window_idx = int(df_15m["ts"].searchsorted(window_start))
        start = max(window_idx, breakout_engine.MIN_BARS_15M)
    else:
        start = breakout_engine.MIN_BARS_15M

    i = start
    while i < len(df_15m) - 1:  # -1: the engine always drops one trailing bar
        bar = df_15m.iloc[i]

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

        # No open position: evaluate as of the close of bar i. The 15m slice
        # ends at bar i+1, which the engine drops as "in progress"; the HTF
        # slices end at the in-progress HTF bar (ts <= bar ts), also dropped.
        # No future data reaches the decision (see docstring).
        if i < blocked_until:
            i += 1
            continue
        bar_ts = bar["ts"]
        win_15m = df_15m.iloc[max(0, i + 2 - _BO_TAIL_15M): i + 2]
        end_1h = int(ts_1h.searchsorted(bar_ts, side="right"))
        end_4h = int(ts_4h.searchsorted(bar_ts, side="right"))
        win_1h = df_1h.iloc[max(0, end_1h - _BO_TAIL_1H):end_1h]
        win_4h = df_4h.iloc[max(0, end_4h - _BO_TAIL_4H):end_4h]
        funding_rate = _funding_as_of(funding_ts, funding_rates, bar_ts)
        ev = breakout_engine.evaluate_breakout(
            win_15m, win_1h, win_4h, funding_rate, params=params,
        )
        if ev["passed_filter"]:
            entry = float(ev["entry_price"])
            sl = float(ev["stop_loss"])
            sl_frac = abs(entry - sl) / entry if entry else 0.0
            position = {
                "direction": ev["direction"],
                "sl": sl,
                "tp1": float(ev["take_profit_1"]),
                "tp2": float(ev["take_profit_2"]),
                "rr1": float(ev["risk_reward_tp1"]),
                "rr2": float(ev["risk_reward_tp2"]),
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
