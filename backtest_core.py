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
from dataclasses import dataclass
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
    if len(pair_marks) != len(btc_logs) or len(pair_marks) < 4:
        return None
    pair_increments = np.diff(pair_marks)
    btc_increments = np.diff(btc_logs)
    finite = np.isfinite(pair_increments) & np.isfinite(btc_increments)
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
    position: dict[str, object], zscore: float | None, alt_close: float, btc_close: float,
) -> None:
    gross_return = _weighted_price_return(position, alt_close, btc_close, slipped=False)
    position["mfe_return"] = max(position["mfe_return"], gross_return)
    position["mae_return"] = min(position["mae_return"], gross_return)
    position["pair_marks"].append(gross_return)
    position["btc_logs"].append(math.log(btc_close))
    if zscore is not None and np.isfinite(zscore):
        pair_direction = -1.0 if position["entry_z"] > 0.0 else 1.0
        z_excursion = pair_direction * (zscore - position["entry_z"])
        position["mfe_z"] = max(position["mfe_z"], z_excursion)
        position["mae_z"] = min(position["mae_z"], z_excursion)
        position["last_z"] = zscore


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
    fee_return = -fee_rate * (
        signal.alt_weight + signal.btc_weight + signal.alt_weight + signal.btc_weight
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
        weighted_beta_numerator += trade.realized_btc_beta * weight
        weighted_beta_denominator += weight
    aggregate_beta = (
        abs(weighted_beta_numerator / weighted_beta_denominator)
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
                _append_mark(position, zscore, close_prices[0], close_prices[1])
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
