#!/usr/bin/env python3
"""In-process parameter sweep for the deterministic signal engine.

Loads candles ONCE per symbol (via backtest_core.load_candles, disk-cached),
then replays a grid of SignalParams overrides by calling
backtest_core.run_backtest directly — no subprocesses, no re-fetching.

Usage (from the rag_pipeline dir, with its venv):
    ./.venv/bin/python sweep.py --symbols BTCUSDT,ETHUSDT --timeframe 4h \
        --train-start-days-ago 365 --train-end-days-ago 95 \
        --grid coarse --top 15

Prints the top-N configs by aggregate profit factor (only configs with
>= MIN_TRADES trades are listed; thinner ones are counted as "thin") and
writes the full results to /tmp/astertrade_sweep_results.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import backtest_core  # noqa: E402
import signal_engine  # noqa: E402

RESULTS_PATH = "/tmp/astertrade_sweep_results.json"
MIN_TRADES = 15

COARSE_GRID = {
    "ATR_SL_MULT": [0.8, 1.0, 1.2, 1.5],
    "ATR_TP1_MULT": [1.5, 2.0, 2.5],
    "ADX_MIN": [15, 20, 25, 30],
    "min_confidence": [60, 70, 80],
    "PULLBACK_ATR_FRAC": [0.35, 0.5, 0.75],
}

FINE_GRID = {
    "ATR_SL_MULT": [0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    "ATR_TP1_MULT": [1.5, 1.75, 2.0, 2.25, 2.5],
    "ADX_MIN": [15, 18, 20, 22, 25],
    "min_confidence": [58, 66, 74],
    "PULLBACK_ATR_FRAC": [0.4, 0.5, 0.6, 0.75],
}


def iter_grid(grid: dict):
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        combo = dict(zip(keys, values))
        min_confidence = combo.pop("min_confidence")
        yield signal_engine.SignalParams(**combo), float(min_confidence)


def short_params(params: signal_engine.SignalParams, min_confidence: float) -> str:
    return (f"SL={params.ATR_SL_MULT} TP1={params.ATR_TP1_MULT} "
            f"ADX={int(params.ADX_MIN)} C={int(min_confidence)} "
            f"PB={params.PULLBACK_ATR_FRAC}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Signal engine parameter sweep")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    ap.add_argument("--timeframe", default="4h")
    ap.add_argument("--train-start-days-ago", type=int, default=365)
    ap.add_argument("--train-end-days-ago", type=int, default=95)
    ap.add_argument("--grid", choices=["coarse", "fine"], default="coarse")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--quote", default="USDT")
    ap.add_argument("--initial-equity", type=float, default=10000.0)
    ap.add_argument("--risk-pct", type=float, default=2.0)
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--cache-dir", default=backtest_core.DEFAULT_CACHE_DIR)
    ap.add_argument("--out", default=RESULTS_PATH)
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    grid = COARSE_GRID if args.grid == "coarse" else FINE_GRID
    combos = list(iter_grid(grid))
    print(f"[info] grid={args.grid}: {len(combos)} combos x {len(symbols)} symbols",
          file=sys.stderr)

    now = pd.Timestamp.now(tz="UTC")
    window_start = now - pd.Timedelta(days=args.train_start_days_ago)
    window_end = now - pd.Timedelta(days=args.train_end_days_ago)

    # ---- load candles once -------------------------------------------------
    data: dict[str, tuple] = {}
    fetch_days = args.train_start_days_ago + backtest_core.warmup_days(args.timeframe)
    fetch_days_1d = args.train_start_days_ago + backtest_core.WARMUP_BARS_1D
    for symbol in symbols:
        df = backtest_core.load_candles(symbol, args.timeframe, fetch_days,
                                        exchange=args.exchange, quote=args.quote,
                                        cache_dir=args.cache_dir)
        df_1d = backtest_core.load_candles(symbol, "1d", fetch_days_1d,
                                           exchange=args.exchange, quote=args.quote,
                                           cache_dir=args.cache_dir)
        df = backtest_core.slice_window(df, window_start, window_end, args.timeframe)
        df_1d = df_1d[df_1d["ts"] < window_end].reset_index(drop=True)
        data[symbol] = (df, df_1d)
        print(f"[info] {symbol}: {len(df)} {args.timeframe} bars, {len(df_1d)} 1d bars "
              f"(window {window_start.date()}..{window_end.date()})", file=sys.stderr)

    # ---- run the grid ------------------------------------------------------
    results = []
    t0 = time.time()
    for idx, (params, min_confidence) in enumerate(combos, 1):
        per_symbol = {}
        for symbol, (df, df_1d) in data.items():
            per_symbol[symbol] = backtest_core.run_backtest(
                df, df_1d,
                min_confidence=min_confidence,
                params=params,
                initial_equity=args.initial_equity,
                risk_pct=args.risk_pct,
                fee_bps=args.fee_bps,
                timeframe=args.timeframe,
                window_start=window_start,
            )
        agg = backtest_core.aggregate(per_symbol, args.initial_equity)
        results.append({
            "params": signal_engine.params_to_dict(params),
            "min_confidence": min_confidence,
            "label": short_params(params, min_confidence),
            "aggregate": agg,
            "per_symbol": {s: {k: v for k, v in m.items() if not k.startswith("_")}
                           for s, m in per_symbol.items()},
        })
        if idx % 50 == 0 or idx == len(combos):
            elapsed = time.time() - t0
            print(f"[info] {idx}/{len(combos)} combos, {elapsed:.0f}s elapsed",
                  file=sys.stderr)

    runtime_s = time.time() - t0

    # ---- rank & report -----------------------------------------------------
    def rank_key(r):
        pf = r["aggregate"]["profit_factor"]
        return pf if pf is not None else float("inf")

    eligible = [r for r in results if r["aggregate"]["trades"] >= MIN_TRADES]
    thin = len(results) - len(eligible)
    eligible.sort(key=rank_key, reverse=True)

    print(f"\nSweep: {args.grid} grid, {len(combos)} combos, "
          f"{args.train_start_days_ago}->{args.train_end_days_ago} days ago, "
          f"{args.timeframe}, fee {args.fee_bps} bps/side, {runtime_s:.0f}s runtime")
    print(f"symbols: {', '.join(symbols)} | window: {window_start.date()}..{window_end.date()}")
    print(f"{thin}/{len(results)} configs are thin (< {MIN_TRADES} trades) and excluded")
    print()
    header = (f"{'#':>3}  {'params':<38} {'trades':>6} {'win_rate':>8} {'PF':>7} "
              f"{'avg_rr':>7} {'max_dd':>7} {'final_eq':>10}")
    print(header)
    print("-" * len(header))
    for rank, r in enumerate(eligible[: args.top], 1):
        a = r["aggregate"]
        pf = f"{a['profit_factor']:.3f}" if a["profit_factor"] is not None else "inf"
        print(f"{rank:>3}  {r['label']:<38} {a['trades']:>6} {a['win_rate']:>8.3f} "
              f"{pf:>7} {a['avg_rr']:>7.3f} {a['max_drawdown_pct']:>7.2f} "
              f"{a['final_equity']:>10.2f}")

    with open(args.out, "w") as fh:
        json.dump({
            "grid": args.grid,
            "symbols": symbols,
            "timeframe": args.timeframe,
            "train_start_days_ago": args.train_start_days_ago,
            "train_end_days_ago": args.train_end_days_ago,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "fee_bps": args.fee_bps,
            "risk_pct": args.risk_pct,
            "initial_equity": args.initial_equity,
            "min_trades": MIN_TRADES,
            "runtime_s": round(runtime_s, 1),
            "n_combos": len(results),
            "n_thin": thin,
            "results": results,
        }, fh, indent=2)
    print(f"\nfull results -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
