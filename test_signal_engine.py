"""Tests for signal_engine — fully offline, synthetic data only.

Two layers:
- unit tests on hand-built frames that already carry the indicator columns
  (evaluate_symbol uses pre-computed columns as-is), so each scenario is
  constructed by setting indicator values directly;
- integration tests that run the full ``add_indicators`` compute path on
  synthetic OHLCV random walks (fixed seeds, searched deterministically).
"""
import json

import numpy as np
import pandas as pd
import pytest

import signal_engine as se


# --------------------------------------------------------------- helpers

N = 70  # rows per hand-built frame (MIN_BARS is 60; last row is dropped)

CLOSE = 100.0
ATR = 2.0


def make_frame(
    *,
    direction="long",
    rsi=55.0,
    adx=30.0,
    di_plus=25.0,
    di_minus=15.0,
    ema9=None,
    ema21=None,
    ema50=None,
    close=CLOSE,
    atr=ATR,
    macd_hist_prev=0.1,
    macd_hist_last=0.2,
    vwap=None,
    volume=1500.0,
    vol_sma=1000.0,
    obv_step=10.0,
    n=N,
):
    """Hand-built frame with indicator columns, all constant per bar except
    macd_hist (rising/fading at the end) and obv (linear). The final row is a
    dummy in-progress candle that evaluate_symbol must drop."""
    if direction == "long":
        ema9 = 100.5 if ema9 is None else ema9
        ema21 = 99.8 if ema21 is None else ema21
        ema50 = 98.0 if ema50 is None else ema50
        vwap = 99.0 if vwap is None else vwap
    else:
        ema9 = 99.5 if ema9 is None else ema9
        ema21 = 100.2 if ema21 is None else ema21
        ema50 = 102.0 if ema50 is None else ema50
        vwap = 101.0 if vwap is None else vwap

    macd_hist = np.full(n, macd_hist_prev)
    macd_hist[-2] = macd_hist_last  # last CLOSED bar (final row is dropped)
    vol = np.full(n, vol_sma)
    vol[-2] = volume

    return pd.DataFrame({
        "open": np.full(n, close),
        "high": np.full(n, close + atr / 2),
        "low": np.full(n, close - atr / 2),
        "close": np.full(n, close),
        "volume": vol,
        "ema_9": np.full(n, ema9),
        "ema_21": np.full(n, ema21),
        "ema_50": np.full(n, ema50),
        "rsi_14": np.full(n, rsi),
        "atr_14": np.full(n, atr),
        "adx": np.full(n, adx),
        "di_plus": np.full(n, di_plus),
        "di_minus": np.full(n, di_minus),
        "macd_hist": macd_hist,
        "obv": np.arange(n, dtype=float) * obv_step,
        "vwap_14": np.full(n, vwap),
        "vol_sma_20": np.full(n, vol_sma),
    })


def make_daily_1d(*, direction="long", n=80):
    """Plain OHLCV daily frame with a clean trend (EMA20 vs EMA50 aligned)."""
    step = 1.0 if direction == "long" else -1.0
    closes = 100.0 + step * np.arange(n, dtype=float)
    return pd.DataFrame({
        "open": closes - step * 0.2,
        "high": closes + 0.5,
        "low": closes - 0.5,
        "close": closes,
        "volume": np.full(n, 1000.0),
    })


PLAN_KEYS = {
    "valid", "invalid_reason", "symbol", "timeframe", "signal", "confidence",
    "entry_price", "stop_loss", "take_profit_1", "take_profit_2",
    "risk_reward_tp1", "risk_reward_tp2", "market_regime", "trend_direction",
    "confluence_indicators", "divergent_indicators", "reasoning",
    "context_factors", "indicators",
}


# ------------------------------------------------------------- unit tests

class TestLongSignal:
    def test_long_pullback_plan_math(self):
        ev = se.evaluate_symbol(make_frame(direction="long"), timeframe="4h")
        assert ev["passed_filter"] is True
        assert ev["direction"] == "LONG"
        assert ev["reasons"] == []
        assert ev["entry_type"] == "pullback_to_ema21"
        assert ev["confidence"] == se.CONFLUENCE_BASE + 4 * se.CONFLUENCE_STEP  # 82

        plan = se.build_trade_plan(ev)
        assert plan is not None
        assert set(plan) == PLAN_KEYS
        assert plan["valid"] is True
        assert plan["invalid_reason"] is None
        assert plan["signal"] == "BUY"
        assert plan["timeframe"] == "4h"
        assert plan["market_regime"] == "trending_long"
        assert plan["trend_direction"] == "LONG"

        entry, atr = CLOSE, ATR
        assert plan["entry_price"] == pytest.approx(entry)
        assert plan["stop_loss"] == pytest.approx(entry - se.ATR_SL_MULT * atr)
        assert plan["take_profit_1"] == pytest.approx(entry + se.ATR_TP1_MULT * atr)
        assert plan["take_profit_2"] == pytest.approx(entry + se.ATR_TP2_MULT * atr)
        assert plan["stop_loss"] < plan["entry_price"] < plan["take_profit_1"] < plan["take_profit_2"]
        assert plan["risk_reward_tp1"] == pytest.approx(se.ATR_TP1_MULT / se.ATR_SL_MULT)  # ~1.67
        assert plan["risk_reward_tp2"] == pytest.approx(se.ATR_TP2_MULT / se.ATR_SL_MULT)  # ~2.92

        assert plan["confluence_indicators"] == [
            "macd_histogram_confirms", "price_above_vwap",
            "volume_above_average", "obv_slope_confirms",
        ]
        assert plan["divergent_indicators"] == []
        assert isinstance(plan["reasoning"], str) and "EMA9" in plan["reasoning"]
        assert plan["indicators"]["close"] == pytest.approx(CLOSE)
        assert plan["indicators"]["volume_ratio"] == pytest.approx(1.5)

    def test_daily_alignment_adds_confluence(self):
        ev = se.evaluate_symbol(make_frame(direction="long"), make_daily_1d(direction="long"))
        assert ev["confidence"] == se.CONFLUENCE_BASE + 5 * se.CONFLUENCE_STEP  # 90
        assert "daily_trend_aligned" in ev["confluence_indicators"]

    def test_daily_divergence_is_recorded_not_fatal(self):
        ev = se.evaluate_symbol(make_frame(direction="long"), make_daily_1d(direction="short"))
        assert ev["passed_filter"] is True
        assert "daily_trend_divergent" in ev["divergent_indicators"]
        assert ev["confidence"] == se.CONFLUENCE_BASE + 4 * se.CONFLUENCE_STEP

    def test_last_candle_is_dropped(self):
        df = make_frame(direction="long")
        # wreck the final (in-progress) row: misaligned EMAs, absurd RSI
        df.loc[df.index[-1], ["ema_9", "ema_21", "ema_50"]] = [90.0, 95.0, 99.0]
        df.loc[df.index[-1], "rsi_14"] = 99.0
        ev = se.evaluate_symbol(df)
        assert ev["passed_filter"] is True
        assert ev["direction"] == "LONG"


class TestRejections:
    def test_ranging_market_no_signal(self):
        # EMAs interleaved and ADX weak: no trend
        ev = se.evaluate_symbol(make_frame(
            direction="long", ema9=100.2, ema21=100.0, ema50=100.4, adx=12.0,
        ))
        assert ev["passed_filter"] is False
        assert ev["direction"] == "NONE"
        assert any("no_ema_alignment" in r for r in ev["reasons"])
        assert any("adx_too_weak" in r for r in ev["reasons"])
        assert se.build_trade_plan(ev) is None

    def test_rsi_overbought_blocks_long(self):
        ev = se.evaluate_symbol(make_frame(direction="long", rsi=75.0))
        assert ev["passed_filter"] is False
        assert any("rsi_outside_guard" in r for r in ev["reasons"])
        assert se.build_trade_plan(ev) is None

    def test_no_entry_timing_blocks_signal(self):
        # close far from EMA21 (trend aligned, RSI fine, but no pullback/cross)
        ev = se.evaluate_symbol(make_frame(direction="long", ema21=95.0, ema50=94.0))
        assert ev["passed_filter"] is False
        assert any("no_entry_timing" in r for r in ev["reasons"])

    def test_confidence_below_min_means_no_plan(self):
        # trend + timing pass, but every confluence check fails -> confidence 50
        ev = se.evaluate_symbol(make_frame(
            direction="long",
            macd_hist_prev=0.3, macd_hist_last=0.1,   # sign ok, magnitude fading
            vwap=101.0,                                # wrong side of VWAP
            volume=1000.0,                             # no volume spike
            obv_step=-10.0,                            # OBV falling
        ))
        assert ev["passed_filter"] is True
        assert ev["confidence"] == se.CONFLUENCE_BASE  # 50
        assert len(ev["divergent_indicators"]) == 4
        assert se.build_trade_plan(ev) is None

    def test_insufficient_bars(self):
        ev = se.evaluate_symbol(make_frame(direction="long", n=50))
        assert ev["passed_filter"] is False
        assert any("insufficient_bars" in r for r in ev["reasons"])


class TestShortSignal:
    def test_short_mirror_plan_math(self):
        frame = make_frame(
            direction="short", rsi=45.0,
            di_plus=15.0, di_minus=25.0,
            macd_hist_prev=-0.1, macd_hist_last=-0.2,
            obv_step=-10.0,
        )
        ev = se.evaluate_symbol(frame)
        assert ev["passed_filter"] is True
        assert ev["direction"] == "SHORT"

        plan = se.build_trade_plan(ev)
        assert plan["signal"] == "SELL"
        assert plan["market_regime"] == "trending_short"
        entry, atr = CLOSE, ATR
        assert plan["stop_loss"] == pytest.approx(entry + se.ATR_SL_MULT * atr)
        assert plan["take_profit_1"] == pytest.approx(entry - se.ATR_TP1_MULT * atr)
        assert plan["take_profit_2"] == pytest.approx(entry - se.ATR_TP2_MULT * atr)
        assert plan["stop_loss"] > plan["entry_price"] > plan["take_profit_1"] > plan["take_profit_2"]
        assert plan["risk_reward_tp1"] == pytest.approx(se.ATR_TP1_MULT / se.ATR_SL_MULT)

    def test_rsi_oversold_blocks_short(self):
        ev = se.evaluate_symbol(make_frame(
            direction="short", rsi=25.0, di_plus=15.0, di_minus=25.0,
        ))
        assert ev["passed_filter"] is False
        assert any("rsi_outside_guard" in r for r in ev["reasons"])


class TestScan:
    def _data(self):
        return {
            # 4 confluences -> 82
            "GOOD/USDT": make_frame(direction="long"),
            # 3 confluences (no volume spike) -> 74
            "OK/USDT": make_frame(direction="long", volume=1000.0),
            # ranging -> no signal
            "FLAT/USDT": make_frame(direction="long", ema9=100.2, ema21=100.0,
                                    ema50=100.4, adx=12.0),
        }

    def test_ranking_picks_higher_confidence(self):
        out = se.scan_symbols(self._data(), timeframe="4h")
        assert out["errors"] == []
        assert out["best"] is not None
        assert out["best"]["symbol"] == "GOOD/USDT"
        assert out["best"]["timeframe"] == "4h"
        assert out["best"]["confidence"] == pytest.approx(82.0)
        by_symbol = {s["symbol"]: s for s in out["scores"]}
        assert by_symbol["FLAT/USDT"]["passed_filter"] is False
        assert by_symbol["OK/USDT"]["confidence"] == pytest.approx(74.0)

    def test_cooldown_symbols_are_skipped(self):
        out = se.scan_symbols(self._data(), cooldown_symbols={"GOOD/USDT"})
        assert out["best"]["symbol"] == "OK/USDT"
        assert "GOOD/USDT" not in {s["symbol"] for s in out["scores"]}

    def test_all_cooled_down_means_no_best(self):
        out = se.scan_symbols(
            self._data(),
            cooldown_symbols={"GOOD/USDT", "OK/USDT", "FLAT/USDT"},
        )
        assert out["best"] is None
        assert out["scores"] == []

    def test_per_symbol_errors_are_collected(self):
        data = self._data()
        data["BROKEN/USDT"] = pd.DataFrame({"close": [1.0, 2.0]})  # missing columns
        out = se.scan_symbols(data)
        assert out["best"] is not None
        assert len(out["errors"]) == 1
        assert out["errors"][0]["symbol"] == "BROKEN/USDT"


# ------------------------------------------------------ integration tests

class TestSignalParams:
    def test_defaults_match_module_constants(self):
        p = se.SignalParams()
        assert p.ATR_SL_MULT == se.ATR_SL_MULT
        assert p.ATR_TP1_MULT == se.ATR_TP1_MULT
        assert p.ATR_TP2_MULT == se.ATR_TP2_MULT
        assert p.ADX_MIN == se.ADX_MIN
        assert p.RSI_LONG == se.RSI_LONG
        assert p.RSI_SHORT == se.RSI_SHORT
        assert p.CONFLUENCE_BASE == se.CONFLUENCE_BASE
        assert p.CONFLUENCE_STEP == se.CONFLUENCE_STEP
        assert p.CONFIDENCE_CAP == se.CONFIDENCE_CAP
        assert p.PULLBACK_ATR_FRAC == se.PULLBACK_ATR_FRAC
        assert p.CROSS_LOOKBACK_BARS == se.CROSS_LOOKBACK_BARS
        assert p.VOLUME_MULT == se.VOLUME_MULT

    def test_wider_sl_changes_plan_prices(self):
        ev = se.evaluate_symbol(make_frame(direction="long"), timeframe="4h")
        default_plan = se.build_trade_plan(ev)
        wide = se.SignalParams(ATR_SL_MULT=se.ATR_SL_MULT + 1.0,
                               ATR_TP1_MULT=se.ATR_TP1_MULT + 1.0)
        wide_plan = se.build_trade_plan(ev, params=wide)
        assert wide_plan["stop_loss"] == pytest.approx(
            CLOSE - (se.ATR_SL_MULT + 1.0) * ATR)
        assert wide_plan["take_profit_1"] == pytest.approx(
            CLOSE + (se.ATR_TP1_MULT + 1.0) * ATR)
        assert wide_plan["stop_loss"] < default_plan["stop_loss"]
        assert wide_plan["risk_reward_tp1"] == pytest.approx(
            (se.ATR_TP1_MULT + 1.0) / (se.ATR_SL_MULT + 1.0))

    def test_params_change_filter_behavior(self):
        # ADX 30 passes the default threshold but not a raised one (35)
        frame = make_frame(direction="long", adx=30.0)
        assert se.evaluate_symbol(frame)["passed_filter"] is True
        ev = se.evaluate_symbol(frame, params=se.SignalParams(ADX_MIN=35))
        assert ev["passed_filter"] is False
        assert any("adx_too_weak" in r for r in ev["reasons"])

    def test_scan_symbols_threads_params(self):
        data = {"GOOD/USDT": make_frame(direction="long")}
        out = se.scan_symbols(data, params=se.SignalParams(ATR_SL_MULT=2.0))
        assert out["best"]["stop_loss"] == pytest.approx(CLOSE - 2.0 * ATR)

    def test_params_to_dict_is_json_safe(self):
        d = se.params_to_dict(se.SignalParams())
        assert d["RSI_LONG"] == [45, 65]
        assert d["ATR_SL_MULT"] == se.ATR_SL_MULT
        json.dumps(d)


# ------------------------------------------------------ integration tests

def make_ohlcv(seed, *, n=320, drift=0.0012, vol=0.008, base=100.0):
    """Synthetic OHLCV random walk with drift (fixed seed -> deterministic)."""
    rng = np.random.default_rng(seed)
    rets = drift + vol * rng.standard_normal(n)
    close = base * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[base], close[:-1]])
    spread = np.abs(rng.standard_normal(n)) * 0.004 * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = 1000.0 * (1.0 + 0.3 * np.abs(rng.standard_normal(n)))
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    })


def _find_setup(direction, drift):
    """First seed whose synthetic series yields a tradeable signal."""
    for seed in range(1000):
        df = make_ohlcv(seed, drift=drift)
        ev = se.evaluate_symbol(df)
        if (ev["passed_filter"] and ev["direction"] == direction
                and ev["confidence"] >= 70.0):
            return df, ev
    raise AssertionError(f"no synthetic {direction} setup found in 1000 seeds")


class TestFullComputePath:
    def test_uptrend_long_from_raw_ohlcv(self):
        df, ev = _find_setup("LONG", drift=0.0012)
        plan = se.build_trade_plan(ev)
        assert plan is not None
        entry = ev["indicators"]["close"]
        atr = ev["indicators"]["atr"]
        assert plan["signal"] == "BUY"
        assert plan["entry_price"] == pytest.approx(entry)
        assert plan["stop_loss"] == pytest.approx(entry - se.ATR_SL_MULT * atr)
        assert plan["take_profit_1"] == pytest.approx(entry + se.ATR_TP1_MULT * atr)
        assert plan["take_profit_2"] == pytest.approx(entry + se.ATR_TP2_MULT * atr)
        assert plan["stop_loss"] < plan["entry_price"] < plan["take_profit_1"] < plan["take_profit_2"]
        assert plan["risk_reward_tp1"] == pytest.approx(se.ATR_TP1_MULT / se.ATR_SL_MULT)
        # indicator values come from the full ta-based compute path
        assert ev["indicators"]["ema9"] > ev["indicators"]["ema21"] > ev["indicators"]["ema50"]
        assert ev["indicators"]["adx"] > se.ADX_MIN

    def test_downtrend_short_from_raw_ohlcv(self):
        df, ev = _find_setup("SHORT", drift=-0.0012)
        plan = se.build_trade_plan(ev)
        assert plan is not None
        entry = ev["indicators"]["close"]
        atr = ev["indicators"]["atr"]
        assert plan["signal"] == "SELL"
        assert plan["stop_loss"] == pytest.approx(entry + se.ATR_SL_MULT * atr)
        assert plan["stop_loss"] > plan["entry_price"] > plan["take_profit_1"] > plan["take_profit_2"]

    def test_flat_market_produces_no_signal(self):
        # zero drift, mean-reverting-ish sinusoid: ADX should stay weak
        n = 320
        t = np.arange(n)
        close = 100.0 + 3.0 * np.sin(t / 8.0)
        df = pd.DataFrame({
            "open": np.roll(close, 1), "high": close + 0.3, "low": close - 0.3,
            "close": close, "volume": np.full(n, 1000.0),
        })
        ev = se.evaluate_symbol(df)
        assert ev["passed_filter"] is False
        assert se.build_trade_plan(ev) is None
