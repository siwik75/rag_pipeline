"""Tests for breakout_engine — fully offline, synthetic data only.

Frames are fabricated scenario-by-scenario: a steady-drift 4h frame (clean
regime, ER well above the floor) or a pure sine (chop, ER ~ 0); a 1h frame
with a long oscillating stretch (high bandwidth), a flat segment (the squeeze
— rolling-20 std collapses, so the bandwidth argmin lands inside it), a small
dip/rise pair (bandwidth starts rising), and a final big expansion bar; a 15m
frame oscillating under/over the 1h consolidation boundary with a volume-
spike breakout bar and a final retest bar. Every builder appends one dummy
in-progress row that evaluate_breakout must drop.
"""
import numpy as np
import pandas as pd
import pytest

import breakout_engine as be


# --------------------------------------------------------------- constants

END = pd.Timestamp("2026-01-08 12:00")  # ts of the dummy in-progress 15m bar

N_4H = 131   # 130 closed + 1 dummy; MIN_BARS_4H = 111
N_1H = 131   # 130 closed + 1 dummy; MIN_BARS_1H = 122
N_15M = 261  # 260 closed + 1 dummy; covers > 2 days for prior-day levels

# LONG happy-path prices (constructed, asserted exactly in the plan test).
L_BREAKOUT_CLOSE = 100.30
L_RETEST_CLOSE = 100.25   # engine entry
L_RETEST_LOW = 100.08
# SHORT mirror.
S_BREAKOUT_CLOSE = 99.70
S_RETEST_CLOSE = 99.75
S_RETEST_HIGH = 99.92

PLAN_KEYS = {
    "valid", "invalid_reason", "symbol", "timeframe", "signal", "confidence",
    "entry_price", "stop_loss", "take_profit_1", "take_profit_2",
    "risk_reward_tp1", "risk_reward_tp2", "market_regime", "trend_direction",
    "confluence_indicators", "divergent_indicators", "reasoning",
    "context_factors", "indicators",
}


# ---------------------------------------------------------------- builders

def _frame(ts, opens, highs, lows, closes, vols):
    return pd.DataFrame({
        "ts": ts, "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": vols,
    })


def make_4h(direction="long", chop=False, n=N_4H):
    """4h regime frame. long/short: steady drift + mild oscillation (ER well
    above 0.3, SMA sloping with price). chop: pure sine, period divides the
    ER window so the net move is ~0 (ER ~ 0)."""
    i = np.arange(n)
    if chop:
        closes = 100.0 + 3.0 * np.sin(2 * np.pi * i / 10)
    elif direction == "long":
        closes = 60.0 + 0.3 * i + 0.5 * np.sin(2 * np.pi * i / 12)
    else:
        closes = 140.0 - 0.3 * i + 0.5 * np.sin(2 * np.pi * i / 12)
    ts = pd.date_range(end=END + pd.Timedelta(hours=4), periods=n, freq="4h")
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    return _frame(ts, opens, closes + 0.4, closes - 0.4, closes,
                  np.full(n, 1000.0))


def make_1h(direction="long", expansion_agree=True, n=N_1H):
    """1h setup frame: oscillation (high bandwidth), 27-bar flat squeeze,
    small dip/rise pair (bandwidth rising), final expansion bar whose
    direction agrees with ``direction`` unless ``expansion_agree=False``.
    Flat-segment highs/lows cap the consolidation boundary, so the 1h
    consolidation high/low is stable regardless of the exact argmin index."""
    i = np.arange(n)
    closes = np.where(i < 100, 96.0 + 0.04 * i + 0.8 * np.sin(2 * np.pi * i / 10), 0.0)
    closes = np.where((i >= 100) & (i <= n - 5),
                      100.0 + 0.01 * np.sin(2 * np.pi * i / 5), closes)
    # n-1 is the dummy in-progress row; closed bars end at n-2.
    if direction == "long":
        closes[n - 4] = 99.94   # dip: bandwidth starts rising
        closes[n - 3] = 99.88
        closes[n - 2] = 101.00 if expansion_agree else 98.80  # expansion bar
        closes[n - 1] = 100.00  # dummy
    else:
        closes[n - 4] = 100.06  # rise: bandwidth starts rising
        closes[n - 3] = 100.12
        closes[n - 2] = 98.80 if expansion_agree else 101.00  # expansion bar
        closes[n - 1] = 100.00  # dummy

    highs = closes + np.where(i < 100, 0.5, 0.06)
    lows = closes - np.where(i < 100, 0.5, 0.06)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    vols = np.full(n, 1000.0)

    exp = n - 2
    if direction == "long":
        opens[exp] = 99.88
        if expansion_agree:
            highs[exp], lows[exp] = 101.10, 99.85   # range 1.25
        else:
            highs[exp], lows[exp] = 99.95, 98.75    # range 1.20, bearish
    else:
        opens[exp] = 100.12
        if expansion_agree:
            highs[exp], lows[exp] = 100.15, 98.75   # range 1.40
        else:
            highs[exp], lows[exp] = 101.05, 99.90   # range 1.15, bullish
    vols[exp] = 3000.0

    ts = pd.date_range(end=END + pd.Timedelta(hours=1), periods=n, freq="1h")
    return _frame(ts, opens, highs, lows, closes, vols)


def make_15m(direction="long", scenario="retest", breakout_volume=3000.0, n=N_15M):
    """15m trigger frame oscillating just inside the 1h consolidation
    boundary, then a volume-spike breakout bar and (scenario="retest") a
    final retest-and-hold bar. scenario="no_retest" leaves the last bar
    elevated; scenario="expired" puts the breakout 7 bars back (outside the
    6-bar retest window) with no retest since."""
    i = np.arange(n)
    base = 99.96 if direction == "long" else 100.04
    closes = base + 0.04 * np.sin(2 * np.pi * i / 8)
    highs = closes + 0.05
    lows = closes - 0.05
    vols = np.full(n, 1000.0)

    r = n - 2  # last closed bar
    b = r - 7 if scenario == "expired" else r - 2  # breakout bar
    if direction == "long":
        closes[b], highs[b], lows[b], vols[b] = L_BREAKOUT_CLOSE, 100.35, 99.90, breakout_volume
        for k in range(b + 1, r + 1):
            closes[k], highs[k], lows[k] = 100.40, 100.45, 100.25
        if scenario == "retest":
            closes[r], highs[r], lows[r] = L_RETEST_CLOSE, 100.42, L_RETEST_LOW
    else:
        closes[b], highs[b], lows[b], vols[b] = S_BREAKOUT_CLOSE, 100.10, 99.65, breakout_volume
        for k in range(b + 1, r + 1):
            closes[k], highs[k], lows[k] = 99.60, 99.65, 99.55
        if scenario == "retest":
            closes[r], highs[r], lows[r] = S_RETEST_CLOSE, S_RETEST_HIGH, 99.55

    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    ts = pd.date_range(end=END, periods=n, freq="15min")
    return _frame(ts, opens, highs, lows, closes, vols)


def happy_data(direction="long", funding=None, **kwargs):
    """scan_breakout-shaped entry for one symbol."""
    return {
        "15m": make_15m(direction, **kwargs),
        "1h": make_1h(direction),
        "4h": make_4h(direction),
        "funding": funding,
    }


# ------------------------------------------------------------------- tests

class TestLongHappyPath:
    def test_full_pipeline_plan_math(self):
        ev = be.evaluate_breakout(
            make_15m("long"), make_1h("long"), make_4h("long"), None,
        )
        assert ev["passed_filter"] is True
        assert ev["direction"] == "LONG"
        assert ev["reasons"] == []
        assert ev["entry_type"] == "consolidation_retest"
        assert ev["market_regime"] == "breakout_long"
        assert ev["funding_unknown"] is True

        plan = be.build_trade_plan(ev)
        assert plan is not None
        assert set(plan) == PLAN_KEYS
        assert plan["signal"] == "BUY"
        assert plan["valid"] is True
        assert plan["invalid_reason"] is None

        ind = ev["indicators"]
        entry = plan["entry_price"]
        assert entry == pytest.approx(L_RETEST_CLOSE)
        assert ind["level_type"] == "consolidation"
        assert ind["level"] < entry

        # Exact structural math: SL beyond the broken level / retest low.
        expected_sl = min(L_RETEST_LOW, ind["level"]) - be.SL_BUFFER_ATR * ind["atr_15m"]
        assert plan["stop_loss"] == pytest.approx(expected_sl)
        assert plan["stop_loss"] < ind["level"]

        risk = entry - plan["stop_loss"]
        assert plan["take_profit_1"] == pytest.approx(entry + be.TP1_RISK_MULT * risk)
        assert plan["take_profit_2"] == pytest.approx(entry + be.TP2_RISK_MULT * risk)
        assert plan["risk_reward_tp1"] == pytest.approx(be.TP1_RISK_MULT)
        assert plan["risk_reward_tp2"] == pytest.approx(be.TP2_RISK_MULT)

        # Confidence: base + ER weight + fast retest + volume spike.
        expected_conf = min(
            be.CONFIDENCE_CAP,
            be.CONFIDENCE_BASE + be.ER_CONFIDENCE_WEIGHT * min(1.0, ind["er"])
            + be.FAST_RETEST_BONUS + be.VOLUME_SPIKE_BONUS,
        )
        assert plan["confidence"] == pytest.approx(expected_conf)

        # Plan extras and reasoning name real values.
        cf = plan["context_factors"]
        assert cf["entry_type"] == "consolidation_retest"
        assert cf["level"] == pytest.approx(ind["level"])
        assert cf["level_type"] == "consolidation"
        assert cf["funding_rate"] is None
        assert cf["er"] == pytest.approx(ind["er"])
        assert cf["funding_unknown"] is True
        assert str(round(ind["level"], 2)) in plan["reasoning"] or f"{ind['level']:.6g}" in plan["reasoning"]

    def test_build_trade_plan_returns_none_on_failure(self):
        ev = be.evaluate_breakout(
            make_15m("long"), make_1h("long"), make_4h(chop=True), None,
        )
        assert ev["passed_filter"] is False
        assert be.build_trade_plan(ev) is None


class TestRegime:
    def test_chop_low_er_rejected(self):
        ev = be.evaluate_breakout(
            make_15m("long"), make_1h("long"), make_4h(chop=True), None,
        )
        assert ev["passed_filter"] is False
        assert ev["direction"] == "NONE"
        assert any("chop_regime" in r for r in ev["reasons"])


class TestSetup:
    def test_expansion_direction_disagreeing_with_regime(self):
        ev = be.evaluate_breakout(
            make_15m("long"), make_1h("long", expansion_agree=False),
            make_4h("long"), None,
        )
        assert ev["passed_filter"] is False
        assert any("expansion_direction_disagrees" in r for r in ev["reasons"])


class TestTrigger:
    def test_breakout_without_retest_is_still_waiting(self):
        ev = be.evaluate_breakout(
            make_15m("long", scenario="no_retest"), make_1h("long"),
            make_4h("long"), None,
        )
        assert ev["passed_filter"] is False
        assert any("awaiting_retest" in r for r in ev["reasons"])

    def test_breakout_older_than_retest_window_expires(self):
        ev = be.evaluate_breakout(
            make_15m("long", scenario="expired"), make_1h("long"),
            make_4h("long"), None,
        )
        assert ev["passed_filter"] is False
        assert any("setup_expired" in r for r in ev["reasons"])


class TestFunding:
    def test_extreme_funding_vetoes_long(self):
        ev = be.evaluate_breakout(
            make_15m("long"), make_1h("long"), make_4h("long"), 0.001,
        )
        assert ev["passed_filter"] is False
        assert any("funding_extreme_long" in r for r in ev["reasons"])

    def test_unknown_funding_allowed_and_flagged(self):
        ev = be.evaluate_breakout(
            make_15m("long"), make_1h("long"), make_4h("long"), None,
        )
        assert ev["passed_filter"] is True
        assert ev["funding_unknown"] is True
        assert "funding_unknown" in ev["divergent_indicators"]

    def test_normal_funding_not_flagged(self):
        ev = be.evaluate_breakout(
            make_15m("long"), make_1h("long"), make_4h("long"), 0.0001,
        )
        assert ev["passed_filter"] is True
        assert ev["funding_unknown"] is False


class TestShortMirror:
    def test_short_happy_path_inverted_math(self):
        ev = be.evaluate_breakout(
            make_15m("short"), make_1h("short"), make_4h("short"), None,
        )
        assert ev["passed_filter"] is True
        assert ev["direction"] == "SHORT"
        assert ev["market_regime"] == "breakout_short"

        plan = be.build_trade_plan(ev)
        assert plan["signal"] == "SELL"
        ind = ev["indicators"]
        entry = plan["entry_price"]
        assert entry == pytest.approx(S_RETEST_CLOSE)

        expected_sl = max(S_RETEST_HIGH, ind["level"]) + be.SL_BUFFER_ATR * ind["atr_15m"]
        assert plan["stop_loss"] == pytest.approx(expected_sl)
        assert plan["stop_loss"] > ind["level"] > entry

        risk = plan["stop_loss"] - entry
        assert plan["take_profit_1"] == pytest.approx(entry - be.TP1_RISK_MULT * risk)
        assert plan["take_profit_2"] == pytest.approx(entry - be.TP2_RISK_MULT * risk)
        # Inverted ordering: SL > entry > TP1 > TP2.
        assert plan["stop_loss"] > entry > plan["take_profit_1"] > plan["take_profit_2"]
        assert plan["risk_reward_tp1"] == pytest.approx(be.TP1_RISK_MULT)
        assert plan["risk_reward_tp2"] == pytest.approx(be.TP2_RISK_MULT)

    def test_extreme_negative_funding_vetoes_short(self):
        ev = be.evaluate_breakout(
            make_15m("short"), make_1h("short"), make_4h("short"), -0.001,
        )
        assert ev["passed_filter"] is False
        assert any("funding_extreme_short" in r for r in ev["reasons"])


class TestInProgressCandleDrop:
    def test_garbage_final_rows_are_dropped(self):
        df15, df1h, df4h = make_15m("long"), make_1h("long"), make_4h("long")
        for df in (df15, df1h, df4h):
            df.iloc[-1, df.columns.get_loc("open")] = 10.0
            df.iloc[-1, df.columns.get_loc("high")] = 10.0
            df.iloc[-1, df.columns.get_loc("low")] = 10.0
            df.iloc[-1, df.columns.get_loc("close")] = 10.0
            df.iloc[-1, df.columns.get_loc("volume")] = 1e12
        ev = be.evaluate_breakout(df15, df1h, df4h, None)
        assert ev["passed_filter"] is True
        assert ev["entry_price"] == pytest.approx(L_RETEST_CLOSE)


class TestScan:
    def _data(self):
        return {
            "AAA": happy_data("long"),  # volume spike -> higher confidence
            "BBB": happy_data("long", breakout_volume=1600.0),  # trigger ok, no spike bonus
            "CCC": {**happy_data("long"), "4h": make_4h(chop=True)},
            "DDD": {"15m": make_15m("long")},  # missing frames -> error
        }

    def test_ranking_errors_and_scores(self):
        out = be.scan_breakout(self._data())

        assert [e["symbol"] for e in out["errors"]] == ["DDD"]
        assert "KeyError" in out["errors"][0]["error"]

        scores = {s["symbol"]: s for s in out["scores"]}
        assert set(scores) == {"AAA", "BBB", "CCC"}  # error symbol has no score
        assert scores["AAA"]["passed_filter"] is True
        assert scores["BBB"]["passed_filter"] is True
        assert scores["CCC"]["passed_filter"] is False
        assert any("chop_regime" in r for r in scores["CCC"]["reasons"])

        # AAA (with spike bonus) outranks BBB; both plans still emitted.
        assert scores["AAA"]["confidence"] > scores["BBB"]["confidence"]
        assert out["best"] is not None
        assert out["best"]["symbol"] == "AAA"
        assert set(out["best"]) == PLAN_KEYS

    def test_cooldown_symbols_are_skipped(self):
        out = be.scan_breakout(self._data(), cooldown_symbols={"AAA", "CCC"})
        assert out["best"]["symbol"] == "BBB"
        scored = {s["symbol"] for s in out["scores"]}
        assert "AAA" not in scored and "CCC" not in scored

    def test_all_cooled_down_or_failing_gives_none(self):
        out = be.scan_breakout(self._data(), cooldown_symbols={"AAA", "BBB", "CCC", "DDD"})
        assert out["best"] is None
        assert out["scores"] == []
        assert out["errors"] == []
