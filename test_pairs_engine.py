"""Tests for the causal hourly relative-value relationship model."""
import numpy as np
import pandas as pd
import pytest

import backtest_core as bc
import pairs_engine as pe


TS0 = pd.Timestamp("2024-01-01T07:30:00Z")
TS1 = pd.Timestamp("2024-01-01T07:45:00Z")
TS2 = pd.Timestamp("2024-01-01T08:00:00Z")
TS3 = pd.Timestamp("2024-01-01T08:15:00Z")
TS_END = pd.Timestamp("2024-01-01T08:30:00Z")
BASE_CONFIG = bc.PairsBacktestConfig(
    initial_equity=10_000.0,
    risk_pct=1.0,
    fee_bps=10.0,
    slippage_bps=2.0,
)


def _fifteen_minute_leg(zscores: list[float], *, btc: bool) -> pd.DataFrame:
    timestamps = pd.date_range(TS0, TS_END, freq="15min")
    closes = np.full(len(timestamps), 100.0)
    if not btc:
        closes = 100.0 * np.exp(np.asarray(zscores) * 0.1)
    return pd.DataFrame({
        "ts": timestamps,
        "open": 100.0,
        "high": np.maximum(100.0, closes),
        "low": np.minimum(100.0, closes),
        "close": closes,
        "volume": 1_000.0,
    })


def _hourly_leg(*, model_z: float, btc: bool) -> pd.DataFrame:
    close = 100.0 if btc else 100.0 * np.exp(model_z * 0.1)
    return pd.DataFrame({
        "ts": [pd.Timestamp("2024-01-01T06:00:00Z")],
        "open": [close],
        "high": [close],
        "low": [close],
        "close": [close],
        "volume": [1_000.0],
        "model_z": [model_z],
    })


def _funding_leg(rate: float) -> pd.DataFrame:
    return pd.DataFrame({"ts": [TS2], "funding_rate": [rate]})


def make_two_pair_market_data(
    *,
    with_funding: bool = False,
    missing_two_bars: bool = False,
    no_exit_signal: bool = False,
    sol_entry: bool = False,
) -> dict[str, bc.PairMarketData]:
    eth_zscores = [2.0, 1.8, 1.5 if no_exit_signal else 0.4, 1.4, 1.3]
    eth_15m = _fifteen_minute_leg(eth_zscores, btc=False)
    if missing_two_bars:
        eth_15m = eth_15m.loc[~eth_15m["ts"].isin([TS2, TS3])].reset_index(drop=True)
    alt_funding_rate = -0.0001 if with_funding else 0.0
    btc_funding_rate = -0.0002 if with_funding else 0.0

    eth = bc.PairMarketData(
        pair="ETHUSDT/BTCUSDT",
        alt_symbol="ETHUSDT",
        btc_symbol="BTCUSDT",
        alt_1h=_hourly_leg(model_z=2.2, btc=False),
        btc_1h=_hourly_leg(model_z=0.0, btc=True),
        alt_15m=eth_15m,
        btc_15m=_fifteen_minute_leg([0.0] * 5, btc=True),
        alt_funding=_funding_leg(alt_funding_rate),
        btc_funding=_funding_leg(btc_funding_rate),
    )
    sol_zscores = eth_zscores if sol_entry else [0.0] * 5
    sol = bc.PairMarketData(
        pair="SOLUSDT/BTCUSDT",
        alt_symbol="SOLUSDT",
        btc_symbol="BTCUSDT",
        alt_1h=_hourly_leg(model_z=2.2 if sol_entry else 0.0, btc=False),
        btc_1h=_hourly_leg(model_z=0.0, btc=True),
        alt_15m=_fifteen_minute_leg(sol_zscores, btc=False),
        btc_15m=_fifteen_minute_leg([0.0] * 5, btc=True),
        alt_funding=_funding_leg(0.0),
        btc_funding=_funding_leg(0.0),
    )
    return {eth.pair: eth, sol.pair: sol}


@pytest.fixture
def replay_observations(monkeypatch):
    def build_observations(hourly, params, *, pair):
        current = hourly.iloc[-1]
        model_z = (
            np.log(float(current["alt_close"])) - np.log(float(current["btc_close"]))
        ) / 0.1
        snapshot = pe.RelationshipSnapshot(
            alpha=0.0,
            beta=1.0,
            residual=model_z * 0.1,
            residual_mean=0.0,
            residual_std=0.1,
            return_correlation=0.9,
            adf_pvalue=0.01,
            half_life_hours=12.0,
            observations=1_440,
            beta_change=0.0,
            stable=True,
            rejection_reason=None,
        )
        direction = None
        if model_z >= params.entry_z:
            direction = "SHORT_ALT_LONG_BTC"
        elif model_z <= -params.entry_z:
            direction = "LONG_ALT_SHORT_BTC"
        observation = pe.PairObservation(
            pair=pair,
            ts=pd.Timestamp(hourly.iloc[-1]["ts"]),
            snapshot=snapshot,
            zscore=model_z,
            direction=direction,
        )
        return pd.DataFrame([{
            "pair": pair,
            "ts": observation.ts,
            "snapshot": snapshot,
            "zscore": model_z,
            "direction": direction,
            "beta": snapshot.beta,
        }])

    monkeypatch.setattr(pe, "build_hourly_observations", build_observations)


def make_observation(
    zscore: float | None,
    beta: float,
    stable: bool = True,
    pair: str = "ETHUSDT/BTCUSDT",
) -> pe.PairObservation:
    snapshot = pe.RelationshipSnapshot(
        alpha=0.3,
        beta=beta,
        residual=0.0,
        residual_mean=0.0,
        residual_std=0.01,
        return_correlation=0.9,
        adf_pvalue=0.01,
        half_life_hours=12.0,
        observations=1_440,
        beta_change=0.0,
        stable=stable,
        rejection_reason=None if stable else "beta_change",
    )
    return pe.PairObservation(
        pair=pair,
        ts=pd.Timestamp("2024-01-01T00:00:00Z"),
        snapshot=snapshot,
        zscore=zscore,
        direction=None,
    )


def make_signal(zscore: float, beta: float) -> pe.PairSignal:
    return pe.PairSignal(
        pair="ETHUSDT/BTCUSDT",
        decision_ts=pd.Timestamp("2024-01-01T00:00:00Z"),
        beta=beta,
        zscore=zscore,
        alt_side="SHORT" if zscore > 0 else "LONG",
        btc_side="LONG" if zscore > 0 else "SHORT",
        alt_weight=1 / (1 + abs(beta)),
        btc_weight=abs(beta) / (1 + abs(beta)),
    )


def exit_case(reason: str) -> dict[str, object]:
    entry_ts = pd.Timestamp("2024-01-01T00:00:00Z")
    cases = {
        "convergence": {
            "zscore": 0.5,
            "stable": True,
            "decision_ts": entry_ts + pd.Timedelta(hours=1),
            "consecutive_missing_bars": 0,
        },
        "divergence_stop": {
            "zscore": 3.5,
            "stable": True,
            "decision_ts": entry_ts + pd.Timedelta(hours=1),
            "consecutive_missing_bars": 0,
        },
        "time_stop": {
            "zscore": 1.0,
            "stable": True,
            "decision_ts": entry_ts + pd.Timedelta(hours=72),
            "consecutive_missing_bars": 0,
        },
        "structural": {
            "zscore": 1.0,
            "stable": False,
            "decision_ts": entry_ts + pd.Timedelta(hours=1),
            "consecutive_missing_bars": 0,
        },
        "data_gap": {
            "zscore": 1.0,
            "stable": True,
            "decision_ts": entry_ts + pd.Timedelta(hours=1),
            "consecutive_missing_bars": 2,
        },
    }
    return {"entry_ts": entry_ts, **cases[reason]}


def make_cointegrated_hourly(hours: int, beta: float = 1.2) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC")
    rng = np.random.default_rng(20260814)
    btc_log = np.log(40_000.0) + np.cumsum(rng.normal(0.0, 0.003, hours))
    phi = np.exp(-np.log(2.0) / 12.0)
    residual = np.zeros(hours)
    for index in range(1, hours):
        residual[index] = phi * residual[index - 1] + rng.normal(0.0, 0.0005)
    return pd.DataFrame({
        "ts": ts,
        "btc_close": np.exp(btc_log),
        "alt_close": np.exp(0.3 + beta * btc_log + residual),
    })


def make_constant_btc_hourly(hours: int) -> pd.DataFrame:
    frame = make_cointegrated_hourly(hours)
    frame["btc_close"] = 40_000.0
    return frame


def make_negative_beta_hourly(hours: int) -> pd.DataFrame:
    return make_cointegrated_hourly(hours, beta=-1.0)


def test_fit_uses_only_history_before_decision():
    hourly = make_cointegrated_hourly(hours=1_500)
    first = pe.build_hourly_observations(hourly, pe.DEFAULT_PARAMS, pair="ETHUSDT/BTCUSDT")
    mutated = hourly.copy()
    mutation_index = 1_470
    mutated.loc[mutated.index >= mutation_index, "alt_close"] *= 100.0
    second = pe.build_hourly_observations(mutated, pe.DEFAULT_PARAMS, pair="ETHUSDT/BTCUSDT")
    compared = first.loc[first["ts"] < hourly.loc[mutation_index, "ts"]]
    assert compared["snapshot"].notna().any()
    pd.testing.assert_series_equal(
        compared["beta"],
        second.loc[second["ts"] < hourly.loc[mutation_index, "ts"], "beta"],
        check_names=False,
    )


def test_positive_cointegrated_history_passes_all_stability_gates():
    snapshot = pe.fit_relationship(make_cointegrated_hourly(hours=1_500), pe.DEFAULT_PARAMS)
    assert snapshot is not None
    assert snapshot.stable is True
    assert snapshot.beta > 0
    assert snapshot.observations == 1_440


def test_observations_wait_for_a_full_formation_window():
    hourly = make_cointegrated_hourly(hours=1_500)
    observations = pe.build_hourly_observations(hourly, pe.DEFAULT_PARAMS, pair="ETHUSDT/BTCUSDT")
    early = observations.loc[observations["ts"] < hourly.loc[1_440, "ts"]]
    assert early["snapshot"].isna().all()


def test_unstable_baseline_cannot_emit_direction():
    hourly = make_cointegrated_hourly(hours=1_500)
    current_index = 1_464
    hourly.loc[current_index, "alt_close"] *= 1.05

    observations = pe.build_hourly_observations(hourly, pe.DEFAULT_PARAMS, pair="ETHUSDT/BTCUSDT")
    baseline = observations.loc[1_440, "snapshot"]
    current = observations.loc[current_index]

    assert baseline is not None
    assert baseline.stable is False
    assert current["snapshot"].stable is True
    assert abs(current["zscore"]) >= pe.DEFAULT_PARAMS.entry_z
    assert current["direction"] is None


def test_short_input_keeps_timestamps_with_null_observations():
    hourly = make_cointegrated_hourly(hours=1_199)
    observations = pe.build_hourly_observations(hourly, pe.DEFAULT_PARAMS, pair="ETHUSDT/BTCUSDT")

    pd.testing.assert_series_equal(observations["ts"], hourly["ts"], check_names=False)
    assert observations["snapshot"].isna().all()
    assert observations["zscore"].isna().all()
    assert observations["direction"].isna().all()


def test_singular_nonpositive_or_short_history_is_rejected():
    assert pe.fit_relationship(make_cointegrated_hourly(hours=1_199), pe.DEFAULT_PARAMS) is None
    assert pe.fit_relationship(make_constant_btc_hourly(hours=1_500), pe.DEFAULT_PARAMS) is None
    assert pe.fit_relationship(make_negative_beta_hourly(hours=1_500), pe.DEFAULT_PARAMS) is None


def test_signal_weights_and_directions_mirror_the_lagged_beta():
    high = pe.make_signal(make_observation(zscore=2.25, beta=2.0), pe.DEFAULT_PARAMS)
    low = pe.make_signal(make_observation(zscore=-2.25, beta=2.0), pe.DEFAULT_PARAMS)
    assert (high.alt_side, high.btc_side, high.alt_weight, high.btc_weight) == ("SHORT", "LONG", 1 / 3, 2 / 3)
    assert (low.alt_side, low.btc_side, low.alt_weight, low.btc_weight) == ("LONG", "SHORT", 1 / 3, 2 / 3)


def test_confirmation_requires_reversion_then_expires_after_four_closed_bars():
    pending = pe.PendingConfirmation(signal=make_signal(zscore=2.0, beta=1.0), bars_seen=0)
    assert pe.advance_confirmation(pending, zscore=1.95, params=pe.DEFAULT_PARAMS).entered is False
    entered = pe.advance_confirmation(pending, zscore=1.90, params=pe.DEFAULT_PARAMS)
    assert entered.entered is True
    for _ in range(4):
        pending = pe.advance_confirmation(pending, zscore=2.10, params=pe.DEFAULT_PARAMS).pending
    assert pending is None


@pytest.mark.parametrize("reason", ["convergence", "divergence_stop", "time_stop", "structural", "data_gap"])
def test_exit_classification_uses_only_pair_level_rules(reason):
    assert pe.classify_exit(**exit_case(reason), params=pe.DEFAULT_PARAMS).reason == reason


@pytest.mark.parametrize("pair", ["ETHUSDT/BTCUSDT", "SOLUSDT/BTCUSDT"])
def test_canonical_pair_identity_propagates_through_observations_and_signals(pair):
    hourly = make_cointegrated_hourly(hours=1_199)
    observations = pe.build_hourly_observations(hourly, pe.DEFAULT_PARAMS, pair=pair)
    signal = pe.make_signal(make_observation(zscore=2.25, beta=2.0, pair=pair), pe.DEFAULT_PARAMS)

    assert observations["pair"].eq(pair).all()
    assert signal.pair == pair


@pytest.mark.parametrize("pair", ["", "ETHUSDT/BTCUSDT ", "XRPUSDT/BTCUSDT"])
def test_pair_observation_rejects_noncanonical_identity(pair):
    with pytest.raises(ValueError, match="pair"):
        make_observation(zscore=2.25, beta=2.0, pair=pair)


def test_hourly_observation_builder_requires_canonical_pair_identity():
    hourly = make_cointegrated_hourly(hours=1_199)

    with pytest.raises(ValueError, match="pair"):
        pe.build_hourly_observations(hourly, pe.DEFAULT_PARAMS, pair="")


def test_exit_classification_prioritizes_structural_then_data_gap_then_divergence():
    entry_ts = pd.Timestamp("2024-01-01T00:00:00Z")
    common = {
        "zscore": 3.5,
        "entry_ts": entry_ts,
        "decision_ts": entry_ts + pd.Timedelta(hours=72),
        "consecutive_missing_bars": 2,
        "params": pe.DEFAULT_PARAMS,
    }

    assert pe.classify_exit(stable=False, **common).reason == "structural"
    assert pe.classify_exit(stable=True, **common).reason == "data_gap"
    assert pe.classify_exit(
        stable=True,
        consecutive_missing_bars=0,
        **{key: value for key, value in common.items() if key != "consecutive_missing_bars"},
    ).reason == "divergence_stop"
    assert pe.classify_exit(
        zscore=0.5,
        stable=True,
        entry_ts=entry_ts,
        decision_ts=entry_ts + pd.Timedelta(hours=72),
        consecutive_missing_bars=0,
        params=pe.DEFAULT_PARAMS,
    ).reason == "convergence"


@pytest.mark.usefixtures("replay_observations")
def test_two_leg_entry_exit_uses_next_opens_and_charges_four_fills():
    result = bc.run_pairs_backtest(
        make_two_pair_market_data(),
        window_start=TS0,
        window_end=TS_END,
        config=BASE_CONFIG,
    )
    trade = result["ETHUSDT/BTCUSDT"].trades[0]
    assert trade.entry_ts == TS1
    assert trade.exit_ts == TS3
    assert trade.fee_return == pytest.approx(-4 * 10 / 10_000 * 0.5)
    assert trade.slippage_return == pytest.approx(-4 * 2 / 10_000 * 0.5)


@pytest.mark.usefixtures("replay_observations")
def test_crossed_funding_is_leg_directional_and_not_asof_repeated():
    trade = bc.run_pairs_backtest(
        make_two_pair_market_data(with_funding=True),
        window_start=TS0,
        window_end=TS_END,
        config=BASE_CONFIG,
    )["ETHUSDT/BTCUSDT"].trades[0]
    assert trade.funding_return == pytest.approx(
        (-0.0001 * trade.alt_weight) + (0.0002 * trade.btc_weight)
    )
    assert trade.funding_events == 2


@pytest.mark.usefixtures("replay_observations")
def test_sustained_missing_execution_data_force_closes_and_marks_trade():
    trade = bc.run_pairs_backtest(
        make_two_pair_market_data(missing_two_bars=True),
        window_start=TS0,
        window_end=TS_END,
        config=BASE_CONFIG,
    )["ETHUSDT/BTCUSDT"].trades[0]
    assert trade.exit_reason == "data_gap"
    assert trade.forced_close is True


@pytest.mark.usefixtures("replay_observations")
def test_window_boundary_marks_open_pair_to_market_instead_of_dropping_it():
    result = bc.run_pairs_backtest(
        make_two_pair_market_data(no_exit_signal=True),
        window_start=TS0,
        window_end=TS_END,
        config=BASE_CONFIG,
    )
    trade = result["ETHUSDT/BTCUSDT"].trades[-1]
    assert trade.exit_reason == "window_boundary"
    assert result["ETHUSDT/BTCUSDT"].metrics["completed_trades"] == len(
        result["ETHUSDT/BTCUSDT"].trades
    )


@pytest.mark.usefixtures("replay_observations")
def test_exact_entry_tie_prefers_eth_and_keeps_only_one_pair_open():
    result = bc.run_pairs_backtest(
        make_two_pair_market_data(sol_entry=True),
        window_start=TS0,
        window_end=TS_END,
        config=BASE_CONFIG,
    )

    assert len(result["ETHUSDT/BTCUSDT"].trades) == 1
    assert result["SOLUSDT/BTCUSDT"].trades == []


@pytest.mark.usefixtures("replay_observations")
def test_risk_sizing_shock_and_currency_pnl_are_separate_and_reconcile():
    config = bc.PairsBacktestConfig(
        initial_equity=10_000.0,
        risk_pct=1.0,
        fee_bps=10.0,
        slippage_bps=2.0,
        one_leg_execution_shock_bps=25.0,
    )
    trade = bc.run_pairs_backtest(
        make_two_pair_market_data(),
        window_start=TS0,
        window_end=TS_END,
        config=config,
    )["ETHUSDT/BTCUSDT"].trades[0]

    assert trade.gross_notional == pytest.approx(10_000 * 0.01 / (0.5 * 1.5 * 0.1))
    assert trade.execution_shock_return == pytest.approx(-25 / 10_000)
    assert trade.net_return == pytest.approx(
        trade.price_return
        + trade.fee_return
        + trade.slippage_return
        + trade.funding_return
        + trade.execution_shock_return
    )
    assert trade.pnl == pytest.approx(trade.gross_notional * trade.net_return)


@pytest.mark.usefixtures("replay_observations")
def test_missing_crossed_funding_timestamp_invalidates_pair_without_dropping_trade():
    data = make_two_pair_market_data()
    eth = data["ETHUSDT/BTCUSDT"]
    data[eth.pair] = bc.PairMarketData(
        **{
            **eth.__dict__,
            "alt_funding": pd.DataFrame(columns=["ts", "funding_rate"]),
        }
    )

    result = bc.run_pairs_backtest(
        data,
        window_start=TS0,
        window_end=TS_END,
        config=BASE_CONFIG,
    )["ETHUSDT/BTCUSDT"]

    assert len(result.trades) == 1
    assert result.invalid_reasons == [
        "missing_funding:ETHUSDT/BTCUSDT:2024-01-01T08:00:00+00:00"
    ]


def test_observation_arriving_while_pair_is_open_is_not_queued_for_later(monkeypatch):
    def build_observations(hourly, params, *, pair):
        decision_ts = (
            pd.Timestamp("2024-01-01T06:00:00Z")
            if pair.startswith("ETH") else TS2 - pd.Timedelta(hours=1)
        )
        snapshot = pe.RelationshipSnapshot(
            alpha=0.0,
            beta=1.0,
            residual=0.22,
            residual_mean=0.0,
            residual_std=0.1,
            return_correlation=0.9,
            adf_pvalue=0.01,
            half_life_hours=12.0,
            observations=1_440,
            beta_change=0.0,
            stable=True,
            rejection_reason=None,
        )
        return pd.DataFrame([{
            "pair": pair,
            "ts": decision_ts,
            "snapshot": snapshot,
            "zscore": 2.2,
            "direction": "SHORT_ALT_LONG_BTC",
            "beta": 1.0,
        }])

    monkeypatch.setattr(pe, "build_hourly_observations", build_observations)
    data = make_two_pair_market_data()
    sol = data["SOLUSDT/BTCUSDT"]
    sol_zscores = [0.0, 0.0, 2.0, 1.8, 1.7]
    data[sol.pair] = bc.PairMarketData(
        **{
            **sol.__dict__,
            "alt_15m": _fifteen_minute_leg(sol_zscores, btc=False),
        }
    )

    result = bc.run_pairs_backtest(
        data,
        window_start=TS0,
        window_end=TS_END,
        config=BASE_CONFIG,
    )

    assert len(result["ETHUSDT/BTCUSDT"].trades) == 1
    assert result["SOLUSDT/BTCUSDT"].trades == []


def test_pre_window_signal_older_than_confirmation_window_is_expired(monkeypatch):
    def build_observations(hourly, params, *, pair):
        snapshot = pe.RelationshipSnapshot(
            alpha=0.0,
            beta=1.0,
            residual=0.22,
            residual_mean=0.0,
            residual_std=0.1,
            return_correlation=0.9,
            adf_pvalue=0.01,
            half_life_hours=12.0,
            observations=1_440,
            beta_change=0.0,
            stable=True,
            rejection_reason=None,
        )
        return pd.DataFrame([{
            "pair": pair,
            "ts": TS0 - pd.Timedelta(hours=2),
            "snapshot": snapshot,
            "zscore": 2.2 if pair.startswith("ETH") else 0.0,
            "direction": "SHORT_ALT_LONG_BTC" if pair.startswith("ETH") else None,
            "beta": 1.0,
        }])

    monkeypatch.setattr(pe, "build_hourly_observations", build_observations)

    result = bc.run_pairs_backtest(
        make_two_pair_market_data(),
        window_start=TS0,
        window_end=TS_END,
        config=BASE_CONFIG,
    )

    assert result["ETHUSDT/BTCUSDT"].trades == []


def test_confirmation_cannot_enter_under_a_new_unstable_snapshot(monkeypatch):
    def snapshot(*, stable: bool) -> pe.RelationshipSnapshot:
        return pe.RelationshipSnapshot(
            alpha=0.0,
            beta=1.0,
            residual=0.22,
            residual_mean=0.0,
            residual_std=0.1,
            return_correlation=0.9,
            adf_pvalue=0.01,
            half_life_hours=12.0,
            observations=1_440,
            beta_change=0.0,
            stable=stable,
            rejection_reason=None if stable else "beta_change",
        )

    def build_observations(hourly, params, *, pair):
        if pair.startswith("SOL"):
            return pd.DataFrame([{
                "pair": pair,
                "ts": TS0 - pd.Timedelta(hours=1, minutes=30),
                "snapshot": snapshot(stable=True),
                "zscore": 0.0,
                "direction": None,
                "beta": 1.0,
            }])
        return pd.DataFrame([
            {
                "pair": pair,
                "ts": TS0 - pd.Timedelta(hours=1, minutes=30),
                "snapshot": snapshot(stable=True),
                "zscore": 2.2,
                "direction": "SHORT_ALT_LONG_BTC",
                "beta": 1.0,
            },
            {
                "pair": pair,
                "ts": TS2 - pd.Timedelta(hours=1),
                "snapshot": snapshot(stable=False),
                "zscore": 1.8,
                "direction": None,
                "beta": 1.0,
            },
        ])

    monkeypatch.setattr(pe, "build_hourly_observations", build_observations)
    data = make_two_pair_market_data()
    eth = data["ETHUSDT/BTCUSDT"]
    data[eth.pair] = bc.PairMarketData(
        **{
            **eth.__dict__,
            "alt_15m": _fifteen_minute_leg([2.15, 2.15, 1.8, 0.4, 0.3], btc=False),
        }
    )

    result = bc.run_pairs_backtest(
        data,
        window_start=TS0,
        window_end=TS_END,
        config=BASE_CONFIG,
    )

    assert result["ETHUSDT/BTCUSDT"].trades == []


@pytest.mark.usefixtures("replay_observations")
def test_unfillable_pending_exit_falls_back_to_boundary_without_backdating_reason():
    data = make_two_pair_market_data()
    eth = data["ETHUSDT/BTCUSDT"]
    data[eth.pair] = bc.PairMarketData(
        **{
            **eth.__dict__,
            "alt_15m": eth.alt_15m.loc[eth.alt_15m["ts"] <= TS1].reset_index(drop=True),
        }
    )

    trade = bc.run_pairs_backtest(
        data,
        window_start=TS0,
        window_end=TS_END,
        config=BASE_CONFIG,
    )["ETHUSDT/BTCUSDT"].trades[0]

    assert trade.exit_ts == trade.entry_ts == TS1
    assert trade.exit_reason == "window_boundary"


def test_entry_confirmed_on_final_bar_does_not_fill_at_exclusive_window_end(monkeypatch):
    def build_observations(hourly, params, *, pair):
        snapshot = pe.RelationshipSnapshot(
            alpha=0.0,
            beta=1.0,
            residual=0.22,
            residual_mean=0.0,
            residual_std=0.1,
            return_correlation=0.9,
            adf_pvalue=0.01,
            half_life_hours=12.0,
            observations=1_440,
            beta_change=0.0,
            stable=True,
            rejection_reason=None,
        )
        return pd.DataFrame([{
            "pair": pair,
            "ts": TS1 - pd.Timedelta(hours=1),
            "snapshot": snapshot,
            "zscore": 2.2 if pair.startswith("ETH") else 0.0,
            "direction": "SHORT_ALT_LONG_BTC" if pair.startswith("ETH") else None,
            "beta": 1.0,
        }])

    monkeypatch.setattr(pe, "build_hourly_observations", build_observations)
    data = make_two_pair_market_data()
    eth = data["ETHUSDT/BTCUSDT"]
    data[eth.pair] = bc.PairMarketData(
        **{
            **eth.__dict__,
            "alt_15m": _fifteen_minute_leg([0.0, 2.2, 2.15, 1.8, 1.7], btc=False),
        }
    )

    result = bc.run_pairs_backtest(
        data,
        window_start=TS0,
        window_end=TS_END,
        config=BASE_CONFIG,
    )

    assert result["ETHUSDT/BTCUSDT"].trades == []


def test_hourly_candle_is_not_available_until_one_hour_after_its_open(monkeypatch):
    def build_observations(hourly, params, *, pair):
        snapshot = pe.RelationshipSnapshot(
            alpha=0.0,
            beta=1.0,
            residual=0.22,
            residual_mean=0.0,
            residual_std=0.1,
            return_correlation=0.9,
            adf_pvalue=0.01,
            half_life_hours=12.0,
            observations=1_440,
            beta_change=0.0,
            stable=True,
            rejection_reason=None,
        )
        return pd.DataFrame([{
            "pair": pair,
            "ts": TS0,
            "snapshot": snapshot,
            "zscore": 2.2 if pair.startswith("ETH") else 0.0,
            "direction": "SHORT_ALT_LONG_BTC" if pair.startswith("ETH") else None,
            "beta": 1.0,
        }])

    monkeypatch.setattr(pe, "build_hourly_observations", build_observations)

    result = bc.run_pairs_backtest(
        make_two_pair_market_data(),
        window_start=TS0,
        window_end=TS_END,
        config=BASE_CONFIG,
    )

    assert result["ETHUSDT/BTCUSDT"].trades == []


@pytest.mark.usefixtures("replay_observations")
def test_market_wide_two_bar_outage_still_triggers_data_gap_exit():
    data = make_two_pair_market_data()
    for pair, market in list(data.items()):
        data[pair] = bc.PairMarketData(
            **{
                **market.__dict__,
                "alt_15m": market.alt_15m.loc[
                    ~market.alt_15m["ts"].isin([TS2, TS3])
                ].reset_index(drop=True),
                "btc_15m": market.btc_15m.loc[
                    ~market.btc_15m["ts"].isin([TS2, TS3])
                ].reset_index(drop=True),
            }
        )

    trade = bc.run_pairs_backtest(
        data,
        window_start=TS0,
        window_end=TS_END,
        config=BASE_CONFIG,
    )["ETHUSDT/BTCUSDT"].trades[0]

    assert trade.exit_reason == "data_gap"
    assert trade.exit_ts == TS_END


@pytest.mark.usefixtures("replay_observations")
def test_each_pair_result_equity_reconciles_with_only_that_pairs_pnl():
    result = bc.run_pairs_backtest(
        make_two_pair_market_data(),
        window_start=TS0,
        window_end=TS_END,
        config=BASE_CONFIG,
    )

    for pair_result in result.values():
        assert pair_result.metrics["final_equity"] == pytest.approx(
            pair_result.metrics["initial_equity"] + pair_result.metrics["total_pnl"]
        )


def test_new_unstable_snapshot_at_fill_time_cancels_pending_entry(monkeypatch):
    stable = pe.RelationshipSnapshot(
        alpha=0.0,
        beta=1.0,
        residual=0.22,
        residual_mean=0.0,
        residual_std=0.1,
        return_correlation=0.9,
        adf_pvalue=0.01,
        half_life_hours=12.0,
        observations=1_440,
        beta_change=0.0,
        stable=True,
        rejection_reason=None,
    )
    unstable = pe.RelationshipSnapshot(
        **{
            **stable.__dict__,
            "stable": False,
            "rejection_reason": "beta_change",
        }
    )

    def build_observations(hourly, params, *, pair):
        if pair.startswith("SOL"):
            return pd.DataFrame([{
                "pair": pair,
                "ts": TS0 - pd.Timedelta(hours=1, minutes=30),
                "snapshot": stable,
                "zscore": 0.0,
                "direction": None,
            }])
        return pd.DataFrame([
            {
                "pair": pair,
                "ts": pd.Timestamp("2024-01-01T06:00:00Z"),
                "snapshot": stable,
                "zscore": 2.2,
                "direction": "SHORT_ALT_LONG_BTC",
            },
            {
                "pair": pair,
                "ts": TS1 - pd.Timedelta(hours=1),
                "snapshot": unstable,
                "zscore": 1.8,
                "direction": None,
            },
        ])

    monkeypatch.setattr(pe, "build_hourly_observations", build_observations)
    result = bc.run_pairs_backtest(
        make_two_pair_market_data(),
        window_start=TS0,
        window_end=TS_END,
        config=BASE_CONFIG,
    )

    assert result["ETHUSDT/BTCUSDT"].trades == []


def test_structural_exit_keeps_valid_price_mark_when_z_is_unavailable(monkeypatch):
    snapshot = pe.RelationshipSnapshot(
        alpha=0.0,
        beta=1.0,
        residual=0.22,
        residual_mean=0.0,
        residual_std=0.1,
        return_correlation=0.9,
        adf_pvalue=0.01,
        half_life_hours=12.0,
        observations=1_440,
        beta_change=0.0,
        stable=True,
        rejection_reason=None,
    )

    def build_observations(hourly, params, *, pair):
        initial_z = 2.2 if pair.startswith("ETH") else 0.0
        direction = "SHORT_ALT_LONG_BTC" if pair.startswith("ETH") else None
        return pd.DataFrame([
            {
                "pair": pair,
                "ts": pd.Timestamp("2024-01-01T06:00:00Z"),
                "snapshot": snapshot,
                "zscore": initial_z,
                "direction": direction,
            },
            {
                "pair": pair,
                "ts": TS3 - pd.Timedelta(hours=1),
                "snapshot": None,
                "zscore": None,
                "direction": None,
            },
        ])

    monkeypatch.setattr(pe, "build_hourly_observations", build_observations)
    data = make_two_pair_market_data()
    eth = data["ETHUSDT/BTCUSDT"]
    eth_15m = eth.alt_15m.copy()
    eth_15m.loc[eth_15m["ts"] == TS2, "close"] = 50.0
    data[eth.pair] = bc.PairMarketData(**{**eth.__dict__, "alt_15m": eth_15m})

    trade = bc.run_pairs_backtest(
        data,
        window_start=TS0,
        window_end=TS_END,
        config=BASE_CONFIG,
    )["ETHUSDT/BTCUSDT"].trades[0]

    assert trade.exit_reason == "structural"
    assert trade.mfe_return == pytest.approx(0.25)
