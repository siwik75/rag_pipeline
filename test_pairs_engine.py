"""Tests for the causal hourly relative-value relationship model."""
import importlib.util
import json
import multiprocessing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

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
COMMON_START = pd.Timestamp("2023-01-01T00:00:00Z")
COMMON_END = pd.Timestamp("2024-01-01T00:00:00Z")


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
    def build_observations(hourly, params, *, pair, universe=None):
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


def make_summary_trade(
    *,
    pair: str = "ETHUSDT/BTCUSDT",
    net_return: float = 0.01,
    alt_side: str = "LONG",
    realized_btc_beta: float = 0.05,
    entry_offset: int = 0,
    entry_ts: pd.Timestamp | None = None,
    holding_hours: float = 12.0,
    gross_notional: float = 1_000.0,
    equity_before: float = 10_000.0,
) -> bc.PairTrade:
    entry_ts = entry_ts or COMMON_START + pd.Timedelta(hours=entry_offset)
    return bc.PairTrade(
        pair=pair,
        entry_ts=entry_ts,
        exit_ts=entry_ts + pd.Timedelta(hours=holding_hours),
        alt_side=alt_side,
        btc_side="SHORT" if alt_side == "LONG" else "LONG",
        alt_weight=0.5,
        btc_weight=0.5,
        entry_z=-2.1 if alt_side == "LONG" else 2.1,
        exit_z=0.4,
        exit_reason="convergence",
        price_return=net_return + 0.0024,
        fee_return=-0.002,
        slippage_return=-0.0004,
        funding_return=0.0,
        execution_shock_return=0.0,
        net_return=net_return,
        realized_btc_beta=realized_btc_beta,
        mfe_z=1.7,
        mae_z=-0.2,
        mfe_return=max(net_return, 0.0) + 0.003,
        mae_return=min(net_return, 0.0) - 0.002,
        funding_events=2,
        forced_close=False,
        gross_notional=gross_notional,
        equity_before=equity_before,
        pnl=gross_notional * net_return,
    )


def make_metrics(**overrides) -> dict[str, object]:
    metrics = {
        "completed_trades": 120,
        "per_pair_trades": {
            "ETHUSDT/BTCUSDT": 60,
            "SOLUSDT/BTCUSDT": 60,
        },
        "profit_factor": 1.5,
        "win_rate": 0.56,
        "mean_net_return": 0.004,
        "median_net_return": 0.003,
        "max_drawdown": 0.10,
        "absolute_realized_btc_beta": 0.10,
        "per_pair": {
            "ETHUSDT/BTCUSDT": {
                "completed_trades": 60,
                "profit_factor": 1.3,
                "mean_net_return": 0.003,
                "gross_profit_contribution": 0.52,
            },
            "SOLUSDT/BTCUSDT": {
                "completed_trades": 60,
                "profit_factor": 1.2,
                "mean_net_return": 0.002,
                "gross_profit_contribution": 0.48,
            },
        },
        "deflated_sharpe_probability": 0.97,
        "bootstrap_mean_net_return_ci_95": [0.001, 0.006],
        "cost_stress": {
            "15bps_fee_5bps_slippage": {"mean_net_return": 0.0001},
        },
        "invalid_reasons": [],
    }
    metrics.update(overrides)
    return metrics


def make_hard_boundary_metrics() -> dict[str, object]:
    metrics = make_metrics(
        completed_trades=100,
        per_pair_trades={"ETHUSDT/BTCUSDT": 50, "SOLUSDT/BTCUSDT": 50},
        profit_factor=1.25,
        win_rate=0.52,
        max_drawdown=0.15,
        absolute_realized_btc_beta=0.15,
        deflated_sharpe_probability=0.95,
        bootstrap_mean_net_return_ci_95=[0.000001, 0.006],
    )
    metrics["per_pair"] = {
        "ETHUSDT/BTCUSDT": {
            "completed_trades": 50,
            "profit_factor": 1.05,
            "mean_net_return": 0.001,
            "gross_profit_contribution": 0.65,
        },
        "SOLUSDT/BTCUSDT": {
            "completed_trades": 50,
            "profit_factor": 1.05,
            "mean_net_return": 0.001,
            "gross_profit_contribution": 0.35,
        },
    }
    metrics["cost_stress"] = {
        "15bps_fee_5bps_slippage": {"mean_net_return": 0.0},
    }
    return metrics


def make_trial(
    phase: str, *, passed: bool, trial_id: str | None = None,
    invalid_reasons: list[str] | None = None,
) -> dict[str, object]:
    return {
        "trial_id": trial_id or f"{phase}-trial",
        "recorded_at": "2026-08-14T00:00:00+00:00",
        "phase": phase,
        "params": {},
        "cost_config": {"fee_bps": 10.0, "slippage_bps": 2.0},
        "dataset_hashes": {"all": "abc"},
        "common_start": COMMON_START.isoformat(),
        "common_end": COMMON_END.isoformat(),
        "metrics": make_metrics(),
        "cost_stress": {},
        "gate": {"passed": passed, "failed_conditions": [] if passed else ["profit_factor"]},
        "invalid_reasons": invalid_reasons or [],
    }


def make_experiment_market_data() -> dict[str, bc.PairMarketData]:
    def prices(symbol: str, timeframe: str) -> pd.DataFrame:
        frequency = "1h" if timeframe == "1h" else "15min"
        timestamps = pd.date_range(
            COMMON_START, COMMON_END, freq=frequency, inclusive="left",
        )
        close = 100.0 + np.arange(len(timestamps), dtype=float) / len(timestamps)
        return pd.DataFrame({
            "ts": timestamps,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(len(timestamps), 1_000.0),
            "symbol": np.full(len(timestamps), symbol),
        })

    funding = pd.DataFrame({
        "ts": pd.date_range(COMMON_START, COMMON_END, freq="8h", inclusive="left"),
        "funding_rate": 0.0,
    })
    btc_1h = prices("BTCUSDT", "1h")
    btc_15m = prices("BTCUSDT", "15m")
    btc_funding = funding.copy()
    result = {}
    for pair in pe.FIXED_PAIRS:
        alt_symbol, btc_symbol = pair.split("/")
        result[pair] = bc.PairMarketData(
            pair=pair,
            alt_symbol=alt_symbol,
            btc_symbol=btc_symbol,
            alt_1h=prices(alt_symbol, "1h"),
            btc_1h=btc_1h.copy(),
            alt_15m=prices(alt_symbol, "15m"),
            btc_15m=btc_15m.copy(),
            alt_funding=funding,
            btc_funding=btc_funding.copy(),
        )
    return result


def make_sparse_experiment_market_data() -> dict[str, bc.PairMarketData]:
    data = make_experiment_market_data()
    for pair, market in list(data.items()):
        data[pair] = bc.PairMarketData(**{
            **market.__dict__,
            "alt_1h": market.alt_1h.iloc[[0, -1]].reset_index(drop=True),
            "btc_1h": market.btc_1h.iloc[[0, -1]].reset_index(drop=True),
            "alt_15m": market.alt_15m.iloc[[0, -1]].reset_index(drop=True),
            "btc_15m": market.btc_15m.iloc[[0, -1]].reset_index(drop=True),
        })
    return data


def passing_experiment_replay(data, *, window_start, window_end, config, params,
                              universe=None):
    results = {}
    for pair_index, pair in enumerate(pe.FIXED_PAIRS):
        trades = [
            make_summary_trade(
                pair=pair,
                net_return=-0.003 if trade_index == 0 else 0.008,
                alt_side="LONG" if trade_index % 2 == 0 else "SHORT",
                entry_ts=window_start + pd.Timedelta(
                    hours=pair_index * 24 + trade_index * 3,
                ),
            )
            for trade_index in range(5)
        ]
        results[pair] = bc.PairsBacktestResult(
            pair=pair,
            trades=trades,
            metrics={},
            invalid_reasons=[],
        )
    return results


_CONCURRENT_HOLDOUT_REPLAYS = None


def concurrent_experiment_replay(data, *, window_start, window_end, config, params,
                                 universe=None):
    if window_end - window_start == pd.Timedelta(days=90):
        with _CONCURRENT_HOLDOUT_REPLAYS.get_lock():
            _CONCURRENT_HOLDOUT_REPLAYS.value += 1
    return passing_experiment_replay(
        data,
        window_start=window_start,
        window_end=window_end,
        config=config,
        params=params,
    )


def run_experiment_process(
    data, ledger_path, start_event, result_queue, holdout_replay_counter,
):
    global _CONCURRENT_HOLDOUT_REPLAYS
    _CONCURRENT_HOLDOUT_REPLAYS = holdout_replay_counter
    bc.run_pairs_backtest = concurrent_experiment_replay
    start_event.wait()
    try:
        report = bc.run_pairs_experiment(
            data,
            config=BASE_CONFIG,
            params=pe.DEFAULT_PARAMS,
            open_holdout=True,
            ledger_path=Path(ledger_path),
        )
    except Exception as exc:  # noqa: BLE001 - child reports exact process outcome
        result_queue.put(("error", type(exc).__name__, str(exc)))
    else:
        result_queue.put(("opened", report["holdout"]["opened"]))


def append_trial_process(ledger_path, trial_id, start_event, result_queue):
    start_event.wait()
    try:
        bc.write_trial_ledger(
            Path(ledger_path),
            make_trial("development", passed=False, trial_id=trial_id),
        )
    except Exception as exc:  # noqa: BLE001 - child reports exact process outcome
        result_queue.put(("error", type(exc).__name__, str(exc)))
    else:
        result_queue.put(("appended", trial_id))


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


@pytest.mark.parametrize(
    "pair",
    ["", "ETHUSDT/BTCUSDT ", "BTCUSDT/BTCUSDT", "ETHUSDT", "ETH/BTCUSDT",
     "ETHUSDT/ETHUSDT", "USDT/BTCUSDT"],
)
def test_pair_observation_rejects_malformed_identity(pair):
    with pytest.raises(ValueError, match="pair"):
        make_observation(zscore=2.25, beta=2.0, pair=pair)


def test_pair_observation_accepts_wellformed_pair_outside_fixed_universe():
    observation = make_observation(zscore=2.25, beta=2.0, pair="XRPUSDT/BTCUSDT")

    assert observation.pair == "XRPUSDT/BTCUSDT"


def test_hourly_observation_builder_requires_wellformed_pair_identity():
    hourly = make_cointegrated_hourly(hours=1_199)

    with pytest.raises(ValueError, match="pair"):
        pe.build_hourly_observations(hourly, pe.DEFAULT_PARAMS, pair="")


def test_hourly_observation_builder_validates_against_caller_provided_universe():
    hourly = make_cointegrated_hourly(hours=1_199)

    # Well-formed but not in the default fixed universe: rejected.
    with pytest.raises(ValueError, match="pair must be one of"):
        pe.build_hourly_observations(
            hourly, pe.DEFAULT_PARAMS, pair="XRPUSDT/BTCUSDT",
        )
    # The same pair is accepted when the caller's universe includes it.
    observations = pe.build_hourly_observations(
        hourly, pe.DEFAULT_PARAMS, pair="XRPUSDT/BTCUSDT",
        universe=("XRPUSDT/BTCUSDT",),
    )
    assert observations["pair"].eq("XRPUSDT/BTCUSDT").all()


def test_expanded_pairs_are_nineteen_wellformed_top_cap_alts():
    assert len(pe.EXPANDED_PAIRS) == 19
    assert pe.validate_pair_universe(pe.EXPANDED_PAIRS) == pe.EXPANDED_PAIRS
    assert set(pe.FIXED_PAIRS) <= set(pe.EXPANDED_PAIRS)
    symbols = pe.universe_symbols(pe.EXPANDED_PAIRS)
    assert symbols[-1] == "BTCUSDT"
    assert len(symbols) == 20
    assert symbols[0] == "ETHUSDT"


def test_validate_pair_universe_rejects_empty_duplicates_and_malformed():
    with pytest.raises(ValueError, match="must not be empty"):
        pe.validate_pair_universe(())
    with pytest.raises(ValueError, match="duplicates"):
        pe.validate_pair_universe(("ETHUSDT/BTCUSDT", "ETHUSDT/BTCUSDT"))
    with pytest.raises(ValueError, match="ALTUSDT/BTCUSDT"):
        pe.validate_pair_universe(("BTCUSDT/BTCUSDT",))
    with pytest.raises(ValueError, match="ALTUSDT/BTCUSDT"):
        pe.validate_pair_universe(("ETHUSDT",))


def test_runner_universe_literals_match_the_engine_universes():
    runner_path = (
        Path(__file__).resolve().parent.parent / "ai_runners" / "backtest.py"
    )
    spec = importlib.util.spec_from_file_location(
        "backtest_runner_universe_check", runner_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.PAIR_UNIVERSE == pe.FIXED_PAIRS
    assert module.EXPANDED_PAIR_UNIVERSE == pe.EXPANDED_PAIRS


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


def test_funding_boundary_is_strict_after_entry_and_inclusive_at_exit():
    entry = pd.Timestamp("2024-01-01T08:00:00Z")
    exit_ts = pd.Timestamp("2024-01-01T16:00:00Z")
    funding = pd.DataFrame({
        "ts": [entry, exit_ts],
        "funding_rate": [0.50, 0.001],
    })
    position = {
        "market": SimpleNamespace(
            pair="ETHUSDT/BTCUSDT",
            alt_funding=funding,
            btc_funding=funding,
        ),
        "signal": SimpleNamespace(
            alt_side="LONG", btc_side="SHORT", alt_weight=0.5, btc_weight=0.5,
        ),
        "entry_ts": entry,
    }

    funding_return, funding_events = bc._trade_funding(position, exit_ts, [])

    assert funding_return == pytest.approx(0.0)
    assert funding_events == 2


@pytest.mark.parametrize("offset_ms", [1, 16])
def test_pairs_funding_normalizes_subsecond_exchange_jitter(offset_ms):
    nominal = pd.Timestamp("2026-01-21T08:00:00Z")
    raw = pd.DataFrame({
        "ts": [nominal + pd.Timedelta(milliseconds=offset_ms)],
        "funding_rate": [0.0001],
    })

    normalized = bc.normalize_pairs_funding(raw, symbol="ETHUSDT")

    assert normalized.to_dict("records") == [
        {"ts": nominal, "funding_rate": 0.0001}
    ]


def test_pairs_funding_rejects_off_schedule_rows_outside_one_second():
    raw = pd.DataFrame({
        "ts": [pd.Timestamp("2026-01-21T08:00:01.001Z")],
        "funding_rate": [0.0001],
    })

    with pytest.raises(ValueError, match="off-schedule funding timestamp.*ETHUSDT"):
        bc.normalize_pairs_funding(raw, symbol="ETHUSDT")


@pytest.mark.parametrize("rates", [[0.0001, 0.0001], [0.0001, 0.0002]])
def test_pairs_funding_rejects_duplicate_or_conflicting_nominal_rows(rates):
    nominal = pd.Timestamp("2026-01-21T08:00:00Z")
    raw = pd.DataFrame({
        "ts": [nominal + pd.Timedelta(milliseconds=1),
               nominal + pd.Timedelta(milliseconds=16)],
        "funding_rate": rates,
    })

    with pytest.raises(ValueError, match="duplicate funding boundary.*ETHUSDT"):
        bc.normalize_pairs_funding(raw, symbol="ETHUSDT")


@pytest.mark.parametrize("offset_ms", [0, 16])
def test_pairs_funding_is_normalized_before_inclusive_window_slice(offset_ms):
    boundary = pd.Timestamp("2026-01-21T08:00:00Z")
    raw = pd.DataFrame({
        "ts": [boundary + pd.Timedelta(milliseconds=offset_ms)],
        "funding_rate": [0.0001],
    })

    prepared = bc.prepare_pairs_funding(
        raw, symbol="ETHUSDT", window_end=boundary,
    )

    assert prepared["ts"].tolist() == [boundary]


def test_pairs_funding_after_window_or_outside_tolerance_does_not_survive():
    boundary = pd.Timestamp("2026-01-21T08:00:00Z")
    next_boundary = boundary + pd.Timedelta(hours=8)
    after_window = pd.DataFrame({
        "ts": [next_boundary + pd.Timedelta(milliseconds=16)],
        "funding_rate": [0.0001],
    })

    assert bc.prepare_pairs_funding(
        after_window, symbol="ETHUSDT", window_end=boundary,
    ).empty
    with pytest.raises(ValueError, match="off-schedule funding timestamp"):
        bc.prepare_pairs_funding(
            pd.DataFrame({
                "ts": [boundary + pd.Timedelta(seconds=1, milliseconds=1)],
                "funding_rate": [0.0001],
            }),
            symbol="ETHUSDT",
            window_end=boundary,
        )


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
    def build_observations(hourly, params, *, pair, universe=None):
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
        window_end=TS_END + pd.Timedelta(minutes=15),
        config=BASE_CONFIG,
    )

    assert len(result["ETHUSDT/BTCUSDT"].trades) == 1
    assert result["SOLUSDT/BTCUSDT"].trades == []


def test_pre_window_signal_older_than_confirmation_window_is_expired(monkeypatch):
    def build_observations(hourly, params, *, pair, universe=None):
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

    def build_observations(hourly, params, *, pair, universe=None):
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
    def build_observations(hourly, params, *, pair, universe=None):
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
    def build_observations(hourly, params, *, pair, universe=None):
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

    def build_observations(hourly, params, *, pair, universe=None):
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

    def build_observations(hourly, params, *, pair, universe=None):
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


def test_absolute_realized_beta_does_not_cancel_opposite_trade_slopes():
    common = {
        "pair": "ETHUSDT/BTCUSDT",
        "entry_ts": TS0,
        "exit_ts": TS_END,
        "alt_side": "SHORT",
        "btc_side": "LONG",
        "alt_weight": 0.25,
        "btc_weight": 0.75,
        "entry_z": 2.0,
        "exit_z": 0.4,
        "exit_reason": "convergence",
        "price_return": 0.01,
        "fee_return": -0.002,
        "slippage_return": -0.0004,
        "funding_return": 0.0,
        "execution_shock_return": 0.0,
        "net_return": 0.0076,
        "mfe_z": 1.6,
        "mae_z": 0.0,
        "mfe_return": 0.01,
        "mae_return": 0.0,
        "funding_events": 0,
        "forced_close": False,
        "gross_notional": 1_000.0,
        "equity_before": 10_000.0,
        "pnl": 7.6,
    }
    trades = [
        bc.PairTrade(**common, realized_btc_beta=0.2),
        bc.PairTrade(**common, realized_btc_beta=-0.2),
    ]

    metrics = bc._pairs_metrics(
        trades,
        initial_equity=10_000.0,
        final_equity=10_015.2,
        rejected_entries=0,
    )

    assert metrics["absolute_realized_btc_beta"] == pytest.approx(0.2)


def test_realized_beta_ignores_nonconsecutive_mark_intervals():
    position = {
        "pair_marks": [0.0, 0.01, 0.04, 0.02],
        "btc_logs": [0.0, 0.01, 0.02, 0.04],
        "mark_ts": [TS0, TS1, TS3, TS_END],
    }

    assert bc._realized_btc_beta(position) is None


def test_replay_rejects_engine_row_with_mismatched_pair_identity(monkeypatch):
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

    def build_observations(hourly, params, *, pair, universe=None):
        row_pair = "SOLUSDT/BTCUSDT" if pair == "ETHUSDT/BTCUSDT" else pair
        return pd.DataFrame([{
            "pair": row_pair,
            "ts": pd.Timestamp("2024-01-01T06:00:00Z"),
            "snapshot": snapshot,
            "zscore": 2.2,
            "direction": "SHORT_ALT_LONG_BTC",
        }])

    monkeypatch.setattr(pe, "build_hourly_observations", build_observations)

    with pytest.raises(ValueError, match="pair"):
        bc.run_pairs_backtest(
            make_two_pair_market_data(),
            window_start=TS0,
            window_end=TS_END,
            config=BASE_CONFIG,
        )


def test_asymmetric_two_trade_accounting_has_real_pnl_beta_and_compounding(monkeypatch):
    start = pd.Timestamp("2024-01-02T07:00:00Z")
    end = pd.Timestamp("2024-01-02T10:00:00Z")
    timestamps = pd.date_range(start, end, freq="15min")
    alpha = -2.0 * np.log(100.0)
    snapshot = pe.RelationshipSnapshot(
        alpha=alpha,
        beta=3.0,
        residual=0.2,
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

    def build_observations(hourly, params, *, pair, universe=None):
        if pair.startswith("SOL"):
            return pd.DataFrame([{
                "pair": pair,
                "ts": start - pd.Timedelta(hours=1),
                "snapshot": snapshot,
                "zscore": 0.0,
                "direction": None,
            }])
        return pd.DataFrame([
            {
                "pair": pair,
                "ts": start - pd.Timedelta(hours=1),
                "snapshot": snapshot,
                "zscore": 2.2,
                "direction": "SHORT_ALT_LONG_BTC",
            },
            {
                "pair": pair,
                "ts": start + pd.Timedelta(minutes=30),
                "snapshot": snapshot,
                "zscore": 2.2,
                "direction": "SHORT_ALT_LONG_BTC",
            },
        ])

    monkeypatch.setattr(pe, "build_hourly_observations", build_observations)
    zscores = [2.0, 1.8, 1.5, 1.0, 0.4, 1.5, 2.0, 1.8, 1.5, 1.0, 0.4, 0.3, 0.2]
    btc_closes = [100.0, 101.0, 99.0, 102.0, 100.0, 100.0, 100.0,
                  101.0, 99.0, 102.0, 100.0, 100.0, 100.0]
    alt_closes = [
        np.exp(alpha + 3.0 * np.log(btc_close) + 0.1 * zscore)
        for btc_close, zscore in zip(btc_closes, zscores)
    ]

    def leg_frame(*, alt: bool) -> pd.DataFrame:
        closes = np.asarray(alt_closes if alt else btc_closes)
        opens = np.full(len(timestamps), 100.0)
        if alt:
            opens[timestamps.get_loc(start + pd.Timedelta(minutes=75))] = 90.0
            opens[timestamps.get_loc(start + pd.Timedelta(minutes=105))] = 120.0
            opens[timestamps.get_loc(start + pd.Timedelta(minutes=165))] = 132.0
        else:
            opens[timestamps.get_loc(start + pd.Timedelta(minutes=75))] = 110.0
            opens[timestamps.get_loc(start + pd.Timedelta(minutes=165))] = 90.0
        return pd.DataFrame({
            "ts": timestamps,
            "open": opens,
            "high": np.maximum(opens, closes),
            "low": np.minimum(opens, closes),
            "close": closes,
            "volume": 1_000.0,
        })

    empty_hourly = _hourly_leg(model_z=0.0, btc=True)
    zero_funding = pd.DataFrame({
        "ts": [pd.Timestamp("2024-01-02T08:00:00Z")],
        "funding_rate": [0.0],
    })
    eth = bc.PairMarketData(
        pair="ETHUSDT/BTCUSDT",
        alt_symbol="ETHUSDT",
        btc_symbol="BTCUSDT",
        alt_1h=empty_hourly,
        btc_1h=empty_hourly,
        alt_15m=leg_frame(alt=True),
        btc_15m=leg_frame(alt=False),
        alt_funding=zero_funding,
        btc_funding=zero_funding,
    )
    sol = bc.PairMarketData(
        pair="SOLUSDT/BTCUSDT",
        alt_symbol="SOLUSDT",
        btc_symbol="BTCUSDT",
        alt_1h=empty_hourly,
        btc_1h=empty_hourly,
        alt_15m=leg_frame(alt=True),
        btc_15m=leg_frame(alt=False),
        alt_funding=zero_funding,
        btc_funding=zero_funding,
    )
    config = bc.PairsBacktestConfig(slippage_bps=0.0)

    trades = bc.run_pairs_backtest(
        {eth.pair: eth, sol.pair: sol},
        window_start=start,
        window_end=end,
        config=config,
    )[eth.pair].trades

    assert len(trades) == 2
    first, second = trades
    assert (first.alt_weight, first.btc_weight) == pytest.approx((0.25, 0.75))
    assert first.price_return == pytest.approx(0.10)
    assert second.price_return == pytest.approx(-0.10)
    assert first.fee_return == pytest.approx(-0.002)
    assert second.fee_return == pytest.approx(-0.002)
    assert first.realized_btc_beta == pytest.approx(-0.42714659602131466)
    assert second.equity_before == pytest.approx(first.equity_before + first.pnl)
    assert second.gross_notional == pytest.approx(
        second.equity_before * 0.01 / (0.25 * 1.5 * 0.1)
    )


def test_four_fill_fee_uses_actual_leg_weights_instead_of_half_constants():
    assert bc._four_fill_fee_return(
        alt_weight=0.25,
        btc_weight=0.50,
        fee_rate=0.001,
    ) == pytest.approx(-0.0015)


def test_walk_forward_reserves_the_final_ninety_common_days():
    schedule = bc.build_walk_forward_schedule(COMMON_START, COMMON_END)

    assert schedule.holdout_start == COMMON_END - pd.Timedelta(days=90)
    assert all(window.end <= schedule.holdout_start for window in schedule.development_windows)
    assert all(
        window.end - window.start == pd.Timedelta(days=30)
        for window in schedule.development_windows
    )
    assert all(
        window.start - window.formation_start == pd.Timedelta(days=60)
        for window in schedule.development_windows
    )


def test_walk_forward_rejects_less_than_three_hundred_sixty_common_days():
    with pytest.raises(
        ValueError,
        match=r"insufficient common history: 359 days; need at least 360",
    ):
        bc.build_walk_forward_schedule(COMMON_START, COMMON_START + pd.Timedelta(days=359))


def test_development_gate_blocks_holdout_when_any_fixed_threshold_fails():
    decision = bc.development_gate(make_metrics(
        completed_trades=59,
        per_pair_trades={"ETHUSDT/BTCUSDT": 20, "SOLUSDT/BTCUSDT": 20},
    ))

    assert decision.passed is False
    assert "completed_trades" in decision.failed_conditions


def test_development_gate_accepts_every_threshold_at_its_inclusive_boundary():
    metrics = make_metrics(
        completed_trades=60,
        per_pair_trades={"ETHUSDT/BTCUSDT": 20, "SOLUSDT/BTCUSDT": 20},
        profit_factor=1.15,
        win_rate=0.50,
        max_drawdown=0.15,
        absolute_realized_btc_beta=0.15,
    )

    assert bc.development_gate(metrics) == bc.GateDecision(True, [])


@pytest.mark.parametrize(
    ("mutation", "failed_condition"),
    [
        ({"profit_factor": 1.1499}, "profit_factor"),
        ({"win_rate": 0.4999}, "win_rate"),
        ({"max_drawdown": 0.1501}, "max_drawdown"),
        ({"absolute_realized_btc_beta": 0.1501}, "absolute_realized_btc_beta"),
        ({"invalid_reasons": ["missing_funding"]}, "invalid_reasons"),
    ],
)
def test_development_gate_uses_inclusive_fixed_thresholds(mutation, failed_condition):
    decision = bc.development_gate(make_metrics(**mutation))

    assert decision.passed is False
    assert failed_condition in decision.failed_conditions


def test_hard_gate_checks_pair_concentration_stress_and_uncertainty():
    metrics = make_metrics()
    metrics["per_pair"] = {
        **metrics["per_pair"],
        "ETHUSDT/BTCUSDT": {
            **metrics["per_pair"]["ETHUSDT/BTCUSDT"],
            "gross_profit_contribution": 0.651,
        },
    }
    metrics["cost_stress"] = {
        "15bps_fee_5bps_slippage": {"mean_net_return": -0.00001},
    }
    metrics["bootstrap_mean_net_return_ci_95"] = [-0.001, 0.006]

    decision = bc.hard_pass_gate(metrics)

    assert decision.passed is False
    assert {
        "gross_profit_concentration",
        "15bps_fee_5bps_slippage",
        "bootstrap_mean_net_return_ci_95",
    }.issubset(decision.failed_conditions)


def test_hard_gate_accepts_inclusive_boundaries_and_rejects_invalid_reasons():
    metrics = make_hard_boundary_metrics()

    assert bc.hard_pass_gate(metrics) == bc.GateDecision(True, [])
    invalid = bc.hard_pass_gate({**metrics, "invalid_reasons": ["missing_funding"]})
    assert invalid.passed is False
    assert "invalid_reasons" in invalid.failed_conditions


@pytest.mark.parametrize(
    ("mutation", "failed_condition"),
    [
        (lambda metrics: metrics.update(completed_trades=99), "completed_trades"),
        (lambda metrics: metrics["per_pair_trades"].update({"ETHUSDT/BTCUSDT": 34}), "per_pair_trades"),
        (lambda metrics: metrics.update(profit_factor=1.2499), "profit_factor"),
        (lambda metrics: metrics.update(win_rate=0.5199), "win_rate"),
        (lambda metrics: metrics.update(mean_net_return=0.0), "mean_net_return"),
        (lambda metrics: metrics.update(median_net_return=0.0), "median_net_return"),
        (lambda metrics: metrics["per_pair"]["ETHUSDT/BTCUSDT"].update(profit_factor=1.0499), "per_pair_quality"),
        (lambda metrics: metrics["per_pair"]["ETHUSDT/BTCUSDT"].update(mean_net_return=0.0), "per_pair_quality"),
        (lambda metrics: metrics.update(max_drawdown=0.1501), "max_drawdown"),
        (lambda metrics: metrics.update(absolute_realized_btc_beta=0.1501), "absolute_realized_btc_beta"),
        (lambda metrics: metrics["per_pair"]["ETHUSDT/BTCUSDT"].update(gross_profit_contribution=0.6501), "gross_profit_concentration"),
        (lambda metrics: metrics["cost_stress"]["15bps_fee_5bps_slippage"].update(mean_net_return=-0.0001), "15bps_fee_5bps_slippage"),
        (lambda metrics: metrics.update(deflated_sharpe_probability=0.9499), "deflated_sharpe_probability"),
        (lambda metrics: metrics.update(bootstrap_mean_net_return_ci_95=[-0.001, 0.001]), "bootstrap_mean_net_return_ci_95"),
    ],
)
def test_every_hard_gate_threshold_rejects_just_outside_boundary(mutation, failed_condition):
    metrics = make_hard_boundary_metrics()
    mutation(metrics)

    decision = bc.hard_pass_gate(metrics)

    assert decision.passed is False
    assert failed_condition in decision.failed_conditions


def test_ledger_retains_failed_trials_and_refuses_second_holdout_open(tmp_path):
    ledger = tmp_path / "ledger.json"
    bc.write_trial_ledger(ledger, make_trial("development", passed=False))
    pending = make_trial(
        "holdout",
        passed=False,
        invalid_reasons=["holdout_replay_pending"],
    )
    bc._reserve_holdout(ledger, pending)

    with pytest.raises(ValueError, match="already opened"):
        bc._reserve_holdout(
            ledger,
            make_trial("holdout", passed=False, trial_id="again"),
        )

    persisted = json.loads(ledger.read_text())
    assert persisted["schema_version"] == 1
    assert persisted["holdout"]["status"] == "opened"
    assert [trial["gate"]["passed"] for trial in persisted["trials"]] == [False, False]
    assert persisted["trials"][1]["invalid_reasons"] == ["holdout_replay_pending"]
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda trial: trial.update(extra=True), "trial keys"),
        (lambda trial: trial.update(recorded_at="2026-08-14T01:00:00+01:00"), "recorded_at"),
        (lambda trial: trial.update(phase="research"), "phase"),
        (lambda trial: trial.update(dataset_hashes={"all": ""}), "dataset_hashes"),
        (lambda trial: trial.update(trial_id=""), "trial_id"),
    ],
)
def test_trial_ledger_rejects_extra_or_malformed_trial_fields(tmp_path, mutation, message):
    ledger = tmp_path / "ledger.json"
    trial = make_trial("development", passed=False)
    mutation(trial)

    with pytest.raises(ValueError, match=message):
        bc.write_trial_ledger(ledger, trial)


@pytest.mark.parametrize("location", ["top", "holdout"])
def test_trial_ledger_rejects_extra_schema_keys(tmp_path, location):
    ledger = tmp_path / "ledger.json"
    document = {
        "schema_version": 1,
        "holdout": {
            "status": "sealed",
            "opened_at": None,
            "dataset_hash": None,
            "trial_id": None,
        },
        "trials": [],
    }
    if location == "top":
        document["extra"] = True
    else:
        document["holdout"]["extra"] = True
    ledger.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="keys"):
        bc.write_trial_ledger(ledger, make_trial("development", passed=False))


def test_trial_ledger_rejects_duplicate_trial_id(tmp_path):
    ledger = tmp_path / "ledger.json"
    trial = make_trial("development", passed=False)
    bc.write_trial_ledger(ledger, trial)

    with pytest.raises(ValueError, match="duplicate trial_id"):
        bc.write_trial_ledger(ledger, trial)


def test_concurrent_process_appends_do_not_lose_trials(tmp_path):
    context = multiprocessing.get_context("fork")
    ledger = tmp_path / "ledger.json"
    bc._reserve_holdout(
        ledger,
        make_trial(
            "holdout",
            passed=False,
            trial_id="pending-holdout",
            invalid_reasons=["holdout_replay_pending"],
        ),
    )
    start_event = context.Event()
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=append_trial_process,
            args=(str(ledger), f"trial-{index}", start_event, result_queue),
        )
        for index in range(8)
    ]
    for process in processes:
        process.start()
    start_event.set()
    results = [result_queue.get(timeout=20) for _ in processes]
    for process in processes:
        process.join(timeout=20)

    assert all(process.exitcode == 0 for process in processes)
    assert all(result[0] == "appended" for result in results)
    persisted = json.loads(ledger.read_text())
    assert persisted["holdout"]["status"] == "opened"
    assert persisted["holdout"]["trial_id"] == "pending-holdout"
    assert len(persisted["trials"]) == 9
    assert {trial["trial_id"] for trial in persisted["trials"]} == {
        "pending-holdout",
        *(f"trial-{index}" for index in range(8)),
    }


def test_trade_summary_bootstrap_is_seeded_and_trial_penalty_is_monotone():
    trades = [
        make_summary_trade(
            pair=pe.FIXED_PAIRS[index % 2],
            net_return=(0.012, -0.004, 0.008, 0.003)[index % 4],
            alt_side="LONG" if index % 2 == 0 else "SHORT",
            entry_offset=index * 24,
        )
        for index in range(40)
    ]

    first = bc.summarize_pair_trades(trades, trial_count=1)
    repeated = bc.summarize_pair_trades(trades, trial_count=1)
    penalized = bc.summarize_pair_trades(trades, trial_count=50)

    assert first["bootstrap_mean_net_return_ci_95"] == repeated[
        "bootstrap_mean_net_return_ci_95"
    ]
    assert penalized["deflated_sharpe_probability"] <= first[
        "deflated_sharpe_probability"
    ]
    assert first["per_pair_trades"] == {
        "ETHUSDT/BTCUSDT": 20,
        "SOLUSDT/BTCUSDT": 20,
    }
    assert set(first["exit_counts"]) == {
        "convergence",
        "divergence_stop",
        "time_stop",
        "structural",
        "data_gap",
        "window_boundary",
    }
    assert set(first["per_pair_diagnostics"]) == set(pe.FIXED_PAIRS)
    assert first["per_pair_diagnostics"]["ETHUSDT/BTCUSDT"]["completed_trades"] == 20


def test_deflated_sharpe_and_drawdown_match_known_values():
    returns = [0.01, -0.005, 0.02, 0.003, -0.002]
    trades = [
        make_summary_trade(
            pair=pe.FIXED_PAIRS[index % 2],
            net_return=net_return,
            entry_offset=index * 24,
        )
        for index, net_return in enumerate(returns)
    ]
    drawdown_trades = [
        make_summary_trade(
            net_return=portfolio_return,
            gross_notional=10_000.0,
            equity_before=10_000.0,
            entry_offset=index * 24,
        )
        for index, portfolio_return in enumerate([0.10, -0.20, 0.05])
    ]

    metrics = bc.summarize_pair_trades(trades, trial_count=4)
    drawdown = bc.summarize_pair_trades(drawdown_trades, trial_count=1)

    assert metrics["deflated_sharpe_probability"] == pytest.approx(0.4929242081662497)
    assert drawdown["max_drawdown"] == pytest.approx(0.20)


def test_exposure_uses_full_evaluated_duration_and_leave_one_out_is_diagnostic():
    trades = [
        make_summary_trade(pair="ETHUSDT/BTCUSDT", holding_hours=12),
        make_summary_trade(pair="SOLUSDT/BTCUSDT", holding_hours=12, entry_offset=24),
    ]

    metrics = bc.summarize_pair_trades(
        trades,
        trial_count=1,
        observation_duration=pd.Timedelta(days=30),
    )

    assert metrics["exposure"] == pytest.approx(24 / (30 * 24))
    eth_omitted = metrics["leave_one_pair_out"]["ETHUSDT/BTCUSDT"]
    assert {
        "completed_trades",
        "wins",
        "win_rate",
        "profit_factor",
        "mean_net_return",
        "median_net_return",
        "sharpe",
        "max_drawdown",
        "exposure",
        "absolute_realized_btc_beta",
        "return_components",
        "exit_counts",
    }.issubset(eth_omitted)
    assert "leave_one_pair_out" not in eth_omitted


def test_no_loss_profit_factor_round_trips_as_explicit_infinity_and_passes_gate(tmp_path):
    trades = [
        make_summary_trade(pair=pe.FIXED_PAIRS[index % 2], entry_offset=index * 24)
        for index in range(60)
    ]
    metrics = bc.summarize_pair_trades(trades, trial_count=1)
    trial = make_trial("development", passed=True)
    trial["metrics"] = metrics
    ledger = tmp_path / "ledger.json"

    assert metrics["profit_factor"] == float("inf")
    assert bc.development_gate({
        **make_metrics(),
        "profit_factor": metrics["profit_factor"],
    }).passed is True
    bc.write_trial_ledger(ledger, trial)

    persisted = json.loads(ledger.read_text())
    assert persisted["trials"][0]["metrics"]["profit_factor"] == "Infinity"


def test_cost_stress_reprices_the_same_entry_exit_ledger():
    trades = [
        make_summary_trade(net_return=0.01),
        make_summary_trade(
            pair="SOLUSDT/BTCUSDT",
            net_return=-0.004,
            alt_side="SHORT",
            entry_offset=24,
        ),
    ]

    repriced = bc._reprice_pair_trades(
        trades,
        base_config=BASE_CONFIG,
        fee_bps=15.0,
        slippage_bps=5.0,
    )

    assert [(trade.pair, trade.entry_ts, trade.exit_ts, trade.exit_reason) for trade in repriced] == [
        (trade.pair, trade.entry_ts, trade.exit_ts, trade.exit_reason) for trade in trades
    ]
    assert all(repriced_trade.net_return < original.net_return for repriced_trade, original in zip(repriced, trades))


def test_cost_stress_grid_is_exactly_zero_two_five_by_five_ten_fifteen():
    stress = bc._cost_stress_metrics(
        [],
        base_config=BASE_CONFIG,
        trial_count=1,
    )

    assert set(stress) == {
        "5bps_fee_0bps_slippage",
        "5bps_fee_2bps_slippage",
        "5bps_fee_5bps_slippage",
        "10bps_fee_0bps_slippage",
        "10bps_fee_2bps_slippage",
        "10bps_fee_5bps_slippage",
        "15bps_fee_0bps_slippage",
        "15bps_fee_2bps_slippage",
        "15bps_fee_5bps_slippage",
    }


def test_dataset_hash_is_order_stable_and_value_sensitive():
    left = pd.DataFrame({"ts": [TS0, TS1], "close": [100.0, 101.0]})
    right = pd.DataFrame({"ts": [TS0, TS1], "close": [50.0, 51.0]})

    expected = bc.dataset_hash({"left": left, "right": right})

    assert bc.dataset_hash({"right": right, "left": left}) == expected
    assert bc.dataset_hash({"left": left, "right": right.assign(close=[50.0, 52.0])}) != expected


def test_sparse_year_spanning_timestamps_do_not_count_as_common_history(tmp_path):
    with pytest.raises(ValueError, match=r"insufficient common coverage: .*need at least 360"):
        bc.run_pairs_experiment(
            make_sparse_experiment_market_data(),
            config=BASE_CONFIG,
            params=pe.DEFAULT_PARAMS,
            open_holdout=False,
            ledger_path=tmp_path / "ledger.json",
        )


@pytest.mark.parametrize(
    ("attribute", "label"),
    [
        ("btc_1h", "1h"),
        ("btc_15m", "15m"),
        ("btc_funding", "funding"),
    ],
)
def test_duplicate_btc_tapes_must_be_identical(tmp_path, monkeypatch, attribute, label):
    data = make_experiment_market_data()
    sol = data["SOLUSDT/BTCUSDT"]
    mismatched = getattr(sol, attribute).copy()
    value_column = "funding_rate" if attribute == "btc_funding" else "close"
    mismatched.loc[mismatched.index[100], value_column] += 0.0001
    data[sol.pair] = bc.PairMarketData(**{**sol.__dict__, attribute: mismatched})
    monkeypatch.setattr(bc, "run_pairs_backtest", lambda data, **kwargs: {
        pair: bc.PairsBacktestResult(pair, [], {}, []) for pair in pe.FIXED_PAIRS
    })

    with pytest.raises(ValueError, match=rf"BTCUSDT {label} tapes differ"):
        bc.run_pairs_experiment(
            data,
            config=BASE_CONFIG,
            params=pe.DEFAULT_PARAMS,
            open_holdout=False,
            ledger_path=tmp_path / "ledger.json",
        )


def test_isolated_common_gaps_are_allowed_when_effective_coverage_is_three_sixty_days(
    tmp_path, monkeypatch,
):
    data = make_experiment_market_data()
    missing_hour = COMMON_START + pd.Timedelta(days=10, hours=3)
    missing_quarter = COMMON_START + pd.Timedelta(days=11, minutes=45)
    for pair, market in list(data.items()):
        data[pair] = bc.PairMarketData(**{
            **market.__dict__,
            "alt_1h": market.alt_1h.loc[market.alt_1h["ts"] != missing_hour],
            "btc_1h": market.btc_1h.loc[market.btc_1h["ts"] != missing_hour],
            "alt_15m": market.alt_15m.loc[market.alt_15m["ts"] != missing_quarter],
            "btc_15m": market.btc_15m.loc[market.btc_15m["ts"] != missing_quarter],
        })
    monkeypatch.setattr(bc, "run_pairs_backtest", lambda data, **kwargs: {
        pair: bc.PairsBacktestResult(pair, [], {}, []) for pair in pe.FIXED_PAIRS
    })

    report = bc.run_pairs_experiment(
        data,
        config=BASE_CONFIG,
        params=pe.DEFAULT_PARAMS,
        open_holdout=False,
        ledger_path=tmp_path / "ledger.json",
    )

    assert report["schedule"]["common_start"] == COMMON_START.isoformat()
    assert report["schedule"]["common_end"] == COMMON_END.isoformat()


def test_each_window_receives_only_its_declared_formation_and_trading_data(monkeypatch):
    data = make_experiment_market_data()
    window = bc.WalkForwardWindow(
        formation_start=COMMON_START + pd.Timedelta(days=30),
        start=COMMON_START + pd.Timedelta(days=90),
        end=COMMON_START + pd.Timedelta(days=120),
    )
    inspected = []

    def replay(sliced, *, window_start, window_end, config, params, universe=None):
        for market in sliced.values():
            for frame in (market.alt_1h, market.btc_1h):
                inspected.append((frame["ts"].min(), frame["ts"].max()))
                assert frame["ts"].min() >= window.formation_start
                assert frame["ts"].max() < window.end
            for frame in (market.alt_15m, market.btc_15m):
                inspected.append((frame["ts"].min(), frame["ts"].max()))
                assert frame["ts"].min() >= window.formation_start
                assert frame["ts"].max() <= window.end
        return {
            pair: bc.PairsBacktestResult(pair, [], {}, []) for pair in pe.FIXED_PAIRS
        }

    monkeypatch.setattr(bc, "run_pairs_backtest", replay)
    bc._run_pairs_windows(
        data,
        windows=[window],
        config=BASE_CONFIG,
        params=pe.DEFAULT_PARAMS,
    )

    assert inspected


def test_experiment_opens_holdout_once_after_development_pass_and_rejects_before_replay(
    tmp_path, monkeypatch,
):
    calls = []

    def replay(data, *, window_start, window_end, config, params, universe=None):
        calls.append((window_start, window_end, config.fee_bps, config.slippage_bps))
        return passing_experiment_replay(
            data,
            window_start=window_start,
            window_end=window_end,
            config=config,
            params=params,
        )

    monkeypatch.setattr(bc, "run_pairs_backtest", replay)
    ledger_path = tmp_path / "ledger.json"

    report = bc.run_pairs_experiment(
        make_experiment_market_data(),
        config=BASE_CONFIG,
        params=pe.DEFAULT_PARAMS,
        open_holdout=True,
        ledger_path=ledger_path,
    )

    assert len(calls) == 8
    assert all((fee_bps, slippage_bps) == (10.0, 2.0) for _, _, fee_bps, slippage_bps in calls)
    assert report["development"]["gate"]["passed"] is True
    assert report["holdout"]["opened"] is True
    persisted = json.loads(ledger_path.read_text())
    assert persisted["holdout"]["status"] == "opened"
    assert [trial["phase"] for trial in persisted["trials"]] == ["development", "holdout"]
    assert persisted["trials"][1]["metrics"]["trial_count"] == 2
    assert len(persisted["trials"][0]["cost_stress"]) == 9

    with pytest.raises(ValueError, match="already opened"):
        bc.run_pairs_experiment(
            make_experiment_market_data(),
            config=BASE_CONFIG,
            params=pe.DEFAULT_PARAMS,
            open_holdout=True,
            ledger_path=ledger_path,
        )
    assert len(calls) == 8


def test_experiment_development_failure_keeps_holdout_sealed(tmp_path, monkeypatch):
    calls = []

    def replay(data, *, window_start, window_end, config, params, universe=None):
        calls.append((window_start, window_end))
        return {
            pair: bc.PairsBacktestResult(
                pair=pair,
                trades=[],
                metrics={},
                invalid_reasons=[],
            )
            for pair in pe.FIXED_PAIRS
        }

    monkeypatch.setattr(bc, "run_pairs_backtest", replay)
    ledger_path = tmp_path / "ledger.json"

    report = bc.run_pairs_experiment(
        make_experiment_market_data(),
        config=BASE_CONFIG,
        params=pe.DEFAULT_PARAMS,
        open_holdout=True,
        ledger_path=ledger_path,
    )

    assert len(calls) == 7
    assert report["development"]["gate"]["passed"] is False
    assert report["holdout"] == {"requested": True, "opened": False, "status": "sealed"}
    assert report["request_error"] == {
        "code": "development_gate_failed",
        "failed_conditions": report["development"]["gate"]["failed_conditions"],
    }
    persisted = json.loads(ledger_path.read_text())
    assert persisted["holdout"]["status"] == "sealed"
    assert len(persisted["trials"]) == 1
    assert persisted["trials"][0]["trial_id"] == report["development"]["trial_id"]
    assert persisted["trials"][0]["gate"]["passed"] is False

    repeated = bc.run_pairs_experiment(
        make_experiment_market_data(),
        config=BASE_CONFIG,
        params=pe.DEFAULT_PARAMS,
        open_holdout=True,
        ledger_path=ledger_path,
    )
    repeated_ledger = json.loads(ledger_path.read_text())
    assert len(repeated_ledger["trials"]) == 1
    assert repeated["development"]["trial_id"] == report["development"]["trial_id"]
    assert repeated["development"]["metrics"] == report["development"]["metrics"]


def test_development_report_preserves_window_metrics_without_stress_duplication(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(bc, "run_pairs_backtest", passing_experiment_replay)

    report = bc.run_pairs_experiment(
        make_experiment_market_data(),
        config=BASE_CONFIG,
        params=pe.DEFAULT_PARAMS,
        open_holdout=False,
        ledger_path=tmp_path / "ledger.json",
    )

    windows = report["development"]["windows"]
    assert len(windows) == len(report["schedule"]["development_windows"])
    assert all(window["metrics"]["completed_trades"] == 10 for window in windows)
    assert all(set(window["metrics"]["per_pair_diagnostics"]) == set(pe.FIXED_PAIRS)
               for window in windows)
    assert "cost_stress" not in report["development"]["metrics"]
    assert "cost_stress" not in report["development"]
    assert len(report["cost_stress"]) == 9


def test_trial_id_is_deterministic_for_fixed_phase_params_costs_and_frames(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(bc, "run_pairs_backtest", passing_experiment_replay)
    data = make_experiment_market_data()

    first = bc.run_pairs_experiment(
        data,
        config=BASE_CONFIG,
        params=pe.DEFAULT_PARAMS,
        open_holdout=False,
        ledger_path=tmp_path / "first.json",
    )
    second = bc.run_pairs_experiment(
        data,
        config=BASE_CONFIG,
        params=pe.DEFAULT_PARAMS,
        open_holdout=False,
        ledger_path=tmp_path / "second.json",
    )

    assert first["development"]["trial_id"] == second["development"]["trial_id"]


def test_trial_id_changes_with_cost_scenario_and_universe(tmp_path, monkeypatch):
    monkeypatch.setattr(bc, "run_pairs_backtest", passing_experiment_replay)
    data = make_experiment_market_data()

    legacy = bc.run_pairs_experiment(
        data,
        config=BASE_CONFIG,
        params=pe.DEFAULT_PARAMS,
        open_holdout=False,
        ledger_path=tmp_path / "legacy.json",
    )
    aster_taker = bc.run_pairs_experiment(
        data,
        config=replace(BASE_CONFIG, fee_bps=4.0),
        params=pe.DEFAULT_PARAMS,
        open_holdout=False,
        ledger_path=tmp_path / "taker.json",
    )
    aster_maker = bc.run_pairs_experiment(
        data,
        config=replace(
            BASE_CONFIG,
            fee_bps=2.0,
            slippage_bps=0.0,
            one_leg_execution_shock_bps=3.0,
        ),
        params=pe.DEFAULT_PARAMS,
        open_holdout=False,
        ledger_path=tmp_path / "maker.json",
    )
    eth_only = bc.run_pairs_experiment(
        {"ETHUSDT/BTCUSDT": data["ETHUSDT/BTCUSDT"]},
        config=BASE_CONFIG,
        params=pe.DEFAULT_PARAMS,
        open_holdout=False,
        ledger_path=tmp_path / "single.json",
        universe=("ETHUSDT/BTCUSDT",),
    )

    trial_ids = {
        legacy["development"]["trial_id"],
        aster_taker["development"]["trial_id"],
        aster_maker["development"]["trial_id"],
        eth_only["development"]["trial_id"],
    }
    assert len(trial_ids) == 4


def test_replay_supports_custom_universe_and_rejects_unknown_pairs():
    data = make_two_pair_market_data()
    eth_only = {"ETHUSDT/BTCUSDT": data["ETHUSDT/BTCUSDT"]}

    result = bc.run_pairs_backtest(
        eth_only,
        window_start=TS0,
        window_end=TS_END,
        config=BASE_CONFIG,
        universe=("ETHUSDT/BTCUSDT",),
    )

    assert set(result) == {"ETHUSDT/BTCUSDT"}
    with pytest.raises(ValueError, match="unsupported pairs"):
        bc.run_pairs_backtest(
            data,
            window_start=TS0,
            window_end=TS_END,
            config=BASE_CONFIG,
            universe=("ETHUSDT/BTCUSDT",),
        )


def test_summary_and_development_gate_follow_the_caller_provided_universe():
    trades = [
        make_summary_trade(pair="XRPUSDT/BTCUSDT", entry_offset=index * 24)
        for index in range(10)
    ]

    metrics = bc.summarize_pair_trades(
        trades, trial_count=1, universe=("XRPUSDT/BTCUSDT",),
    )

    assert set(metrics["per_pair"]) == {"XRPUSDT/BTCUSDT"}
    assert metrics["per_pair_trades"] == {"XRPUSDT/BTCUSDT": 10}
    assert set(metrics["per_pair_diagnostics"]) == {"XRPUSDT/BTCUSDT"}
    assert set(metrics["leave_one_pair_out"]) == {"XRPUSDT/BTCUSDT"}

    gate_metrics = make_metrics(
        per_pair_trades={"XRPUSDT/BTCUSDT": 60},
        per_pair={
            "XRPUSDT/BTCUSDT": {
                "completed_trades": 60,
                "profit_factor": 1.3,
                "mean_net_return": 0.003,
                "gross_profit_contribution": 1.0,
            },
        },
    )
    assert bc.development_gate(
        gate_metrics, universe=("XRPUSDT/BTCUSDT",),
    ).passed is True
    # The default fixed universe still requires ETH and SOL rows.
    assert bc.development_gate(gate_metrics).passed is False


def test_same_cached_frames_produce_identical_report_and_dataset_hash(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(bc, "run_pairs_backtest", passing_experiment_replay)
    data = make_experiment_market_data()

    first = bc.run_pairs_experiment(
        data,
        config=BASE_CONFIG,
        params=pe.DEFAULT_PARAMS,
        open_holdout=False,
        ledger_path=tmp_path / "one.json",
    )
    second = bc.run_pairs_experiment(
        data,
        config=BASE_CONFIG,
        params=pe.DEFAULT_PARAMS,
        open_holdout=False,
        ledger_path=tmp_path / "two.json",
    )

    assert first["dataset_hashes"] == second["dataset_hashes"]
    assert first["development"]["metrics"] == second["development"]["metrics"]


def test_identical_development_rerun_is_idempotent_in_one_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(bc, "run_pairs_backtest", passing_experiment_replay)
    ledger = tmp_path / "ledger.json"
    data = make_experiment_market_data()

    first = bc.run_pairs_experiment(
        data, config=BASE_CONFIG, params=pe.DEFAULT_PARAMS,
        open_holdout=False, ledger_path=ledger,
    )
    second = bc.run_pairs_experiment(
        data, config=BASE_CONFIG, params=pe.DEFAULT_PARAMS,
        open_holdout=False, ledger_path=ledger,
    )

    assert second["development"]["trial_id"] == first["development"]["trial_id"]
    assert second["development"]["metrics"] == first["development"]["metrics"]
    assert second["development"]["windows"] == first["development"]["windows"]
    assert len(json.loads(ledger.read_text())["trials"]) == 1


def test_cost_scenarios_coexist_idempotently_in_one_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(bc, "run_pairs_backtest", passing_experiment_replay)
    ledger = tmp_path / "ledger.json"
    data = make_experiment_market_data()

    for config in (BASE_CONFIG, replace(BASE_CONFIG, fee_bps=4.0)):
        for _ in range(2):
            bc.run_pairs_experiment(
                data, config=config, params=pe.DEFAULT_PARAMS,
                open_holdout=False, ledger_path=ledger,
            )

    persisted = json.loads(ledger.read_text())
    assert len(persisted["trials"]) == 2
    assert {trial["cost_config"]["fee_bps"] for trial in persisted["trials"]} == {
        10.0, 4.0,
    }
    assert persisted["holdout"]["status"] == "sealed"


def test_holdout_replay_exception_updates_pending_trial_before_reraising(tmp_path, monkeypatch):
    calls = []

    def replay(data, *, window_start, window_end, config, params, universe=None):
        calls.append((window_start, window_end))
        if window_end - window_start == pd.Timedelta(days=90):
            raise RuntimeError("synthetic holdout failure")
        return passing_experiment_replay(
            data,
            window_start=window_start,
            window_end=window_end,
            config=config,
            params=params,
        )

    monkeypatch.setattr(bc, "run_pairs_backtest", replay)
    ledger = tmp_path / "ledger.json"

    with pytest.raises(RuntimeError, match="synthetic holdout failure"):
        bc.run_pairs_experiment(
            make_experiment_market_data(),
            config=BASE_CONFIG,
            params=pe.DEFAULT_PARAMS,
            open_holdout=True,
            ledger_path=ledger,
        )

    persisted = json.loads(ledger.read_text())
    assert persisted["holdout"]["status"] == "opened"
    assert [trial["phase"] for trial in persisted["trials"]] == ["development", "holdout"]
    failed_holdout = persisted["trials"][1]
    assert failed_holdout["gate"] == {
        "passed": False,
        "failed_conditions": ["holdout_replay_failed"],
    }
    assert failed_holdout["invalid_reasons"] == [
        "holdout_replay_failed:RuntimeError:synthetic holdout failure"
    ]
    assert len(calls) == 8
    with pytest.raises(ValueError, match="already opened"):
        bc.run_pairs_experiment(
            make_experiment_market_data(),
            config=BASE_CONFIG,
            params=pe.DEFAULT_PARAMS,
            open_holdout=True,
            ledger_path=ledger,
        )
    assert len(calls) == 8
    later_development = bc.run_pairs_experiment(
        make_experiment_market_data(),
        config=BASE_CONFIG,
        params=pe.DEFAULT_PARAMS,
        open_holdout=False,
        ledger_path=ledger,
    )
    assert later_development["development"]["metrics"]["trial_count"] == 2


def test_hard_crash_leaves_pending_holdout_trial_as_consumed(tmp_path, monkeypatch):
    def replay(data, *, window_start, window_end, config, params, universe=None):
        if window_end - window_start == pd.Timedelta(days=90):
            raise KeyboardInterrupt("synthetic hard crash")
        return passing_experiment_replay(
            data,
            window_start=window_start,
            window_end=window_end,
            config=config,
            params=params,
        )

    monkeypatch.setattr(bc, "run_pairs_backtest", replay)
    ledger = tmp_path / "ledger.json"

    with pytest.raises(KeyboardInterrupt, match="synthetic hard crash"):
        bc.run_pairs_experiment(
            make_experiment_market_data(),
            config=BASE_CONFIG,
            params=pe.DEFAULT_PARAMS,
            open_holdout=True,
            ledger_path=ledger,
        )

    persisted = json.loads(ledger.read_text())
    assert persisted["holdout"]["status"] == "opened"
    assert persisted["trials"][-1]["phase"] == "holdout"
    assert persisted["trials"][-1]["invalid_reasons"] == ["holdout_replay_pending"]

    with pytest.raises(ValueError, match="already opened"):
        bc.run_pairs_experiment(
            make_experiment_market_data(),
            config=BASE_CONFIG,
            params=pe.DEFAULT_PARAMS,
            open_holdout=True,
            ledger_path=ledger,
        )


def test_concurrent_holdout_requests_reserve_and_replay_exactly_once(tmp_path):
    context = multiprocessing.get_context("spawn")
    global _CONCURRENT_HOLDOUT_REPLAYS
    _CONCURRENT_HOLDOUT_REPLAYS = context.Value("i", 0)
    ledger = tmp_path / "ledger.json"
    start_event = context.Event()
    result_queue = context.Queue()
    data = make_experiment_market_data()
    processes = [
        context.Process(
            target=run_experiment_process,
            args=(
                data,
                str(ledger),
                start_event,
                result_queue,
                _CONCURRENT_HOLDOUT_REPLAYS,
            ),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start_event.set()
    results = [result_queue.get(timeout=60) for _ in processes]
    for process in processes:
        process.join(timeout=60)

    assert all(process.exitcode == 0 for process in processes)
    assert sorted(result[0] for result in results) == ["error", "opened"]
    assert _CONCURRENT_HOLDOUT_REPLAYS.value == 1
    persisted = json.loads(ledger.read_text())
    assert persisted["holdout"]["status"] == "opened"
    assert sum(trial["phase"] == "holdout" for trial in persisted["trials"]) == 1
