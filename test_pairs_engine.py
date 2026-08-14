"""Tests for the causal hourly relative-value relationship model."""
import numpy as np
import pandas as pd

import pairs_engine as pe


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
    first = pe.build_hourly_observations(hourly, pe.DEFAULT_PARAMS)
    mutated = hourly.copy()
    mutated.loc[mutated.index >= 1_440, "alt_close"] *= 100.0
    second = pe.build_hourly_observations(mutated, pe.DEFAULT_PARAMS)
    pd.testing.assert_series_equal(
        first.loc[first["ts"] < hourly.loc[1_440, "ts"], "beta"],
        second.loc[second["ts"] < hourly.loc[1_440, "ts"], "beta"],
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
    observations = pe.build_hourly_observations(hourly, pe.DEFAULT_PARAMS)
    early = observations.loc[observations["ts"] < hourly.loc[1_440, "ts"]]
    assert early["snapshot"].isna().all()


def test_singular_nonpositive_or_short_history_is_rejected():
    assert pe.fit_relationship(make_cointegrated_hourly(hours=1_199), pe.DEFAULT_PARAMS) is None
    assert pe.fit_relationship(make_constant_btc_hourly(hours=1_500), pe.DEFAULT_PARAMS) is None
    assert pe.fit_relationship(make_negative_beta_hourly(hours=1_500), pe.DEFAULT_PARAMS) is None
