"""Causal hourly relationship estimates for BTC-neutral pairs research.

This module is deliberately offline-only.  It fits each observation using
completed hourly history and stores the residual distribution used for later
intrabar replay, so no estimate can use future prices.
"""
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairsParams:
    formation_hours: int = 60 * 24
    residual_history_hours: int = 30 * 24
    min_observations: int = 1_200
    min_return_correlation: float = 0.60
    max_adf_pvalue: float = 0.05
    min_half_life_hours: float = 2.0
    max_half_life_hours: float = 72.0
    max_beta_change: float = 0.30
    entry_z: float = 2.0
    confirm_reversion: float = 0.10
    confirmation_bars: int = 4
    convergence_z: float = 0.50
    stop_z: float = 3.50
    max_holding_hours: int = 72


DEFAULT_PARAMS = PairsParams()


@dataclass(frozen=True)
class RelationshipSnapshot:
    alpha: float
    beta: float
    residual: float
    residual_mean: float
    residual_std: float
    return_correlation: float
    adf_pvalue: float
    half_life_hours: float
    observations: int
    beta_change: float | None
    stable: bool
    rejection_reason: str | None


@dataclass(frozen=True)
class PairObservation:
    ts: pd.Timestamp
    snapshot: RelationshipSnapshot | None
    zscore: float | None
    direction: str | None


def align_hourly_prices(alt_1h: pd.DataFrame, btc_1h: pd.DataFrame) -> pd.DataFrame:
    """Return finite positive alt and BTC closes joined on exact hourly timestamps."""
    required = {"ts", "close"}
    if not required.issubset(alt_1h.columns) or not required.issubset(btc_1h.columns):
        raise ValueError("hourly prices require ts and close columns")

    alt = alt_1h.loc[:, ["ts", "close"]].rename(columns={"close": "alt_close"}).copy()
    btc = btc_1h.loc[:, ["ts", "close"]].rename(columns={"close": "btc_close"}).copy()
    alt["ts"] = pd.to_datetime(alt["ts"], utc=True)
    btc["ts"] = pd.to_datetime(btc["ts"], utc=True)
    hourly = alt.merge(btc, how="inner", on="ts", validate="one_to_one")
    hourly["alt_close"] = pd.to_numeric(hourly["alt_close"], errors="coerce")
    hourly["btc_close"] = pd.to_numeric(hourly["btc_close"], errors="coerce")
    valid = (
        np.isfinite(hourly["alt_close"])
        & np.isfinite(hourly["btc_close"])
        & (hourly["alt_close"] > 0.0)
        & (hourly["btc_close"] > 0.0)
    )
    return hourly.loc[valid].sort_values("ts").reset_index(drop=True)


def _valid_history(history: pd.DataFrame, params: PairsParams) -> pd.DataFrame | None:
    required = {"ts", "alt_close", "btc_close"}
    if not required.issubset(history.columns):
        return None
    clean = history.loc[:, ["ts", "alt_close", "btc_close"]].copy()
    clean["alt_close"] = pd.to_numeric(clean["alt_close"], errors="coerce")
    clean["btc_close"] = pd.to_numeric(clean["btc_close"], errors="coerce")
    valid = (
        np.isfinite(clean["alt_close"])
        & np.isfinite(clean["btc_close"])
        & (clean["alt_close"] > 0.0)
        & (clean["btc_close"] > 0.0)
    )
    clean = clean.loc[valid].sort_values("ts").tail(params.formation_hours).reset_index(drop=True)
    if len(clean) < params.min_observations:
        return None
    return clean


def fit_relationship(
    history: pd.DataFrame,
    params: PairsParams,
    prior_beta: float | None = None,
) -> RelationshipSnapshot | None:
    """Fit a lagged log-price relationship from completed hourly history."""
    clean = _valid_history(history, params)
    if clean is None:
        return None

    alt_log = np.log(clean["alt_close"].to_numpy(dtype=float))
    btc_log = np.log(clean["btc_close"].to_numpy(dtype=float))
    returns = np.diff(np.column_stack((alt_log, btc_log)), axis=0)
    if returns.shape[0] < 2:
        return None
    if np.any(np.std(returns, axis=0) <= 0.0):
        return None
    return_correlation = float(np.corrcoef(returns[:, 0], returns[:, 1])[0, 1])
    if not np.isfinite(return_correlation):
        return None

    design = np.column_stack((np.ones(len(btc_log)), btc_log))
    try:
        alpha, beta = np.linalg.lstsq(design, alt_log, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    residuals = alt_log - (alpha + beta * btc_log)
    if not np.all(np.isfinite(residuals)) or not np.isfinite(beta) or beta <= 0.0:
        return None

    lagged = residuals[:-1]
    delta = np.diff(residuals)
    try:
        _, ar_lambda = np.linalg.lstsq(
            np.column_stack((np.ones(len(lagged)), lagged)), delta, rcond=None,
        )[0]
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(ar_lambda) or ar_lambda >= 0.0:
        return None
    half_life_hours = float(-np.log(2.0) / ar_lambda)
    if not np.isfinite(half_life_hours) or half_life_hours <= 0.0:
        return None

    try:
        from statsmodels.tsa.stattools import adfuller

        adf_pvalue = float(adfuller(residuals, maxlag=24, autolag=None)[1])
    except (ValueError, OverflowError, np.linalg.LinAlgError):
        return None
    if not np.isfinite(adf_pvalue):
        return None

    residual_window = residuals[-params.residual_history_hours:]
    residual_mean = float(np.mean(residual_window))
    residual_std = float(np.std(residual_window, ddof=0))
    if not np.isfinite(residual_std) or residual_std <= 0.0:
        return None

    beta_change = None
    beta_change_ok = True
    if prior_beta is not None:
        if not np.isfinite(prior_beta) or prior_beta == 0.0:
            beta_change_ok = False
        else:
            beta_change = float(abs(beta - prior_beta) / abs(prior_beta))
            beta_change_ok = beta_change <= params.max_beta_change

    gates = (
        return_correlation >= params.min_return_correlation,
        adf_pvalue <= params.max_adf_pvalue,
        params.min_half_life_hours <= half_life_hours <= params.max_half_life_hours,
        beta_change_ok,
    )
    reasons = (
        "return_correlation",
        "adf_pvalue",
        "half_life",
        "beta_change",
    )
    rejection_reason = next((reason for gate, reason in zip(gates, reasons) if not gate), None)
    return RelationshipSnapshot(
        alpha=float(alpha),
        beta=float(beta),
        residual=float(residuals[-1]),
        residual_mean=residual_mean,
        residual_std=residual_std,
        return_correlation=return_correlation,
        adf_pvalue=adf_pvalue,
        half_life_hours=half_life_hours,
        observations=len(clean),
        beta_change=beta_change,
        stable=all(gates),
        rejection_reason=rejection_reason,
    )


def _snapshot_row(observation: PairObservation) -> dict[str, object]:
    snapshot = observation.snapshot
    row: dict[str, object] = {
        "ts": observation.ts,
        "zscore": observation.zscore,
        "direction": observation.direction,
        "snapshot": snapshot,
    }
    for name in RelationshipSnapshot.__dataclass_fields__:
        row[name] = getattr(snapshot, name) if snapshot is not None else np.nan
    return row


def build_hourly_observations(hourly: pd.DataFrame, params: PairsParams) -> pd.DataFrame:
    """Build one causal relationship observation for each completed hourly bar.

    At timestamp ``t`` the model sees only the 1,440 rows immediately before
    ``t``; ``t`` is used solely to calculate the current residual and z-score.
    """
    clean = _valid_history(hourly, replace(
        params,
        formation_hours=max(len(hourly), params.formation_hours),
        min_observations=0,
    ))
    if clean is None:
        return pd.DataFrame(columns=["ts", "zscore", "direction", "snapshot", "beta"])

    snapshots: dict[pd.Timestamp, RelationshipSnapshot | None] = {}
    rows: list[dict[str, object]] = []
    for index, current in clean.iterrows():
        ts = current["ts"]
        if index < params.formation_hours:
            snapshots[ts] = None
            rows.append(_snapshot_row(PairObservation(ts, None, None, None)))
            continue
        history = clean.iloc[max(0, index - params.formation_hours):index]
        baseline = snapshots.get(ts - pd.Timedelta(hours=24))
        baseline_stable = baseline is not None and baseline.stable
        snapshot = fit_relationship(
            history,
            params,
            prior_beta=baseline.beta if baseline_stable else None,
        )
        if snapshot is not None and baseline is None:
            snapshot = replace(
                snapshot,
                stable=False,
                rejection_reason="missing_prior_snapshot",
            )
        snapshots[ts] = snapshot

        zscore = None
        direction = None
        if snapshot is not None and snapshot.stable:
            residual = float(
                np.log(float(current["alt_close"]))
                - (snapshot.alpha + snapshot.beta * np.log(float(current["btc_close"])))
            )
            zscore_value = (residual - snapshot.residual_mean) / snapshot.residual_std
            if np.isfinite(zscore_value):
                zscore = float(zscore_value)
                if baseline_stable and zscore >= params.entry_z:
                    direction = "SHORT_ALT_LONG_BTC"
                elif baseline_stable and zscore <= -params.entry_z:
                    direction = "LONG_ALT_SHORT_BTC"
        rows.append(_snapshot_row(PairObservation(ts, snapshot, zscore, direction)))

    return pd.DataFrame(rows)
