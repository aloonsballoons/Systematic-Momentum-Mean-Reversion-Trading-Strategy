"""Statistical tests: Lo-adjusted Sharpe, Deflated Sharpe Ratio, ADF."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller


def lo_adjusted_sharpe_test(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
) -> dict[str, float]:
    """
    Lo (2002) autocorrelation-adjusted Sharpe ratio test.

    Standard Sharpe ratio is biased when returns are autocorrelated.
    This adjusts the standard error to account for serial correlation.

    Returns
    -------
    dict with 'sharpe', 'adjusted_sharpe', 'se', 't_stat', 'p_value'
    """
    ret = returns.dropna()
    n = len(ret)

    # Standard Sharpe
    excess_mean = ret.mean() * 252 - risk_free_rate
    annual_vol = ret.std() * np.sqrt(252)
    sharpe = excess_mean / annual_vol if annual_vol > 0 else 0

    # Autocorrelation adjustment (Lo 2002)
    # η(q) = 1 + 2 * Σ(k=1 to q) (1 - k/(q+1)) * ρ(k)
    q = min(6, n // 4)  # Truncation lag
    autocorrs = [ret.autocorr(lag=k) for k in range(1, q + 1)]
    eta = 1 + 2 * sum(
        (1 - k / (q + 1)) * rho
        for k, rho in enumerate(autocorrs, 1)
        if not np.isnan(rho)
    )
    eta = max(eta, 0.01)  # Safety floor

    # Adjusted Sharpe
    adjusted_sharpe = sharpe / np.sqrt(eta)

    # Standard error under H0: SR = 0
    se = np.sqrt((1 + 0.5 * sharpe ** 2) / n * eta)
    t_stat = sharpe / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    return {
        "sharpe": sharpe,
        "adjusted_sharpe": adjusted_sharpe,
        "eta": eta,
        "se": se,
        "t_stat": t_stat,
        "p_value": p_value,
    }


def deflated_sharpe_ratio(
    sharpe_observed: float,
    n_trials: int,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    sharpe_benchmark: float = 0.0,
) -> dict[str, float]:
    """
    Deflated Sharpe Ratio (Bailey & de Prado, 2014).

    Adjusts for multiple testing: given N strategy trials, what is the
    probability that the best Sharpe ratio is genuine (not due to luck)?

    Parameters
    ----------
    sharpe_observed : best observed Sharpe ratio (annualized)
    n_trials : number of strategy variants tested
    n_obs : number of return observations
    skewness : return skewness
    kurtosis : return kurtosis (excess, so normal = 0)
    sharpe_benchmark : expected Sharpe under null (usually 0)

    Returns
    -------
    dict with 'dsr', 'expected_max_sharpe', 'p_value'
    """
    if n_trials <= 0 or n_obs <= 0:
        return {"dsr": 0, "expected_max_sharpe": 0, "p_value": 1.0}

    # Expected maximum Sharpe ratio from n_trials independent trials
    # E[max(SR)] ≈ (1 - γ) * Φ^{-1}(1 - 1/N) + γ * Φ^{-1}(1 - 1/(N*e))
    # Simplified approximation:
    from scipy.special import erfinv

    z = stats.norm.ppf(1 - 1 / n_trials)
    expected_max = sharpe_benchmark + (
        z * np.sqrt(1 / n_obs)
        * np.sqrt(1 + (skewness * z) / 3 + ((kurtosis - 3) * z**2) / 24)
    )

    # Standard error of Sharpe ratio
    se = np.sqrt(
        (1 - skewness * sharpe_observed + (kurtosis - 1) / 4 * sharpe_observed**2)
        / (n_obs - 1)
    )

    # DSR: probability that observed SR > expected max from random trials
    if se > 0:
        t_stat = (sharpe_observed - expected_max) / se
        dsr = stats.norm.cdf(t_stat)
    else:
        dsr = 0.0

    p_value = 1 - dsr

    return {
        "dsr": dsr,
        "expected_max_sharpe": expected_max,
        "p_value": p_value,
        "n_trials": n_trials,
    }


def adf_test(series: pd.Series) -> dict[str, float]:
    """
    Augmented Dickey-Fuller stationarity test.

    Tests null hypothesis that the series has a unit root (non-stationary).
    """
    clean = series.dropna()
    if len(clean) < 20:
        return {"adf_statistic": np.nan, "p_value": 1.0, "is_stationary": False}

    result = adfuller(clean, autolag="AIC")
    return {
        "adf_statistic": result[0],
        "p_value": result[1],
        "is_stationary": result[1] < 0.05,
        "n_lags": result[2],
    }


def information_coefficient(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> pd.Series:
    """
    Compute daily cross-sectional rank IC (Spearman correlation).

    Parameters
    ----------
    signal : alpha signal values (date × ticker)
    forward_returns : next-period returns (date × ticker)

    Returns
    -------
    Series of daily IC values
    """
    common_idx = signal.index.intersection(forward_returns.index)
    common_cols = signal.columns.intersection(forward_returns.columns)

    sig = signal.loc[common_idx, common_cols]
    fwd = forward_returns.loc[common_idx, common_cols]

    ic = sig.corrwith(fwd, axis=1, method="spearman")
    return ic
