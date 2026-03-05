"""Covariance estimation with Ledoit-Wolf shrinkage."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def estimate_covariance(
    returns: pd.DataFrame,
    lookback: int = 252,
) -> pd.DataFrame:
    """
    Estimate covariance matrix using Ledoit-Wolf shrinkage.

    Uses trailing `lookback` days of returns. Returns annualized covariance.
    """
    recent = returns.tail(lookback).dropna(axis=1, how="any")

    if recent.shape[0] < 30 or recent.shape[1] < 2:
        # Fallback to diagonal (variance only)
        var = returns.tail(lookback).var() * 252
        return pd.DataFrame(
            np.diag(var.reindex(returns.columns).fillna(var.mean()).values),
            index=returns.columns,
            columns=returns.columns,
        )

    lw = LedoitWolf()
    lw.fit(recent.values)
    cov = lw.covariance_ * 252  # Annualize

    result = pd.DataFrame(cov, index=recent.columns, columns=recent.columns)
    return result.reindex(index=returns.columns, columns=returns.columns).fillna(0)
