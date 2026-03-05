"""Signal normalization: cross-sectional rank and z-score transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rank_normalize(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional rank normalization.

    Maps each row to [-1, 1] via percentile rank, centered at 0.
    Robust to outliers — the primary normalization method.
    """
    ranked = signals.rank(axis=1, pct=True)  # 0 to 1
    return 2 * ranked - 1  # -1 to 1


def zscore_normalize(signals: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """
    Cross-sectional z-score normalization (clipped).

    Subtracts row mean and divides by row std, then clips at ±clip.
    """
    row_mean = signals.mean(axis=1)
    row_std = signals.std(axis=1).replace(0, np.nan)
    z = signals.sub(row_mean, axis=0).div(row_std, axis=0)
    return z.clip(-clip, clip)
