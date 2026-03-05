"""Momentum alpha signals: XSMOM, TSMOM, Composite."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MomentumConfig
from src.features.base import Feature


class XSMOM(Feature):
    """
    Cross-sectional momentum with skip period.

    Formula: Return(t-skip-lookback, t-skip) / Vol(t, lookback)
    Skips the most recent month to avoid short-term reversal contamination.
    """

    def __init__(self, lookback: int, skip: int = 21):
        self._lookback = lookback
        self._skip = skip

    @property
    def name(self) -> str:
        return f"xsmom_{self._lookback}"

    def compute(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        # Cumulative return from t-skip-lookback to t-skip
        total_return = prices.shift(self._skip) / prices.shift(self._skip + self._lookback) - 1

        # Realized vol over lookback period
        vol = returns.rolling(window=self._lookback, min_periods=self._lookback // 2).std() * np.sqrt(252)
        vol = vol.replace(0, np.nan)

        return total_return / vol


class TSMOM(Feature):
    """
    Time-series momentum.

    Formula: sign(Return(t-L, t)) × vol_target / Vol(t, L)
    Trend-following signal scaled to target volatility.
    """

    def __init__(self, lookback: int = 252, vol_target: float = 0.15):
        self._lookback = lookback
        self._vol_target = vol_target

    @property
    def name(self) -> str:
        return "tsmom"

    def compute(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        # Past return sign
        past_return = prices / prices.shift(self._lookback) - 1
        sign = np.sign(past_return)

        # Realized vol
        vol = returns.rolling(window=self._lookback, min_periods=self._lookback // 2).std() * np.sqrt(252)
        vol = vol.replace(0, np.nan)

        return sign * self._vol_target / vol


class CompositeMomentum(Feature):
    """
    Composite momentum: average of cross-sectional ranks across multiple lookbacks.

    Formula: mean(rank(XSMOM_L)) for L in lookbacks
    """

    def __init__(self, cfg: MomentumConfig):
        self._cfg = cfg
        self._xsmom_signals = [
            XSMOM(lookback=lb, skip=cfg.xsmom_skip)
            for lb in cfg.xsmom_lookbacks
        ]

    @property
    def name(self) -> str:
        return "composite_momentum"

    def compute(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        ranks = []
        for signal in self._xsmom_signals:
            raw = signal.compute(prices, returns)
            # Cross-sectional rank (pct=True gives 0-1 range)
            ranked = raw.rank(axis=1, pct=True)
            ranks.append(ranked)

        # Average of ranks
        return pd.concat(ranks).groupby(level=0).mean().reindex(prices.index)
