"""Volatility features: EWMA Vol, Vol-of-Vol."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import VolatilityConfig
from src.features.base import Feature


class EWMAVol(Feature):
    """
    Exponentially weighted moving average volatility.

    Formula: sqrt(EWMA(returns², halflife)) — annualized.
    """

    def __init__(self, halflife: int = 21):
        self._halflife = halflife

    @property
    def name(self) -> str:
        return "ewma_vol"

    def compute(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        variance = (returns ** 2).ewm(halflife=self._halflife, min_periods=self._halflife).mean()
        return np.sqrt(variance) * np.sqrt(252)


class VolOfVol(Feature):
    """
    Volatility of volatility — dispersion of realized vol across windows.

    Formula: Std(RealizedVol_w for w in windows)
    Higher vol-of-vol indicates unstable volatility regime.
    """

    def __init__(self, windows: tuple[int, ...] = (21, 63, 126)):
        self._windows = windows

    @property
    def name(self) -> str:
        return "vol_of_vol"

    def compute(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        vols = []
        for w in self._windows:
            vol = returns.rolling(window=w, min_periods=w // 2).std() * np.sqrt(252)
            vols.append(vol)

        # Stack and compute cross-window std for each date/ticker
        stacked = pd.concat(vols, keys=range(len(vols)))
        return stacked.groupby(level=1).std()
