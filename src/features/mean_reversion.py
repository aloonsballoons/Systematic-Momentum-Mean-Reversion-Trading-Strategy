"""Mean-reversion alpha signals: Bollinger, RSI, Short-Term Reversal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MeanReversionConfig
from src.features.base import Feature


class BollingerMeanReversion(Feature):
    """
    Bollinger Band mean-reversion signal.

    Formula: -(Price - SMA(window)) / (k × StdDev(window))
    Negative sign: oversold → positive signal (buy), overbought → negative (sell).
    """

    def __init__(self, window: int = 20, num_std: float = 2.0):
        self._window = window
        self._num_std = num_std

    @property
    def name(self) -> str:
        return "bollinger_mr"

    def compute(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        sma = prices.rolling(window=self._window, min_periods=self._window).mean()
        std = prices.rolling(window=self._window, min_periods=self._window).std()
        std = std.replace(0, np.nan)

        z_score = (prices - sma) / (self._num_std * std)
        return -z_score


class RSIMeanReversion(Feature):
    """
    RSI-based mean-reversion signal.

    Formula: -(RSI(window) - 50) / 50
    Maps RSI to [-1, 1] range with mean-reversion sign convention.
    """

    def __init__(self, window: int = 14):
        self._window = window

    @property
    def name(self) -> str:
        return "rsi_mr"

    def compute(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        # Compute RSI
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(span=self._window, adjust=False).mean()
        avg_loss = loss.ewm(span=self._window, adjust=False).mean()
        avg_loss = avg_loss.replace(0, np.nan)

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return -(rsi - 50) / 50


class ShortTermReversal(Feature):
    """
    Short-term reversal signal.

    Formula: -Return(t-window, t) / Vol(t, vol_window)
    Bets against recent short-term moves, normalized by volatility.
    """

    def __init__(self, window: int = 5, vol_window: int = 21):
        self._window = window
        self._vol_window = vol_window

    @property
    def name(self) -> str:
        return "short_term_reversal"

    def compute(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        recent_return = prices / prices.shift(self._window) - 1

        vol = returns.rolling(window=self._vol_window, min_periods=self._vol_window // 2).std() * np.sqrt(252)
        vol = vol.replace(0, np.nan)

        return -recent_return / vol
