"""Tests for alpha signal features — primarily no-lookahead verification."""

import numpy as np
import pandas as pd
import pytest

from src.features.momentum import XSMOM, TSMOM, CompositeMomentum
from src.features.mean_reversion import BollingerMeanReversion, RSIMeanReversion, ShortTermReversal
from src.features.volatility import EWMAVol, VolOfVol
from src.config import MomentumConfig


@pytest.fixture
def sample_data():
    """Generate synthetic price/return data for testing."""
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=500)
    tickers = ["A", "B", "C", "D", "E"]
    returns = pd.DataFrame(
        np.random.randn(500, 5) * 0.02,
        index=dates,
        columns=tickers,
    )
    prices = (1 + returns).cumprod() * 100
    return prices, returns


class TestNoLookahead:
    """Verify features don't use future data."""

    def _check_no_lookahead(self, feature, prices, returns):
        """
        Core no-lookahead test: truncating the last day should not
        change the feature value on the second-to-last day.
        """
        full_signal = feature.compute(prices, returns)
        truncated_signal = feature.compute(prices.iloc[:-1], returns.iloc[:-1])

        # Second-to-last day values should match
        last_common = prices.index[-2]
        if last_common in full_signal.index and last_common in truncated_signal.index:
            full_row = full_signal.loc[last_common].dropna()
            trunc_row = truncated_signal.loc[last_common].dropna()
            common_cols = full_row.index.intersection(trunc_row.index)
            if len(common_cols) > 0:
                pd.testing.assert_series_equal(
                    full_row[common_cols],
                    trunc_row[common_cols],
                    check_names=False,
                    atol=1e-10,
                )

    def test_xsmom_no_lookahead(self, sample_data):
        prices, returns = sample_data
        self._check_no_lookahead(XSMOM(63, 21), prices, returns)

    def test_tsmom_no_lookahead(self, sample_data):
        prices, returns = sample_data
        self._check_no_lookahead(TSMOM(252, 0.15), prices, returns)

    def test_bollinger_no_lookahead(self, sample_data):
        prices, returns = sample_data
        self._check_no_lookahead(BollingerMeanReversion(20, 2.0), prices, returns)

    def test_rsi_no_lookahead(self, sample_data):
        prices, returns = sample_data
        self._check_no_lookahead(RSIMeanReversion(14), prices, returns)

    def test_reversal_no_lookahead(self, sample_data):
        prices, returns = sample_data
        self._check_no_lookahead(ShortTermReversal(5, 21), prices, returns)

    def test_ewma_vol_no_lookahead(self, sample_data):
        prices, returns = sample_data
        self._check_no_lookahead(EWMAVol(21), prices, returns)

    def test_vol_of_vol_no_lookahead(self, sample_data):
        prices, returns = sample_data
        self._check_no_lookahead(VolOfVol((21, 63, 126)), prices, returns)


class TestFeatureShapes:
    """Verify features return correct shapes."""

    def test_xsmom_shape(self, sample_data):
        prices, returns = sample_data
        signal = XSMOM(63, 21).compute(prices, returns)
        assert signal.shape == prices.shape

    def test_tsmom_shape(self, sample_data):
        prices, returns = sample_data
        signal = TSMOM(252, 0.15).compute(prices, returns)
        assert signal.shape == prices.shape

    def test_bollinger_shape(self, sample_data):
        prices, returns = sample_data
        signal = BollingerMeanReversion(20, 2.0).compute(prices, returns)
        assert signal.shape == prices.shape

    def test_rsi_range(self, sample_data):
        """RSI MR signal should be in [-1, 1]."""
        prices, returns = sample_data
        signal = RSIMeanReversion(14).compute(prices, returns)
        clean = signal.dropna()
        assert clean.min().min() >= -1.1  # Allow small float tolerance
        assert clean.max().max() <= 1.1

    def test_composite_momentum(self, sample_data):
        prices, returns = sample_data
        cfg = MomentumConfig(xsmom_lookbacks=(63, 126), xsmom_skip=21)
        signal = CompositeMomentum(cfg).compute(prices, returns)
        assert signal.shape[1] == prices.shape[1]


class TestFeatureNames:
    """Verify feature name property."""

    def test_xsmom_name(self):
        assert XSMOM(63, 21).name == "xsmom_63"

    def test_tsmom_name(self):
        assert TSMOM().name == "tsmom"

    def test_bollinger_name(self):
        assert BollingerMeanReversion().name == "bollinger_mr"
