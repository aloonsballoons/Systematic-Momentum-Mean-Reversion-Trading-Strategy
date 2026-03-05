"""End-to-end integration smoke test with synthetic data."""

import numpy as np
import pandas as pd
import pytest

from src.config import StrategyConfig, load_config
from src.data.cleaner import DataCleaner
from src.features.registry import build_features
from src.signals.combiner import SignalCombiner
from src.portfolio.constructor import PortfolioConstructor
from src.backtest.engine import BacktestEngine
from src.risk.manager import RiskManager
from src.analytics.performance import compute_metrics
from src.analytics.statistics import (
    lo_adjusted_sharpe_test,
    deflated_sharpe_ratio,
    adf_test,
    information_coefficient,
)
from src.analytics.report import generate_report


@pytest.fixture
def synthetic_data():
    """Generate realistic synthetic data for integration testing."""
    np.random.seed(42)
    n_days = 1000
    n_tickers = 20
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    tickers = [f"TICK{i:02d}" for i in range(n_tickers)]

    # Generate correlated returns
    corr = np.eye(n_tickers) * 0.5 + 0.5 / n_tickers
    chol = np.linalg.cholesky(corr)
    raw = np.random.randn(n_days, n_tickers) @ chol.T * 0.015

    returns = pd.DataFrame(raw, index=dates, columns=tickers)
    prices = (1 + returns).cumprod() * 100

    return prices, returns


@pytest.fixture
def config():
    """Default config for testing."""
    return StrategyConfig()


class TestFullPipeline:
    """End-to-end pipeline test."""

    def test_pipeline_runs(self, synthetic_data, config):
        """Full pipeline should run without errors on synthetic data."""
        prices, returns = synthetic_data

        # Step 1: Compute features
        features = build_features(config.features)
        signal_dict = {}
        for feature in features:
            signal_dict[feature.name] = feature.compute(prices, returns)

        # Step 2: Combine signals
        combiner = SignalCombiner(config.signals)
        combined = combiner.combine(signal_dict, returns)
        assert combined.shape[1] == prices.shape[1]

        # Step 3: Portfolio construction
        constructor = PortfolioConstructor(config.portfolio)
        weights = constructor.construct(combined, returns)
        assert weights.shape == combined.shape

        # Step 4: Risk management
        risk_mgr = RiskManager(config.risk)
        adjusted = risk_mgr.apply(weights, returns)

        # Step 5: Backtest
        engine = BacktestEngine(config.backtest, config.execution)
        result = engine.run(adjusted, returns)

        # Step 6: Analytics
        metrics = compute_metrics(result.net_returns, result.turnover, result.total_costs)
        assert metrics.sharpe_ratio is not None
        assert not np.isnan(metrics.sharpe_ratio)

        # Step 7: Report
        report = generate_report(result)
        assert len(report) > 100
        assert "Sharpe" in report

    def test_weights_within_constraints(self, synthetic_data, config):
        """Constructed weights should respect position limits."""
        prices, returns = synthetic_data

        features = build_features(config.features)
        signal_dict = {}
        for feature in features:
            signal_dict[feature.name] = feature.compute(prices, returns)

        combiner = SignalCombiner(config.signals)
        combined = combiner.combine(signal_dict, returns)

        constructor = PortfolioConstructor(config.portfolio)
        weights = constructor.construct(combined, returns)

        # Check position limits
        max_pos = weights.abs().max().max()
        assert max_pos <= config.portfolio.max_position_weight + 1e-10

        # Check leverage
        max_leverage = weights.abs().sum(axis=1).max()
        assert max_leverage <= config.portfolio.max_leverage + 1e-10


class TestStatisticalTests:
    """Test statistical test functions."""

    def test_lo_adjusted_sharpe(self, synthetic_data):
        _, returns = synthetic_data
        port_ret = returns.mean(axis=1)
        result = lo_adjusted_sharpe_test(port_ret)

        assert "sharpe" in result
        assert "adjusted_sharpe" in result
        assert "p_value" in result
        assert 0 <= result["p_value"] <= 1

    def test_deflated_sharpe_ratio(self):
        result = deflated_sharpe_ratio(
            sharpe_observed=0.8,
            n_trials=6,
            n_obs=2520,
            skewness=-0.5,
            kurtosis=4.0,
        )
        assert "dsr" in result
        assert 0 <= result["dsr"] <= 1

    def test_adf_test(self, synthetic_data):
        _, returns = synthetic_data
        result = adf_test(returns.mean(axis=1))
        assert "p_value" in result
        assert "is_stationary" in result

    def test_information_coefficient(self, synthetic_data):
        prices, returns = synthetic_data
        # Use random signal
        signal = pd.DataFrame(
            np.random.randn(*prices.shape),
            index=prices.index,
            columns=prices.columns,
        )
        fwd = returns.shift(-1)
        ic = information_coefficient(signal, fwd)
        assert len(ic) == len(prices)
        # Random signal IC should be close to 0
        assert abs(ic.mean()) < 0.1
