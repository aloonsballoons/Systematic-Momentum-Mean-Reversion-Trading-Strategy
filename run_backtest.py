#!/usr/bin/env python3
"""
Single entry point for the momentum/mean-reversion backtest pipeline.

Usage:
    python run_backtest.py                          # Use default config.yaml
    python run_backtest.py --config my_config.yaml  # Custom config
    python run_backtest.py --no-download            # Skip data download (use cache)
    python run_backtest.py --walkforward            # Run walk-forward validation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import pandas as pd
from loguru import logger

from src.config import load_config, StrategyConfig
from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner
from src.data.universe import UniverseProvider
from src.features.registry import build_features
from src.signals.combiner import SignalCombiner
from src.portfolio.constructor import PortfolioConstructor
from src.backtest.engine import BacktestEngine, BacktestResult
from src.backtest.walkforward import WalkForwardOptimizer
from src.risk.manager import RiskManager
from src.analytics.performance import compute_metrics
from src.analytics.report import generate_report
from src.visualization.tearsheet import create_tearsheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run momentum/MR backtest")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip data download, use cached data only")
    parser.add_argument("--walkforward", action="store_true",
                        help="Run walk-forward validation")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory for output files")
    return parser.parse_args()


def load_data(cfg: StrategyConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load, clean, and filter market data."""
    # Get universe
    universe = UniverseProvider(cfg.data)
    tickers = universe.get_tickers()

    # Download / load from cache
    loader = DataLoader(cfg.data)
    raw_data = loader.load_universe(tickers)

    # Build matrices
    prices = loader.build_price_matrix(raw_data)
    volumes = loader.build_volume_matrix(raw_data)

    # Filter universe by liquidity
    valid_tickers = universe.filter_universe(prices, volumes)
    prices = prices[valid_tickers]
    volumes = volumes[valid_tickers]

    # Clean
    cleaner = DataCleaner(cfg.data)
    prices, returns = cleaner.clean(prices, volumes)

    return prices, returns, volumes


def compute_signals(
    cfg: StrategyConfig,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Compute all features and combine into a single signal."""
    features = build_features(cfg.features)

    signal_dict = {}
    for feature in features:
        logger.info(f"Computing feature: {feature.name}")
        signal_dict[feature.name] = feature.compute(prices, returns)

    # Combine signals
    combiner = SignalCombiner(cfg.signals)
    combined = combiner.combine(signal_dict, returns)
    return combined


def smooth_weights(weights: pd.DataFrame, blend: float) -> pd.DataFrame:
    """Exponentially smooth weights to reduce turnover.

    w_t = blend * target_t + (1 - blend) * w_{t-1}
    Applied as the LAST step before the engine to capture all sources of turnover.
    """
    if blend >= 1.0:
        return weights
    smoothed = weights.copy()
    for i in range(1, len(smoothed)):
        smoothed.iloc[i] = (
            blend * weights.iloc[i] + (1 - blend) * smoothed.iloc[i - 1]
        )
    return smoothed


def run_full_backtest(
    cfg: StrategyConfig,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    combined_signal: pd.DataFrame,
) -> BacktestResult:
    """Run the full backtest pipeline."""
    # Portfolio construction
    constructor = PortfolioConstructor(cfg.portfolio)
    target_weights = constructor.construct(combined_signal, returns)

    # Risk management: vol targeting + iterative drawdown breaker
    risk_mgr = RiskManager(cfg.risk)
    risk_adjusted = risk_mgr.apply(target_weights, returns)

    # Smooth final weights to control turnover (last step before engine)
    final_weights = smooth_weights(risk_adjusted, cfg.portfolio.weight_blend)

    # Run backtest
    engine = BacktestEngine(cfg.backtest, cfg.execution)
    result = engine.run(final_weights, returns)

    return result


def run_walkforward(
    cfg: StrategyConfig,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
) -> BacktestResult:
    """Run walk-forward validation."""
    wf = WalkForwardOptimizer(cfg.backtest.walk_forward)
    windows = wf.generate_windows(len(prices))

    oos_returns_list = []

    for i, window in enumerate(windows):
        logger.info(f"Walk-forward window {i + 1}/{len(windows)}")

        # Train period: compute signals and optimize
        train_prices = prices.iloc[window.train_start:window.train_end]
        train_returns = returns.iloc[window.train_start:window.train_end]

        # Compute signals on train period
        combined = compute_signals(cfg, train_prices, train_returns)

        # Test period: apply trained signals
        test_prices = prices.iloc[window.test_start:window.test_end]
        test_returns = returns.iloc[window.test_start:window.test_end]

        # Re-compute features on test period
        test_combined = compute_signals(cfg, test_prices, test_returns)

        # Portfolio construction
        constructor = PortfolioConstructor(cfg.portfolio)
        test_weights = constructor.construct(test_combined, test_returns)

        # Risk management
        risk_mgr = RiskManager(cfg.risk)
        test_adjusted = risk_mgr.apply(test_weights, test_returns)

        # Run backtest on test period (no warmup for OOS windows)
        from src.config import BacktestConfig
        import dataclasses
        test_bt_cfg = dataclasses.replace(cfg.backtest, warmup_days=0)
        engine = BacktestEngine(test_bt_cfg, cfg.execution)
        test_result = engine.run(test_adjusted, test_returns)

        oos_returns_list.append(test_result.net_returns)

    # Concatenate OOS returns
    oos_returns = wf.concatenate_oos(oos_returns_list)
    oos_equity = (1 + oos_returns).cumprod()

    # Build a BacktestResult from concatenated OOS
    from src.execution.costs import CostModel
    cost_model = CostModel(cfg.execution)

    return BacktestResult(
        gross_returns=oos_returns,  # Approximate
        net_returns=oos_returns,
        equity_curve=oos_equity,
        weights=pd.DataFrame(),  # Not tracked in WF
        turnover=pd.Series(0, index=oos_returns.index),
        total_costs=pd.Series(0, index=oos_returns.index),
        cost_per_period=pd.Series(0, index=oos_returns.index),
        start_date=str(oos_returns.index[0].date()),
        end_date=str(oos_returns.index[-1].date()),
        n_assets=prices.shape[1],
    )


def main():
    args = parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    cfg = load_config(args.config)
    logger.info("Configuration loaded")

    # Step 1: Load data
    logger.info("=== Step 1: Loading Data ===")
    prices, returns, volumes = load_data(cfg)

    # Step 2: Compute signals
    logger.info("=== Step 2: Computing Signals ===")
    combined_signal = compute_signals(cfg, prices, returns)

    if args.walkforward:
        # Walk-forward validation
        logger.info("=== Running Walk-Forward Validation ===")
        result = run_walkforward(cfg, prices, returns)
    else:
        # Full in-sample backtest
        logger.info("=== Step 3: Running Backtest ===")
        result = run_full_backtest(cfg, prices, returns, combined_signal)

    # Step 4: Generate report
    logger.info("=== Step 4: Generating Report ===")
    report = generate_report(result, cfg.analytics.risk_free_rate)
    print(report)

    # Save report
    report_path = output_dir / "report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")

    # Step 5: Generate tearsheet
    if not args.no_plot:
        logger.info("=== Step 5: Generating Tearsheet ===")
        tearsheet_path = output_dir / "tearsheet.png"
        create_tearsheet(result, save_path=str(tearsheet_path),
                         risk_free_rate=cfg.analytics.risk_free_rate)
        logger.info(f"Tearsheet saved to {tearsheet_path}")

    logger.info("Pipeline complete!")
    return result


if __name__ == "__main__":
    main()
