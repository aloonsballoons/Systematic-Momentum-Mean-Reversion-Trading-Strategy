"""Vectorized backtest engine with 1-day execution lag."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

from src.config import BacktestConfig, ExecutionConfig
from src.execution.costs import CostModel


@dataclass
class BacktestResult:
    """Container for all backtest outputs."""

    gross_returns: pd.Series
    net_returns: pd.Series
    equity_curve: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    total_costs: pd.Series
    cost_per_period: pd.Series

    # Metadata
    start_date: str = ""
    end_date: str = ""
    n_assets: int = 0
    warmup_days: int = 0


class BacktestEngine:
    """
    Vectorized backtest engine for daily equity strategies.

    Critical implementation details:
    - 1-day execution lag: signals computed at close of day t
      are executed at close of day t+1 via target_weights.shift(1)
    - Warmup period: first N days are discarded for feature computation
    - Returns both gross and net (after costs) performance
    """

    def __init__(self, backtest_cfg: BacktestConfig, execution_cfg: ExecutionConfig):
        self.backtest_cfg = backtest_cfg
        self.cost_model = CostModel(execution_cfg)

    def run(
        self,
        target_weights: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> BacktestResult:
        """
        Run vectorized backtest.

        Parameters
        ----------
        target_weights : signal-based target weights (date × ticker)
        returns : asset returns (date × ticker)

        Returns
        -------
        BacktestResult with equity curve, returns, costs, etc.
        """
        # Align indices
        common_idx = target_weights.index.intersection(returns.index)
        common_cols = target_weights.columns.intersection(returns.columns)
        target_weights = target_weights.loc[common_idx, common_cols]
        returns = returns.loc[common_idx, common_cols]

        # CRITICAL: 1-day execution lag
        # Signals computed at close of day t → trade at close of day t+1
        executed_weights = target_weights.shift(1)

        # Apply warmup
        warmup = self.backtest_cfg.warmup_days
        executed_weights = executed_weights.iloc[warmup:]
        returns = returns.iloc[warmup:]

        # Fill NaN weights with 0 (no position)
        executed_weights = executed_weights.fillna(0)
        returns = returns.fillna(0)

        # Gross returns: sum of weight × return across assets
        gross_returns = (executed_weights * returns).sum(axis=1)

        # Compute costs
        # Drift-adjusted weights: after returns, weights shift
        prev_weights = executed_weights.shift(1).fillna(0)
        # Account for drift: prev_weight * (1 + return) / (1 + portfolio_return)
        port_ret = gross_returns.shift(1).fillna(0)
        drifted_weights = prev_weights.mul(1 + returns.shift(1).fillna(0)).div(
            1 + port_ret, axis=0
        )
        drifted_weights = drifted_weights.fillna(0)

        # Cost of trading from drifted to new target
        cost_matrix = self.cost_model.compute_costs(executed_weights, drifted_weights)
        total_costs = cost_matrix.sum(axis=1)

        # Net returns
        net_returns = gross_returns - total_costs

        # Equity curve (cumulative net returns)
        equity_curve = (1 + net_returns).cumprod()

        # Turnover
        turnover = self.cost_model.compute_turnover(executed_weights)

        result = BacktestResult(
            gross_returns=gross_returns,
            net_returns=net_returns,
            equity_curve=equity_curve,
            weights=executed_weights,
            turnover=turnover,
            total_costs=total_costs,
            cost_per_period=total_costs,
            start_date=str(executed_weights.index[0].date()),
            end_date=str(executed_weights.index[-1].date()),
            n_assets=len(common_cols),
            warmup_days=warmup,
        )

        logger.info(
            f"Backtest complete: {result.start_date} to {result.end_date}, "
            f"{result.n_assets} assets, "
            f"final equity = {equity_curve.iloc[-1]:.4f}"
        )
        return result
