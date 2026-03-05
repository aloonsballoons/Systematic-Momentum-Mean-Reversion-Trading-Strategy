"""Transaction cost model: fixed costs + Almgren-Chriss square-root market impact."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ExecutionConfig


class CostModel:
    """
    Realistic transaction cost model.

    Total cost per rebalance = fixed_cost + market_impact
    - Fixed: fixed_cost_bps × |turnover|
    - Market impact: impact_bps × sqrt(|turnover|) (Almgren-Chriss)
    """

    def __init__(self, cfg: ExecutionConfig):
        self.cfg = cfg

    def compute_costs(
        self,
        weights: pd.DataFrame,
        prev_weights: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute transaction costs from weight changes.

        Parameters
        ----------
        weights : target weights after rebalance
        prev_weights : weights before rebalance (after return drift)

        Returns
        -------
        DataFrame of costs (same shape, always non-negative)
        """
        trades = (weights - prev_weights).abs()

        # Filter minimum trade size
        trades = trades.where(trades >= self.cfg.min_trade_pct, 0)

        # Fixed cost component
        fixed = trades * (self.cfg.fixed_cost_bps / 10_000)

        # Market impact: sqrt of TOTAL turnover, not per-position
        # Almgren-Chriss: impact ∝ sqrt(total_participation)
        total_turnover = trades.sum(axis=1) / 2  # one-way turnover
        impact_per_day = np.sqrt(total_turnover) * (self.cfg.market_impact_bps / 10_000)

        # Distribute impact proportionally across positions for the cost matrix
        trade_share = trades.div(trades.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        impact = trade_share.mul(impact_per_day, axis=0)

        return fixed + impact

    def compute_turnover(
        self,
        weights: pd.DataFrame,
    ) -> pd.Series:
        """Compute one-way turnover per period."""
        weight_changes = weights.diff().abs()
        return weight_changes.sum(axis=1) / 2  # One-way

    def cost_summary(
        self,
        total_costs: pd.Series,
        turnover: pd.Series,
    ) -> dict[str, float]:
        """Summary statistics for costs and turnover."""
        return {
            "avg_daily_cost_bps": total_costs.mean() * 10_000,
            "annual_cost_bps": total_costs.mean() * 252 * 10_000,
            "avg_daily_turnover": turnover.mean(),
            "annual_turnover": turnover.mean() * 252,
        }
