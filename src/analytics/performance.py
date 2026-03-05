"""Performance metrics: Sharpe, Sortino, max DD, Calmar, VaR, CVaR, etc."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Container for all performance metrics."""

    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    skewness: float
    kurtosis: float
    var_95: float
    cvar_95: float
    avg_turnover: float
    annual_turnover: float
    avg_daily_cost_bps: float
    annual_cost_bps: float

    def to_dict(self) -> dict[str, float]:
        return {
            "Total Return": f"{self.total_return:.2%}",
            "Annual Return": f"{self.annual_return:.2%}",
            "Annual Volatility": f"{self.annual_volatility:.2%}",
            "Sharpe Ratio": f"{self.sharpe_ratio:.3f}",
            "Sortino Ratio": f"{self.sortino_ratio:.3f}",
            "Max Drawdown": f"{self.max_drawdown:.2%}",
            "Calmar Ratio": f"{self.calmar_ratio:.3f}",
            "Win Rate": f"{self.win_rate:.2%}",
            "Profit Factor": f"{self.profit_factor:.3f}",
            "Skewness": f"{self.skewness:.3f}",
            "Kurtosis": f"{self.kurtosis:.3f}",
            "VaR (95%)": f"{self.var_95:.4f}",
            "CVaR (95%)": f"{self.cvar_95:.4f}",
            "Avg Daily Turnover": f"{self.avg_turnover:.4f}",
            "Annual Turnover": f"{self.annual_turnover:.1%}",
            "Avg Daily Cost (bps)": f"{self.avg_daily_cost_bps:.2f}",
            "Annual Cost (bps)": f"{self.annual_cost_bps:.1f}",
        }


def compute_metrics(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    costs: pd.Series | None = None,
    risk_free_rate: float = 0.02,
    confidence_level: float = 0.95,
) -> PerformanceMetrics:
    """Compute all performance metrics from a return series."""
    ret = returns.dropna()
    n_days = len(ret)
    n_years = n_days / 252

    # Basic returns
    total_return = (1 + ret).prod() - 1
    annual_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
    annual_vol = ret.std() * np.sqrt(252)

    # Sharpe ratio
    excess = ret.mean() * 252 - risk_free_rate
    sharpe = excess / annual_vol if annual_vol > 0 else 0

    # Sortino ratio (downside deviation)
    downside = ret[ret < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else annual_vol
    sortino = excess / downside_std if downside_std > 0 else 0

    # Max drawdown
    equity = (1 + ret).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()

    # Calmar
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

    # Win rate and profit factor
    wins = ret[ret > 0]
    losses = ret[ret < 0]
    win_rate = len(wins) / max(len(ret), 1)
    profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")

    # Distribution
    skew = ret.skew()
    kurt = ret.kurtosis()

    # VaR and CVaR
    alpha = 1 - confidence_level
    var = ret.quantile(alpha)
    cvar = ret[ret <= var].mean() if (ret <= var).any() else var

    # Turnover and costs
    avg_turnover = turnover.mean() if turnover is not None else 0
    annual_to = avg_turnover * 252
    avg_cost = costs.mean() * 10_000 if costs is not None else 0
    annual_cost = avg_cost * 252

    return PerformanceMetrics(
        total_return=total_return,
        annual_return=annual_return,
        annual_volatility=annual_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        calmar_ratio=calmar,
        win_rate=win_rate,
        profit_factor=profit_factor,
        skewness=skew,
        kurtosis=kurt,
        var_95=var,
        cvar_95=cvar,
        avg_turnover=avg_turnover,
        annual_turnover=annual_to,
        avg_daily_cost_bps=avg_cost,
        annual_cost_bps=annual_cost,
    )
