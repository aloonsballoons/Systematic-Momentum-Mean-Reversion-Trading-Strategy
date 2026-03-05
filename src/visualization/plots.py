"""Visualization: equity curves, rolling Sharpe, quintile analysis, IC plots."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.backtest.engine import BacktestResult


def plot_equity_curve(
    result: BacktestResult,
    benchmark: pd.Series | None = None,
    title: str = "Strategy Equity Curve",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot equity curve with drawdown subplot (log scale).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1],
                                     sharex=True, gridspec_kw={"hspace": 0.05})

    # Equity curve
    ax1.semilogy(result.equity_curve.index, result.equity_curve.values,
                 label="Strategy (net)", color="#1f77b4", linewidth=1.5)

    if benchmark is not None:
        bench_aligned = benchmark.reindex(result.equity_curve.index).dropna()
        bench_norm = bench_aligned / bench_aligned.iloc[0] * result.equity_curve.iloc[0]
        ax1.semilogy(bench_norm.index, bench_norm.values,
                     label="Benchmark", color="#ff7f0e", alpha=0.7, linewidth=1)

    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.set_ylabel("Equity (log scale)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Drawdown
    equity = result.equity_curve
    running_max = equity.cummax()
    drawdown = equity / running_max - 1

    ax2.fill_between(drawdown.index, drawdown.values, 0,
                     color="#d62728", alpha=0.4, label="Drawdown")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_rolling_sharpe(
    returns: pd.Series,
    window: int = 252,
    risk_free_rate: float = 0.02,
    title: str = "Rolling 1-Year Sharpe Ratio",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot rolling Sharpe ratio."""
    excess = returns - risk_free_rate / 252
    rolling_mean = excess.rolling(window, min_periods=window // 2).mean() * 252
    rolling_std = returns.rolling(window, min_periods=window // 2).std() * np.sqrt(252)
    rolling_sharpe = rolling_mean / rolling_std.replace(0, np.nan)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, color="#1f77b4", linewidth=1)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.axhline(y=rolling_sharpe.mean(), color="#ff7f0e", linestyle="--",
               linewidth=1, label=f"Mean = {rolling_sharpe.mean():.2f}")
    ax.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                    where=rolling_sharpe > 0, alpha=0.15, color="green")
    ax.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                    where=rolling_sharpe < 0, alpha=0.15, color="red")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_quintile_returns(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
    title: str = "Signal Quintile Returns",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Quintile analysis: average forward return by signal quintile.
    A monotonic pattern indicates signal has predictive power.
    """
    common_idx = signal.index.intersection(forward_returns.index)
    common_cols = signal.columns.intersection(forward_returns.columns)
    sig = signal.loc[common_idx, common_cols]
    fwd = forward_returns.loc[common_idx, common_cols]

    # Assign quintiles cross-sectionally
    quintiles = sig.rank(axis=1, pct=True).apply(
        lambda row: pd.cut(row, bins=5, labels=[1, 2, 3, 4, 5]), axis=1
    )

    # Average forward return per quintile
    avg_returns = {}
    for q in range(1, 6):
        mask = quintiles == q
        avg_returns[f"Q{q}"] = (fwd * mask).sum(axis=1) / mask.sum(axis=1)

    avg_df = pd.DataFrame(avg_returns)
    annual_returns = avg_df.mean() * 252

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d62728", "#ff7f0e", "#7f7f7f", "#2ca02c", "#1f77b4"]
    ax.bar(annual_returns.index, annual_returns.values * 100, color=colors)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Signal Quintile (1=lowest, 5=highest)")
    ax.set_ylabel("Annualized Return (%)")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # Add long-short return
    ls_return = annual_returns["Q5"] - annual_returns["Q1"]
    ax.annotate(f"Q5-Q1 spread: {ls_return * 100:.1f}%",
                xy=(0.98, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_ic_timeseries(
    ic_series: pd.Series,
    signal_name: str = "Signal",
    window: int = 63,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot IC time series with rolling mean."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.bar(ic_series.index, ic_series.values, alpha=0.3, color="#1f77b4", width=1)
    rolling_ic = ic_series.rolling(window, min_periods=window // 2).mean()
    ax.plot(rolling_ic.index, rolling_ic.values, color="#d62728", linewidth=1.5,
            label=f"Rolling {window}d mean")
    ax.axhline(y=0, color="black", linewidth=0.5)

    mean_ic = ic_series.mean()
    ax.axhline(y=mean_ic, color="#ff7f0e", linestyle="--",
               label=f"Overall IC = {mean_ic:.4f}")

    ax.set_title(f"Information Coefficient — {signal_name}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Rank IC")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_cost_sensitivity(
    returns_by_cost: dict[str, pd.Series],
    title: str = "Cost Sensitivity Analysis",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot equity curves under different cost assumptions."""
    fig, ax = plt.subplots(figsize=(14, 6))

    for label, ret in returns_by_cost.items():
        equity = (1 + ret).cumprod()
        ax.semilogy(equity.index, equity.values, label=label, linewidth=1.2)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Equity (log scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
