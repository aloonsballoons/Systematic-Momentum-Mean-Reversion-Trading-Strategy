"""Full-page strategy tearsheet: single-figure summary."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

from src.analytics.performance import compute_metrics
from src.analytics.statistics import lo_adjusted_sharpe_test, deflated_sharpe_ratio
from src.backtest.engine import BacktestResult


def create_tearsheet(
    result: BacktestResult,
    benchmark: pd.Series | None = None,
    risk_free_rate: float = 0.02,
    n_trials: int = 6,
    title: str = "Strategy Tearsheet",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Create a comprehensive single-page tearsheet.

    Layout:
    - Top left: equity curve with drawdown
    - Top right: key metrics table
    - Bottom left: rolling Sharpe
    - Bottom right: monthly returns heatmap
    """
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Compute metrics
    metrics = compute_metrics(
        result.net_returns, result.turnover, result.total_costs,
        risk_free_rate=risk_free_rate,
    )
    lo = lo_adjusted_sharpe_test(result.net_returns, risk_free_rate)
    dsr = deflated_sharpe_ratio(
        metrics.sharpe_ratio, n_trials, len(result.net_returns),
        metrics.skewness, metrics.kurtosis + 3,
    )

    # --- 1. Equity curve + drawdown ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(result.equity_curve.index, result.equity_curve.values,
                 color="#1f77b4", linewidth=1.5, label="Strategy (net)")
    if benchmark is not None:
        b = benchmark.reindex(result.equity_curve.index).dropna()
        b = b / b.iloc[0]
        ax1.semilogy(b.index, b.values, color="#ff7f0e", alpha=0.7, label="Benchmark")
    ax1.set_title("Equity Curve (log scale)", fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Drawdown inset
    equity = result.equity_curve
    dd = equity / equity.cummax() - 1
    ax1b = ax1.twinx()
    ax1b.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.2)
    ax1b.set_ylabel("Drawdown", color="#d62728", fontsize=9)
    ax1b.set_ylim(dd.min() * 1.5, 0.05)

    # --- 2. Metrics table ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    table_data = [
        ["Annual Return", f"{metrics.annual_return:.2%}"],
        ["Annual Vol", f"{metrics.annual_volatility:.2%}"],
        ["Sharpe Ratio", f"{metrics.sharpe_ratio:.3f}"],
        ["Lo-Adj Sharpe", f"{lo['adjusted_sharpe']:.3f}"],
        ["Sortino Ratio", f"{metrics.sortino_ratio:.3f}"],
        ["Max Drawdown", f"{metrics.max_drawdown:.2%}"],
        ["Calmar Ratio", f"{metrics.calmar_ratio:.3f}"],
        ["Win Rate", f"{metrics.win_rate:.1%}"],
        ["Profit Factor", f"{metrics.profit_factor:.2f}"],
        ["Skewness", f"{metrics.skewness:.3f}"],
        ["VaR (95%)", f"{metrics.var_95:.4f}"],
        ["Annual Turnover", f"{metrics.annual_turnover:.0%}"],
        ["Annual Cost (bps)", f"{metrics.annual_cost_bps:.0f}"],
        ["DSR", f"{dsr['dsr']:.3f}"],
        ["DSR p-value", f"{dsr['p_value']:.3f}"],
    ]
    table = ax2.table(cellText=table_data, colLabels=["Metric", "Value"],
                      loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.4)
    ax2.set_title("Key Metrics", fontweight="bold", pad=20)

    # --- 3. Rolling Sharpe ---
    ax3 = fig.add_subplot(gs[1, 0])
    excess = result.net_returns - risk_free_rate / 252
    rolling_mean = excess.rolling(252, min_periods=126).mean() * 252
    rolling_std = result.net_returns.rolling(252, min_periods=126).std() * np.sqrt(252)
    rolling_sharpe = rolling_mean / rolling_std.replace(0, np.nan)

    ax3.plot(rolling_sharpe.index, rolling_sharpe.values, color="#1f77b4", linewidth=1)
    ax3.axhline(y=0, color="black", linewidth=0.5)
    ax3.axhline(y=rolling_sharpe.mean(), color="#ff7f0e", linestyle="--",
                label=f"Mean = {rolling_sharpe.mean():.2f}")
    ax3.fill_between(rolling_sharpe.index, rolling_sharpe, 0,
                     where=rolling_sharpe > 0, alpha=0.15, color="green")
    ax3.fill_between(rolling_sharpe.index, rolling_sharpe, 0,
                     where=rolling_sharpe < 0, alpha=0.15, color="red")
    ax3.set_title("Rolling 1-Year Sharpe Ratio", fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # --- 4. Monthly returns heatmap ---
    ax4 = fig.add_subplot(gs[1, 1])
    monthly = result.net_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly_pivot = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })
    heatmap_data = monthly_pivot.pivot(index="year", columns="month", values="return")

    im = ax4.imshow(heatmap_data.values * 100, cmap="RdYlGn", aspect="auto",
                    vmin=-10, vmax=10)
    ax4.set_xticks(range(12))
    ax4.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], fontsize=8)
    ax4.set_yticks(range(len(heatmap_data.index)))
    ax4.set_yticklabels(heatmap_data.index, fontsize=8)
    ax4.set_title("Monthly Returns (%)", fontweight="bold")
    plt.colorbar(im, ax=ax4, shrink=0.8)

    # --- 5. Turnover and costs ---
    ax5 = fig.add_subplot(gs[2, 0])
    rolling_turnover = result.turnover.rolling(63, min_periods=21).mean() * 252
    ax5.plot(rolling_turnover.index, rolling_turnover.values, color="#2ca02c", linewidth=1)
    ax5.set_title("Rolling Annual Turnover", fontweight="bold")
    ax5.set_ylabel("Annual Turnover")
    ax5.grid(True, alpha=0.3)

    # --- 6. Return distribution ---
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.hist(result.net_returns.dropna().values * 100, bins=100, color="#1f77b4",
             alpha=0.7, edgecolor="white", linewidth=0.5)
    ax6.axvline(x=0, color="black", linewidth=0.5)
    ax6.set_title("Daily Return Distribution (%)", fontweight="bold")
    ax6.set_xlabel("Daily Return (%)")
    ax6.set_ylabel("Frequency")
    ax6.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
