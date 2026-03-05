"""Report generation: summary tables and formatted output."""

from __future__ import annotations

import pandas as pd
from tabulate import tabulate

from src.analytics.performance import PerformanceMetrics, compute_metrics
from src.analytics.statistics import (
    adf_test,
    deflated_sharpe_ratio,
    lo_adjusted_sharpe_test,
)
from src.backtest.engine import BacktestResult


def generate_report(
    result: BacktestResult,
    risk_free_rate: float = 0.02,
    n_trials: int = 6,
) -> str:
    """
    Generate full performance report.

    Parameters
    ----------
    result : backtest result
    risk_free_rate : for Sharpe calculation
    n_trials : number of strategy variants tested (for DSR)

    Returns
    -------
    Formatted report string
    """
    # Performance metrics
    metrics = compute_metrics(
        result.net_returns,
        turnover=result.turnover,
        costs=result.total_costs,
        risk_free_rate=risk_free_rate,
    )

    # Statistical tests
    lo_test = lo_adjusted_sharpe_test(result.net_returns, risk_free_rate)
    dsr = deflated_sharpe_ratio(
        sharpe_observed=metrics.sharpe_ratio,
        n_trials=n_trials,
        n_obs=len(result.net_returns),
        skewness=metrics.skewness,
        kurtosis=metrics.kurtosis + 3,  # Convert excess to raw
    )
    adf = adf_test(result.net_returns)

    # Build report
    lines = []
    lines.append("=" * 70)
    lines.append("STRATEGY PERFORMANCE REPORT")
    lines.append("=" * 70)
    lines.append(f"Period: {result.start_date} to {result.end_date}")
    lines.append(f"Assets: {result.n_assets}")
    lines.append("")

    # Performance table
    lines.append("--- Performance Metrics ---")
    perf_table = [[k, v] for k, v in metrics.to_dict().items()]
    lines.append(tabulate(perf_table, headers=["Metric", "Value"], tablefmt="simple"))
    lines.append("")

    # Statistical tests
    lines.append("--- Statistical Tests ---")
    stat_table = [
        ["Lo-Adjusted Sharpe", f"{lo_test['adjusted_sharpe']:.3f}"],
        ["Lo Sharpe p-value", f"{lo_test['p_value']:.4f}"],
        ["Autocorrelation Factor (η)", f"{lo_test['eta']:.3f}"],
        ["Deflated Sharpe Ratio", f"{dsr['dsr']:.4f}"],
        ["DSR p-value", f"{dsr['p_value']:.4f}"],
        ["Expected Max Sharpe (random)", f"{dsr['expected_max_sharpe']:.3f}"],
        ["N trials tested", f"{dsr['n_trials']}"],
        ["ADF Statistic", f"{adf['adf_statistic']:.3f}"],
        ["ADF p-value", f"{adf['p_value']:.4f}"],
        ["Returns Stationary", "Yes" if adf['is_stationary'] else "No"],
    ]
    lines.append(tabulate(stat_table, headers=["Test", "Value"], tablefmt="simple"))
    lines.append("")

    # Sanity checks
    lines.append("--- Sanity Checks ---")
    if metrics.sharpe_ratio > 2.0:
        lines.append("⚠ WARNING: Sharpe > 2.0 — likely a bug or overfitting!")
    if metrics.max_drawdown < -0.50:
        lines.append("⚠ WARNING: Max drawdown > 50% — extreme risk")
    if metrics.annual_turnover > 30:
        lines.append("⚠ WARNING: Annual turnover > 3000% — excessive trading")

    lines.append("=" * 70)

    return "\n".join(lines)


def generate_comparison_table(
    results: dict[str, BacktestResult],
    risk_free_rate: float = 0.02,
) -> str:
    """Compare multiple strategy variants side-by-side."""
    rows = []
    for name, result in results.items():
        m = compute_metrics(
            result.net_returns,
            turnover=result.turnover,
            costs=result.total_costs,
            risk_free_rate=risk_free_rate,
        )
        rows.append({
            "Strategy": name,
            "Return": f"{m.annual_return:.2%}",
            "Vol": f"{m.annual_volatility:.2%}",
            "Sharpe": f"{m.sharpe_ratio:.3f}",
            "Sortino": f"{m.sortino_ratio:.3f}",
            "Max DD": f"{m.max_drawdown:.2%}",
            "Calmar": f"{m.calmar_ratio:.3f}",
            "Turnover": f"{m.annual_turnover:.0%}",
        })

    df = pd.DataFrame(rows)
    return tabulate(df, headers="keys", tablefmt="simple", showindex=False)
