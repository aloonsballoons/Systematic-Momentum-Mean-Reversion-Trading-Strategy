# Systematic Momentum/Mean-Reversion Trading Strategy

A complete quant research pipeline implementing momentum and mean-reversion signals on ~100 liquid US large-cap equities over 20 years of daily data, with realistic transaction costs, walk-forward validation, and statistical significance testing.

## Quick Start

```bash
# Set up isolated virtual environment (requires uv)
bash setup_env.sh
source .venv/bin/activate

# Run full backtest (downloads data on first run)
python run_backtest.py

# Run with walk-forward validation (primary reported result)
python run_backtest.py --walkforward

# Skip plot generation
python run_backtest.py --no-plot

# Custom config
python run_backtest.py --config my_config.yaml

# Run tests
pytest tests/ -v
```

## Methodology

### Alpha Signals

| Signal | Formula | Type | Source |
|--------|---------|------|--------|
| XSMOM | `Return(t-S-L, t-S) / Vol(t, L)` | Momentum | Jegadeesh & Titman (1993) |
| TSMOM | `sign(Return(t-L, t)) × vol_target / Vol(t, L)` | Momentum | Moskowitz et al. (2012) |
| Composite MOM | `mean(rank(XSMOM_L))` for L in {63, 126, 252} | Momentum | - |
| Bollinger MR | `-(Price - SMA) / (k × StdDev)` | Mean Reversion | Bollinger (1992) |
| RSI MR | `-(RSI(14) - 50) / 50` | Mean Reversion | Wilder (1978) |
| Short-Term Reversal | `-Return(t-5, t) / Vol(t, 21)` | Mean Reversion | Jegadeesh (1990) |
| Vol-of-Vol | `Std(RealizedVol across windows)` | Volatility | - |
| EWMA Vol | `sqrt(EWMA(returns², halflife))` | Volatility | RiskMetrics (1996) |

All signals use published academic formulas — no data mining on this dataset.

### Signal Combination

**Equal-weight** combination (default). Fewest degrees of freedom = most robust out-of-sample. IC-weighted and PCA options available but flagged as additional DoF.

### Portfolio Construction

- **Signal-proportional** weights (baseline)
- Risk parity with signal tilt
- Mean-variance with turnover regularization
- Ledoit-Wolf shrinkage covariance estimation
- Constraints: 5% max position, 25% max sector, 1.0x max leverage

### Transaction Costs

- Fixed: 5 bps per side
- Market impact: 10 bps × √(turnover) (Almgren-Chriss square-root law)
- Minimum trade filter to avoid churn

### Risk Management

- Volatility targeting: scale exposure to 15% annualized target
- Drawdown circuit breaker: go flat when DD > 15%, stay flat 21 days
- All overlays use `.shift(1)` to avoid look-ahead bias

### Validation

- **Walk-forward**: 3-year train → 21-day purge → 1-year test, rolling
- **Purged K-fold CV**: 5-fold with 21-day purge gap
- **Deflated Sharpe Ratio**: adjusts for multiple testing

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| All params in `config.yaml` | Reviewer can count degrees of freedom (~17 total) |
| Equal-weight signal combination | Most robust OOS; resists overfitting |
| Walk-forward with purge gap | Never reports in-sample as results |
| Deflated Sharpe Ratio | Adjusts for multiple testing |
| 10 bps realistic costs + sensitivity | Bridges gap between backtest and reality |
| Published academic signals only | Not data-mined from this dataset |
| 1-day execution lag | Realistic trading latency |

## Project Structure

```
├── config.yaml                 # All tunable parameters
├── run_backtest.py             # Single entry point
├── src/
│   ├── config.py               # Frozen dataclass config
│   ├── data/                   # Download, clean, universe
│   ├── features/               # Alpha signals (momentum, MR, vol)
│   ├── signals/                # Normalization and combination
│   ├── portfolio/              # Construction, risk model, constraints
│   ├── execution/              # Transaction cost model
│   ├── backtest/               # Engine, walk-forward, CV
│   ├── risk/                   # Vol targeting, circuit breaker
│   ├── analytics/              # Performance, statistics, reports
│   └── visualization/          # Plots and tearsheet
├── tests/                      # Unit + integration tests
└── notebooks/                  # Research notebooks (1-4)
```

## Realistic Expectations

- **Sharpe**: 0.3–0.8 (higher suggests a bug)
- **Max DD**: 15–30%
- **Annual turnover**: 500–1500%
- **OOS Sharpe**: ~0.5–0.8× in-sample Sharpe
- **DSR < 0.95** with 6 trials indicates genuine signal

If Sharpe > 2.0, there's a bug.

## Requirements

Python 3.10+ and [uv](https://docs.astral.sh/uv/). Key dependencies: numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib, plotly, yfinance, PyYAML.

```bash
# One-step setup (creates .venv and installs everything)
bash setup_env.sh
```
