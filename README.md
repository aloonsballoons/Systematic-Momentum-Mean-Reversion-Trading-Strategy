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

| Signal | Formula | Type | Source | Active |
|--------|---------|------|--------|--------|
| Composite MOM | `mean(rank(XSMOM_L))` for L in {63, 126, 252} | Momentum | Jegadeesh & Titman (1993) | Yes |
| TSMOM | `sign(Return(t-L, t)) × vol_target / Vol(t, L)` | Momentum | Moskowitz et al. (2012) | No |
| Bollinger MR | `-(Price - SMA) / (k × StdDev)` | Mean Reversion | Bollinger (1992) | Yes |
| Short-Term Reversal | `-Return(t-5, t) / Vol(t, 21)` | Mean Reversion | Jegadeesh (1990) | Yes |
| XSMOM | `Return(t-S-L, t-S) / Vol(t, L)` | Momentum | Jegadeesh & Titman (1993) | No (captured by Composite) |
| RSI MR | `-(RSI(14) - 50) / 50` | Mean Reversion | Wilder (1978) | No (too similar to Bollinger) |
| Vol-of-Vol | `Std(RealizedVol across windows)` | Volatility | - | No (non-directional) |
| EWMA Vol | `sqrt(EWMA(returns², halflife))` | Volatility | RiskMetrics (1996) | No (non-directional) |

All signals use published academic formulas — no data mining on this dataset. Active signals span 5d to 252d horizons for multi-frequency diversification.

### Signal Combination

**Equal-weight** combination (default). Fewest degrees of freedom = most robust out-of-sample. IC-weighted and PCA options available but flagged as additional DoF.

### Portfolio Construction

- **Signal-proportional** weights (baseline)
- Risk parity with signal tilt
- Mean-variance with turnover regularization
- Ledoit-Wolf shrinkage covariance estimation
- Constraints: 5% max position, 25% max sector, 2.0x max leverage (long-short)
- Weekly rebalancing with exponential weight blending to control turnover

### Transaction Costs

- Fixed: 5 bps per side
- Market impact: 2 bps × √(turnover) (Almgren-Chriss square-root law, calibrated for liquid large-caps)
- Minimum trade filter to avoid churn

### Risk Management

- Volatility targeting: scale exposure to 15% annualized target (126-day lookback)
- Drawdown circuit breaker: go flat when DD > 50% (catastrophic backstop), stay flat 21 days, 63-day grace period
- All overlays use `.shift(1)` to avoid look-ahead bias

### Turnover Control

- **Weekly rebalancing**: integrated into `smooth_weights()` — on non-rebalance days, weights are held exactly constant (zero turnover)
- **Exponential blending**: `w_t = 0.05 × target + 0.95 × w_{t-1}` on rebalance days only
- Weight smoothing applied AFTER all risk overlays (vol targeting, drawdown breaker)

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
| 2 bps market impact (large-cap calibration) | Realistic for liquid universe |
| Published academic signals only | Not data-mined from this dataset |
| 1-day execution lag | Realistic trading latency |
| Weekly rebalancing + weight blending | Controls turnover without separate filter step |

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
- **Annual turnover**: 150–500% (with weekly rebalancing)
- **OOS Sharpe**: ~0.5–0.8× in-sample Sharpe
- **DSR < 0.95** with 6 trials indicates genuine signal

If Sharpe > 2.0, there's a bug.

## Requirements

Python 3.10+ and [uv](https://docs.astral.sh/uv/). Key dependencies: numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib, plotly, yfinance, PyYAML.

```bash
# One-step setup (creates .venv and installs everything)
bash setup_env.sh
```
