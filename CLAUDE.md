# Systematic Momentum/Mean-Reversion Trading Strategy

## Overview
Quant research pipeline trading momentum and mean-reversion signals on ~100 liquid US large-cap equities (2005–2024) with realistic costs, walk-forward validation, and statistical testing. Entry point: `run_backtest.py`, all parameters in `config.yaml`.

## Pipeline
`config.yaml` → `src/data/` (load, clean, filter) → `src/features/` (momentum, MR, vol signals) → `src/signals/` (rank normalize, combine) → `src/portfolio/` (construct weights, apply constraints) → `src/risk/` (vol targeting, drawdown breaker) → `run_backtest.py:smooth_weights()` → `src/backtest/engine.py` (1-day lag, costs) → `src/analytics/` (metrics, statistical tests) → `src/visualization/` (tearsheet)

## Key Design Decisions
- All params in `config.yaml` — reviewer can count degrees of freedom instantly
- Equal-weight signal combination (fewest DOF, most robust OOS)
- Walk-forward with purge gap — never reports in-sample as results
- Deflated Sharpe Ratio — adjusts for multiple testing
- Published academic signals only — not data-mined from this dataset
- Vectorized backtest with 1-day execution lag via `target_weights.shift(1)`

## Signal Formulas
| Signal | Formula | Horizon |
|--------|---------|---------|
| XSMOM | `Return(t-S-L, t-S) / Vol(t, L)`, skip=21d | 63/126/252d |
| TSMOM | `sign(Return(t-L, t)) × vol_target / Vol(t, L)` | 252d |
| Composite MOM | `mean(rank(XSMOM_L))` for L ∈ {63, 126, 252} | Multi |
| Bollinger MR | `-(Price - SMA) / (k × StdDev)` | 20d |
| RSI MR | `-(RSI(14) - 50) / 50` | 14d |
| Short-Term Reversal | `-Return(t-5, t) / Vol(t, 21)` | 5d |
| Vol-of-Vol / EWMA Vol | Non-directional — risk model inputs only | — |

## Active Signal Selection (in `src/features/registry.py`)
CompositeMomentum, TSMOM, BollingerMR, ShortTermReversal — 4 signals spanning 5d to 252d.

## Verification
- SPY close on 2020-03-23 ≈ $222; NaN% < 1% per ticker
- Buy-and-hold test: portfolio return = equal-weight avg to 6 decimals
- Zero costs: gross = net exactly; execution lag: alternating signal verifies 1-day delay
- Realistic expectations: Sharpe 0.3–0.8, max DD 15–30%. If Sharpe > 2.0, there's a bug.
- `pytest tests/ -v` passes

## Known Pitfalls (do NOT re-introduce)

**Costs & Turnover:**
1. `market_impact_bps` applies to `sqrt(total_turnover)` — use 2.0 for liquid large-caps (10.0 generates ~500 bps/yr)
2. `vol_lookback: 126` minimum — shorter causes daily scale factor oscillation → massive turnover
3. `weight_blend: 0.15` (half-life ~4 days) — too low (0.05) kills alpha via lag, too high (0.30+) causes excess turnover
4. `max_leverage: 2.0` for long-short (gross exposure = |long| + |short|)

**Risk Management:**
5. Weight smoothing MUST be applied AFTER all risk overlays (vol targeting, drawdown breaker)
6. Drawdown breaker needs `grace_days: 63` after flat period to prevent immediate re-triggering
7. `max_drawdown: 0.50` as catastrophic backstop — 0.35 triggers on routine drawdowns (25-30% normal for 15% vol target with 2x leverage), keeps strategy flat 24%+ of days

**Signal Selection:**
8. EWMAVol and VolOfVol are non-directional — exclude from alpha signal combination
9. Signals at similar frequencies cancel (momentum 63d vs RSI 14d vs Bollinger 20d). Signals at different frequencies complement (momentum 252d + Bollinger 20d + reversal 5d).
10. Use CompositeMomentum (not individual XSMOMs) — avoids 4 redundant correlated signals, composite already captures multi-lookback momentum
