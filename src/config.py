"""Configuration management — loads config.yaml into frozen dataclasses."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Dataclass hierarchy — mirrors config.yaml structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataConfig:
    start_date: str = "2005-01-01"
    end_date: str = "2024-12-31"
    universe_file: str = "data/universes/sp500_constituents.csv"
    cache_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    min_price: float = 5.0
    min_adv: float = 1_000_000.0
    max_missing_pct: float = 0.10
    max_forward_fill: int = 5
    winsorize_returns: float = 0.25


@dataclass(frozen=True)
class MomentumConfig:
    xsmom_lookbacks: tuple[int, ...] = (63, 126, 252)
    xsmom_skip: int = 21
    tsmom_lookback: int = 252
    tsmom_vol_target: float = 0.15


@dataclass(frozen=True)
class MeanReversionConfig:
    bollinger_window: int = 20
    bollinger_num_std: float = 2.0
    rsi_window: int = 14
    reversal_window: int = 5
    reversal_vol_window: int = 21


@dataclass(frozen=True)
class VolatilityConfig:
    ewma_halflife: int = 21
    vol_of_vol_windows: tuple[int, ...] = (21, 63, 126)


@dataclass(frozen=True)
class FeaturesConfig:
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    mean_reversion: MeanReversionConfig = field(default_factory=MeanReversionConfig)
    volatility: VolatilityConfig = field(default_factory=VolatilityConfig)


@dataclass(frozen=True)
class SignalsConfig:
    combination_method: str = "equal_weight"
    rank_normalize: bool = True
    ic_lookback: int = 252


@dataclass(frozen=True)
class PortfolioConfig:
    method: str = "signal_proportional"
    max_position_weight: float = 0.05
    max_sector_weight: float = 0.25
    max_leverage: float = 1.0
    turnover_penalty: float = 0.001
    rebalance_frequency: str = "daily"
    weight_blend: float = 0.15  # Blend factor: w_t = blend*target + (1-blend)*w_{t-1}


@dataclass(frozen=True)
class ExecutionConfig:
    fixed_cost_bps: float = 5.0
    market_impact_bps: float = 10.0
    min_trade_pct: float = 0.001


@dataclass(frozen=True)
class RiskConfig:
    vol_target: float = 0.15
    vol_lookback: int = 126
    max_drawdown: float = 0.50
    flat_days: int = 21
    grace_days: int = 63
    max_leverage: float = 2.0


@dataclass(frozen=True)
class WalkForwardConfig:
    train_days: int = 756
    purge_days: int = 21
    test_days: int = 252


@dataclass(frozen=True)
class CrossValidationConfig:
    n_splits: int = 5
    purge_days: int = 21


@dataclass(frozen=True)
class BacktestConfig:
    warmup_days: int = 252
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    cross_validation: CrossValidationConfig = field(default_factory=CrossValidationConfig)


@dataclass(frozen=True)
class AnalyticsConfig:
    risk_free_rate: float = 0.02
    confidence_level: float = 0.95
    rolling_window: int = 252


@dataclass(frozen=True)
class StrategyConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    signals: SignalsConfig = field(default_factory=SignalsConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _build_nested(cls, raw: dict):
    """Recursively build frozen dataclasses from a raw dict."""
    import dataclasses
    import typing

    # Resolve string annotations from `from __future__ import annotations`
    type_hints = typing.get_type_hints(cls)

    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in raw:
            continue
        val = raw[f.name]
        resolved_type = type_hints.get(f.name)
        if isinstance(resolved_type, type) and dataclasses.is_dataclass(resolved_type):
            val = _build_nested(resolved_type, val)
        elif isinstance(val, list):
            val = tuple(val)
        kwargs[f.name] = val
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> StrategyConfig:
    """Load strategy configuration from YAML file."""
    if path is None:
        path = Path(os.environ.get("QUANT_CONFIG", "config.yaml"))
    path = Path(path)

    if not path.exists():
        return StrategyConfig()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    return _build_nested(StrategyConfig, raw)
