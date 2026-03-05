"""Portfolio constraints: position, sector, and leverage limits."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.config import PortfolioConfig


def apply_constraints(
    weights: pd.DataFrame,
    cfg: PortfolioConfig,
) -> pd.DataFrame:
    """
    Apply portfolio constraints to target weights.

    Constraints applied sequentially:
    1. Max position weight (per stock)
    2. Max leverage (gross exposure)

    Note: Sector constraints require sector mapping data and are applied
    when sector information is available.
    """
    w = weights.copy()

    # 1. Position limits: clip individual weights
    w = w.clip(-cfg.max_position_weight, cfg.max_position_weight)

    # 2. Leverage constraint: scale down if gross exposure exceeds limit
    gross = w.abs().sum(axis=1)
    scale = (cfg.max_leverage / gross).clip(upper=1.0)
    w = w.mul(scale, axis=0)

    return w


def apply_sector_constraints(
    weights: pd.DataFrame,
    sector_map: dict[str, str],
    max_sector_weight: float = 0.25,
) -> pd.DataFrame:
    """
    Apply sector exposure limits.

    Parameters
    ----------
    weights : target weights
    sector_map : {ticker: sector_name}
    max_sector_weight : max absolute weight per sector
    """
    w = weights.copy()

    for date_idx in w.index:
        row = w.loc[date_idx]
        sectors = pd.Series(sector_map).reindex(row.index)
        sector_exposure = row.groupby(sectors).sum()

        for sector, exposure in sector_exposure.items():
            if abs(exposure) > max_sector_weight:
                sector_tickers = sectors[sectors == sector].index
                scale = max_sector_weight / abs(exposure)
                w.loc[date_idx, sector_tickers] *= scale

    return w
