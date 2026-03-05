"""Data cleaning: missing data handling, outlier treatment, alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.config import DataConfig


class DataCleaner:
    """Cleans price/volume matrices for backtest consumption."""

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg

    def clean(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Full cleaning pipeline.

        Returns
        -------
        prices : cleaned price matrix
        returns : cleaned return matrix (winsorized)
        """
        prices = self._drop_high_missing(prices)
        prices = self._forward_fill(prices)
        prices = self._drop_remaining_na_tickers(prices)

        returns = prices.pct_change()
        returns = self._winsorize_returns(returns)

        logger.info(
            f"Cleaned data: {prices.shape[1]} tickers, "
            f"{prices.shape[0]} days, "
            f"NaN% = {prices.isna().mean().mean():.4%}"
        )
        return prices, returns

    def _drop_high_missing(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Drop tickers with too much missing data."""
        missing_pct = prices.isna().mean()
        keep = missing_pct[missing_pct <= self.cfg.max_missing_pct].index
        dropped = prices.shape[1] - len(keep)
        if dropped > 0:
            logger.info(f"Dropped {dropped} tickers with >{self.cfg.max_missing_pct:.0%} missing")
        return prices[keep]

    def _forward_fill(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill up to max_forward_fill consecutive days."""
        return prices.ffill(limit=self.cfg.max_forward_fill)

    def _drop_remaining_na_tickers(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Drop tickers that still have NaN after forward-fill (e.g., late IPOs)."""
        # Only require data from the first non-NaN date per ticker
        first_valid = prices.apply(lambda s: s.first_valid_index())
        # Drop tickers where >5% is still NaN after their first valid date
        keep = []
        for col in prices.columns:
            if first_valid[col] is None:
                continue
            series = prices.loc[first_valid[col]:, col]
            if series.isna().mean() <= 0.05:
                keep.append(col)
        return prices[keep]

    def _winsorize_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Clip daily returns at ±winsorize_returns threshold."""
        cap = self.cfg.winsorize_returns
        clipped = returns.clip(-cap, cap)
        n_clipped = ((returns.abs() > cap) & returns.notna()).sum().sum()
        if n_clipped > 0:
            logger.info(f"Winsorized {n_clipped} return observations at ±{cap:.0%}")
        return clipped
