"""Walk-forward optimization with purge gap."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from loguru import logger

from src.config import WalkForwardConfig


@dataclass
class WalkForwardWindow:
    """A single walk-forward train/test window."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int


class WalkForwardOptimizer:
    """
    Walk-forward validation.

    3-year train → 21-day purge → 1-year test, rolling.
    Concatenated OOS equity curve is the primary reported result.
    """

    def __init__(self, cfg: WalkForwardConfig):
        self.cfg = cfg

    def generate_windows(self, n_periods: int) -> list[WalkForwardWindow]:
        """Generate walk-forward windows for the given number of periods."""
        windows = []
        train = self.cfg.train_days
        purge = self.cfg.purge_days
        test = self.cfg.test_days

        start = 0
        while start + train + purge + test <= n_periods:
            train_start = start
            train_end = start + train
            test_start = train_end + purge
            test_end = test_start + test

            if test_end > n_periods:
                test_end = n_periods

            windows.append(WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            ))

            # Roll forward by test period
            start += test

        logger.info(f"Generated {len(windows)} walk-forward windows")
        return windows

    def concatenate_oos(
        self,
        oos_returns: list[pd.Series],
    ) -> pd.Series:
        """Concatenate out-of-sample returns from all windows."""
        combined = pd.concat(oos_returns)
        # Remove duplicates if windows overlap
        combined = combined[~combined.index.duplicated(keep="first")]
        return combined.sort_index()
