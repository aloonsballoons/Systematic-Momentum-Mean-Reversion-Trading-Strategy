"""Risk management: volatility targeting and drawdown circuit breaker."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.config import RiskConfig


class RiskManager:
    """
    Risk overlays applied to portfolio weights.

    All overlays use .shift(1) to avoid look-ahead bias.
    """

    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg

    def apply(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply all risk overlays to target weights.

        Parameters
        ----------
        weights : raw target weights (date × ticker)
        returns : asset returns for vol estimation

        Returns
        -------
        Risk-adjusted weights
        """
        w = self.apply_vol_target(weights, returns)
        w = self.apply_drawdown_breaker(w, returns)
        return w

    def apply_vol_target(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Scale portfolio exposure so realized vol ≈ target.

        Estimates portfolio vol from the hypothetical portfolio returns
        using the target weights, then computes a scale factor to reach
        the target vol. Uses .shift(1) to avoid look-ahead bias.
        """
        w = weights.copy()

        # Estimate portfolio vol from hypothetical portfolio returns
        port_ret = (weights.shift(1) * returns).sum(axis=1)
        realized_vol = (
            port_ret.rolling(self.cfg.vol_lookback, min_periods=21).std()
            * np.sqrt(252)
        )

        # Scale factor: target / realized, with safety clamp
        # Use .shift(1) — we use yesterday's vol estimate for today's sizing
        scale = (self.cfg.vol_target / realized_vol.replace(0, np.nan)).shift(1)
        scale = scale.clip(0.1, 3.0).fillna(1.0)

        w = w.mul(scale, axis=0)

        # Re-enforce leverage limit after vol scaling
        gross = w.abs().sum(axis=1)
        leverage_scale = (self.cfg.max_leverage / gross).clip(upper=1.0)
        w = w.mul(leverage_scale, axis=0)

        return w

    def apply_drawdown_breaker(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Go flat when drawdown exceeds threshold, stay flat for flat_days.

        Iteratively tracks its own equity curve to avoid the two-pass
        feedback problem. Uses previous day's drawdown (no look-ahead).

        After each flat period ends, resets the high-water mark to the
        current equity level so the same drawdown doesn't re-trigger
        the breaker indefinitely.
        """
        w = weights.copy()

        equity = 1.0
        peak = 1.0
        flat_counter = 0
        grace_counter = 0  # Grace period after flat: breaker cannot re-trigger
        n_flat = 0

        grace_days = getattr(self.cfg, 'grace_days', 63)

        for i in range(len(w)):
            # Check if we're in a forced-flat period
            if flat_counter > 0:
                w.iloc[i] = 0.0
                flat_counter -= 1
                n_flat += 1
                # Reset peak when flat period ends so the same drawdown
                # doesn't immediately re-trigger the breaker
                if flat_counter == 0:
                    peak = equity
                    grace_counter = grace_days  # Start grace period
            elif grace_counter > 0:
                # During grace period: trade normally but don't check drawdown
                grace_counter -= 1
            else:
                # Check drawdown from previous state (no look-ahead)
                dd = equity / peak - 1 if peak > 0 else 0
                if dd < -self.cfg.max_drawdown:
                    w.iloc[i] = 0.0
                    flat_counter = self.cfg.flat_days - 1  # -1 because today counts
                    n_flat += 1

            # Update equity based on today's actual weights and returns
            day_ret = (w.iloc[i] * returns.iloc[i]).sum()
            equity *= (1 + day_ret)
            peak = max(peak, equity)

        if n_flat > 0:
            logger.info(
                f"Drawdown circuit breaker: flat for {n_flat} days "
                f"({n_flat / len(w) * 100:.1f}%)"
            )

        return w
