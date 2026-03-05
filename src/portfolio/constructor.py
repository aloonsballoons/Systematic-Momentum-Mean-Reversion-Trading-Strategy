"""Portfolio construction: signal-proportional, risk parity, mean-variance."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize

from src.config import PortfolioConfig
from src.portfolio.constraints import apply_constraints
from src.portfolio.risk_model import estimate_covariance


class PortfolioConstructor:
    """Converts signals into portfolio weights."""

    def __init__(self, cfg: PortfolioConfig):
        self.cfg = cfg

    def construct(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build portfolio weights from combined signals.

        Parameters
        ----------
        signals : combined alpha signal (date × ticker)
        returns : historical returns for covariance estimation

        Returns
        -------
        target weights : DataFrame (date × ticker)
        """
        method = self.cfg.method
        if method == "signal_proportional":
            weights = self._signal_proportional(signals)
        elif method == "risk_parity":
            weights = self._risk_parity(signals, returns)
        elif method == "mean_variance":
            weights = self._mean_variance(signals, returns)
        else:
            raise ValueError(f"Unknown portfolio method: {method}")

        weights = apply_constraints(weights, self.cfg)
        logger.info(f"Constructed portfolio using {method}")
        return weights

    def _signal_proportional(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Weights proportional to signal strength, normalized to sum to target leverage.

        This is the simplest and most robust approach.
        """
        # Normalize: weights sum of abs = 1 (dollar-neutral)
        abs_sum = signals.abs().sum(axis=1).replace(0, np.nan)
        weights = signals.div(abs_sum, axis=0)
        return weights.fillna(0)

    def _risk_parity(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Risk parity with signal tilt.

        Base allocation inversely proportional to volatility,
        then tilted by signal direction.
        """
        # Inverse-vol weights
        vol = returns.rolling(63, min_periods=30).std() * np.sqrt(252)
        inv_vol = (1 / vol.replace(0, np.nan)).fillna(0)

        # Tilt by signal sign and magnitude
        signal_sign = np.sign(signals)
        tilted = inv_vol * signal_sign * signals.abs()

        # Normalize
        abs_sum = tilted.abs().sum(axis=1).replace(0, np.nan)
        weights = tilted.div(abs_sum, axis=0)
        return weights.fillna(0)

    def _mean_variance(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Mean-variance optimization with turnover regularization.

        Solved periodically (weekly) to reduce computation,
        interpolated for daily weights.
        """
        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
        prev_w = np.zeros(len(signals.columns))
        rebal_freq = 5  # Weekly

        for i in range(252, len(signals), rebal_freq):
            date = signals.index[i]
            sig = signals.iloc[i].values
            tickers = signals.columns

            # Get recent returns for covariance
            recent_ret = returns.iloc[max(0, i - 252):i]
            valid_cols = recent_ret.dropna(axis=1, how="any").columns
            if len(valid_cols) < 5:
                weights.iloc[i:i + rebal_freq] = prev_w
                continue

            # Subset to valid columns
            mask = np.isin(tickers, valid_cols)
            n = mask.sum()
            sig_valid = sig[mask]
            ret_valid = recent_ret[valid_cols]

            cov = estimate_covariance(ret_valid, lookback=252)

            # Objective: -alpha'w + lambda * w'Σw + penalty * ||w - w_prev||
            lam = 1.0
            penalty = self.cfg.turnover_penalty
            prev_valid = prev_w[mask]

            def objective(w):
                port_var = w @ cov.values @ w
                alpha_return = sig_valid @ w
                turnover = np.sum((w - prev_valid) ** 2)
                return -alpha_return + lam * port_var + penalty * turnover

            # Constraints
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 0},  # Dollar neutral
            ]
            bounds = [(-self.cfg.max_position_weight, self.cfg.max_position_weight)] * n

            w0 = prev_valid if len(prev_valid) == n else np.zeros(n)
            result = minimize(objective, w0, method="SLSQP", bounds=bounds,
                              constraints=constraints, options={"maxiter": 200})

            opt_w = np.zeros(len(tickers))
            opt_w[mask] = result.x
            prev_w = opt_w

            end_idx = min(i + rebal_freq, len(signals))
            for j in range(i, end_idx):
                weights.iloc[j] = opt_w

        # Forward-fill any remaining rows
        weights = weights.replace(0, np.nan).ffill().fillna(0)
        return weights
