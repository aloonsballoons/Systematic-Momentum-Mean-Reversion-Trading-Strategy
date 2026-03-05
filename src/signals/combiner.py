"""Signal combination: equal-weight, IC-weight, PCA."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.config import SignalsConfig
from src.signals.normalizer import rank_normalize


class SignalCombiner:
    """Combines multiple alpha signals into a single composite signal."""

    def __init__(self, cfg: SignalsConfig):
        self.cfg = cfg

    def combine(
        self,
        signal_dict: dict[str, pd.DataFrame],
        returns: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Combine signals using the configured method.

        Parameters
        ----------
        signal_dict : {signal_name: DataFrame} of raw signals
        returns : forward returns (needed for IC-weighted combination)

        Returns
        -------
        DataFrame of combined signal values
        """
        # Normalize each signal
        normalized = {}
        for name, sig in signal_dict.items():
            if self.cfg.rank_normalize:
                normalized[name] = rank_normalize(sig)
            else:
                normalized[name] = sig

        method = self.cfg.combination_method
        if method == "equal_weight":
            return self._equal_weight(normalized)
        elif method == "ic_weight":
            return self._ic_weight(normalized, returns)
        elif method == "pca":
            return self._pca_combine(normalized)
        else:
            raise ValueError(f"Unknown combination method: {method}")

    def _equal_weight(self, signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Simple average — fewest degrees of freedom, most robust OOS."""
        stacked = pd.concat(signals.values(), keys=signals.keys())
        combined = stacked.groupby(level=1).mean()
        logger.info(f"Combined {len(signals)} signals with equal weight")
        return combined

    def _ic_weight(
        self,
        signals: dict[str, pd.DataFrame],
        returns: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """
        Weight signals by trailing information coefficient.

        WARNING: Additional degrees of freedom — use with caution.
        """
        if returns is None:
            logger.warning("IC-weight requires returns; falling back to equal weight")
            return self._equal_weight(signals)

        lookback = self.cfg.ic_lookback
        # Compute trailing IC for each signal
        ic_weights = {}
        for name, sig in signals.items():
            # Rolling rank IC (Spearman correlation with next-day returns)
            # Use .shift(1) on returns to align signal with forward return
            fwd = returns.shift(-1)
            ic = sig.corrwith(fwd, axis=1).rolling(lookback, min_periods=lookback // 2).mean()
            ic_weights[name] = ic.clip(lower=0)  # Only use positive IC signals

        # Normalize weights to sum to 1
        ic_df = pd.DataFrame(ic_weights)
        ic_sum = ic_df.sum(axis=1).replace(0, np.nan)
        ic_normalized = ic_df.div(ic_sum, axis=0).fillna(1 / len(signals))

        # Weight each signal by its IC
        combined = pd.DataFrame(0.0, index=list(signals.values())[0].index,
                                columns=list(signals.values())[0].columns)
        for name, sig in signals.items():
            combined = combined.add(sig.mul(ic_normalized[name], axis=0), fill_value=0)

        logger.info(f"Combined {len(signals)} signals with IC weighting (lookback={lookback})")
        return combined

    def _pca_combine(self, signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        First principal component of signals.

        WARNING: Additional degrees of freedom — use with caution.
        """
        from sklearn.decomposition import PCA

        # Stack all signals: each signal becomes a "feature" per date-ticker
        names = list(signals.keys())
        first = signals[names[0]]
        combined = pd.DataFrame(0.0, index=first.index, columns=first.columns)

        # For each date, run PCA across stocks × signals
        # Simplified: use rolling window approach on panel
        stacked = np.stack([signals[n].values for n in names], axis=-1)  # T × N × K
        T, N, K = stacked.shape

        pca = PCA(n_components=1)
        result = np.full((T, N), np.nan)

        for t in range(max(252, K), T):
            window = stacked[t - 252:t]  # 252 × N × K
            flat = window.reshape(-1, K)
            mask = ~np.isnan(flat).any(axis=1)
            if mask.sum() < K * 2:
                continue
            pca.fit(flat[mask])
            row = stacked[t]
            row_mask = ~np.isnan(row).any(axis=1)
            if row_mask.any():
                result[t, row_mask] = pca.transform(row[row_mask]).flatten()

        combined = pd.DataFrame(result, index=first.index, columns=first.columns)
        logger.info(f"Combined {len(signals)} signals with PCA")
        return combined
