"""Purged K-fold cross-validation for signal evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger

from src.config import CrossValidationConfig


@dataclass
class CVFold:
    """Train and test indices for one CV fold."""

    train_indices: np.ndarray
    test_indices: np.ndarray


class PurgedKFoldCV:
    """
    Purged K-fold cross-validation.

    Inserts a purge gap between train and test sets to prevent
    information leakage from overlapping feature windows.

    Reference: de Prado (2018), "Advances in Financial Machine Learning"
    """

    def __init__(self, cfg: CrossValidationConfig):
        self.cfg = cfg

    def split(self, n_samples: int) -> list[CVFold]:
        """
        Generate purged K-fold splits.

        Parameters
        ----------
        n_samples : total number of time-series observations

        Returns
        -------
        List of CVFold with train/test indices
        """
        n_splits = self.cfg.n_splits
        purge = self.cfg.purge_days
        indices = np.arange(n_samples)
        fold_size = n_samples // n_splits

        folds = []
        for i in range(n_splits):
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n_samples)
            test_idx = indices[test_start:test_end]

            # Train: everything outside test ± purge gap
            purge_start = max(0, test_start - purge)
            purge_end = min(n_samples, test_end + purge)

            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[purge_start:purge_end] = False
            train_idx = indices[train_mask]

            folds.append(CVFold(train_indices=train_idx, test_indices=test_idx))

        logger.info(
            f"Generated {n_splits} purged K-fold splits "
            f"(purge={purge} days, fold_size={fold_size})"
        )
        return folds
