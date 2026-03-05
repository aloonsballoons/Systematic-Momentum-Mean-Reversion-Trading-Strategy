"""Abstract base class for all features/alpha signals."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Feature(ABC):
    """
    Base class for alpha signal computation.

    All features must:
    - Use only past data (no look-ahead)
    - Return a DataFrame aligned with the input price index
    - Handle NaN gracefully (warmup period produces NaN)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this feature."""

    @abstractmethod
    def compute(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Compute the feature signal.

        Parameters
        ----------
        prices : DataFrame (date × ticker) of adjusted close prices
        returns : DataFrame (date × ticker) of daily returns

        Returns
        -------
        DataFrame (date × ticker) of signal values
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
