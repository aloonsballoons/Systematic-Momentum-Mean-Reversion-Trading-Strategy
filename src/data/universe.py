"""Point-in-time universe filtering to avoid survivorship bias."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import DataConfig


# A representative list of ~100 liquid S&P 500 large-cap names.
# Using a static list avoids survivorship bias from only picking current members.
DEFAULT_UNIVERSE = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AFL",
    "AIG", "AMAT", "AMD", "AMGN", "AMZN", "ANTM", "AON", "APD", "AVGO", "AXP",
    "BA", "BAC", "BDX", "BK", "BKNG", "BLK", "BMY", "BRK-B", "C", "CAT",
    "CCI", "CHTR", "CI", "CL", "CMCSA", "CME", "COF", "COP", "COST", "CRM",
    "CSCO", "CVS", "CVX", "D", "DD", "DE", "DHR", "DIS", "DOW", "DUK",
    "ECL", "EL", "EMR", "EOG", "EXC", "F", "FDX", "GD", "GE", "GILD",
    "GM", "GOOG", "GS", "HD", "HON", "IBM", "ICE", "INTC", "INTU", "ISRG",
    "ITW", "JNJ", "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA",
    "MCD", "MDLZ", "MDT", "MET", "MMC", "MMM", "MO", "MRK", "MS", "MSFT",
    "NEE", "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "PYPL",
    "QCOM", "RTX", "SBUX", "SCHW", "SHW", "SLB", "SO", "SPG", "T", "TGT",
    "TMO", "TMUS", "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WBA",
    "WFC", "WMT", "XOM",
]


class UniverseProvider:
    """Provides the investable universe, optionally filtered by price and volume."""

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg

    def get_tickers(self) -> list[str]:
        """Load ticker universe from CSV or use default list."""
        universe_path = Path(self.cfg.universe_file)
        if universe_path.exists():
            df = pd.read_csv(universe_path)
            col = df.columns[0]
            tickers = df[col].dropna().str.strip().tolist()
            logger.info(f"Loaded {len(tickers)} tickers from {universe_path}")
            return tickers

        logger.info(f"Using default universe of {len(DEFAULT_UNIVERSE)} tickers")
        return DEFAULT_UNIVERSE.copy()

    def filter_universe(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
    ) -> list[str]:
        """
        Filter tickers by minimum price and average daily volume.
        Uses trailing 63-day averages (point-in-time).
        """
        # Use last available data for filtering
        avg_price = prices.tail(63).mean()
        avg_volume = volumes.tail(63).mean()
        avg_dollar_volume = avg_price * avg_volume

        mask = (avg_price >= self.cfg.min_price) & (avg_dollar_volume >= self.cfg.min_adv)
        filtered = mask[mask].index.tolist()
        logger.info(
            f"Universe filter: {len(filtered)}/{prices.shape[1]} tickers pass "
            f"(min price=${self.cfg.min_price}, min ADV=${self.cfg.min_adv:,.0f})"
        )
        return filtered
