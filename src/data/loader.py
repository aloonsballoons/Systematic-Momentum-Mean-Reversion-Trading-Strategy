"""Download and cache market data via yfinance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger
from tqdm import tqdm

from src.config import DataConfig


class DataLoader:
    """Downloads OHLCV data for a list of tickers and caches as parquet."""

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.cache_dir = Path(cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker}.parquet"

    def download_ticker(self, ticker: str) -> pd.DataFrame | None:
        """Download a single ticker, cache as parquet. Returns None on failure."""
        cache = self._cache_path(ticker)
        if cache.exists():
            return pd.read_parquet(cache)

        try:
            df = yf.download(
                ticker,
                start=self.cfg.start_date,
                end=self.cfg.end_date,
                auto_adjust=True,
                progress=False,
            )
            if df.empty:
                logger.warning(f"No data for {ticker}")
                return None

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.index.name = "date"
            df.to_parquet(cache)
            return df
        except Exception as e:
            logger.warning(f"Failed to download {ticker}: {e}")
            return None

    def load_universe(self, tickers: list[str]) -> dict[str, pd.DataFrame]:
        """Download/load all tickers. Returns dict of ticker → OHLCV DataFrame."""
        data = {}
        for ticker in tqdm(tickers, desc="Loading data"):
            df = self.download_ticker(ticker)
            if df is not None:
                data[ticker] = df
        logger.info(f"Loaded {len(data)}/{len(tickers)} tickers")
        return data

    def build_price_matrix(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Build aligned adjusted close price matrix (date × ticker)."""
        prices = {}
        for ticker, df in data.items():
            if "Close" in df.columns:
                prices[ticker] = df["Close"]
        return pd.DataFrame(prices).sort_index()

    def build_volume_matrix(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Build aligned volume matrix (date × ticker)."""
        volumes = {}
        for ticker, df in data.items():
            if "Volume" in df.columns:
                volumes[ticker] = df["Volume"]
        return pd.DataFrame(volumes).sort_index()
