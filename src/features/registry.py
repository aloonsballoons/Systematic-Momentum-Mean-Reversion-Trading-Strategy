"""Feature registry: maps feature names to configured instances."""

from __future__ import annotations

from src.config import FeaturesConfig
from src.features.base import Feature
from src.features.mean_reversion import (
    BollingerMeanReversion,
    RSIMeanReversion,
    ShortTermReversal,
)
from src.features.momentum import CompositeMomentum, TSMOM, XSMOM
from src.features.volatility import EWMAVol, VolOfVol


def build_features(cfg: FeaturesConfig) -> list[Feature]:
    """Create all configured feature instances."""
    mom = cfg.momentum
    mr = cfg.mean_reversion
    vol = cfg.volatility

    features = [
        # Momentum (multi-lookback composite: 63, 126, 252 days)
        # Single composite avoids redundancy of individual XSMOM signals
        CompositeMomentum(mom),

        # Trend-following (252-day, distinct from cross-sectional momentum)
        TSMOM(lookback=mom.tsmom_lookback, vol_target=mom.tsmom_vol_target),

        # Bollinger mean-reversion (20-day horizon) — complements momentum
        # at a different frequency. Stocks trending UP that dip short-term
        # get reinforced by both momentum AND Bollinger.
        BollingerMeanReversion(window=mr.bollinger_window, num_std=mr.bollinger_num_std),

        # Short-term reversal (5-day horizon) — complementary to momentum
        ShortTermReversal(window=mr.reversal_window, vol_window=mr.reversal_vol_window),

        # Note: TSMOM excluded — binary signal (sign only), redundant with CompositeMomentum.
        # Note: Individual XSMOMs excluded — composite already captures them.
        # Note: RSI_MR excluded — too similar to Bollinger at 14d vs 20d.
        # Note: EWMAVol and VolOfVol excluded — non-directional.
    ]
    return features


FEATURE_REGISTRY: dict[str, type[Feature]] = {
    "xsmom": XSMOM,
    "tsmom": TSMOM,
    "composite_momentum": CompositeMomentum,
    "bollinger_mr": BollingerMeanReversion,
    "rsi_mr": RSIMeanReversion,
    "short_term_reversal": ShortTermReversal,
    "ewma_vol": EWMAVol,
    "vol_of_vol": VolOfVol,
}
