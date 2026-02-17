"""
Strategies basees sur les EMA (Exponential Moving Average)
Plus reactives que les SMA pour les actifs volatils (crypto, tech)
"""

import pandas as pd
import numpy as np
from src.indicators.technical import TechnicalIndicators
from src.strategies.base import BaseStrategy


class EMACrossover(BaseStrategy):
    """
    EMA Crossover - version rapide du MA Crossover classique.
    L'EMA pondere plus les prix recents, capte les retournements plus vite.
    """

    def __init__(self, fast_period: int = 12, slow_period: int = 26):
        super().__init__(f"EMA Crossover ({fast_period}/{slow_period})")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0

        signals['ema_fast'] = TechnicalIndicators.ema(data['close'], self.fast_period)
        signals['ema_slow'] = TechnicalIndicators.ema(data['close'], self.slow_period)

        signals['position'] = 0
        signals.loc[signals['ema_fast'] > signals['ema_slow'], 'position'] = 1
        signals.loc[signals['ema_fast'] < signals['ema_slow'], 'position'] = -1

        signals['signal'] = signals['position'].diff()
        return signals
