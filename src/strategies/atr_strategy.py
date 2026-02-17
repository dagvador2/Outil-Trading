"""
Strategie ATR Volatility Breakout (Keltner Channel)
Utilise l'ATR pour definir des canaux adaptatifs a la volatilite
Auto-adaptatif : les seuils s'ajustent a la volatilite de chaque actif
"""

import pandas as pd
import numpy as np
from src.indicators.technical import TechnicalIndicators
from src.strategies.base import BaseStrategy


class ATRBreakoutStrategy(BaseStrategy):
    """
    ATR Volatility Breakout / Keltner Channel
    LONG quand le prix casse au-dessus de SMA + N*ATR (breakout haussier)
    SHORT quand le prix casse en-dessous de SMA - N*ATR (breakout baissier)
    EXIT quand le prix revient a la SMA (retour a la moyenne)
    """

    def __init__(self, sma_period: int = 20, atr_period: int = 14,
                 atr_multiplier: float = 2.0):
        super().__init__(f"ATR Breakout ({sma_period}, {atr_multiplier}x)")
        self.sma_period = sma_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0

        sma = TechnicalIndicators.sma(data['close'], self.sma_period)
        atr = TechnicalIndicators.atr(
            data['high'], data['low'], data['close'], self.atr_period
        )

        upper_channel = sma + self.atr_multiplier * atr
        lower_channel = sma - self.atr_multiplier * atr

        signals['position'] = 0

        # LONG : prix casse au-dessus du canal superieur
        signals.loc[data['close'] > upper_channel, 'position'] = 1

        # SHORT : prix casse en-dessous du canal inferieur
        signals.loc[data['close'] < lower_channel, 'position'] = -1

        # FLAT : entre les canaux (retour a la moyenne)
        in_channel = (data['close'] >= lower_channel) & (data['close'] <= upper_channel)
        signals.loc[in_channel, 'position'] = 0

        signals['signal'] = signals['position'].diff()
        return signals
