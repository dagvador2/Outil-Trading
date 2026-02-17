"""
Strategie basee sur le Stochastic Oscillator
Mesure la position du close par rapport au range haut-bas sur N periodes
Signal different du RSI : le Stochastic capture le momentum de prix relatif
"""

import pandas as pd
import numpy as np
from src.indicators.technical import TechnicalIndicators
from src.strategies.base import BaseStrategy


class StochasticStrategy(BaseStrategy):
    """
    Stochastic Oscillator - Mean Reversion
    LONG quand %K < oversold et croise au-dessus de %D (momentum haussier depuis survente)
    SHORT quand %K > overbought et croise en-dessous de %D (momentum baissier depuis surachat)
    """

    def __init__(self, k_period: int = 14, d_period: int = 3,
                 oversold: int = 20, overbought: int = 80):
        super().__init__(f"Stochastic ({k_period}, {oversold}/{overbought})")
        self.k_period = k_period
        self.d_period = d_period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0

        stoch = TechnicalIndicators.stochastic(
            data['high'], data['low'], data['close'],
            self.k_period, self.d_period
        )
        signals['stoch_k'] = stoch['k']
        signals['stoch_d'] = stoch['d']

        signals['position'] = 0

        # LONG : %K < oversold ET %K croise au-dessus de %D
        long_condition = (
            (signals['stoch_k'] < self.oversold) &
            (signals['stoch_k'] > signals['stoch_d'])
        )
        signals.loc[long_condition, 'position'] = 1

        # SHORT : %K > overbought ET %K croise en-dessous de %D
        short_condition = (
            (signals['stoch_k'] > self.overbought) &
            (signals['stoch_k'] < signals['stoch_d'])
        )
        signals.loc[short_condition, 'position'] = -1

        signals['signal'] = signals['position'].diff()
        return signals
