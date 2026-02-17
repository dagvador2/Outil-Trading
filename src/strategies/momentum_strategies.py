"""
Strategies basees sur le momentum
- MACD Histogram Reversal : detecte les retournements d'histogramme (plus rapide que le crossover)
- RSI + Stochastic Convergence : double confirmation pour des entrees de haute qualite
"""

import pandas as pd
import numpy as np
from src.indicators.technical import TechnicalIndicators
from src.strategies.base import BaseStrategy


class MACDHistogramStrategy(BaseStrategy):
    """
    MACD Histogram Reversal
    Trade les retournements de l'histogramme MACD, qui precedent le crossover de 1-3 barres.
    LONG quand l'histogramme passe de negatif a positif avec momentum croissant
    SHORT quand l'histogramme passe de positif a negatif avec momentum decroissant
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(f"MACD Histogram ({fast}/{slow}/{signal})")
        self.fast = fast
        self.slow = slow
        self.signal_period = signal

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0

        macd_data = TechnicalIndicators.macd(
            data['close'], self.fast, self.slow, self.signal_period
        )
        hist = macd_data['histogram']

        signals['position'] = 0

        # Detecter les retournements avec 2 barres de confirmation
        hist_rising = (hist > hist.shift(1)) & (hist.shift(1) > hist.shift(2))
        hist_falling = (hist < hist.shift(1)) & (hist.shift(1) < hist.shift(2))

        # LONG : histogramme etait negatif et maintenant monte (2 barres de momentum haussier)
        signals.loc[(hist.shift(2) < 0) & hist_rising, 'position'] = 1

        # SHORT : histogramme etait positif et maintenant descend (2 barres de momentum baissier)
        signals.loc[(hist.shift(2) > 0) & hist_falling, 'position'] = -1

        signals['signal'] = signals['position'].diff()
        return signals


class RSIStochasticConvergence(BaseStrategy):
    """
    RSI + Stochastic Convergence
    Double confirmation : entre SEULEMENT quand RSI ET Stochastic sont d'accord
    Moins de trades mais meilleur win rate attendu
    """

    def __init__(self, rsi_period: int = 14, stoch_k: int = 14, stoch_d: int = 3,
                 rsi_oversold: int = 35, rsi_overbought: int = 65,
                 stoch_oversold: int = 20, stoch_overbought: int = 80):
        super().__init__(f"RSI+Stochastic ({rsi_period}, {stoch_k})")
        self.rsi_period = rsi_period
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.stoch_oversold = stoch_oversold
        self.stoch_overbought = stoch_overbought

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0

        rsi = TechnicalIndicators.rsi(data['close'], self.rsi_period)
        stoch = TechnicalIndicators.stochastic(
            data['high'], data['low'], data['close'],
            self.stoch_k, self.stoch_d
        )

        signals['position'] = 0

        # LONG : RSI en survente ET Stochastic en survente avec crossover haussier
        long_cond = (
            (rsi < self.rsi_oversold) &
            (stoch['k'] < self.stoch_oversold) &
            (stoch['k'] > stoch['d'])
        )
        signals.loc[long_cond, 'position'] = 1

        # SHORT : RSI en surachat ET Stochastic en surachat avec crossover baissier
        short_cond = (
            (rsi > self.rsi_overbought) &
            (stoch['k'] > self.stoch_overbought) &
            (stoch['k'] < stoch['d'])
        )
        signals.loc[short_cond, 'position'] = -1

        signals['signal'] = signals['position'].diff()
        return signals
