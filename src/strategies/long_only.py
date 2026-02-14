"""
Stratégies LONG-ONLY optimisées pour crypto (marchés haussiers)
Ne prennent PAS de positions SHORT
"""

import pandas as pd
import numpy as np
from src.indicators.technical import TechnicalIndicators
from src.strategies.base import BaseStrategy


class MovingAverageCrossoverLongOnly(BaseStrategy):
    """
    MA Crossover - LONG ONLY
    Signal d'achat: MA rapide > MA lente
    Signal de sortie: MA rapide < MA lente (pas de SHORT, juste cash)
    """

    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        super().__init__(f"MA Crossover LONG ({fast_period}/{slow_period})")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0

        # Calculer les moyennes mobiles
        signals['ma_fast'] = TechnicalIndicators.sma(data['close'], self.fast_period)
        signals['ma_slow'] = TechnicalIndicators.sma(data['close'], self.slow_period)

        # Position LONG uniquement
        signals['position'] = 0
        signals.loc[signals['ma_fast'] > signals['ma_slow'], 'position'] = 1  # LONG
        # Pas de -1 (SHORT), juste 0 ou 1

        # Détecter les changements
        signals['signal'] = signals['position'].diff()

        return signals


class RSIStrategyLongOnly(BaseStrategy):
    """
    RSI - LONG ONLY
    Signal d'achat: RSI < seuil de survente
    Signal de sortie: RSI > seuil de surachat (pas de SHORT)
    """

    def __init__(self, period: int = 14, oversold: int = 35, overbought: int = 80):
        super().__init__(f"RSI LONG ({period}, {oversold}/{overbought})")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0

        # Calculer le RSI
        signals['rsi'] = TechnicalIndicators.rsi(data['close'], self.period)

        # Position LONG uniquement
        signals['position'] = 0
        signals.loc[signals['rsi'] < self.oversold, 'position'] = 1  # LONG en survente
        # Pas de position SHORT, juste sortir en surachat

        # Détecter les changements
        signals['signal'] = signals['position'].diff()

        return signals


class TrendFollowingLongOnly(BaseStrategy):
    """
    Trend Following avec filtre MA 200 - LONG ONLY

    Règles:
    - LONG uniquement si prix > MA 200 (tendance haussière confirmée)
    - Signal d'entrée: MA rapide croise au-dessus MA lente + prix > MA 200
    - Signal de sortie: MA rapide croise en-dessous MA lente OU prix < MA 200
    """

    def __init__(self, fast_period: int = 20, slow_period: int = 50, trend_period: int = 200):
        super().__init__(f"Trend Following LONG ({fast_period}/{slow_period}/{trend_period})")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.trend_period = trend_period

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0

        # Calculer les MAs
        signals['ma_fast'] = TechnicalIndicators.sma(data['close'], self.fast_period)
        signals['ma_slow'] = TechnicalIndicators.sma(data['close'], self.slow_period)
        signals['ma_trend'] = TechnicalIndicators.sma(data['close'], self.trend_period)

        # Conditions pour LONG
        bullish_cross = signals['ma_fast'] > signals['ma_slow']
        above_trend = data['close'] > signals['ma_trend']

        # LONG uniquement si tendance haussière confirmée
        signals['position'] = 0
        signals.loc[bullish_cross & above_trend, 'position'] = 1

        # Détecter les changements
        signals['signal'] = signals['position'].diff()

        return signals


class BollingerLongOnly(BaseStrategy):
    """
    Bollinger Bands - LONG ONLY
    Signal d'achat: Prix touche bande basse (rebond attendu)
    Signal de sortie: Prix touche bande haute (sortir, pas SHORT)
    """

    def __init__(self, period: int = 20, num_std: float = 2):
        super().__init__(f"Bollinger LONG ({period}, {num_std}σ)")
        self.period = period
        self.num_std = num_std

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0

        # Calculer les Bollinger Bands
        bb = TechnicalIndicators.bollinger_bands(data['close'], self.period, self.num_std)
        signals['bb_upper'] = bb['upper']
        signals['bb_lower'] = bb['lower']
        signals['bb_middle'] = bb['middle']

        # LONG uniquement
        signals['position'] = 0

        # Acheter quand prix touche bande basse
        signals.loc[data['close'] <= signals['bb_lower'], 'position'] = 1

        # Sortir quand prix > middle (pas de SHORT sur bande haute)
        signals.loc[data['close'] >= signals['bb_middle'], 'position'] = 0

        # Détecter les changements
        signals['signal'] = signals['position'].diff()

        return signals


class HybridTrendMomentumLongOnly(BaseStrategy):
    """
    Stratégie hybride optimisée pour crypto - LONG ONLY

    Combine:
    - Trend: MA 50/200 pour direction
    - Momentum: RSI pour timing d'entrée
    - Volatilité: ATR pour stop-loss dynamique
    """

    def __init__(self, ma_fast: int = 50, ma_slow: int = 200, rsi_period: int = 14, rsi_level: int = 40):
        super().__init__("Hybrid Trend-Momentum LONG")
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.rsi_period = rsi_period
        self.rsi_level = rsi_level

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0

        # Indicateurs
        signals['ma_fast'] = TechnicalIndicators.sma(data['close'], self.ma_fast)
        signals['ma_slow'] = TechnicalIndicators.sma(data['close'], self.ma_slow)
        signals['rsi'] = TechnicalIndicators.rsi(data['close'], self.rsi_period)

        # Tendance haussière
        uptrend = (signals['ma_fast'] > signals['ma_slow']) & (data['close'] > signals['ma_fast'])

        # RSI en zone de pullback (pas trop suracheté)
        rsi_ok = signals['rsi'] < 70

        # LONG si tendance haussière ET RSI acceptable
        signals['position'] = 0
        signals.loc[uptrend & rsi_ok, 'position'] = 1

        # Détecter les changements
        signals['signal'] = signals['position'].diff()

        return signals
